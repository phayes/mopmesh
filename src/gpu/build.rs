//! Mesh construction on the GPU from caller-owned position and triangle buffers.
use super::pipeline::{self, check_binding};
use super::{GpuError, GpuMesh, scan};
use crate::{BuildError, MeshId};
use wgpu::util::DeviceExt;

/// Marks an unset word in the error buffer.
const NONE: u32 = u32::MAX;

/// Words in the error buffer shared by all stages; decode with [`build_error`].
const ERROR_WORDS: usize = 11;

/// Mesh data already on the GPU, for [`GpuMeshBuilder::build`].
///
/// Both buffers need `STORAGE` usage and must hold at least `count` values and
/// at least 16 bytes, even for an empty mesh. The same rules apply as for
/// [`MeshBuilder`](crate::MeshBuilder): vertex and face IDs are zero-based
/// indices into these buffers.
#[derive(Clone, Copy, Debug)]
pub struct GpuMeshInput<'a> {
    /// One `[f32; 4]` position per vertex (WGSL `vec4<f32>`); the fourth lane
    /// is ignored.
    pub positions: &'a wgpu::Buffer,
    /// Three vertex IDs per face, stored as `[u32; 4]` (WGSL `vec4<u32>`);
    /// the fourth lane is ignored.
    pub triangles: &'a wgpu::Buffer,
    pub vertex_count: u32,
    pub face_count: u32,
}

/// Builds [`GpuMesh`] connectivity on the GPU from mesh data already in GPU
/// buffers, without a CPU [`Mesh`](crate::Mesh).
///
/// Creating a builder compiles its shaders, so create one per device and reuse
/// it. The resulting mesh answers the same queries as an uploaded one, but it
/// may split faces into patches differently, so the order of faces and edges
/// within a patch can differ from [`GpuMesh::upload`]. Building the same input
/// on the same device gives the same result every time.
pub struct GpuMeshBuilder {
    pipelines: GpuBuilder,
    patch_size: u32,
}

impl GpuMeshBuilder {
    /// Creates a builder for `device` with the default patch size of 512 faces.
    pub fn new(device: &wgpu::Device) -> Self {
        Self {
            pipelines: GpuBuilder::new(device),
            patch_size: 512,
        }
    }

    /// Sets the largest number of faces each patch owns. Must be at least 1;
    /// [`build`](Self::build) returns [`BuildError::InvalidPatchSize`] otherwise.
    pub fn patch_size(mut self, size: u32) -> Self {
        self.patch_size = size;
        self
    }

    /// Validates `input` and builds its connectivity on the GPU, submitting its
    /// own commands to `queue`.
    ///
    /// Blocks until the GPU has validated the input and counted the mesh's
    /// elements; the rest of the build is submitted but not waited for, and
    /// later submissions on `queue` see its result. Returns
    /// [`GpuBuildError::Input`] with the error [`MeshBuilder`](crate::MeshBuilder)
    /// reports for the same data, [`GpuBuildError::Gpu`] if an input buffer is
    /// too small or lacks `STORAGE` usage or the mesh exceeds device limits, or
    /// [`GpuBuildError::Device`] if waiting for the GPU fails.
    ///
    /// The new mesh has its own [`MeshId`] and no CPU attributes. Create
    /// attributes for it with [`GpuAttribute::from_buffer`](super::GpuAttribute::from_buffer),
    /// for example from `input.positions`, or [`GpuAttribute::new`](super::GpuAttribute::new).
    pub fn build(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &GpuMeshInput,
    ) -> Result<GpuMesh, GpuBuildError> {
        let built =
            self.pipelines
                .build(device, queue, input, GpuSplit::Morton, self.patch_size)?;
        Ok(GpuMesh {
            arena: built.arena,
            masks: built.masks,
            face_count: input.face_count,
            vertex_count: input.vertex_count,
            edge_count: built.summary.edges,
            patch_count: built.split.patch_count,
            mesh: MeshId::next(),
        })
    }
}

/// Results of the element stage.
pub(super) struct ElementStage {
    /// [`ERROR_WORDS`] words that every stage writes its errors into. Decode
    /// with [`build_error`] after all stages have run.
    pub errors: wgpu::Buffer,
    /// `V + 1` row offsets, then the `3F` corner IDs of each vertex, in an order
    /// that may change between runs. Offsets index the corner region. Valid
    /// only if `errors` reports no error.
    pub rows: wgpu::Buffer,
}

/// Results of the edge stage. Valid only if the shared error buffer reports no
/// error.
pub(super) struct EdgeStage {
    /// `V + 1` words: the first edge ID whose smaller endpoint is each vertex,
    /// then the edge count. Edge IDs follow ascending (smaller, larger) endpoints.
    pub edge_offsets: wgpu::Buffer,
    /// Per edge: `[smaller vertex, larger vertex, first face, second face]`,
    /// faces in input order, the second `u32::MAX` on a boundary. Sized for
    /// `3F` edges; only the first edge-count records are written.
    pub edges: wgpu::Buffer,
    /// Per corner (face `c / 3`, slot `c % 3`): `[edge ID, reversed 0/1]` for
    /// the edge from that corner's vertex to the next corner's vertex.
    pub face_edges: wgpu::Buffer,
}

/// The bind-group layout of one build shader module and its entry points'
/// pipelines, in the order the entry points were named.
struct Stage<const N: usize> {
    layout: wgpu::BindGroupLayout,
    pipelines: [wgpu::ComputePipeline; N],
}

impl<const N: usize> Stage<N> {
    /// Compiles `wgsl` and creates one pipeline per entry point. `writable`
    /// gives, per storage binding after the uniform at binding 0, whether the
    /// shader writes it.
    fn new(
        device: &wgpu::Device,
        name: &str,
        wgsl: &'static str,
        writable: &[bool],
        entries: [&str; N],
    ) -> Self {
        let module = pipeline::shader_module(device, &format!("MOPMesh build {name}"), wgsl);
        let bindings: Vec<_> = std::iter::once(wgpu::BufferBindingType::Uniform)
            .chain(
                writable
                    .iter()
                    .map(|&write| wgpu::BufferBindingType::Storage { read_only: !write }),
            )
            .collect();
        let (layout, pipeline_layout) =
            pipeline::buffer_layout(device, &format!("MOPMesh build {name} layout"), &bindings);
        let pipelines = entries.map(|entry| {
            let label = format!("MOPMesh build {entry}");
            pipeline::compute_pipeline(device, &label, &module, entry, Some(&pipeline_layout))
        });
        Self { layout, pipelines }
    }

    fn bind(&self, device: &wgpu::Device, buffers: &[&wgpu::Buffer]) -> wgpu::BindGroup {
        pipeline::bind_buffers(device, "MOPMesh build bindings", &self.layout, buffers)
    }
}

/// Cached pipelines for GPU mesh construction. Create once per device.
pub(super) struct GpuBuilder {
    scan: scan::ScanPipeline,
    elements: Stage<6>,
    connectivity: Stage<6>,
    members: Stage<3>,
    rows: Stage<4>,
    local: Stage<1>,
    arena: Stage<4>,
    split: Stage<4>,
    radix: Stage<2>,
}

/// How the GPU build splits faces into patches.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum GpuSplit {
    /// Consecutive runs of `patch_size` faces in input order.
    InputOrder,
    /// Consecutive runs of `patch_size` faces in Morton order of their centroids.
    Morton,
}

/// The owning patch of each face, from [`GpuBuilder::encode_split`].
pub(super) struct FaceSplit {
    /// One `u32` patch ID per face (at least 16 bytes).
    pub face_patch: wgpu::Buffer,
    pub patch_count: u32,
}

impl FaceSplit {
    pub(super) fn partition(&self) -> Partition<'_> {
        Partition {
            face_patch: &self.face_patch,
            patch_count: self.patch_count,
        }
    }
}

impl GpuBuilder {
    pub(super) fn new(device: &wgpu::Device) -> Self {
        let elements = Stage::new(
            device,
            "elements",
            include_str!(concat!(env!("OUT_DIR"), "/build_elements.wgsl")),
            &[false, false, true, true, true],
            [
                "check_positions",
                "check_faces",
                "describe_face_error",
                "count_corners",
                "store_offsets",
                "fill_corners",
            ],
        );
        let connectivity = Stage::new(
            device,
            "connectivity",
            include_str!(concat!(env!("OUT_DIR"), "/build_connectivity.wgsl")),
            &[false, false, true, true, true, true, true],
            [
                "find_duplicates",
                "group_corners",
                "count_edges",
                "number_edges",
                "describe_edge_error",
                "check_fans",
            ],
        );
        let members = Stage::new(
            device,
            "patch members",
            include_str!(concat!(env!("OUT_DIR"), "/build_patch_members.wgsl")),
            &[false, false, false, false, false, true, true, true],
            ["collect_vertex_patches", "count_members", "write_members"],
        );
        let rows = Stage::new(
            device,
            "patch rows",
            include_str!(concat!(env!("OUT_DIR"), "/build_patch_rows.wgsl")),
            &[false, true, true, true],
            ["count_rows", "store_offsets", "fill_rows", "sort_rows"],
        );
        let local = Stage::new(
            device,
            "patch local",
            include_str!(concat!(env!("OUT_DIR"), "/build_patch_local.wgsl")),
            &[false, false, false, false, false, false, true, true],
            ["derive_local"],
        );
        let arena = Stage::new(
            device,
            "arena",
            include_str!(concat!(env!("OUT_DIR"), "/build_arena.wgsl")),
            &[false, false, false, true, true, true, true],
            [
                "write_header",
                "size_patches",
                "write_records",
                "write_arena",
            ],
        );
        let split = Stage::new(
            device,
            "split",
            include_str!(concat!(env!("OUT_DIR"), "/build_split.wgsl")),
            &[false, false, true, true, true],
            [
                "split_input_order",
                "centroid_bounds",
                "morton_keys",
                "assign_patches",
            ],
        );
        let radix = Stage::new(
            device,
            "radix sort",
            include_str!(concat!(env!("OUT_DIR"), "/radix_sort.wgsl")),
            &[false, true, true],
            ["count_digits", "scatter"],
        );
        Self {
            scan: scan::ScanPipeline::new(device),
            elements,
            connectivity,
            members,
            rows,
            local,
            arena,
            split,
            radix,
        }
    }

    /// Records the face split: each face's owning patch, with at most
    /// `patch_size` faces per patch and `ceil(F / patch_size)` patches. The
    /// result is the same on every run on one device, but may differ from the
    /// CPU's split of the same method. Returns [`GpuError::LimitsExceeded`]
    /// before recording if a buffer or dispatch is too large; panics if
    /// `patch_size` is zero.
    pub(super) fn encode_split(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuMeshInput,
        method: GpuSplit,
        patch_size: u32,
    ) -> Result<FaceSplit, GpuError> {
        // Check output, scratch and dispatch limits.
        const RESOURCE: &str = "GPU build face split";
        let faces = input.face_count;
        let limits = device.limits();
        check_binding(&limits, (8 * u64::from(faces)).max(16), RESOURCE)?;
        let blocks = pipeline::workgroup_grid(&limits, faces, RESOURCE)?;
        let histogram_plan = scan::ScanPlan::new(
            16u32
                .checked_mul(faces.div_ceil(pipeline::WORKGROUP_SIZE))
                .ok_or(GpuError::LimitsExceeded { resource: RESOURCE })?,
            &limits,
        )?;
        let split = FaceSplit {
            face_patch: pipeline::output_buffer(
                device,
                "MOPMesh build face patches",
                (4 * u64::from(faces)).max(16),
            ),
            patch_count: faces.div_ceil(patch_size),
        };
        if faces == 0 {
            return Ok(split);
        }

        // Bind the inputs; a mesh without vertices has already failed validation.
        let positions = match input.vertex_count {
            0 => pipeline::output_buffer(device, "MOPMesh build placeholder", 16),
            _ => input.positions.clone(),
        };
        let params = uniform(device, &[faces, patch_size]);
        let bounds = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh build centroid bounds"),
            contents: bytemuck::cast_slice(&[u32::MAX, u32::MAX, u32::MAX, 0, 0, 0]),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let pair_bytes = 8 * u64::from(faces);
        let pairs = [
            pipeline::output_buffer(device, "MOPMesh build sort pairs", pair_bytes),
            pipeline::output_buffer(device, "MOPMesh build sort pairs", pair_bytes),
        ];
        let bindings = self.split.bind(
            device,
            &[
                &params,
                &positions,
                input.triangles,
                &bounds,
                &pairs[0],
                &split.face_patch,
            ],
        );
        let [input_order, centroid_bounds, morton_keys, assign] = &self.split.pipelines;
        let run = |encoder: &mut wgpu::CommandEncoder, label, pipeline, groups| {
            pipeline::dispatch_grid(encoder, label, pipeline, &bindings, groups)
        };
        if method == GpuSplit::InputOrder {
            run(
                encoder,
                "MOPMesh build split input order",
                input_order,
                blocks,
            );
            return Ok(split);
        }

        // Morton: bound the centroids and key each face by its code.
        run(
            encoder,
            "MOPMesh build centroid bounds",
            centroid_bounds,
            blocks,
        );
        run(encoder, "MOPMesh build Morton keys", morton_keys, blocks);

        // Sort by code, then give consecutive runs of the sorted faces a patch.
        self.encode_sort(device, encoder, &pairs, faces, blocks, &histogram_plan);
        run(encoder, "MOPMesh build assign patches", assign, blocks);
        Ok(split)
    }

    /// Records a stable sort of the first `count` `[key, value]` pairs in
    /// `pairs[0]` by key, using `pairs[1]` as scratch; the result is in
    /// `pairs[0]`. `grid` must come from [`pipeline::workgroup_grid`] for
    /// `count`, and `histogram_plan` must be a scan plan for `16 * ceil(count /
    /// 64)` values, both checked against the device limits.
    fn encode_sort(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pairs: &[wgpu::Buffer; 2],
        count: u32,
        grid: [u32; 2],
        histogram_plan: &scan::ScanPlan,
    ) {
        let blocks = count.div_ceil(pipeline::WORKGROUP_SIZE);
        let histogram = pipeline::output_buffer(
            device,
            "MOPMesh build sort histogram",
            histogram_plan.byte_size(),
        );

        // Four bits per pass; an even number of passes ends in pairs[0].
        let [count_digits, scatter] = &self.radix.pipelines;
        for pass in 0..8usize {
            let (source, sorted) = (&pairs[pass % 2], &pairs[1 - pass % 2]);
            let params = uniform(device, &[count, 4 * pass as u32, blocks]);
            let sort = self
                .radix
                .bind(device, &[&params, source, sorted, &histogram]);
            let label = "MOPMesh build sort count";
            pipeline::dispatch_grid(encoder, label, count_digits, &sort, grid);
            self.scan
                .encode(device, encoder, &histogram, histogram_plan);
            let label = "MOPMesh build sort scatter";
            pipeline::dispatch_grid(encoder, label, scatter, &sort, grid);
        }
    }

    /// Records the element checks and the vertex→corner rows.
    /// Returns [`GpuError::LimitsExceeded`] before recording if a buffer or
    /// dispatch is too large. Panics if an input buffer is smaller than its count
    /// requires (16 bytes per vertex or face).
    pub(super) fn encode_elements(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuMeshInput,
    ) -> Result<ElementStage, GpuError> {
        // Check input sizes, then output, scratch and dispatch limits.
        const RESOURCE: &str = "GPU build corner rows";
        let (vertices, faces) = (input.vertex_count, input.face_count);
        check_input(input.positions, vertices, "GPU build positions")?;
        check_input(input.triangles, faces, "GPU build triangles")?;
        let limits = device.limits();
        let row_words = u64::from(vertices) + 1 + 3 * u64::from(faces);
        check_binding(&limits, row_words * 4, RESOURCE)?;
        check_binding(&limits, input.positions.size(), "GPU build positions")?;
        check_binding(&limits, input.triangles.size(), "GPU build triangles")?;
        let scan_plan = scan::ScanPlan::new(vertices + 1, &limits)?;
        let face_groups = pipeline::workgroup_grid(&limits, faces, RESOURCE)?;
        let row_groups = pipeline::workgroup_grid(&limits, vertices + 1, RESOURCE)?;
        let vertex_groups = pipeline::workgroup_grid(&limits, vertices, RESOURCE)?;

        // Errors start unset; rows start as zeros, which is correct for an empty mesh.
        let stage = ElementStage {
            errors: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("MOPMesh build errors"),
                contents: bytemuck::cast_slice(&[NONE; ERROR_WORDS]),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            }),
            rows: pipeline::output_buffer(device, "MOPMesh build corner rows", row_words * 4),
        };
        if vertices == 0 && faces == 0 {
            return Ok(stage);
        }

        // Bind the inputs, substituting a placeholder for an empty one.
        let placeholder = |buffer: &wgpu::Buffer, count: u32| match count {
            0 => pipeline::output_buffer(device, "MOPMesh build placeholder", 16),
            _ => buffer.clone(),
        };
        let positions = placeholder(input.positions, vertices);
        let triangles = placeholder(input.triangles, faces);
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh build sizes"),
            contents: bytemuck::cast_slice(&[vertices, faces]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let cursors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh build corner cursors"),
            size: scan_plan.byte_size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bindings = self.elements.bind(
            device,
            &[
                &params,
                &positions,
                &triangles,
                &stage.errors,
                &cursors,
                &stage.rows,
            ],
        );

        // Check positions and faces, then record why the first failing face failed.
        let [
            check_positions,
            check_faces,
            describe,
            count,
            store_offsets,
            fill,
        ] = &self.elements.pipelines;
        let run = |encoder: &mut wgpu::CommandEncoder, label, pipeline, groups| {
            pipeline::dispatch_grid(encoder, label, pipeline, &bindings, groups)
        };
        run(
            encoder,
            "MOPMesh build check positions",
            check_positions,
            vertex_groups,
        );
        run(
            encoder,
            "MOPMesh build check faces",
            check_faces,
            face_groups,
        );
        run(
            encoder,
            "MOPMesh build describe face error",
            describe,
            [1, 1],
        );

        // Count corners per vertex, scan the counts into row offsets, then fill.
        run(encoder, "MOPMesh build count corners", count, face_groups);
        self.scan.encode(device, encoder, &cursors, &scan_plan);
        run(
            encoder,
            "MOPMesh build store offsets",
            store_offsets,
            row_groups,
        );
        run(encoder, "MOPMesh build fill corners", fill, face_groups);
        Ok(stage)
    }

    /// Records duplicate-face and edge checks, numbers the edges, and checks
    /// each vertex's fan. `elements` must come from
    /// [`GpuBuilder::encode_elements`] for the same `input`. Returns
    /// [`GpuError::LimitsExceeded`] before recording if a buffer or dispatch is
    /// too large.
    pub(super) fn encode_connectivity(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuMeshInput,
        elements: &ElementStage,
    ) -> Result<EdgeStage, GpuError> {
        // Check output, scratch and dispatch limits.
        const RESOURCE: &str = "GPU build edges";
        let (vertices, faces) = (input.vertex_count, input.face_count);
        let corners = u64::from(faces) * 3;
        let limits = device.limits();
        check_binding(&limits, (corners * 16).max(16), RESOURCE)?;
        check_binding(&limits, (corners * 8).max(16), RESOURCE)?;
        let scan_plan = scan::ScanPlan::new(vertices + 1, &limits)?;
        let corner_groups = pipeline::workgroup_grid(&limits, corners as u32, RESOURCE)?;
        let face_groups = pipeline::workgroup_grid(&limits, faces, RESOURCE)?;
        let vertex_groups = pipeline::workgroup_grid(&limits, vertices, RESOURCE)?;

        // Edge offsets start as zeros, which is correct for a mesh without faces.
        let stage = EdgeStage {
            edge_offsets: pipeline::output_buffer(
                device,
                "MOPMesh build edge offsets",
                scan_plan.byte_size(),
            ),
            edges: pipeline::output_buffer(device, "MOPMesh build edges", (corners * 16).max(16)),
            face_edges: pipeline::output_buffer(
                device,
                "MOPMesh build face edges",
                (corners * 8).max(16),
            ),
        };
        // Without vertices, every face has already failed the element checks.
        // Without faces, the fan check still reports isolated vertices.
        if vertices == 0 {
            return Ok(stage);
        }
        let triangles = match faces {
            0 => pipeline::output_buffer(device, "MOPMesh build placeholder", 16),
            _ => input.triangles.clone(),
        };
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh build sizes"),
            contents: bytemuck::cast_slice(&[vertices, faces]),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let leaders = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh build edge leaders"),
            size: (corners * 4).max(16),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bindings = self.connectivity.bind(
            device,
            &[
                &params,
                &triangles,
                &elements.rows,
                &elements.errors,
                &leaders,
                &stage.edge_offsets,
                &stage.edges,
                &stage.face_edges,
            ],
        );
        let [duplicates, group, count, number, describe, fans] = &self.connectivity.pipelines;
        let run = |encoder: &mut wgpu::CommandEncoder, label, pipeline, groups| {
            pipeline::dispatch_grid(encoder, label, pipeline, &bindings, groups)
        };

        // Find duplicate faces, group corners by edge, and record the first edge error.
        run(
            encoder,
            "MOPMesh build find duplicates",
            duplicates,
            face_groups,
        );
        run(encoder, "MOPMesh build group corners", group, corner_groups);
        run(
            encoder,
            "MOPMesh build describe edge error",
            describe,
            [1, 1],
        );

        // Count edges per smaller endpoint, scan into first IDs, then number and link edges.
        run(encoder, "MOPMesh build count edges", count, vertex_groups);
        self.scan
            .encode(device, encoder, &stage.edge_offsets, &scan_plan);
        run(encoder, "MOPMesh build number edges", number, corner_groups);

        // Check that every vertex is used and that its faces form one fan.
        run(encoder, "MOPMesh build check fans", fans, vertex_groups);
        Ok(stage)
    }

    /// Records the first phase's last step: each patch's member counts. Then
    /// copies the error words and the totals into [`MemberCounts::summary`],
    /// which the caller reads back after submitting, to size the second phase.
    /// Returns [`GpuError::LimitsExceeded`] before recording if a buffer or
    /// dispatch is too large.
    pub(super) fn encode_member_counts(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuMeshInput,
        elements: &ElementStage,
        connectivity: &EdgeStage,
        partition: &Partition,
    ) -> Result<MemberCounts, GpuError> {
        // Check scratch and dispatch limits. Count slots cover every face, the
        // 3F edge bound and every vertex, plus one for the total.
        const RESOURCE: &str = "GPU build patch members";
        let (vertices, faces) = (input.vertex_count, input.face_count);
        let limits = device.limits();
        let slots = 4 * u64::from(faces) + u64::from(vertices) + 1;
        let slots =
            u32::try_from(slots).map_err(|_| GpuError::LimitsExceeded { resource: RESOURCE })?;
        let set_words = u64::from(vertices) + 3 * u64::from(faces);
        check_binding(&limits, (set_words * 4).max(16), RESOURCE)?;
        let scan_plan = scan::ScanPlan::new(slots, &limits)?;
        let slot_groups = pipeline::workgroup_grid(&limits, slots, RESOURCE)?;
        let vertex_groups = pipeline::workgroup_grid(&limits, vertices, RESOURCE)?;

        let counts = MemberCounts {
            vertex_patches: pipeline::output_buffer(
                device,
                "MOPMesh build vertex patches",
                (set_words * 4).max(16),
            ),
            counts: pipeline::output_buffer(
                device,
                "MOPMesh build member counts",
                scan_plan.byte_size(),
            ),
            summary: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("MOPMesh build summary"),
                size: 4 * SUMMARY_WORDS as u64,
                usage: wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
        };

        // Collect each vertex's owning patches, count members, and scan the counts.
        if vertices > 0 && faces > 0 {
            let params = self.member_params(device, input, partition);
            let placeholder = pipeline::output_buffer(device, "MOPMesh build placeholder", 16);
            let bindings = self.members.bind(
                device,
                &[
                    &params,
                    input.triangles,
                    &elements.rows,
                    partition.face_patch,
                    &connectivity.edges,
                    &connectivity.edge_offsets,
                    &counts.vertex_patches,
                    &counts.counts,
                    &placeholder,
                ],
            );
            let [collect, count, _] = &self.members.pipelines;
            let label = "MOPMesh build collect vertex patches";
            pipeline::dispatch_grid(encoder, label, collect, &bindings, vertex_groups);
            let label = "MOPMesh build count members";
            pipeline::dispatch_grid(encoder, label, count, &bindings, slot_groups);
            self.scan
                .encode(device, encoder, &counts.counts, &scan_plan);
        }

        // Copy the errors, the edge count, and the member prefixes at the face,
        // edge and vertex boundaries into the summary.
        let word = |i: u64| 4 * i;
        let (f, v) = (u64::from(faces), u64::from(vertices));
        encoder.copy_buffer_to_buffer(
            &elements.errors,
            0,
            &counts.summary,
            0,
            word(ERROR_WORDS as u64),
        );
        let at = ERROR_WORDS as u64;
        encoder.copy_buffer_to_buffer(
            &connectivity.edge_offsets,
            word(v),
            &counts.summary,
            word(at),
            4,
        );
        for (i, slot) in [f, 4 * f, 4 * f + v].into_iter().enumerate() {
            let to = word(at + 1 + i as u64);
            encoder.copy_buffer_to_buffer(&counts.counts, word(slot), &counts.summary, to, 4);
        }
        Ok(counts)
    }

    /// Records the second phase: each patch's sorted face, edge and vertex
    /// lists, patch-local connectivity and owners. `first` must hold the first
    /// phase's results for the same inputs, with a summary that reports no
    /// error. Returns [`GpuError::LimitsExceeded`] before recording if a buffer
    /// or dispatch is too large.
    pub(super) fn encode_patches(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuMeshInput,
        first: &FirstPhase,
    ) -> Result<PatchStage, GpuError> {
        let FirstPhase {
            elements,
            connectivity,
            partition,
            counts,
            summary,
        } = first;
        // Check output, scratch and dispatch limits.
        const RESOURCE: &str = "GPU build patch contents";
        let limits = device.limits();
        let too_large = GpuError::LimitsExceeded { resource: RESOURCE };
        let (fm, em, vm) = (
            u64::from(summary.face_members),
            u64::from(summary.edge_members),
            u64::from(summary.vertex_members),
        );
        let members = fm + em + vm;
        let members_u32 = u32::try_from(members).map_err(|_| too_large)?;
        let rows = 3 * u64::from(partition.patch_count);
        let rows_u32 = u32::try_from(rows).map_err(|_| too_large)?;
        let grouped_words = rows + 1 + members;
        let local_words = 2 * em + 6 * fm;
        for bytes in [8 * members, 4 * grouped_words, 4 * local_words] {
            check_binding(&limits, bytes.max(16), RESOURCE)?;
        }
        let row_plan = scan::ScanPlan::new(rows_u32 + 1, &limits)?;
        let member_groups = pipeline::workgroup_grid(&limits, members_u32, RESOURCE)?;
        let row_groups = pipeline::workgroup_grid(&limits, rows_u32 + 1, RESOURCE)?;
        // The row sort runs one 64-wide workgroup per row.
        let sort_invocations = rows_u32
            .checked_mul(pipeline::WORKGROUP_SIZE)
            .ok_or(too_large)?;
        let sort_groups = pipeline::workgroup_grid(&limits, sort_invocations, RESOURCE)?;
        let slots = 4 * input.face_count + input.vertex_count + 1;
        let slot_groups = pipeline::workgroup_grid(&limits, slots, RESOURCE)?;

        let stage = PatchStage {
            grouped: pipeline::output_buffer(device, "MOPMesh build patch rows", 4 * grouped_words),
            local_ids: pipeline::output_buffer(
                device,
                "MOPMesh build patch local IDs",
                (4 * local_words).max(16),
            ),
            owners: pipeline::output_buffer(
                device,
                "MOPMesh build patch owners",
                (8 * members).max(16),
            ),
        };
        if members == 0 {
            return Ok(stage);
        }

        // Write each patch member at its counted position.
        let member_list = pipeline::output_buffer(device, "MOPMesh build members", 8 * members);
        let params = self.member_params(device, input, partition);
        let bindings = self.members.bind(
            device,
            &[
                &params,
                input.triangles,
                &elements.rows,
                partition.face_patch,
                &connectivity.edges,
                &connectivity.edge_offsets,
                &counts.vertex_patches,
                &counts.counts,
                &member_list,
            ],
        );
        let write = &self.members.pipelines[2];
        let label = "MOPMesh build write members";
        pipeline::dispatch_grid(encoder, label, write, &bindings, slot_groups);

        // Group members by row: count, scan into offsets, fill, then sort each row.
        let params = uniform(device, &[rows_u32, members_u32]);
        let cursors = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh build patch row cursors"),
            size: row_plan.byte_size(),
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let unsorted = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh build unsorted patch rows"),
            size: 4 * members,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let bindings = self.rows.bind(
            device,
            &[&params, &member_list, &cursors, &stage.grouped, &unsorted],
        );
        let [count, store, fill, sort] = &self.rows.pipelines;
        let run = |encoder: &mut wgpu::CommandEncoder, label, pipeline, groups| {
            pipeline::dispatch_grid(encoder, label, pipeline, &bindings, groups)
        };
        run(
            encoder,
            "MOPMesh build count patch rows",
            count,
            member_groups,
        );
        self.scan.encode(device, encoder, &cursors, &row_plan);
        run(encoder, "MOPMesh build store patch rows", store, row_groups);
        run(
            encoder,
            "MOPMesh build fill patch rows",
            fill,
            member_groups,
        );
        run(encoder, "MOPMesh build sort patch rows", sort, sort_groups);

        // Derive local connectivity and owners for every member.
        let params = uniform(
            device,
            &[
                input.vertex_count,
                partition.patch_count,
                members_u32,
                summary.face_members,
                summary.edge_members,
                0,
                0,
                0,
            ],
        );
        let bindings = self.local.bind(
            device,
            &[
                &params,
                &stage.grouped,
                partition.face_patch,
                &counts.vertex_patches,
                &elements.rows,
                &connectivity.edges,
                &connectivity.face_edges,
                &stage.local_ids,
                &stage.owners,
            ],
        );
        let label = "MOPMesh build derive patch local IDs";
        pipeline::dispatch_grid(
            encoder,
            label,
            &self.local.pipelines[0],
            &bindings,
            member_groups,
        );
        Ok(stage)
    }

    /// Records commands that turn the prepared patch data into connectivity and
    /// mask buffers for GPU queries. For the same mesh and partition, the result
    /// matches [`super::GpuMesh::upload`], with possible extra zero words at the
    /// end of the mask buffer. Both buffers support `STORAGE | COPY_SRC`.
    /// Returns [`GpuError::LimitsExceeded`] before recording commands if a
    /// required buffer or dispatch is too large.
    pub(super) fn encode_arena(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        input: &GpuMeshInput,
        partition: &Partition,
        summary: &Summary,
        patches: &PatchStage,
    ) -> Result<[wgpu::Buffer; 2], GpuError> {
        use super::patch_layout::{HEADER_WORDS, RECORD_WORDS};
        // Compute exact connectivity storage and allow for mask rounding in
        // every patch. Check buffer and dispatch limits before recording work.
        const RESOURCE: &str = "connectivity arena";
        const MASKS: &str = "connectivity masks";
        let limits = device.limits();
        let too_large = GpuError::LimitsExceeded { resource: RESOURCE };
        let patch_count = u64::from(partition.patch_count);
        let (fm, em, vm) = (
            u64::from(summary.face_members),
            u64::from(summary.edge_members),
            u64::from(summary.vertex_members),
        );
        let words = HEADER_WORDS as u64 + RECORD_WORDS as u64 * patch_count + vm + 3 * em + 7 * fm;
        check_binding(&limits, 4 * words, RESOURCE)?;
        let mask_bound = |members: u64| (members + 31 * patch_count) / 32;
        let mask_words =
            (3 * patch_count + 2 * (mask_bound(vm) + mask_bound(em) + mask_bound(fm))).max(1);
        check_binding(&limits, 4 * mask_words, MASKS)?;
        let members = u32::try_from(fm + em + vm).map_err(|_| too_large)?;
        let region_plan = scan::ScanPlan::new(partition.patch_count + 1, &limits)?;
        let member_groups = pipeline::workgroup_grid(&limits, members, RESOURCE)?;
        let patch_groups = pipeline::workgroup_grid(&limits, partition.patch_count, RESOURCE)?;

        let arena = pipeline::output_buffer(device, "MOPMesh static connectivity", 4 * words);
        let masks =
            pipeline::output_buffer(device, "MOPMesh static connectivity masks", 4 * mask_words);
        let params = uniform(
            device,
            &[
                input.vertex_count,
                summary.edges,
                input.face_count,
                partition.patch_count,
                members,
                summary.face_members,
                summary.edge_members,
                0,
            ],
        );
        let scratch = |label| pipeline::output_buffer(device, label, region_plan.byte_size());
        let regions = scratch("MOPMesh build arena regions");
        let mask_regions = scratch("MOPMesh build mask regions");
        let bindings = self.arena.bind(
            device,
            &[
                &params,
                &patches.grouped,
                &patches.local_ids,
                &patches.owners,
                &regions,
                &mask_regions,
                &arena,
                &masks,
            ],
        );
        let [header, size, records, write] = &self.arena.pipelines;
        let run = |encoder: &mut wgpu::CommandEncoder, label, pipeline, groups| {
            pipeline::dispatch_grid(encoder, label, pipeline, &bindings, groups)
        };

        // Write the header even for an empty mesh. For nonempty meshes,
        // calculate patch sizes, turn them into offsets, and fill the buffers.
        run(encoder, "MOPMesh build arena header", header, [1, 1]);
        if members == 0 {
            return Ok([arena, masks]);
        }
        run(encoder, "MOPMesh build size patches", size, patch_groups);
        self.scan.encode(device, encoder, &regions, &region_plan);
        self.scan
            .encode(device, encoder, &mask_regions, &region_plan);
        run(
            encoder,
            "MOPMesh build write patch records",
            records,
            patch_groups,
        );
        run(encoder, "MOPMesh build write arena", write, member_groups);
        Ok([arena, masks])
    }

    /// Builds the connectivity arena on the GPU in two submissions, reading
    /// back a small summary between them to size the second. Returns the
    /// input's [`BuildError`] if it is invalid, or a limit or device error.
    /// Blocks until the GPU has finished the first phase; the second is
    /// submitted but not waited for.
    pub(super) fn build(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: &GpuMeshInput,
        method: GpuSplit,
        patch_size: u32,
    ) -> Result<GpuBuild, GpuBuildError> {
        // Reject what the CPU builder rejects before looking at the data.
        if patch_size == 0 {
            return Err(GpuBuildError::Input(BuildError::InvalidPatchSize));
        }
        if input.face_count > u32::MAX / 3 {
            return Err(GpuBuildError::Input(BuildError::TooManyElements));
        }

        // Phase 1: validation, connectivity, the split and the member counts.
        let mut encoder = device.create_command_encoder(&Default::default());
        let elements = self.encode_elements(device, &mut encoder, input)?;
        let connectivity = self.encode_connectivity(device, &mut encoder, input, &elements)?;
        let split = self.encode_split(device, &mut encoder, input, method, patch_size)?;
        let counts = self.encode_member_counts(
            device,
            &mut encoder,
            input,
            &elements,
            &connectivity,
            &split.partition(),
        )?;
        let words = read_back(device, queue, encoder, &counts.summary)?;
        let summary = Summary::decode(words.try_into().unwrap());
        if let Some(error) = summary.error.clone() {
            return Err(GpuBuildError::Input(error));
        }

        // Phase 2: patch contents and the arena.
        let mut encoder = device.create_command_encoder(&Default::default());
        let first = FirstPhase {
            elements: &elements,
            connectivity: &connectivity,
            partition: split.partition(),
            counts: &counts,
            summary: &summary,
        };
        let patches = self.encode_patches(device, &mut encoder, input, &first)?;
        let [arena, masks] = self.encode_arena(
            device,
            &mut encoder,
            input,
            &first.partition,
            &summary,
            &patches,
        )?;
        queue.submit([encoder.finish()]);
        Ok(GpuBuild {
            arena,
            masks,
            split,
            patches,
            summary,
        })
    }

    fn member_params(
        &self,
        device: &wgpu::Device,
        input: &GpuMeshInput,
        partition: &Partition,
    ) -> wgpu::Buffer {
        uniform(
            device,
            &[
                input.vertex_count,
                input.face_count,
                partition.patch_count,
                0,
            ],
        )
    }
}

/// A finished GPU build: the arena plus the intermediate results that
/// describe it.
pub(super) struct GpuBuild {
    /// The connectivity arena, with `STORAGE | COPY_SRC` usage.
    pub arena: wgpu::Buffer,
    /// Slot bounds and bitmasks identifying active and owned elements.
    /// Supports `STORAGE | COPY_SRC`; trailing zero words may be unused.
    pub masks: wgpu::Buffer,
    /// Read only by tests, which check the split and patches against the CPU.
    #[cfg_attr(not(test), allow(dead_code))]
    pub split: FaceSplit,
    #[cfg_attr(not(test), allow(dead_code))]
    pub patches: PatchStage,
    pub summary: Summary,
}

/// Why [`GpuMeshBuilder::build`] failed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GpuBuildError {
    /// The mesh data is invalid. This is the error
    /// [`MeshBuilder::build`](crate::MeshBuilder::build) returns for the same
    /// positions, triangles and patch size.
    Input(BuildError),
    /// An input buffer is too small or lacks `STORAGE` usage, or the mesh needs
    /// buffers or dispatches beyond the device limits.
    Gpu(GpuError),
    /// Waiting for the GPU or reading back its results failed, with the
    /// device's message. The device may have been lost.
    Device(String),
}

impl std::fmt::Display for GpuBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Input(error) => write!(f, "invalid mesh input: {error}"),
            Self::Gpu(error) => error.fmt(f),
            Self::Device(message) => write!(f, "GPU mesh build failed: {message}"),
        }
    }
}

impl std::error::Error for GpuBuildError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Input(error) => Some(error),
            Self::Gpu(error) => Some(error),
            Self::Device(_) => None,
        }
    }
}

impl From<GpuError> for GpuBuildError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

/// Submits `encoder` with a copy of `buffer` (which needs `COPY_SRC`) to the
/// host, waits for it, and returns the buffer's words.
fn read_back(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    mut encoder: wgpu::CommandEncoder,
    buffer: &wgpu::Buffer,
) -> Result<Vec<u32>, GpuBuildError> {
    let readback = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("MOPMesh build readback"),
        size: buffer.size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    encoder.copy_buffer_to_buffer(buffer, 0, &readback, 0, buffer.size());
    queue.submit([encoder.finish()]);
    let (tx, rx) = std::sync::mpsc::channel();
    readback
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            let _ = tx.send(result);
        });
    let device_error = |e: &dyn std::fmt::Display| GpuBuildError::Device(e.to_string());
    device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .map_err(|e| device_error(&e))?;
    rx.recv()
        .map_err(|e| device_error(&e))?
        .map_err(|e| device_error(&e))?;
    let words = bytemuck::cast_slice(
        &readback
            .slice(..)
            .get_mapped_range()
            .map_err(|e| device_error(&e))?,
    )
    .to_vec();
    readback.unmap();
    Ok(words)
}

/// Checks that `buffer` has `STORAGE` usage and holds `count` 16-byte values,
/// and at least 16 bytes.
fn check_input(buffer: &wgpu::Buffer, count: u32, resource: &'static str) -> Result<(), GpuError> {
    let large_enough = buffer.size() >= 16 * u64::from(count.max(1));
    if large_enough && buffer.usage().contains(wgpu::BufferUsages::STORAGE) {
        Ok(())
    } else {
        Err(GpuError::InvalidBuffer { resource })
    }
}

/// Creates a uniform buffer holding `words`.
fn uniform(device: &wgpu::Device, words: &[u32]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("MOPMesh build parameters"),
        contents: bytemuck::cast_slice(words),
        usage: wgpu::BufferUsages::UNIFORM,
    })
}

/// The first phase's results that the second phase reads.
pub(super) struct FirstPhase<'a> {
    pub elements: &'a ElementStage,
    pub connectivity: &'a EdgeStage,
    pub partition: Partition<'a>,
    pub counts: &'a MemberCounts,
    /// Decoded from `counts.summary`; must report no error.
    pub summary: &'a Summary,
}

/// A face partition: the patch that owns each face.
pub(super) struct Partition<'a> {
    /// One `u32` patch ID per face, each below `patch_count`; needs `STORAGE`
    /// usage. Any buffer of at least 16 bytes when there are no faces.
    pub face_patch: &'a wgpu::Buffer,
    pub patch_count: u32,
}

/// Words in [`MemberCounts::summary`]; decode with [`Summary::decode`].
pub(super) const SUMMARY_WORDS: usize = ERROR_WORDS + 4;

/// Results of the first phase's member counting.
pub(super) struct MemberCounts {
    /// `[V set sizes, then each vertex's owning patches ascending]`, in the
    /// slots of that vertex's corner row.
    vertex_patches: wgpu::Buffer,
    /// Member counts per face, edge and vertex, scanned into first indices.
    counts: wgpu::Buffer,
    /// [`SUMMARY_WORDS`] words: the error words, then the edge count and the
    /// member prefixes at the face, edge and vertex boundaries.
    pub summary: wgpu::Buffer,
}

/// The first phase's result, read back from [`MemberCounts::summary`].
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Summary {
    pub error: Option<BuildError>,
    pub edges: u32,
    /// Patch members of each kind, summed over all patches.
    pub face_members: u32,
    pub edge_members: u32,
    pub vertex_members: u32,
}

impl Summary {
    pub(super) fn decode(words: [u32; SUMMARY_WORDS]) -> Self {
        let [edges, faces_end, edges_end, vertices_end]: [u32; 4] =
            words[ERROR_WORDS..].try_into().unwrap();
        Self {
            error: build_error(words[..ERROR_WORDS].try_into().unwrap()),
            edges,
            face_members: faces_end,
            edge_members: edges_end.wrapping_sub(faces_end),
            vertex_members: vertices_end.wrapping_sub(edges_end),
        }
    }
}

/// Results of the second phase. Members are grouped in rows: rows `0..P` hold
/// each patch's faces, `P..2P` its edges and `2P..3P` its vertices, each in
/// ascending global ID order; a member's local ID is its position in its row.
pub(super) struct PatchStage {
    /// `[3P + 1 row offsets, global IDs]`. Offsets index the ID region.
    pub grouped: wgpu::Buffer,
    /// Two local vertex IDs per edge member, then three `(local edge,
    /// reversed 0/1)` pairs per face member, each in grouped order.
    pub local_ids: wgpu::Buffer,
    /// `[owner patch, owner-local ID]` per member, in grouped order.
    pub owners: wgpu::Buffer,
}

/// Decodes the shared error buffer into the error the CPU builder reports for
/// the same input, or `None` if every stage that ran found the input valid.
pub(super) fn build_error(words: [u32; ERROR_WORDS]) -> Option<BuildError> {
    let [
        vertex,
        face,
        kind,
        missing,
        duplicate,
        corner,
        edge_kind,
        low,
        high,
        isolated,
        non_manifold,
    ] = words;

    // Element errors come first: non-finite positions, then the first bad face.
    if vertex != NONE {
        return Some(BuildError::NonFinitePosition(vertex));
    }
    match (face, kind) {
        (NONE, _) => {}
        (face, 0) => {
            return Some(BuildError::InvalidVertex {
                face,
                vertex: missing,
            });
        }
        (face, _) => return Some(BuildError::DegenerateFace(face)),
    }

    // Then the earlier of a duplicate face and an edge error; a duplicate wins a tie.
    let edge_face = if corner == NONE { NONE } else { corner / 3 };
    if duplicate != NONE && duplicate <= edge_face {
        return Some(BuildError::DuplicateFace(duplicate));
    }
    match (corner, edge_kind) {
        (NONE, _) => {}
        (_, 0) => return Some(BuildError::InconsistentWinding([low, high])),
        _ => return Some(BuildError::NonManifoldEdge([low, high])),
    }

    // Then the lowest vertex that is isolated or has more than one fan.
    match (isolated, non_manifold) {
        (NONE, NONE) => None,
        (v, w) if v < w => Some(BuildError::IsolatedVertex(v)),
        (_, w) => Some(BuildError::NonManifoldVertex(w)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshBuilder;
    use crate::gpu::test_support::{self, fixtures};
    use crate::mesh::build::{Rows, check_edges, check_elements, check_fans};

    #[test]
    fn errors_decode_with_cpu_precedence() {
        let words = |set: &[(usize, u32)]| {
            let mut words = [NONE; ERROR_WORDS];
            for &(i, value) in set {
                words[i] = value;
            }
            words
        };
        assert_eq!(build_error(words(&[])), None);

        // Element errors, then duplicates and edge errors in face order.
        let face_error = [(1, 1), (2, 0), (3, 9)];
        let both = [&face_error[..], &[(0, 3)]].concat();
        assert_eq!(
            build_error(words(&both)),
            Some(BuildError::NonFinitePosition(3))
        );
        let with_edge = [&face_error[..], &[(4, 0), (5, 0)]].concat();
        assert_eq!(
            build_error(words(&with_edge)),
            Some(BuildError::InvalidVertex { face: 1, vertex: 9 })
        );
        assert_eq!(
            build_error(words(&[(1, 2), (2, 1)])),
            Some(BuildError::DegenerateFace(2))
        );
        // Corner 7 is in face 2: a duplicate face 2 wins the tie, face 3 loses.
        let edge = [(5, 7), (6, 0), (7, 4), (8, 5)];
        assert_eq!(
            build_error(words(&[&edge[..], &[(4, 2)]].concat())),
            Some(BuildError::DuplicateFace(2))
        );
        assert_eq!(
            build_error(words(&[&edge[..], &[(4, 3)]].concat())),
            Some(BuildError::InconsistentWinding([4, 5]))
        );
        assert_eq!(
            build_error(words(&[(5, 7), (6, 1), (7, 4), (8, 5)])),
            Some(BuildError::NonManifoldEdge([4, 5]))
        );

        // Vertex errors come last; the lower vertex wins.
        assert_eq!(
            build_error(words(&[(4, 0), (9, 2)])),
            Some(BuildError::DuplicateFace(0))
        );
        assert_eq!(
            build_error(words(&[(9, 2), (10, 1)])),
            Some(BuildError::NonManifoldVertex(1))
        );
        assert_eq!(
            build_error(words(&[(9, 2), (10, 3)])),
            Some(BuildError::IsolatedVertex(2))
        );
        assert_eq!(
            build_error(words(&[(10, 3)])),
            Some(BuildError::NonManifoldVertex(3))
        );
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn element_and_connectivity_stages_match_the_cpu_builder() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let builder = GpuBuilder::new(&device);
        let read = |buffer: &wgpu::Buffer| {
            let encoder = device.create_command_encoder(&Default::default());
            test_support::read_words(&device, &queue, encoder, buffer)
        };
        for fixture in fixtures::valid().into_iter().chain(fixtures::invalid()) {
            let name = fixture.name;
            let (vertex_count, face_count) = (fixture.positions.len(), fixture.triangles.len());
            let (positions, triangles) = test_support::input_buffers(&device, &fixture);
            let input = GpuMeshInput {
                positions: &positions,
                triangles: &triangles,
                vertex_count: vertex_count as u32,
                face_count: face_count as u32,
            };
            let mut encoder = device.create_command_encoder(&Default::default());
            let elements = builder
                .encode_elements(&device, &mut encoder, &input)
                .unwrap();
            let edges = builder
                .encode_connectivity(&device, &mut encoder, &input, &elements)
                .unwrap();
            queue.submit([encoder.finish()]);

            // Errors must match the CPU checks, including which one is reported.
            let errors = read(&elements.errors);
            let triangles = &fixture.triangles;
            let group_rows =
                || Rows::group(vertex_count, 3 * face_count, |c| triangles[c / 3][c % 3]);
            let expected = check_elements(&fixture.positions, triangles).and_then(|()| {
                let edges = check_edges(vertex_count, triangles)?;
                check_fans(&group_rows(), triangles, &edges)?;
                Ok(edges)
            });
            let expected = match expected {
                Ok(edges) => edges,
                Err(error) => {
                    let actual = build_error(errors.try_into().unwrap());
                    assert_eq!(actual, Some(error), "{name}");
                    continue;
                }
            };
            assert_eq!(build_error(errors.try_into().unwrap()), None, "{name}");

            // Corner rows must hold the CPU grouping's corners, in any order
            // within each row.
            let rows = group_rows();
            let mut actual_rows = read(&elements.rows);
            let (offsets, corners) = actual_rows.split_at_mut(vertex_count + 1);
            assert_eq!(offsets, rows.offsets(), "{name}: row offsets");
            for v in 0..vertex_count {
                let row = &mut corners[offsets[v] as usize..offsets[v + 1] as usize];
                row.sort_unstable();
                assert_eq!(row, rows.row(v), "{name}: corners of vertex {v}");
            }

            // Edge IDs, endpoints, faces and face edges must match too.
            let edge_count = expected.endpoints.len();
            assert_eq!(
                read(&edges.edge_offsets)[vertex_count] as usize,
                edge_count,
                "{name}: edge count"
            );
            let expected_edges: Vec<u32> = expected
                .endpoints
                .iter()
                .zip(&expected.faces)
                .flat_map(|(&[a, b], &[f, g])| [a, b, f, g])
                .collect();
            let actual_edges = read(&edges.edges);
            assert_eq!(
                actual_edges[..4 * edge_count],
                expected_edges,
                "{name}: edges"
            );
            let expected_face_edges: Vec<u32> = expected
                .face_edges
                .iter()
                .flatten()
                .flat_map(|d| [d.edge, u32::from(d.reversed)])
                .collect();
            let actual_face_edges = read(&edges.face_edges);
            assert_eq!(
                actual_face_edges[..2 * 3 * face_count],
                expected_face_edges,
                "{name}: face edges"
            );
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn element_stage_checks_dispatch_limits_before_recording() {
        let (device, _queue) = test_support::device(wgpu::Limits {
            max_compute_workgroups_per_dimension: 1,
            ..Default::default()
        });
        let builder = GpuBuilder::new(&device);
        let (positions, triangles) = fixtures::grid(9);
        let fixture = fixtures::Fixture {
            name: "grid 9",
            positions,
            triangles,
        };
        let (positions, triangles) = test_support::input_buffers(&device, &fixture);
        let input = GpuMeshInput {
            positions: &positions,
            triangles: &triangles,
            vertex_count: fixture.positions.len() as u32,
            face_count: fixture.triangles.len() as u32,
        };
        // 128 faces need two workgroups, one more than this device allows.
        let mut encoder = device.create_command_encoder(&Default::default());
        assert!(matches!(
            builder.encode_elements(&device, &mut encoder, &input),
            Err(GpuError::LimitsExceeded { .. })
        ));
    }

    /// The patch that owns each face of `mesh`.
    fn face_patches(mesh: &crate::Mesh) -> Vec<u32> {
        let mut face_patch = vec![u32::MAX; mesh.face_count()];
        for p in mesh.patches() {
            for (local, owner) in p.face_owners().iter().enumerate() {
                if owner.patch == p.id() {
                    face_patch[p.face_ids()[local] as usize] = p.id();
                }
            }
        }
        face_patch
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn patch_contents_and_arena_match_the_cpu_builder() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let builder = GpuBuilder::new(&device);
        let read = |buffer: &wgpu::Buffer| {
            let encoder = device.create_command_encoder(&Default::default());
            test_support::read_words(&device, &queue, encoder, buffer)
        };
        let upload = |words: &[u32]| {
            let mut words = words.to_vec();
            words.resize(words.len().max(4), 0);
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test face patches"),
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::STORAGE,
            })
        };

        // Records the first phase and returns its stages and decoded summary.
        let phase_one = |fixture: &fixtures::Fixture, face_patch: &wgpu::Buffer, patch_count| {
            let (positions, triangles) = test_support::input_buffers(&device, fixture);
            let input = GpuMeshInput {
                positions: &positions,
                triangles: &triangles,
                vertex_count: fixture.positions.len() as u32,
                face_count: fixture.triangles.len() as u32,
            };
            let partition = Partition {
                face_patch,
                patch_count,
            };
            let mut encoder = device.create_command_encoder(&Default::default());
            let elements = builder
                .encode_elements(&device, &mut encoder, &input)
                .unwrap();
            let connectivity = builder
                .encode_connectivity(&device, &mut encoder, &input, &elements)
                .unwrap();
            let counts = builder
                .encode_member_counts(
                    &device,
                    &mut encoder,
                    &input,
                    &elements,
                    &connectivity,
                    &partition,
                )
                .unwrap();
            queue.submit([encoder.finish()]);
            let summary = Summary::decode(read(&counts.summary).try_into().unwrap());
            (
                positions,
                triangles,
                elements,
                connectivity,
                counts,
                summary,
            )
        };

        // Invalid inputs, with every face in patch 0, report the CPU error.
        for fixture in fixtures::invalid() {
            let expected = MeshBuilder::new()
                .positions(&fixture.positions)
                .triangles(&fixture.triangles)
                .build()
                .unwrap_err();
            let face_patch = upload(&vec![0; fixture.triangles.len()]);
            let (.., summary) = phase_one(&fixture, &face_patch, 1);
            assert_eq!(summary.error, Some(expected), "{}", fixture.name);
        }

        for fixture in fixtures::valid() {
            for patch_size in fixtures::PATCH_SIZES {
                let name = format!("{}, patch size {patch_size}", fixture.name);
                let (mesh, arena) =
                    test_support::reference(&fixture, patch_size, &device.limits()).unwrap();
                let patches = mesh.patches();
                let patch_count = patches.len() as u32;
                let face_patch = upload(&face_patches(&mesh));
                let (positions, triangles, elements, connectivity, counts, summary) =
                    phase_one(&fixture, &face_patch, patch_count);

                // The totals must match the CPU patches and give the arena size.
                let sum = |len: fn(&crate::Patch) -> usize| patches.iter().map(len).sum::<usize>();
                let expected = Summary {
                    error: None,
                    edges: mesh.edge_count() as u32,
                    face_members: sum(|p| p.face_ids().len()) as u32,
                    edge_members: sum(|p| p.edge_ids().len()) as u32,
                    vertex_members: sum(|p| p.vertex_ids().len()) as u32,
                };
                assert_eq!(summary, expected, "{name}");
                let words = 8
                    + 16 * patch_count
                    + summary.vertex_members
                    + 3 * summary.edge_members
                    + 7 * summary.face_members;
                assert_eq!(words as usize, arena.arena.len(), "{name}: arena size");

                // Record and run the second phase.
                let input = GpuMeshInput {
                    positions: &positions,
                    triangles: &triangles,
                    vertex_count: fixture.positions.len() as u32,
                    face_count: fixture.triangles.len() as u32,
                };
                let first = FirstPhase {
                    elements: &elements,
                    connectivity: &connectivity,
                    partition: Partition {
                        face_patch: &face_patch,
                        patch_count,
                    },
                    counts: &counts,
                    summary: &summary,
                };
                let mut encoder = device.create_command_encoder(&Default::default());
                let stage = builder
                    .encode_patches(&device, &mut encoder, &input, &first)
                    .unwrap();
                let partition = &first.partition;
                let [gpu_arena, gpu_masks] = builder
                    .encode_arena(&device, &mut encoder, &input, partition, &summary, &stage)
                    .unwrap();
                queue.submit([encoder.finish()]);

                // The arena and masks must match GpuMesh::upload's word for word.
                let (gpu_arena, gpu_masks) = (read(&gpu_arena), read(&gpu_masks));
                test_support::assert_arena_eq(&name, &mesh, &gpu_arena, &gpu_masks, &arena);
                if patches.is_empty() {
                    continue;
                }

                // Rows: faces, then edges, then vertices of each patch.
                let lists: [fn(&crate::Patch) -> &[u32]; 3] =
                    [|p| p.face_ids(), |p| p.edge_ids(), |p| p.vertex_ids()];
                let mut offsets = vec![0u32];
                let mut ids = Vec::new();
                for list in lists {
                    for p in patches {
                        ids.extend_from_slice(list(p));
                        offsets.push(ids.len() as u32);
                    }
                }
                let expected_rows: Vec<u32> = offsets.into_iter().chain(ids).collect();
                assert_eq!(read(&stage.grouped), expected_rows, "{name}: rows");

                // Local IDs: every patch's EV, then every patch's FE.
                let ev = patches
                    .iter()
                    .flat_map(|p| p.edge_vertices().iter().flatten().copied());
                let fe = patches.iter().flat_map(|p| {
                    p.face_edges()
                        .iter()
                        .flatten()
                        .flat_map(|d| [d.edge, u32::from(d.reversed)])
                });
                let expected_local: Vec<u32> = ev.chain(fe).collect();
                let local = read(&stage.local_ids);
                assert_eq!(
                    local[..expected_local.len()],
                    expected_local,
                    "{name}: local IDs"
                );

                // Owners: faces, then edges, then vertices, in row order.
                let mut expected_owners = Vec::new();
                for p in patches {
                    expected_owners.extend(p.face_owners().iter().flat_map(|h| [h.patch, h.local]));
                }
                for p in patches {
                    expected_owners.extend(p.edge_owners().iter().flat_map(|h| [h.patch, h.local]));
                }
                for p in patches {
                    expected_owners
                        .extend(p.vertex_owners().iter().flat_map(|h| [h.patch, h.local]));
                }
                let owners = read(&stage.owners);
                assert_eq!(
                    owners[..expected_owners.len()],
                    expected_owners,
                    "{name}: owners"
                );
            }
        }
    }

    /// The output of a full GPU build, read back for checking.
    struct Built {
        arena: Vec<u32>,
        masks: Vec<u32>,
        face_patch: Vec<u32>,
        patch_sizes: Vec<crate::gpu::validate::PatchSizes>,
    }

    /// Runs both phases of a GPU build of `fixture`, splitting with `method`.
    fn gpu_build(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        builder: &GpuBuilder,
        fixture: &fixtures::Fixture,
        method: GpuSplit,
        patch_size: u32,
    ) -> Result<Built, BuildError> {
        let read = |buffer: &wgpu::Buffer| {
            let encoder = device.create_command_encoder(&Default::default());
            test_support::read_words(device, queue, encoder, buffer)
        };
        let (positions, triangles) = test_support::input_buffers(device, fixture);
        let input = GpuMeshInput {
            positions: &positions,
            triangles: &triangles,
            vertex_count: fixture.positions.len() as u32,
            face_count: fixture.triangles.len() as u32,
        };
        let GpuBuild {
            arena,
            masks,
            split,
            patches,
            ..
        } = match builder.build(device, queue, &input, method, patch_size) {
            Ok(build) => build,
            Err(GpuBuildError::Input(error)) => return Err(error),
            Err(error) => panic!("GPU build failed: {error:?}"),
        };

        // Region sizes come from the patch rows: faces, edges, vertices per patch.
        let p = split.patch_count as usize;
        let grouped = read(&patches.grouped);
        let row = |r: usize| grouped[r + 1] - grouped[r];
        let patch_sizes = (0..p)
            .map(|q| crate::gpu::validate::PatchSizes {
                vertices: row(2 * p + q),
                edges: row(p + q),
                faces: row(q),
            })
            .collect();
        let mut face_patch = read(&split.face_patch);
        face_patch.truncate(fixture.triangles.len());
        Ok(Built {
            arena: read(&arena),
            masks: read(&masks),
            face_patch,
            patch_sizes,
        })
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn radix_sort_is_stable_and_sorts_by_key() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let builder = GpuBuilder::new(&device);
        // Partial blocks, keys using every digit, and many repeated keys.
        for count in [1u32, 63, 64, 65, 1000, 70_000] {
            let mut state = 0x9e37_79b9u32;
            let pairs: Vec<[u32; 2]> = (0..count)
                .map(|i| {
                    state ^= state << 13;
                    state ^= state >> 17;
                    state ^= state << 5;
                    let key = if i % 3 == 0 { state % 7 } else { state };
                    [key, i]
                })
                .collect();
            let mut expected = pairs.clone();
            expected.sort_by_key(|pair| pair[0]);

            let buffers = [0, 1].map(|_| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("test sort pairs"),
                    contents: bytemuck::cast_slice(&pairs),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
            });
            let blocks = count.div_ceil(pipeline::WORKGROUP_SIZE);
            let plan = scan::ScanPlan::new(16 * blocks, &device.limits()).unwrap();
            let mut encoder = device.create_command_encoder(&Default::default());
            let grid = pipeline::workgroup_grid(&device.limits(), count, "test sort").unwrap();
            builder.encode_sort(&device, &mut encoder, &buffers, count, grid, &plan);
            let actual = test_support::read_words(&device, &queue, encoder, &buffers[0]);
            let expected: Vec<u32> = expected.into_iter().flatten().collect();
            assert_eq!(actual, expected, "{count} pairs");
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn two_dimensional_dispatch_gives_the_same_arena() {
        // At most 64 workgroups per dimension forces two-dimensional grids,
        // with partial last rows, for every stage of an 8,192-face build.
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let (small, small_queue) = test_support::device(wgpu::Limits {
            max_compute_workgroups_per_dimension: 64,
            ..Default::default()
        });
        let (positions, triangles) = fixtures::grid(65);
        let fixture = fixtures::Fixture {
            name: "grid 65",
            positions,
            triangles,
        };
        for method in [GpuSplit::InputOrder, GpuSplit::Morton] {
            let wide = gpu_build(
                &device,
                &queue,
                &GpuBuilder::new(&device),
                &fixture,
                method,
                128,
            )
            .unwrap();
            let grid = gpu_build(
                &small,
                &small_queue,
                &GpuBuilder::new(&small),
                &fixture,
                method,
                128,
            )
            .unwrap();
            assert_eq!(grid.face_patch, wide.face_patch, "{method:?}: split");
            assert_eq!(grid.arena, wide.arena, "{method:?}: arena");
        }
    }

    #[test]
    #[ignore = "requires a GPU adapter; run cargo test --lib -- --ignored --nocapture"]
    fn gpu_split_builds_meet_the_contract() {
        let (device, queue) = test_support::device(wgpu::Limits::default());
        let builder = GpuBuilder::new(&device);

        // Invalid inputs report the CPU's error, whatever the split.
        for fixture in fixtures::invalid() {
            let expected = MeshBuilder::new()
                .positions(&fixture.positions)
                .triangles(&fixture.triangles)
                .build()
                .unwrap_err();
            let built = gpu_build(&device, &queue, &builder, &fixture, GpuSplit::Morton, 3);
            assert_eq!(built.err(), Some(expected), "{}", fixture.name);
        }

        for fixture in fixtures::valid() {
            for patch_size in fixtures::PATCH_SIZES {
                for method in [GpuSplit::InputOrder, GpuSplit::Morton] {
                    let name = format!("{}, patch size {patch_size}, {method:?}", fixture.name);
                    let size = patch_size as u32;
                    let built =
                        gpu_build(&device, &queue, &builder, &fixture, method, size).unwrap();

                    // The split: every patch nonempty and within patch_size.
                    let patch_count = built.patch_sizes.len();
                    let mut owned = vec![0; patch_count];
                    for &p in &built.face_patch {
                        owned[p as usize] += 1;
                    }
                    assert!(
                        owned.iter().all(|&n| n > 0 && n <= patch_size),
                        "{name}: patch sizes {owned:?}"
                    );
                    if method == GpuSplit::InputOrder {
                        let expected: Vec<u32> = (0..fixture.triangles.len() as u32)
                            .map(|f| f / size)
                            .collect();
                        assert_eq!(built.face_patch, expected, "{name}: split");
                    }

                    // The arena meets the contract.
                    let valid = crate::gpu::validate::validate_arena(
                        &built.arena,
                        &built.masks,
                        fixture.positions.len(),
                        &fixture.triangles,
                        &built.patch_sizes,
                        patch_size,
                    );
                    assert_eq!(valid, Ok(()), "{name}");

                    // The CPU builder, given the same split, makes the same arena.
                    let cpu = MeshBuilder::new()
                        .patch_size(patch_size)
                        .face_patches(&built.face_patch);
                    let (mesh, words) =
                        test_support::reference_with(&fixture, cpu, &device.limits()).unwrap();
                    test_support::assert_arena_eq(&name, &mesh, &built.arena, &built.masks, &words);

                    // Quality: patch faces including ribbons, per input face. The
                    // GPU Morton split may round differently from the CPU's, but
                    // must be about as compact.
                    if method == GpuSplit::Morton && !fixture.triangles.is_empty() {
                        let held = |mesh: &crate::Mesh| {
                            mesh.patches()
                                .iter()
                                .map(|p| p.face_ids().len())
                                .sum::<usize>()
                        };
                        let cpu = |method| {
                            let builder =
                                MeshBuilder::new().patch_size(patch_size).partition(method);
                            let limits = device.limits();
                            held(
                                &test_support::reference_with(&fixture, builder, &limits)
                                    .unwrap()
                                    .0,
                            )
                        };
                        let gpu: usize = built.patch_sizes.iter().map(|p| p.faces as usize).sum();
                        let cpu_morton = cpu(crate::mesh::Partition::Morton);
                        let breadth_first = cpu(crate::mesh::Partition::BreadthFirst);
                        let per_face = |n: usize| n as f64 / fixture.triangles.len() as f64;
                        eprintln!(
                            "{name}: faces held per face: GPU Morton {:.2}, CPU Morton {:.2}, breadth-first {:.2}",
                            per_face(gpu),
                            per_face(cpu_morton),
                            per_face(breadth_first)
                        );
                        assert!(
                            gpu as f64 <= 1.25 * cpu_morton as f64,
                            "{name}: GPU Morton holds {gpu} faces, CPU Morton {cpu_morton}"
                        );
                    }

                    // A second build gives the same result.
                    let again =
                        gpu_build(&device, &queue, &builder, &fixture, method, size).unwrap();
                    assert_eq!(again.face_patch, built.face_patch, "{name}: repeat split");
                    assert_eq!(again.arena, built.arena, "{name}: repeat arena");
                    assert_eq!(again.masks, built.masks, "{name}: repeat masks");
                }
            }
        }
    }
}
