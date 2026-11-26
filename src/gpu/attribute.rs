use super::{FaceVertexQuery, GpuMesh};
use crate::VertexAttribute;
use std::{marker::PhantomData, sync::Arc};
use wgpu::util::DeviceExt;

mod sealed {
    pub trait Sealed {}
    macro_rules! values {
        ($($ty:ty),*) => { $(impl Sealed for $ty {})* };
    }
    values!(
        f32, u32, i32, [f32; 2], [u32; 2], [i32; 2], [f32; 4], [u32; 4], [i32; 4]
    );
}

/// Supported storage-array elements: f32/u32/i32 scalars and their vec2/vec4
/// equivalents. Rust arrays of length three are deliberately excluded: WGSL
/// vec3 array stride is 16 bytes, whereas a Rust [f32; 3] occupies 12 bytes.
/// This trait is sealed to keep the storage ABI explicit.
pub trait GpuValue: sealed::Sealed + bytemuck::Pod {}
impl<T: sealed::Sealed + bytemuck::Pod> GpuValue for T {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuAttributeError {
    MeshMismatch,
    LimitsExceeded,
}
impl std::fmt::Display for GpuAttributeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::MeshMismatch => "attribute belongs to another mesh",
            Self::LimitsExceeded => "attribute or result exceeds GPU buffer limits",
        })
    }
}
impl std::error::Error for GpuAttributeError {}

/// One value per original input vertex, shared by every ribbon reference.
/// Bind as an array of the matching WGSL scalar/vector type. Usage is STORAGE,
/// COPY_SRC, and COPY_DST. Device/queue consistency is the caller's responsibility.
/// Raw-buffer writes must preserve the element layout and canonical indexing.
pub struct GpuVertexAttribute<T: GpuValue> {
    buffer: wgpu::Buffer,
    identity: Arc<()>,
    len: u32,
    marker: PhantomData<T>,
}

impl GpuMesh {
    pub fn upload_vertex_attribute<T: GpuValue>(
        &self,
        device: &wgpu::Device,
        attribute: &VertexAttribute<T>,
    ) -> Result<GpuVertexAttribute<T>, GpuAttributeError> {
        if !Arc::ptr_eq(&self.identity, &attribute.identity) {
            return Err(GpuAttributeError::MeshMismatch);
        }
        let bytes = bytemuck::cast_slice(&attribute.values);
        let size = bytes.len().max(std::mem::size_of::<T>()) as u64;
        check_size(device, size)?;
        let zero = T::zeroed();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("MOPMesh canonical vertex attribute"),
            contents: if bytes.is_empty() {
                bytemuck::bytes_of(&zero)
            } else {
                bytes
            },
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });
        Ok(GpuVertexAttribute {
            buffer,
            identity: self.identity.clone(),
            len: self.vertex_count,
            marker: PhantomData,
        })
    }
}

impl<T: GpuValue> GpuVertexAttribute<T> {
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
    pub fn len(&self) -> u32 {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Queue a full replacement from a host attribute of the same mesh. Host
    /// and GPU attributes are snapshots; neither automatically updates the other.
    /// Queue writes become visible on the next submission, including to commands
    /// already encoded. Submit earlier readers before calling this method.
    pub fn write(
        &mut self,
        queue: &wgpu::Queue,
        attribute: &VertexAttribute<T>,
    ) -> Result<(), GpuAttributeError> {
        if !Arc::ptr_eq(&self.identity, &attribute.identity) {
            return Err(GpuAttributeError::MeshMismatch);
        }
        if !self.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&attribute.values));
        }
        Ok(())
    }
}

fn check_size(device: &wgpu::Device, size: u64) -> Result<(), GpuAttributeError> {
    let limits = device.limits();
    if size
        > limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
    {
        return Err(GpuAttributeError::LimitsExceeded);
    }
    Ok(())
}

/// A first attribute-consuming geometry query. Cache once per device.
pub struct FaceCentroidQuery {
    face_vertices: FaceVertexQuery,
    pipeline: wgpu::ComputePipeline,
}
impl FaceCentroidQuery {
    pub fn new(device: &wgpu::Device) -> Self {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("MOPMesh face centroids"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!(concat!(env!("OUT_DIR"), "/face_centroids.wgsl")).into(),
            ),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("MOPMesh face centroids"),
            layout: None,
            module: &module,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            face_vertices: FaceVertexQuery::new(device),
            pipeline,
        }
    }

    /// Returns one vec4<f32> (centroid xyz, zero w) per original face ID with
    /// STORAGE | COPY_SRC usage. Position w lanes are ignored. Empty meshes
    /// return one zero vec4 placeholder without dispatch. Checks identity and
    /// output limits before recording commands. Uses three storage bindings.
    pub fn encode(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        mesh: &GpuMesh,
        positions: &GpuVertexAttribute<[f32; 4]>,
    ) -> Result<wgpu::Buffer, GpuAttributeError> {
        if !Arc::ptr_eq(&mesh.identity, &positions.identity) {
            return Err(GpuAttributeError::MeshMismatch);
        }
        let size = (u64::from(mesh.face_count) * 16).max(16);
        check_size(device, size)?;
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("MOPMesh face centroid result"),
            size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        if mesh.face_count == 0 {
            return Ok(output);
        }
        let faces = self.face_vertices.encode(device, encoder, mesh);
        let buffers = [&faces, positions.buffer(), &output];
        let entries: Vec<_> = buffers
            .iter()
            .enumerate()
            .map(|(binding, buffer)| wgpu::BindGroupEntry {
                binding: binding as u32,
                resource: buffer.as_entire_binding(),
            })
            .collect();
        let bindings = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("MOPMesh face centroid bindings"),
            layout: &self.pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("MOPMesh face centroids"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bindings, &[]);
            pass.dispatch_workgroups(mesh.face_count.div_ceil(64), 1, 1);
        }
        Ok(output)
    }
}
