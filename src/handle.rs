macro_rules! handle {
    ($name:ident) => {
        /// A patch-local address. Only valid for the mesh that created it.
        /// Future migration or slot reuse will invalidate these addresses.
        #[repr(C)]
        #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
        pub struct $name {
            pub patch: u32,
            pub local: u32,
        }
    };
}
handle!(VertexHandle);
handle!(EdgeHandle);
handle!(FaceHandle);
