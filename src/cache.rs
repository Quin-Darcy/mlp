use ndarray::Array1;

pub struct Entry {
    pub input: Array1<f32>,
    pub pre_activations: Array1<f32>,
}

#[derive(Default)]
pub struct ForwardCache {
    pub entries: Vec<Entry>,
}

impl ForwardCache {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn with_capacity(num_layers: usize) -> Self {
        Self {
            entries: Vec::with_capacity(num_layers),
        }
    }
}
