use ndarray::Array1;

pub struct Entry {
    pub input: Array1<f32>,
    pub pre_activations: Array1<f32>,
}

pub struct ForwardCache {
    pub entries: Vec<Entry>,
}

impl ForwardCache {
    #[must_use]
    pub fn new() -> Self {
        ForwardCache { entries: vec![] }
    }
}
