use ndarray::{Array1, Array2};

#[derive(Debug, PartialEq)]
pub struct ForwardEntry {
    pub input: Array1<f32>,
    pub pre_activation: Array1<f32>,
}

#[derive(Debug, PartialEq)]
pub struct BackwardEntry {
    pub bias_gradient: Array1<f32>,
    pub weight_gradient: Array2<f32>,
}

#[derive(Default, Debug, PartialEq)]
pub struct ForwardCache {
    pub entries: Vec<ForwardEntry>,
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

#[derive(Default, Debug, PartialEq)]
pub struct BackwardCache {
    pub entries: Vec<BackwardEntry>,
}

impl BackwardCache {
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
