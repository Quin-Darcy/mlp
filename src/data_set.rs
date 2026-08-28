use ndarray::Array1;

#[derive(Default)]
pub struct DataSet {
    samples: Vec<Array1<f32>>,
    labels: Vec<Array1<f32>>,
}

impl DataSet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

// TODO: add error type and validate samples and labels are same size
