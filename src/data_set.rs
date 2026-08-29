use ndarray::Array1;

#[derive(Debug)]
pub enum DataError {
    EmptySamples(String),
    EmptyLabels(String),
    InvalidSizes(String),
}

#[derive(Default)]
pub struct DataSet {
    pub samples: Vec<Array1<f32>>,
    pub labels: Vec<Array1<f32>>,
}

impl DataSet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn from_data(
        samples: Vec<Array1<f32>>,
        labels: Vec<Array1<f32>>,
    ) -> Result<Self, DataError> {
        if samples.is_empty() {
            return Err(DataError::EmptySamples(
                "Samples cannot be empty".to_string(),
            ));
        }

        if labels.is_empty() {
            return Err(DataError::EmptyLabels("Labels cannot be empty".to_string()));
        }

        if samples.len() != labels.len() {
            return Err(DataError::InvalidSizes(
                "There must be an equal number of samples and labels".to_string(),
            ));
        }

        Ok(Self { samples, labels })
    }
}
