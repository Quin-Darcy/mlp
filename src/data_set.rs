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

    // TODO: maybe add method which allows data to exist as plain vectors
    // instead of ndarray types and the method converts those
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_data_set_empty_samples() {
        let test_samples: Vec<Array1<f32>> = Vec::new();
        let test_labels: Vec<Array1<f32>> = vec![
            array![1.0, 0.0]
        ];

        let result = DataSet::from_data(test_samples, test_labels);

        assert!(result.is_err());
    }

    #[test]
    fn test_data_set_empty_lables() {
        let test_labels: Vec<Array1<f32>> = Vec::new();
        let test_samples: Vec<Array1<f32>> = vec![
            array![1.0, 0.0]
        ];

        let result = DataSet::from_data(test_samples, test_labels);

        assert!(result.is_err());
    }

    #[test]
    fn test_data_set_invalid_sizes() {
        let test_samples: Vec<Array1<f32>> = vec![
            array![1.0, 2.0],
            array![2.0, 1.0]
        ];
        let test_labels: Vec<Array1<f32>> = vec![
            array![1.0, 0.0]
        ];

        let result = DataSet::from_data(test_samples, test_labels);

        assert!(result.is_err());
    }

    #[test]
    fn test_data_set_valid_args() {
        let test_samples: Vec<Array1<f32>> = vec![
            array![1.0, 2.0],
        ];
        let test_labels: Vec<Array1<f32>> = vec![
            array![1.0, 0.0]
        ];

        let result = DataSet::from_data(test_samples, test_labels);

        assert!(result.is_ok());
    }
}
