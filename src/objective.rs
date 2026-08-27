use ndarray::Array1;


#[derive(Debug)]
pub enum ObjectiveError {
    InvalidArgDimensions(String),
}

pub enum Objective {
    MSE,
}

impl Objective {
    pub fn compute(&self, output: &Array1<f32>, target: &Array1<f32>) -> Result<f32, ObjectiveError> {
        if output.dim() != target.dim() {
            return Err(ObjectiveError::InvalidArgDimensions(
                "output and target vectors must be the same length".to_string()
            ));
        }

        match self {
            Self::MSE => {
                let diff = target - output;
                Ok(diff.dot(&diff) / diff.len() as f32)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_objective_mse_valid_args() {
        let test_output: Array1<f32> = array![1.0, 0.0];
        let test_target: Array1<f32> = array![1.0, 0.0];
        let test_objective = Objective::MSE;
        let result = test_objective.compute(&test_output, &test_target);

        assert!(result.is_ok());
    }

    #[test]
    fn test_objective_mse_invalid_args() {
        let test_output: Array1<f32> = array![1.0, 0.0, 0.0];
        let test_target: Array1<f32> = array![1.0, 0.0];
        let test_objective = Objective::MSE;
        let result = test_objective.compute(&test_output, &test_target);

        assert!(result.is_err());
    }

    #[test]
    fn test_objective_mse() {
        let test_output: Array1<f32> = array![1.0, 2.0];
        let test_target: Array1<f32> = array![3.0, 0.0];
        let test_objective = Objective::MSE;
        let test_expected_value: f32 = 4.0;
        let test_value: f32 = test_objective.compute(&test_output, &test_target).unwrap();

        assert_eq!(test_expected_value, test_value);
    }
}
