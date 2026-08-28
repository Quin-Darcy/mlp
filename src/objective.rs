use ndarray::Array1;

#[derive(Debug)]
pub enum ObjectiveError {
    InvalidArgDimensions(String),
}

pub enum Objective {
    MSE,
}

impl Objective {
    pub fn compute(
        &self,
        input: &Array1<f32>,
        target: &Array1<f32>,
    ) -> Result<f32, ObjectiveError> {
        if input.dim() != target.dim() {
            return Err(ObjectiveError::InvalidArgDimensions(
                "input and target vectors must be the same length".to_string(),
            ));
        }

        match self {
            Self::MSE => {
                let diff: Array1<f32> = target - input;
                Ok(diff.dot(&diff) / diff.len() as f32)
            }
        }
    }

    pub fn gradient(
        &self,
        input: &Array1<f32>,
        target: &Array1<f32>,
    ) -> Result<Array1<f32>, ObjectiveError> {
        if input.dim() != target.dim() {
            return Err(ObjectiveError::InvalidArgDimensions(
                "input and target vectors must be the same length".to_string(),
            ));
        }

        match self {
            Self::MSE => {
                let scalar: f32 = -2.0 / (input.dim() as f32);
                let diff: Array1<f32> = target - input;
                Ok(scalar * diff)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_objective_mse_compute_valid_args() {
        let test_input: Array1<f32> = array![1.0, 0.0];
        let test_target: Array1<f32> = array![1.0, 0.0];
        let test_objective = Objective::MSE;
        let result = test_objective.compute(&test_input, &test_target);

        assert!(result.is_ok());
    }

    #[test]
    fn test_objective_mse_compute_invalid_args() {
        let test_input: Array1<f32> = array![1.0, 0.0, 0.0];
        let test_target: Array1<f32> = array![1.0, 0.0];
        let test_objective = Objective::MSE;
        let result = test_objective.compute(&test_input, &test_target);

        assert!(result.is_err());
    }

    #[test]
    fn test_objective_mse_compute() {
        let test_input: Array1<f32> = array![1.0, 2.0];
        let test_target: Array1<f32> = array![3.0, 0.0];
        let test_objective = Objective::MSE;
        let test_expected_value: f32 = 4.0;
        let test_value: f32 = test_objective.compute(&test_input, &test_target).unwrap();

        assert_eq!(test_expected_value, test_value);
    }

    #[test]
    fn test_objective_mse_gradient_valid_args() {
        let test_input: Array1<f32> = array![1.0, 0.0];
        let test_target: Array1<f32> = array![1.0, 0.0];
        let test_objective = Objective::MSE;
        let result = test_objective.gradient(&test_input, &test_target);

        assert!(result.is_ok());
    }

    #[test]
    fn test_objective_mse_gradient_invalid_args() {
        let test_input: Array1<f32> = array![1.0, 0.0, 0.0];
        let test_target: Array1<f32> = array![1.0, 0.0];
        let test_objective = Objective::MSE;
        let result = test_objective.gradient(&test_input, &test_target);

        assert!(result.is_err());
    }

    #[test]
    fn test_objective_mse_gradient() {
        let test_input: Array1<f32> = array![1.0, 2.0];
        let test_target: Array1<f32> = array![3.0, 0.0];
        let test_objective = Objective::MSE;
        let test_expected_value: Array1<f32> = array![-2.0, 2.0];
        let test_value: Array1<f32> = test_objective.gradient(&test_input, &test_target).unwrap();

        assert_eq!(test_expected_value, test_value);
    }
}
