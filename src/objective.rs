use ndarray::Array1;

#[derive(Debug)]
pub enum ObjectiveError {
    InvalidArgDimensions(String),
}

pub enum Objective {
    MSE,
    NLL,
}

impl Objective {
    pub fn compute(
        &self,
        input: &Array1<f32>,
        target: &Array1<f32>,
    ) -> Result<f32, ObjectiveError> {
        match self {
            Self::MSE => {
                if input.dim() != target.dim() {
                    return Err(ObjectiveError::InvalidArgDimensions(
                        "input and target vectors must be the same length".to_string(),
                    ));
                }

                let diff: Array1<f32> = target - input;
                Ok(diff.dot(&diff) / diff.len() as f32)
            },
            Self::NLL => {
               if target.dim() != 1 {
                    return Err(ObjectiveError::InvalidArgDimensions(
                        "target must have only one element".to_string(),
                    ));
               }

               let index: usize = target[0] as usize;
               if index >= input.dim() {
                    return Err(ObjectiveError::InvalidArgDimensions(
                        "target index cannot be greater than the number of elements in input".to_string(),
                    ));
               }
               Ok(-input[index].ln())
            }
        }
    }

    pub fn gradient(
        &self,
        input: &Array1<f32>,
        target: &Array1<f32>,
    ) -> Result<Array1<f32>, ObjectiveError> {
        match self {
            Self::MSE => {
                if input.dim() != target.dim() {
                    return Err(ObjectiveError::InvalidArgDimensions(
                        "input and target vectors must be the same length".to_string(),
                    ));
                }

                let scalar: f32 = -2.0 / (input.dim() as f32);
                let diff: Array1<f32> = target - input;
                Ok(scalar * diff)
            },
            Self::NLL => {
                if target.dim() != 1 {
                    return Err(ObjectiveError::InvalidArgDimensions(
                        "target must have only one element".to_string(),
                    ));
                }

                let index: usize = target[0] as usize;
                if index >= input.dim() {
                    return Err(ObjectiveError::InvalidArgDimensions(
                        "target index cannot be greater than the number of elements in input".to_string(),
                    ));
                }
                
                let mut v: Vec<f32> = Vec::with_capacity(input.dim());
                for i in 0..input.dim() {
                    if i == index {
                        v.push(input[i] - 1.0_f32);
                    } else {
                        v.push(input[i]);
                    }
                }

                Ok(Array1::from_vec(v))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    const EPSILON: f32 = 0.0001;

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

        assert!((test_value - test_expected_value).abs() < EPSILON);
    }

    #[test]
    fn test_objective_nll_compute_valid_args() {
        let test_input: Array1<f32> = array![1.0];
        let test_target: Array1<f32> = array![0.0];
        let test_objective = Objective::NLL;
        let result = test_objective.compute(&test_input, &test_target);

        assert!(result.is_ok());
    }

    #[test]
    fn test_objective_nll_compute_invalid_target_dim() {
        let test_input: Array1<f32> = array![1.0, 0.0];
        let test_target: Array1<f32> = array![1.0, 0.0];
        let test_objective = Objective::NLL;
        let result = test_objective.compute(&test_input, &test_target);

        assert!(result.is_err());
    }

    #[test]
    fn test_objective_nll_compute_invalid_target_index() {
        let test_input: Array1<f32> = array![1.0, 0.0];
        let test_target: Array1<f32> = array![4.0];
        let test_objective = Objective::NLL;
        let result = test_objective.compute(&test_input, &test_target);

        assert!(result.is_err());
    }

    #[test]
    fn test_objective_nll_compute() {
        let test_input: Array1<f32> = array![(7.0_f32).exp(), 2.0];
        let test_target: Array1<f32> = array![0.0];
        let test_objective = Objective::NLL;
        let test_expected_value: f32 = -7.0;
        let test_value: f32 = test_objective.compute(&test_input, &test_target).unwrap();

        assert!((test_value - test_expected_value).abs() < EPSILON);
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

        assert!(
            (&test_value - &test_expected_value)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }
}
