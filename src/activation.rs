use ndarray::Array1;

pub enum Activation {
    RELU,
    IDENTITY,
}

impl Activation {
    #[must_use]
    pub fn apply(&self, pre_activation: &Array1<f32>) -> Array1<f32> {
        match self {
            Self::RELU => pre_activation.mapv(|x| x.max(0.0)),
            Self::IDENTITY => pre_activation.clone(),
        }
    }

    pub fn jacobian(&self, pre_activation: &Array1<f32>) -> Array1<f32> {
        /* RELU and the IDENTITY act component wise and so the jacobian
         * is always a diagnonal matrix. That is why we only return a row
         * vector and we will replace the matrix multiplication with a
         * Hadamard product to get the same result, just more efficiently
         */
        let dim: usize = pre_activation.dim();
        let mut jacobian: Array1<f32> = Array1::ones(dim);

        match self {
            Self::RELU => {
                for i in 0..dim {
                    if pre_activation[i] < 0.0 {
                        jacobian[i] = 0.0;
                    }
                }
            }
            Self::IDENTITY => {}
        }
        jacobian
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    // Add new test for each new activation function added

    #[test]
    fn test_activation_relu() {
        let test_activation = Activation::RELU;

        let test_input1: Array1<f32> = array![1.0, 0.0];
        let test_expected1: Array1<f32> = array![1.0, 0.0];
        assert_eq!(test_activation.apply(&test_input1), test_expected1);

        let test_input2: Array1<f32> = array![-1.0, 0.0];
        let test_expected2: Array1<f32> = array![0.0, 0.0];
        assert_eq!(test_activation.apply(&test_input2), test_expected2);
    }

    #[test]
    fn test_activation_identity() {
        let test_activation = Activation::IDENTITY;

        let test_input: Array1<f32> = array![23.4, 0.5, -9.9];
        let test_expected: Array1<f32> = test_input.clone();
        assert_eq!(test_activation.apply(&test_input), test_expected);
    }

    #[test]
    fn test_activation_relu_jacobian() {
        let test_activation = Activation::RELU;

        let test_pre_activation: Array1<f32> = array![2.0, -1.0, -0.1, 0.1, 0.0];
        let test_expected_result: Array1<f32> = array![1.0, 0.0, 0.0, 1.0, 1.0];

        assert_eq!(
            test_activation.jacobian(&test_pre_activation),
            test_expected_result
        );
    }

    #[test]
    fn test_activation_identity_jacobian() {
        let test_activation = Activation::IDENTITY;

        let test_pre_activation: Array1<f32> = array![2.0, -1.0, -0.1, 0.1, 0.0];
        let test_expected_result: Array1<f32> = array![1.0, 1.0, 1.0, 1.0, 1.0];

        assert_eq!(
            test_activation.jacobian(&test_pre_activation),
            test_expected_result
        );
    }
}
