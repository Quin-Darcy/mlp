use ndarray::{Array1, array};

pub enum Activation {
    RELU,
    IDENTITY,
}

impl Activation {
    pub fn apply(&self, pre_activations: &Array1<f32>) -> Array1<f32> {
        match self {
            Self::RELU => pre_activations.mapv(|x| x.max(0.0)),
            Self::IDENTITY => pre_activations.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
