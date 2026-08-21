use ndarray::{Array1, Array2};

use crate::activation::Activation;

#[derive(Debug)]
pub enum LayerError {
    InvalidArgDimensions(String),
}

pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
}

impl Layer {
    pub fn new(
        weights: Array2<f32>,
        biases: Array1<f32>,
        activation: Activation,
    ) -> Result<Self, LayerError> {
        if biases.dim() != weights.dim().0 {
            return Err(LayerError::InvalidArgDimensions(
                "Weight matrix rows must equal number of biases".to_string(),
            ));
        }

        Ok(Self {
            weights,
            biases,
            activation,
        })
    }

    #[must_use]
    pub fn forward_pass(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let pre_activations: Array1<f32> = self.weights.dot(input) + &self.biases;
        let post_activations: Array1<f32> = self.activation.apply(&pre_activations);
        (pre_activations, post_activations)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_layer_new_valid_arg() {
        let test_weights: Array2<f32> = array![
            [1.0, 0.3, 1.4],
            [0.0, 1.0, 3.2]
        ];
        let test_biases: Array1<f32> = array![1.0, 1.0];
        let test_activation = Activation::RELU;
        let test_layer = Layer::new(test_weights, test_biases, test_activation);

        assert!(test_layer.is_ok());
    }

    #[test]
    fn test_layer_new_invalid_args() {
        let test_weights: Array2<f32> = array![
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0]
        ];
        let test_biases: Array1<f32> = array![1.0, 1.0, 1.0];
        let test_activation = Activation::RELU;
        let test_layer = Layer::new(test_weights, test_biases, test_activation);

        assert!(test_layer.is_err());
    }

    #[test]
    fn test_layer_forward_pass() {
        let test_activation_relu = Activation::RELU;
        let test_activation_iden = Activation::IDENTITY;

        // Test 1
        let test_weights1: Array2<f32> = array![
            [1.0, 0.5, -1.0, -0.5],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, -1.0]
        ];
        let test_biases1: Array1<f32> = array![-3.0, 0.5, 1.0];
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation_relu).unwrap();

        let test_input1: Array1<f32> = array![1.0, 0.0, 0.0, 0.0];
        let test_expected_pre1: Array1<f32> = array![-2.0, 0.5, 2.0];
        let test_expected_post1: Array1<f32> = array![0.0, 0.5, 2.0];
        assert_eq!(test_layer1.forward_pass(&test_input1), (test_expected_pre1, test_expected_post1));

        // Test 2
        let test_weights2: Array2<f32> = array![];
        let test_biases2: Array1<f32> = array![];
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation_iden).unwrap();
        
        let test_input2: Array1<f32> = array![0.0, -1.0, 1.0, 0.3];
        let test_expected_pre2: Array1<f32> = array![];
        let test_expected_post2: Array1<f32> = array![];
        assert_eq!(test_layer2.forward_pass(&test_input2), (test_expected_pre2, test_expected_post2));

        // Test 3
        let test_weights3: Array2<f32> = array![];
        let test_biases3: Array1<f32> = array![];
        let test_layer3 = Layer::new(test_weights3, test_biases3, test_activation_relu).unwrap();

        let test_input3: Array1<f32> = array![2.5, 0.0, 0.5, 1.0];
        let test_expected_pre3: Array1<f32> = array![];
        let test_expected_post3: Array1<f32> = array![];
        assert_eq!(test_layer3.forward_pass(&test_input3), (test_expected_pre3, test_expected_post3));
    }
}
