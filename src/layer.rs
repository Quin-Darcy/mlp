use ndarray::{Array1, Array2};

use crate::activation::Activation;

#[derive(Debug)]
pub enum LayerError {
    InvalidArgDimensions(String),
}

#[derive(Debug, PartialEq)]
pub struct LayerOutput {
    pub pre_activation: Array1<f32>,
    pub post_activation: Array1<f32>,
}

pub struct Layer {
    pub weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
}

impl Layer {
    #[must_use]
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
    pub fn forward_pass(&self, input: &Array1<f32>) -> Result<LayerOutput, LayerError> {
        // Validate input against dimensions of layer's weights
        if input.dim() != self.weights.dim().1 {
            return Err(LayerError::InvalidArgDimensions(
                "Length of input must equal number of columns in weight matrix".to_string(),
            ));
        }
        let pre_activation: Array1<f32> = self.weights.dot(input) + &self.biases;
        let post_activation: Array1<f32> = self.activation.apply(&pre_activation);
        Ok(LayerOutput { 
            pre_activation, 
            post_activation
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn test_layer_new_valid_args() {
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
    fn test_layer_forward_pass_valid_args() {
        let test_weights: Array2<f32> = array![
            [1.0, 0.5, -1.0, -0.5],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, -1.0]
        ];
        let test_biases: Array1<f32> = array![-3.0, 0.5, 1.0];
        let test_activation = Activation::RELU;
        let test_layer = Layer::new(test_weights, test_biases, test_activation).unwrap();

        let test_input: Array1<f32> = array![1.0, 0.0, 0.0, 0.0];
        let test_result = test_layer.forward_pass(&test_input);
        assert!(test_result.is_ok());
    }

    #[test]
    fn test_layer_forward_pass_invalid_args() {
        let test_weights: Array2<f32> = array![
            [1.0, 0.5, -1.0, -0.5],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, -1.0]
        ];
        let test_biases: Array1<f32> = array![-3.0, 0.5, 1.0];
        let test_activation = Activation::RELU;
        let test_layer = Layer::new(test_weights, test_biases, test_activation).unwrap();

        let test_input: Array1<f32> = array![1.0, 0.0];
        let test_result = test_layer.forward_pass(&test_input);
        assert!(test_result.is_err());
    }

    #[test]
    fn test_layer_forward_pass() {
        // Test 1
        let test_weights1: Array2<f32> = array![
            [1.0, 0.5, -1.0, -0.5],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, -1.0]
        ];
        let test_biases1: Array1<f32> = array![-3.0, 0.5, 1.0];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_input1: Array1<f32> = array![1.0, 0.0, 0.0, 0.0];
        let test_expected_pre1: Array1<f32> = array![-2.0, 0.5, 2.0];
        let test_expected_post1: Array1<f32> = array![0.0, 0.5, 2.0];
        let test_expected_output1 = LayerOutput {
            pre_activation: test_expected_pre1,
            post_activation: test_expected_post1
        };
        assert_eq!(
            test_layer1.forward_pass(&test_input1).unwrap(), 
            test_expected_output1
        );

        // Test 2
        let test_weights2: Array2<f32> = array![
            [1.0, -1.0, 2.0, -2.0],
            [-2.0, 2.0, -1.0, 1.0]
        ];
        let test_biases2: Array1<f32> = array![-2.4, 2.7];
        let test_activation2 = Activation::IDENTITY;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();
        
        let test_input2: Array1<f32> = array![0.0, -1.0, 1.0, 0.3];
        let test_expected_pre2: Array1<f32> = array![0.0, 0.0];
        let test_expected_post2: Array1<f32> = array![0.0, 0.0];
        let test_expected_output2 = LayerOutput {
            pre_activation: test_expected_pre2,
            post_activation: test_expected_post2
        };
        assert_eq!(
            test_layer2.forward_pass(&test_input2).unwrap(), 
            test_expected_output2
        );

        // Test 3
        let test_weights3: Array2<f32> = array![
            [-1.0, 2.0, -3.0, 4.0, -5.0],
            [1.0, -2.0, 3.0, -4.0, 5.0],
            [1.0, 0.0, 1.0, 0.0, 1.0]
        ];
        let test_biases3: Array1<f32> = array![26.0, -26.0, 1.0];
        let test_activation3 = Activation::RELU;
        let test_layer3 = Layer::new(test_weights3, test_biases3, test_activation3).unwrap();

        let test_input3: Array1<f32> = array![2.5, 0.0, 0.5, 1.0, 5.0];
        let test_expected_pre3: Array1<f32> = array![1.0, -1.0, 9.0];
        let test_expected_post3: Array1<f32> = array![1.0, 0.0, 9.0];
        let test_expected_output3 = LayerOutput {
            pre_activation: test_expected_pre3,
            post_activation: test_expected_post3
        };
        assert_eq!(
            test_layer3.forward_pass(&test_input3).unwrap(), 
            test_expected_output3
        );
    }
}
