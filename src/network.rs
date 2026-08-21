use ndarray::Array1;

use crate::cache::{Entry, ForwardCache};
use crate::layer::{LayerError, LayerOutput, Layer};


#[derive(Debug)]
pub enum NetworkError {
    InvalidArgDimensions(String),
}

#[derive(Debug, PartialEq)]
pub struct NetworkOutput {
    pub cache: ForwardCache,
    pub output: Array1<f32>,
}

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    #[must_use]
    pub fn new(layers: Vec<Layer>) -> Result<Self, NetworkError> {
        // Validate that the dimensions of consecutive layers
        for i in 0..layers.len() - 1 {
            if layers[i].weights.dim().0 != layers[i+1].weights.dim().1 {
                return Err(NetworkError::InvalidArgDimensions(
                    "The number of rows in one layer's weight matrix must equal the number of columns in the next layer's weight matrix.".to_string(),
                ));
            }
        }

        Ok(Self {
            layers
        })
    }

    #[must_use]
    pub fn forward_pass(&self, input: &Array1<f32>) -> Result<NetworkOutput, LayerError> {
        let mut layer_input: Array1<f32> = input.clone();
        let mut layer_output: LayerOutput;
        let mut cache = ForwardCache::new();

        for layer in &self.layers {
            layer_output = layer.forward_pass(&layer_input)?;
            cache.entries.push(Entry {
                input: layer_input.clone(),
                pre_activation: layer_output.pre_activation,
            });
            layer_input = layer_output.post_activation;
        }
        Ok(NetworkOutput {
            cache: cache,
            output: layer_input
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};
    use crate::activation::Activation;

    #[test]
    fn test_network_valid_args() {
        let test_weights1: Array2<f32> = array![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let test_biases1: Array1<f32> = array![0.0, 0.0];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ];
        let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0];
        let test_activation2 = Activation::RELU;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_weights3: Array2<f32> = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let test_biases3: Array1<f32> = array![0.0, 0.0];
        let test_activation3 = Activation::RELU;
        let test_layer3 = Layer::new(test_weights3, test_biases3, test_activation3).unwrap();

        let layers: Vec<Layer> = vec![test_layer1, test_layer2, test_layer3];
        
        let result = Network::new(layers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_network_invalid_args() {
        let test_weights1: Array2<f32> = array![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let test_biases1: Array1<f32> = array![0.0, 0.0];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0]
        ];
        let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0];
        let test_activation2 = Activation::RELU;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_weights3: Array2<f32> = array![
            [1.0, 0.0],
            [0.0, 1.0],
        ];
        let test_biases3: Array1<f32> = array![0.0, 0.0];
        let test_activation3 = Activation::RELU;
        let test_layer3 = Layer::new(test_weights3, test_biases3, test_activation3).unwrap();

        let layers: Vec<Layer> = vec![test_layer1, test_layer2, test_layer3];
        let result = Network::new(layers);
        assert!(result.is_err());
    }

    #[test]
    fn test_network_forward_pass() {
       let test_weights1: Array2<f32> = array![
            [1.0, 0.0, 0.5, 0.5],
            [0.0, 1.0, 0.5, -0.5]
       ];
       let test_biases1: Array1<f32> = array![0.0, 0.0];
       let test_activation1 = Activation::RELU;
       let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

       let test_weights2: Array2<f32> = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.5, -0.5]
       ];
       let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0, 0.0];
       let test_activation2 = Activation::RELU;
       let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

       let layers: Vec<Layer> = vec![test_layer1, test_layer2];
       let test_network = Network::new(layers).unwrap();

       let test_input: Array1<f32> = array![0.0, 0.0, 0.0, 1.0];
       let result = test_network.forward_pass(&test_input);
    }
}
