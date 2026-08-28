use ndarray::{Array1, Array2, Axis};

use crate::cache::{BackwardCache, BackwardEntry, ForwardCache, ForwardEntry};
use crate::layer::{Layer, LayerError, LayerOutput};

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
            if layers[i].weights.dim().0 != layers[i + 1].weights.dim().1 {
                return Err(NetworkError::InvalidArgDimensions(
                    "The number of rows in one layer's weight matrix must equal the number of columns in the next layer's weight matrix.".to_string(),
                ));
            }
        }

        Ok(Self { layers })
    }

    #[must_use]
    pub fn forward_pass(&self, input: &Array1<f32>) -> Result<NetworkOutput, LayerError> {
        let mut layer_input: Array1<f32> = input.clone();
        let mut layer_output: LayerOutput;
        let mut cache = ForwardCache::new();

        for layer in &self.layers {
            layer_output = layer.forward_pass(&layer_input)?; // match to prop error type?
            cache.entries.push(ForwardEntry {
                input: layer_input.clone(),
                pre_activation: layer_output.pre_activation,
            });
            layer_input = layer_output.post_activation;
        }
        Ok(NetworkOutput {
            cache,
            output: layer_input, // this is the activated output of the last layer
        })
    }

    #[must_use]
    pub fn backward_pass(
        &self,
        network_output: &NetworkOutput,
        objective_gradient: &Array1<f32>,
    ) -> BackwardCache {
        let mut cache = BackwardCache::new();
        let mut carryover: Array1<f32> = objective_gradient.clone();

        let num_layers: usize = self.layers.len();
        for i in (0..num_layers).rev() {
            let delta: Array1<f32> = self.layers[i]
                .activation
                .jacobian(&network_output.cache.entries[i].pre_activation)
                * carryover;
            let activation: Array1<f32> = network_output.cache.entries[i].input.clone();
            let weight_gradient: Array2<f32> =
                &delta.view().insert_axis(Axis(1)) * &activation.view().insert_axis(Axis(0));

            cache.entries.push(BackwardEntry {
                weight_gradient,
                bias_gradient: delta.clone(),
            });

            carryover = delta.dot(&self.layers[i].weights);
        }

        cache.entries.reverse();
        cache
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use ndarray::{Array2, array};

    #[test]
    fn test_network_valid_args() {
        let test_weights1: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0]];
        let test_biases1: Array1<f32> = array![0.0, 0.0];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0];
        let test_activation2 = Activation::RELU;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_weights3: Array2<f32> = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],];
        let test_biases3: Array1<f32> = array![0.0, 0.0];
        let test_activation3 = Activation::RELU;
        let test_layer3 = Layer::new(test_weights3, test_biases3, test_activation3).unwrap();

        let layers: Vec<Layer> = vec![test_layer1, test_layer2, test_layer3];

        let result = Network::new(layers);
        assert!(result.is_ok());
    }

    #[test]
    fn test_network_invalid_args() {
        let test_weights1: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0]];
        let test_biases1: Array1<f32> = array![0.0, 0.0];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0];
        let test_activation2 = Activation::RELU;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_weights3: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0],];
        let test_biases3: Array1<f32> = array![0.0, 0.0];
        let test_activation3 = Activation::RELU;
        let test_layer3 = Layer::new(test_weights3, test_biases3, test_activation3).unwrap();

        let test_layers: Vec<Layer> = vec![test_layer1, test_layer2, test_layer3];
        let result = Network::new(test_layers);
        assert!(result.is_err());
    }

    #[test]
    fn test_network_forward_pass() {
        // Test 1
        let test_weights11: Array2<f32> = array![[1.0, 0.0, 0.5, 0.5], [0.0, 1.0, 0.5, -0.5]];
        let test_biases11: Array1<f32> = array![0.0, 0.0];
        let test_activation11 = Activation::RELU;
        let test_layer11 = Layer::new(test_weights11, test_biases11, test_activation11).unwrap();

        let test_weights12: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.5, -0.5]];
        let test_biases12: Array1<f32> = array![0.0, 0.0, 0.0, 0.0];
        let test_activation12 = Activation::IDENTITY;
        let test_layer12 = Layer::new(test_weights12, test_biases12, test_activation12).unwrap();

        let test_layers1: Vec<Layer> = vec![test_layer11, test_layer12];
        let test_network1 = Network::new(test_layers1).unwrap();

        let test_input1: Array1<f32> = array![0.0, 0.0, 0.0, 1.0];
        let test_expected_output1: Array1<f32> = array![0.5, 0.0, 0.25, 0.25];
        let result1 = test_network1.forward_pass(&test_input1).unwrap();

        assert_eq!(result1.output, test_expected_output1);

        // Test 2
        let test_weights21: Array2<f32> = array![[1.0, 2.0], [0.0, 5.0], [1.0, 1.0]];
        let test_biases21: Array1<f32> = array![-3.0, 1.0, -32.0];
        let test_activation21 = Activation::RELU;
        let test_layer21 = Layer::new(test_weights21, test_biases21, test_activation21).unwrap();

        let test_weights22: Array2<f32> = array![[-1.0, 0.0, 1.0], [0.0, 1.0, 0.0]];
        let test_biases22: Array1<f32> = array![7.0, -20.0];
        let test_activation22 = Activation::RELU;
        let test_layer22 = Layer::new(test_weights22, test_biases22, test_activation22).unwrap();

        let test_weights23: Array2<f32> = array![[2.0, -2.0], [1.0, -1.0], [-2.0, -1.0]];
        let test_biases23: Array1<f32> = array![3.0, -6.0, 5.2];
        let test_activation23 = Activation::RELU;
        let test_layer23 = Layer::new(test_weights23, test_biases23, test_activation23).unwrap();

        let test_layers2: Vec<Layer> = vec![test_layer21, test_layer22, test_layer23];
        let test_network2 = Network::new(test_layers2).unwrap();

        let test_input2: Array1<f32> = array![3.0, 4.0];
        let test_expected_output2: Array1<f32> = array![1.0, 0.0, 4.2];
        let result2 = test_network2.forward_pass(&test_input2).unwrap();

        assert_eq!(result2.output, test_expected_output2);
    }
}
