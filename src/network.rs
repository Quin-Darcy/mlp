use ndarray::{Array1, Array2};

use crate::cache::{BackwardCache, BackwardEntry, ForwardCache, ForwardEntry};
use crate::layer::{Layer, LayerError, LayerOutput};
use crate::objective::{Objective, ObjectiveError};
use crate::updater::Updater;

#[derive(Debug)]
pub enum NetworkError {
    InvalidArgDimensions(String),
    BadForwardPass(String),
    BadObjectiveGradient(String),
}

impl From<LayerError> for NetworkError {
    fn from(e: LayerError) -> Self {
        NetworkError::BadForwardPass(format!("{e:?}"))
    }
}

impl From<ObjectiveError> for NetworkError {
    fn from(e: ObjectiveError) -> Self {
        NetworkError::BadObjectiveGradient(format!("{e:?}"))
    }
}

#[derive(Debug, PartialEq)]
pub struct NetworkOutput {
    pub cache: ForwardCache,
    pub output: Array1<f32>,
}

pub struct Network {
    pub layers: Vec<Layer>,
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
    pub fn forward_pass(&self, input: &Array1<f32>) -> Result<NetworkOutput, NetworkError> {
        let mut layer_input: Array1<f32> = input.clone();
        let mut layer_output: LayerOutput;
        let mut cache = ForwardCache::new();

        for layer in &self.layers {
            layer_output = layer.forward_pass(&layer_input)?;
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
        // This cache will hold the weight and bias gradients
        // accumulated through the back prop
        let mut backward_cache = BackwardCache::new();

        // This is the intermediate value which gets carried backwards
        // from one layer to the next. For the last layer (first in
        // back prop), it holds the gradient of the objective function.
        // For all subsequent layers, it holds the product of the error
        // term (delta) and the current layer's weight matrix
        //
        // TODO: Add more intuition and better naming
        let mut carryover: Array1<f32> = objective_gradient.clone();

        // The three primary objects updated each layer
        let mut delta: Array1<f32>;
        let mut weight_gradient: Array2<f32>;
        let mut bias_gradient: Array1<f32>;

        // main back prop loop
        for (layer, forward_cache_entry) in self
            .layers
            .iter()
            .rev()
            .zip(network_output.cache.entries.iter().rev())
        {
            delta = layer
                .activation
                .jacobian(&forward_cache_entry.pre_activation)
                * carryover;
            weight_gradient = outer_product(&delta, &forward_cache_entry.input);
            bias_gradient = delta.clone();

            backward_cache.entries.push(BackwardEntry {
                bias_gradient,
                weight_gradient,
            });

            carryover = delta.dot(&layer.weights);
        }

        backward_cache.entries.reverse();
        backward_cache
    }

    pub fn train(
        &mut self,
        epochs: usize,
        updater: &mut Updater,
        objective: &Objective,
        data: &[(Array1<f32>, Array1<f32>)], // maybe better as custom type?
    ) -> Result<(), NetworkError> {
        for _ in 0..epochs {
            // below seems hardcoded to singleton sgd. need to revise to allow
            // for batching depending on optimizer choice
            for item in data {
                let network_out: NetworkOutput = self.forward_pass(&item.0)?;
                let gradient_objective = objective.gradient(&network_out.output, &item.1)?;
                let network_back_prop: BackwardCache =
                    self.backward_pass(&network_out, &gradient_objective);
                updater.update(&network_back_prop, &mut self.layers);
            }
        }
        Ok(())
    }
}

fn outer_product(delta: &Array1<f32>, input: &Array1<f32>) -> Array2<f32> {
    Array2::from_shape_fn((delta.dim(), input.dim()), |(i, j)| delta[i] * input[j])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use ndarray::{Array2, array};

    const EPSILON: f32 = 0.0001;

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

        assert!(
            (&result1.output - &test_expected_output1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );

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

        assert!(
            (&result2.output - &test_expected_output2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }

    #[test]
    fn test_network_backward_pass() {
        // Test 1
        let test_weights1: Array2<f32> = array![[0.5, -0.3], [0.2, 0.8]];
        let test_biases1: Array1<f32> = array![0.2, -0.1];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_layers1: Vec<Layer> = vec![test_layer1];
        let test_network1 = Network::new(test_layers1).unwrap();

        let test_input1: Array1<f32> = array![1.0, 2.0];
        let test_output1: NetworkOutput = test_network1.forward_pass(&test_input1).unwrap();

        let test_target1: Array1<f32> = array![1.0, 3.0];
        let test_objective = Objective::MSE;
        let test_objective_gradient = test_objective
            .gradient(&test_output1.output, &test_target1)
            .unwrap();

        let test_back_prop: BackwardCache =
            test_network1.backward_pass(&test_output1, &test_objective_gradient);

        let test_expected_weight_gradient1: Array2<f32> = array![[-0.9, -1.8], [-1.3, -2.6]];
        let test_expected_bias_gradient1: Array1<f32> = array![-0.9, -1.3];

        assert!(
            (&test_back_prop.entries[0].weight_gradient - &test_expected_weight_gradient1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_back_prop.entries[0].bias_gradient - &test_expected_bias_gradient1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );

        // Test 2
        let test_weights21: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let test_biases21: Array1<f32> = array![0.5, 0.5, 0.5];
        let test_activation21 = Activation::RELU;
        let test_layer21 = Layer::new(test_weights21, test_biases21, test_activation21).unwrap();

        let test_weights22: Array2<f32> = array![[1.0, 1.0, 1.0], [-1.0, 0.0, 2.0]];
        let test_biases22: Array1<f32> = array![0.0, 0.0];
        let test_activation22 = Activation::RELU;
        let test_layer22 = Layer::new(test_weights22, test_biases22, test_activation22).unwrap();

        let test_weights23: Array2<f32> = array![[1.0, 2.0], [0.5, -1.0]];
        let test_biases23: Array1<f32> = array![0.0, 1.0];
        let test_activation23 = Activation::IDENTITY;
        let test_layer23 = Layer::new(test_weights23, test_biases23, test_activation23).unwrap();

        let test_layers2: Vec<Layer> = vec![test_layer21, test_layer22, test_layer23];
        let test_network2 = Network::new(test_layers2).unwrap();

        let test_input2: Array1<f32> = array![1.0, -1.0];
        let test_output2: NetworkOutput = test_network2.forward_pass(&test_input2).unwrap();

        let test_target2: Array1<f32> = array![1.0, 0.0];
        let test_objective_gradient2 = test_objective
            .gradient(&test_output2.output, &test_target2)
            .unwrap();

        let test_back_prop2: BackwardCache =
            test_network2.backward_pass(&test_output2, &test_objective_gradient2);

        let test_expected_weight_gradient21: Array2<f32> =
            array![[2.0, -2.0], [0.0, 0.0], [2.0, -2.0]];
        let test_expected_bias_gradient21: Array1<f32> = array![2.0, 0.0, 2.0];
        let test_expected_weight_gradient22: Array2<f32> = array![[3.0, 0.0, 1.0], [0.0, 0.0, 0.0]];
        let test_expected_bias_gradient22: Array1<f32> = array![2.0, 0.0];
        let test_expected_weight_gradient23: Array2<f32> = array![[2.0, 0.0], [4.0, 0.0]];
        let test_expected_bias_gradient23: Array1<f32> = array![1.0, 2.0];

        assert_eq!(test_back_prop2.entries.len(), 3);
        assert!(
            (&test_back_prop2.entries[0].weight_gradient - &test_expected_weight_gradient21)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_back_prop2.entries[0].bias_gradient - &test_expected_bias_gradient21)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_back_prop2.entries[1].weight_gradient - &test_expected_weight_gradient22)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_back_prop2.entries[1].bias_gradient - &test_expected_bias_gradient22)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_back_prop2.entries[2].weight_gradient - &test_expected_weight_gradient23)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_back_prop2.entries[2].bias_gradient - &test_expected_bias_gradient23)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }

    #[test]
    fn test_network_backward_pass_update() {
        let test_weights1: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let test_biases1: Array1<f32> = array![0.5, 0.5, 0.5];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![[1.0, 1.0, 1.0], [-1.0, 0.0, 2.0]];
        let test_biases2: Array1<f32> = array![0.0, 0.0];
        let test_activation2 = Activation::RELU;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_weights3: Array2<f32> = array![[1.0, 2.0], [0.5, -1.0]];
        let test_biases3: Array1<f32> = array![0.0, 1.0];
        let test_activation3 = Activation::IDENTITY;
        let test_layer3 = Layer::new(test_weights3, test_biases3, test_activation3).unwrap();

        let test_layers: Vec<Layer> = vec![test_layer1, test_layer2, test_layer3];
        let mut test_network = Network::new(test_layers).unwrap();

        let test_input: Array1<f32> = array![1.0, -1.0];
        let test_output: NetworkOutput = test_network.forward_pass(&test_input).unwrap();

        let test_target: Array1<f32> = array![1.0, 0.0];
        let test_objective = Objective::MSE;
        let test_objective_gradient = test_objective
            .gradient(&test_output.output, &test_target)
            .unwrap();

        let test_back_prop: BackwardCache =
            test_network.backward_pass(&test_output, &test_objective_gradient);

        let test_learning_rate: f32 = 0.01;
        let mut test_updater = Updater::GD {
            learning_rate: test_learning_rate,
        };

        let test_pre_update_objective: f32 = test_objective
            .compute(&test_output.output, &test_target)
            .unwrap();

        // Update parameters
        test_updater.update(&test_back_prop, &mut test_network.layers);

        let test_post_update_output: NetworkOutput =
            test_network.forward_pass(&test_input).unwrap();
        let test_post_update_objective: f32 = test_objective
            .compute(&test_post_update_output.output, &test_target)
            .unwrap();

        assert!(test_post_update_objective < test_pre_update_objective);
    }

    #[test]
    fn test_network_util_outer_product() {
        let test_vec1: Array1<f32> = array![-1.8, -2.6];
        let test_vec2: Array1<f32> = array![1.0, 2.0];
        let test_expected: Array2<f32> = array![[-1.8, -3.6], [-2.6, -5.2]];
        let test_result: Array2<f32> = outer_product(&test_vec1, &test_vec2);

        assert!(
            (&test_result - &test_expected)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }
}
