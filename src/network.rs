use ndarray::{Array1, Array2, Axis};

use crate::cache::{BackwardCache, BackwardEntry, ForwardCache, ForwardEntry};
use crate::layer::{Layer, LayerError, LayerOutput};
use crate::objective::Objective;
use crate::optimizer::Optimizer;

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

    pub fn train(
        &mut self,
        epochs: usize,
        optimizer: Optimizer,
        objective: Objective,
        learning_rate: f32,
        data: Vec<(Array1<f32>, Array1<f32>)>   // maybe better as custom type?
    ) -> Result<(), NetworkError> {
        for _ in 0..epochs {
            // below seems hardcoded to singleton sgd. need to revise to allow 
            // for batching depending on optimizer choice
            for i in 0..data.len() {
                let network_out: NetworkOutput = self.forward_pass(&data[i].0).unwrap();
                let gradient_objective = objective.gradient(&network_out.output, &data[i].1).unwrap();
                let network_back_prop: BackwardCache = self.backward_pass(&network_out, &gradient_objective);
                optimizer.update(learning_rate, &network_back_prop, &mut self.layers);
            }
        }
        Ok(())
    }
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
    fn test_network_backward_pass_optimize() {
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

        let test_optimizer = Optimizer::SGD_SINGLE;
        let test_learning_rate: f32 = 0.01;

        let test_pre_update_objective: f32 = test_objective
            .compute(&test_output.output, &test_target)
            .unwrap();

        // Update parameters
        test_optimizer.update(
            test_learning_rate,
            &test_back_prop,
            &mut test_network.layers,
        );

        let test_post_update_output: NetworkOutput =
            test_network.forward_pass(&test_input).unwrap();
        let test_post_update_objective: f32 = test_objective
            .compute(&test_post_update_output.output, &test_target)
            .unwrap();

        assert!(test_post_update_objective < test_pre_update_objective);
    }
}
