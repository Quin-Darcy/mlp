/*
 * Below I implement the various gradient descent algorithms discussed in the following paper:
 * [1] https://arxiv.org/pdf/1609.04747
 */

use ndarray::{Array1, Array2};

use crate::cache::{BackwardCache, BackwardEntry};
use crate::network::Network;

pub enum Updater {
    // simple gradient descent. whether stochastic, batch, or mini-batch
    // depends only on batch size used
    SGD_SIMPLE {
        learning_rate: f32,
    },
    SGD_MOMENTUM {
        learning_rate: f32,
        gamma: f32,
        update_vector: BackwardCache,
    },
    SGD_NAG {
        learning_rate: f32,
        gamma: f32,
        update_vector: BackwardCache,
    },
}

impl Updater {
    pub fn update(&mut self, cache: &BackwardCache, network: &mut Network) {
        match self {
            /*
             * Letting
             * - theta = network.layers
             * - gradient of objective function = cache
             *
             * then here we are computing
             *
             * theta = theta - (learning_rate) * (gradient of objective function)
             */
            Self::SGD_SIMPLE { learning_rate } => {
                for (layer, entry) in network.layers.iter_mut().zip(cache.entries.iter()) {
                    layer
                        .weights
                        .scaled_add(-*learning_rate, &entry.weight_gradient);
                    layer
                        .biases
                        .scaled_add(-*learning_rate, &entry.bias_gradient);
                }
            }
            /*
             * update_vector = (gamma) * update_vector + (learning_rate) * (gradient of objective
             * function)
             */
            Self::SGD_MOMENTUM {
                learning_rate,
                gamma,
                update_vector,
            } => {
                // For first call we need to initialize update vector
                if update_vector.entries.is_empty() {
                    for entry in &cache.entries {
                        update_vector.entries.push(BackwardEntry {
                            weight_gradient: Array2::<f32>::zeros(entry.weight_gradient.dim()),
                            bias_gradient: Array1::<f32>::zeros(entry.bias_gradient.dim()),
                        });
                    }
                }
                // Update the update vector
                for (uv_entry, c_entry) in
                    update_vector.entries.iter_mut().zip(cache.entries.iter())
                {
                    uv_entry.weight_gradient *= *gamma;
                    uv_entry
                        .weight_gradient
                        .scaled_add(*learning_rate, &c_entry.weight_gradient);

                    uv_entry.bias_gradient *= *gamma;
                    uv_entry
                        .bias_gradient
                        .scaled_add(*learning_rate, &c_entry.bias_gradient);
                }

                // Apply the update vector
                for (layer, uv_entry) in network.layers.iter_mut().zip(update_vector.entries.iter())
                {
                    layer.weights -= &uv_entry.weight_gradient;
                    layer.biases -= &uv_entry.bias_gradient;
                }
            }
            /*
             * TODO: add comment
             */
            Self::SGD_NAG {
                learning_rate,
                gamma,
                update_vector,
            } => {
                if update_vector.entries.is_empty() {
                    for entry in &cache.entries {
                        update_vector.entries.push(BackwardEntry {
                            weight_gradient: Array2::<f32>::zeros(entry.weight_gradient.dim()),
                            bias_gradient: Array1::<f32>::zeros(entry.bias_gradient.dim()),
                        });
                    }
                }

                for (uv_entry, c_entry) in
                    update_vector.entries.iter_mut().zip(cache.entries.iter())
                {
                    uv_entry.weight_gradient *= *gamma;
                    uv_entry
                        .weight_gradient
                        .scaled_add(*learning_rate, &c_entry.weight_gradient);

                    uv_entry.bias_gradient *= *gamma;
                    uv_entry
                        .bias_gradient
                        .scaled_add(*learning_rate, &c_entry.bias_gradient);
                }

                for ((layer, uv_entry), c_entry) in network
                    .layers
                    .iter_mut()
                    .zip(update_vector.entries.iter())
                    .zip(cache.entries.iter())
                {
                    layer.weights.scaled_add(-*gamma, &uv_entry.weight_gradient);

                    layer
                        .weights
                        .scaled_add(-*learning_rate, &c_entry.weight_gradient);

                    layer.biases.scaled_add(-*gamma, &uv_entry.bias_gradient);

                    layer
                        .biases
                        .scaled_add(-*learning_rate, &c_entry.bias_gradient);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::cache::BackwardEntry;
    use crate::layer::Layer;
    use ndarray::{Array1, Array2, array};

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_updater_sgd_simple() {
        let test_weights1: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let test_biases1: Array1<f32> = array![0.5, -0.5];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0];
        let test_activation2 = Activation::IDENTITY;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_layers: Vec<Layer> = vec![test_layer1, test_layer2];
        let mut test_network = Network::new(test_layers).unwrap();

        let test_cache = BackwardCache {
            entries: vec![
                BackwardEntry {
                    weight_gradient: array![[1.0, -1.0], [2.0, 0.0]],
                    bias_gradient: array![1.0, -2.0],
                },
                BackwardEntry {
                    weight_gradient: array![[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
                    bias_gradient: array![1.0, 2.0, 3.0],
                },
            ],
        };

        let test_learning_rate: f32 = 0.1;
        let mut test_updater = Updater::SGD_SIMPLE {
            learning_rate: test_learning_rate,
        };
        test_updater.update(&test_cache, &mut test_network);

        // Each parameter moves by -learning_rate * gradient
        let test_expected_weights1: Array2<f32> = array![[0.9, 2.1], [2.8, 4.0]];
        let test_expected_biases1: Array1<f32> = array![0.4, -0.3];
        let test_expected_weights2: Array2<f32> = array![[1.0, -0.1], [-0.1, 1.0], [0.8, 0.8]];
        let test_expected_biases2: Array1<f32> = array![-0.1, -0.2, -0.3];

        assert!(
            (&test_network.layers[0].weights - &test_expected_weights1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[0].biases - &test_expected_biases1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].weights - &test_expected_weights2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].biases - &test_expected_biases2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }

    #[test]
    fn test_updater_sgd_momentum() {
        let test_weights1: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let test_biases1: Array1<f32> = array![0.5, -0.5];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0];
        let test_activation2 = Activation::IDENTITY;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_layers: Vec<Layer> = vec![test_layer1, test_layer2];
        let mut test_network = Network::new(test_layers).unwrap();

        let test_cache = BackwardCache {
            entries: vec![
                BackwardEntry {
                    weight_gradient: array![[1.0, -1.0], [2.0, 0.0]],
                    bias_gradient: array![1.0, -2.0],
                },
                BackwardEntry {
                    weight_gradient: array![[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
                    bias_gradient: array![1.0, 2.0, 3.0],
                },
            ],
        };

        let test_learning_rate: f32 = 0.1;
        let test_gamma: f32 = 0.9;
        let mut test_updater = Updater::SGD_MOMENTUM {
            learning_rate: test_learning_rate,
            gamma: test_gamma,
            update_vector: BackwardCache::new(),
        };

        // the update vector starts at zero, so v = lr * g and the
        // step is the same as simple gradient descent
        test_updater.update(&test_cache, &mut test_network);

        let test_expected_weights1: Array2<f32> = array![[0.9, 2.1], [2.8, 4.0]];
        let test_expected_biases1: Array1<f32> = array![0.4, -0.3];
        let test_expected_weights2: Array2<f32> = array![[1.0, -0.1], [-0.1, 1.0], [0.8, 0.8]];
        let test_expected_biases2: Array1<f32> = array![-0.1, -0.2, -0.3];

        assert!(
            (&test_network.layers[0].weights - &test_expected_weights1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[0].biases - &test_expected_biases1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].weights - &test_expected_weights2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].biases - &test_expected_biases2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );

        test_updater.update(&test_cache, &mut test_network);

        let test_expected_weights1: Array2<f32> = array![[0.71, 2.29], [2.42, 4.0]];
        let test_expected_biases1: Array1<f32> = array![0.21, 0.08];
        let test_expected_weights2: Array2<f32> = array![[1.0, -0.29], [-0.29, 1.0], [0.42, 0.42]];
        let test_expected_biases2: Array1<f32> = array![-0.29, -0.58, -0.87];

        assert!(
            (&test_network.layers[0].weights - &test_expected_weights1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[0].biases - &test_expected_biases1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].weights - &test_expected_weights2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].biases - &test_expected_biases2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }

    #[test]
    fn test_updater_sgd_nag() {
        let test_weights1: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let test_biases1: Array1<f32> = array![0.5, -0.5];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0];
        let test_activation2 = Activation::IDENTITY;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_layers: Vec<Layer> = vec![test_layer1, test_layer2];
        let mut test_network = Network::new(test_layers).unwrap();

        let test_cache = BackwardCache {
            entries: vec![
                BackwardEntry {
                    weight_gradient: array![[1.0, -1.0], [2.0, 0.0]],
                    bias_gradient: array![1.0, -2.0],
                },
                BackwardEntry {
                    weight_gradient: array![[0.0, 1.0], [1.0, 0.0], [2.0, 2.0]],
                    bias_gradient: array![1.0, 2.0, 3.0],
                },
            ],
        };

        let test_learning_rate: f32 = 0.1;
        let test_gamma: f32 = 0.9;
        let mut test_updater = Updater::SGD_NAG {
            learning_rate: test_learning_rate,
            gamma: test_gamma,
            update_vector: BackwardCache::new(),
        };

        // on first call the update vector starts at zero, so v = lr * g. same update
        // vector caclulation as in momentum variant. But here we need to update parameters
        // so that they become theta_t - gamma*update_vector which requires us update the
        // parameters by -gamma * update_vector - lr * g
        test_updater.update(&test_cache, &mut test_network);

        let test_expected_weights1: Array2<f32> = array![[0.81, 2.19], [2.62, 4.0]];
        let test_expected_biases1: Array1<f32> = array![0.31, -0.12];
        let test_expected_weights2: Array2<f32> = array![[1.0, -0.19], [-0.19, 1.0], [0.62, 0.62]];
        let test_expected_biases2: Array1<f32> = array![-0.19, -0.38, -0.57];

        assert!(
            (&test_network.layers[0].weights - &test_expected_weights1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[0].biases - &test_expected_biases1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].weights - &test_expected_weights2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].biases - &test_expected_biases2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );

        // Second call with the same gradient: v = 0.9 * 0.1g + 0.1g = 0.19g,
        // applied step = 0.9 * 0.19g + 0.1g = 0.271g, so the parameters end
        // at original - (0.19 + 0.271) * g = original - 0.461 * g
        test_updater.update(&test_cache, &mut test_network);

        let test_expected_weights1: Array2<f32> = array![[0.539, 2.461], [2.078, 4.0]];
        let test_expected_biases1: Array1<f32> = array![0.039, 0.422];
        let test_expected_weights2: Array2<f32> =
            array![[1.0, -0.461], [-0.461, 1.0], [0.078, 0.078]];
        let test_expected_biases2: Array1<f32> = array![-0.461, -0.922, -1.383];

        assert!(
            (&test_network.layers[0].weights - &test_expected_weights1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[0].biases - &test_expected_biases1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].weights - &test_expected_weights2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_network.layers[1].biases - &test_expected_biases2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }
}
