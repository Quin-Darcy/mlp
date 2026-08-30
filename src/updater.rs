use ndarray::{Array1, Array2};

use crate::cache::{BackwardCache, BackwardEntry};
use crate::layer::Layer;

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
}

impl Updater {
    pub fn update(&mut self, cache: &BackwardCache, layers: &mut [Layer]) {
        match self {
            Self::SGD_SIMPLE { learning_rate } => {
                for (layer, entry) in layers.iter_mut().zip(cache.entries.iter()) {
                    layer
                        .weights
                        .scaled_add(-*learning_rate, &entry.weight_gradient);
                    layer
                        .biases
                        .scaled_add(-*learning_rate, &entry.bias_gradient);
                }
            }
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
                for (layer, uv_entry) in layers.iter_mut().zip(update_vector.entries.iter()) {
                    layer.weights -= &uv_entry.weight_gradient;
                    layer.biases -= &uv_entry.bias_gradient;
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
    use ndarray::{Array1, Array2, array};

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_updater_sgd_single_update() {
        let test_weights1: Array2<f32> = array![[1.0, 2.0], [3.0, 4.0]];
        let test_biases1: Array1<f32> = array![0.5, -0.5];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let test_biases2: Array1<f32> = array![0.0, 0.0, 0.0];
        let test_activation2 = Activation::IDENTITY;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let mut test_layers: Vec<Layer> = vec![test_layer1, test_layer2];

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
        test_updater.update(&test_cache, &mut test_layers);

        // Each parameter moves by -learning_rate * gradient
        let test_expected_weights1: Array2<f32> = array![[0.9, 2.1], [2.8, 4.0]];
        let test_expected_biases1: Array1<f32> = array![0.4, -0.3];
        let test_expected_weights2: Array2<f32> = array![[1.0, -0.1], [-0.1, 1.0], [0.8, 0.8]];
        let test_expected_biases2: Array1<f32> = array![-0.1, -0.2, -0.3];

        assert!(
            (&test_layers[0].weights - &test_expected_weights1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_layers[0].biases - &test_expected_biases1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_layers[1].weights - &test_expected_weights2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_layers[1].biases - &test_expected_biases2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }
}
