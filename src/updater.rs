use crate::cache::BackwardCache;
use crate::layer::Layer;


pub struct UpdaterState { /* TBD */ }

pub enum Updater {
    SGD_SINGLE { learning_rate: f32 }, // stochastic gradient descent for single input
    TBD { learning_rate: f32, state: UpdaterState },
}

impl Updater {
    pub fn update(&mut self, cache: &BackwardCache, layers: &mut [Layer]) {
        match self {
            Self::SGD_SINGLE { learning_rate } => {
                for (layer, entry) in layers.iter_mut().zip(cache.entries.iter()) {
                    layer
                        .weights
                        .scaled_add(-*learning_rate, &entry.weight_gradient);
                    layer
                        .biases
                        .scaled_add(-*learning_rate, &entry.bias_gradient);
                }
            },
            Self::TBD { learning_rate, state } => {}
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
        let test_updater = Updater::SGD_SINGLE {
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
