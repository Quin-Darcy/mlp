use ndarray::{Array1, Array2};

use crate::cache::{BackwardCache, BackwardEntry};
use crate::data_set::DataSet;
use crate::network::{Network, NetworkError, NetworkOutput};
use crate::objective::{Objective, ObjectiveError};
use crate::updater::Updater;

#[derive(Debug)]
pub enum TrainerError {
    BadForwardPass(String),
    BadObjectiveGradient(String),
    EmptyBatch(String),
    InvalidBatchSize(String),
    UninitializedCache(String),
    EmptyDataSet(String),
}

impl From<NetworkError> for TrainerError {
    fn from(e: NetworkError) -> Self {
        TrainerError::BadForwardPass(format!("{e:?}"))
    }
}

impl From<ObjectiveError> for TrainerError {
    fn from(e: ObjectiveError) -> Self {
        TrainerError::BadObjectiveGradient(format!("{e:?}"))
    }
}

#[derive(Default, Debug)]
pub struct Batch {
    items: Vec<BackwardCache>,
}

impl Batch {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

pub struct Trainer {
    pub network: Network,
    pub objective: Objective,
    pub updater: Updater,
}

impl Trainer {
    #[must_use]
    pub fn new(network: Network, objective: Objective, updater: Updater) -> Self {
        Self {
            network,
            objective,
            updater,
        }
    }

    pub fn run(
        &mut self,
        data: &DataSet,
        batch_size: usize,
        epochs: usize,
    ) -> Result<(), TrainerError> {
        if data.samples.len() < batch_size {
            return Err(TrainerError::InvalidBatchSize(
                "Batch size cannot exceed number of samples in data set".to_string(),
            ));
        }

        // Handle when batch size doesn't divide num samples. reject it for now
        // TODO: consider handling this more elaborately/permissively
        if !data.samples.len().is_multiple_of(batch_size) {
            return Err(TrainerError::InvalidBatchSize(
                "Batch size must divide the number of samples in data set".to_string(),
            ));
        }

        let mut batch = Batch::new();
        let mut aggregated_gradients = BackwardCache::new();
        let data_size: usize = data.samples.len();
        for _ in 0..epochs {
            // todo: learn rayon enough to figure out how to parallelize this part
            for i in (0..data_size).step_by(batch_size) {
                batch.items.clear();
                for j in 0..batch_size {
                    let network_out: NetworkOutput =
                        self.network.forward_pass(&data.samples[i + j])?;
                    let gradient_objective: Array1<f32> = self
                        .objective
                        .gradient(&network_out.output, &data.labels[i + j])?;

                    batch.items.push(
                        self.network
                            .backward_pass(&network_out, &gradient_objective),
                    );
                }

                // initialize the aggregated_gradients only once after we
                // have a backwards cache struct whose shapes we can copy
                if aggregated_gradients.entries.is_empty() && !batch.items.is_empty() {
                    for entry in &batch.items[0].entries {
                        aggregated_gradients.entries.push(BackwardEntry {
                            weight_gradient: Array2::<f32>::zeros(entry.weight_gradient.dim()),
                            bias_gradient: Array1::<f32>::zeros(entry.bias_gradient.dim()),
                        });
                    }
                }

                aggregate_batch(&batch, &mut aggregated_gradients)?;
                self.updater
                    .update(&aggregated_gradients, &mut self.network.layers);
            }
        }

        Ok(())
    }
}

fn aggregate_batch(
    batch: &Batch,
    aggregated_gradients: &mut BackwardCache,
) -> Result<(), TrainerError> {
    if batch.items.is_empty() {
        return Err(TrainerError::EmptyBatch(
            "Cannot aggregate empty batch".to_string(),
        ));
    }

    if aggregated_gradients.entries.is_empty() {
        return Err(TrainerError::UninitializedCache(
            "Aggregated gradients cache struct must be initialized".to_string(),
        ));
    }

    // zeroize the entries before accumulating
    for entry in &mut aggregated_gradients.entries {
        entry.weight_gradient.fill(0.0);
        entry.bias_gradient.fill(0.0);
    }

    let num_items: usize = batch.items.len();
    let num_cache_entries: usize = batch.items[0].entries.len();

    for item in &batch.items {
        for i in 0..num_cache_entries {
            aggregated_gradients.entries[i].weight_gradient += &item.entries[i].weight_gradient;
            aggregated_gradients.entries[i].bias_gradient += &item.entries[i].bias_gradient;
        }
    }

    let scale = num_items as f32;
    for entry in &mut aggregated_gradients.entries {
        entry.weight_gradient /= scale;
        entry.bias_gradient /= scale;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::Activation;
    use crate::layer::Layer;
    use ndarray::array;

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_trainer_run_invalid_args_batch_too_large() {
        let test_weights: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0]];
        let test_biases: Array1<f32> = array![0.0, 0.0];
        let test_activation = Activation::IDENTITY;
        let test_layer = Layer::new(test_weights, test_biases, test_activation).unwrap();
        let test_network = Network::new(vec![test_layer]).unwrap();

        let test_updater = Updater::GD { learning_rate: 0.1 };
        let mut test_trainer = Trainer::new(test_network, Objective::MSE, test_updater);

        let mut test_data = DataSet::new();
        test_data.samples = vec![array![1.0, 0.0], array![0.0, 1.0]];
        test_data.labels = vec![array![1.0, 0.0], array![0.0, 1.0]];

        // Batch size exceeds the number of samples
        let result = test_trainer.run(&test_data, 3, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_trainer_run_invalid_args_batch_not_divisor() {
        let test_weights: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0]];
        let test_biases: Array1<f32> = array![0.0, 0.0];
        let test_activation = Activation::IDENTITY;
        let test_layer = Layer::new(test_weights, test_biases, test_activation).unwrap();
        let test_network = Network::new(vec![test_layer]).unwrap();

        let test_updater = Updater::GD { learning_rate: 0.1 };
        let mut test_trainer = Trainer::new(test_network, Objective::MSE, test_updater);

        let mut test_data = DataSet::new();
        test_data.samples = vec![
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
            array![0.0, 0.0],
        ];
        test_data.labels = vec![
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
            array![0.0, 0.0],
        ];

        // Batch size does not divide the number of samples
        let result = test_trainer.run(&test_data, 3, 1);
        assert!(result.is_err());
    }

    #[test]
    fn test_trainer_run_valid_args() {
        let test_weights: Array2<f32> = array![[1.0, 0.0], [0.0, 1.0]];
        let test_biases: Array1<f32> = array![0.0, 0.0];
        let test_activation = Activation::IDENTITY;
        let test_layer = Layer::new(test_weights, test_biases, test_activation).unwrap();
        let test_network = Network::new(vec![test_layer]).unwrap();

        let test_updater = Updater::GD { learning_rate: 0.1 };
        let mut test_trainer = Trainer::new(test_network, Objective::MSE, test_updater);

        let mut test_data = DataSet::new();
        test_data.samples = vec![
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
            array![0.0, 0.0],
        ];
        test_data.labels = vec![
            array![1.0, 0.0],
            array![0.0, 1.0],
            array![1.0, 1.0],
            array![0.0, 0.0],
        ];

        let result = test_trainer.run(&test_data, 2, 1);
        assert!(result.is_ok());
    }

    #[test]
    fn test_trainer_run() {
        let test_weights1: Array2<f32> = array![[0.5, -0.5], [0.5, 0.5]];
        let test_biases1: Array1<f32> = array![0.1, -0.1];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_weights2: Array2<f32> = array![[1.0, -1.0], [0.5, 0.5]];
        let test_biases2: Array1<f32> = array![0.0, 0.2];
        let test_activation2 = Activation::IDENTITY;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_network = Network::new(vec![test_layer1, test_layer2]).unwrap();
        let test_updater = Updater::GD { learning_rate: 0.1 };
        let mut test_trainer = Trainer::new(test_network, Objective::MSE, test_updater);

        let mut test_data = DataSet::new();
        test_data.samples = vec![array![1.0, 0.0], array![0.0, 1.0]];
        test_data.labels = vec![array![1.0, 0.0], array![0.0, 1.0]];

        test_trainer.run(&test_data, 2, 2).unwrap();

        let test_expected_weights1: Array2<f32> =
            array![[0.535_394, -0.5], [0.399_502_8, 0.498_012_9]];
        let test_expected_biases1: Array1<f32> = array![0.135_394, -0.202_484_3];
        let test_expected_weights2: Array2<f32> =
            array![[1.041_827_8, -0.963_863_2], [0.458_250_5, 0.499_622_7]];
        let test_expected_biases2: Array1<f32> = array![0.100_866, 0.194_801];

        assert!(
            (&test_trainer.network.layers[0].weights - &test_expected_weights1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_trainer.network.layers[0].biases - &test_expected_biases1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_trainer.network.layers[1].weights - &test_expected_weights2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_trainer.network.layers[1].biases - &test_expected_biases2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }

    #[test]
    fn test_trainer_aggregate_batch() {
        let test_weight_gradient11: Array2<f32> = array![[1.0, 2.0], [-1.0, 0.0]];
        let test_bias_gradient11: Array1<f32> = array![3.2, -1.1];
        let test_backward_entry11 = BackwardEntry {
            weight_gradient: test_weight_gradient11,
            bias_gradient: test_bias_gradient11,
        };

        let test_weight_gradient12: Array2<f32> = array![[0.0, 1.0], [2.0, -2.0]];
        let test_bias_gradient12: Array1<f32> = array![2.2, 4.1];
        let test_backward_entry12 = BackwardEntry {
            weight_gradient: test_weight_gradient12,
            bias_gradient: test_bias_gradient12,
        };

        let test_backward_cache1 = BackwardCache {
            entries: vec![test_backward_entry11, test_backward_entry12],
        };

        let test_weight_gradient21: Array2<f32> = array![[3.0, -1.5], [4.1, 2.0]];
        let test_bias_gradient21: Array1<f32> = array![1.0, -0.6];
        let test_backward_entry21 = BackwardEntry {
            weight_gradient: test_weight_gradient21,
            bias_gradient: test_bias_gradient21,
        };

        let test_weight_gradient22: Array2<f32> = array![[2.5, 0.0], [-1.1, 3.5]];
        let test_bias_gradient22: Array1<f32> = array![3.0, -2.2];
        let test_backward_entry22 = BackwardEntry {
            weight_gradient: test_weight_gradient22,
            bias_gradient: test_bias_gradient22,
        };

        let test_backward_cache2 = BackwardCache {
            entries: vec![test_backward_entry21, test_backward_entry22],
        };

        let test_batch = Batch {
            items: vec![test_backward_cache1, test_backward_cache2],
        };

        let mut result = BackwardCache::new();
        for entry in &test_batch.items[0].entries {
            result.entries.push(BackwardEntry {
                weight_gradient: Array2::<f32>::zeros(entry.weight_gradient.dim()),
                bias_gradient: Array1::<f32>::zeros(entry.bias_gradient.dim()),
            });
        }

        aggregate_batch(&test_batch, &mut result).unwrap();

        let test_expected_weight_aggregate1: Array2<f32> = array![[2.0, 0.25], [1.55, 1.0]];
        let test_expected_bias_aggregate1: Array1<f32> = array![2.1, -0.85];
        let test_expected_weight_aggregate2: Array2<f32> = array![[1.25, 0.5], [0.45, 0.75]];
        let test_expected_bias_aggregate2: Array1<f32> = array![2.6, 0.95];

        assert!(
            (&result.entries[0].weight_gradient - &test_expected_weight_aggregate1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&result.entries[0].bias_gradient - &test_expected_bias_aggregate1)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&result.entries[1].weight_gradient - &test_expected_weight_aggregate2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&result.entries[1].bias_gradient - &test_expected_bias_aggregate2)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }

    #[test]
    fn test_trainer_aggregate_batch_invalid_args_empty_batch() {
        let test_batch = Batch::new();

        let mut test_aggregated_gradients = BackwardCache {
            entries: vec![BackwardEntry {
                weight_gradient: Array2::<f32>::zeros((2, 2)),
                bias_gradient: Array1::<f32>::zeros(2),
            }],
        };

        let result = aggregate_batch(&test_batch, &mut test_aggregated_gradients);
        assert!(result.is_err());
    }

    #[test]
    fn test_trainer_aggregate_batch_invalid_args_uninitialized_cache() {
        let test_backward_cache = BackwardCache {
            entries: vec![BackwardEntry {
                weight_gradient: array![[1.0, 2.0], [-1.0, 0.0]],
                bias_gradient: array![3.2, -1.1],
            }],
        };
        let test_batch = Batch {
            items: vec![test_backward_cache],
        };

        // Aggregation target has no entries to accumulate into
        let mut test_aggregated_gradients = BackwardCache::new();

        let result = aggregate_batch(&test_batch, &mut test_aggregated_gradients);
        assert!(result.is_err());
    }

    #[test]
    fn test_trainer_aggregate_batch_valid_args() {
        let test_backward_cache = BackwardCache {
            entries: vec![BackwardEntry {
                weight_gradient: array![[1.0, 2.0], [-1.0, 0.0]],
                bias_gradient: array![3.2, -1.1],
            }],
        };
        let test_batch = Batch {
            items: vec![test_backward_cache],
        };

        let mut test_aggregated_gradients = BackwardCache {
            entries: vec![BackwardEntry {
                weight_gradient: Array2::<f32>::zeros((2, 2)),
                bias_gradient: Array1::<f32>::zeros(2),
            }],
        };

        let result = aggregate_batch(&test_batch, &mut test_aggregated_gradients);
        assert!(result.is_ok());
    }
}
