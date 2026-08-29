use ndarray::Array1;

use crate::data_set::DataSet;
use crate::network::{Network, NetworkOutput, NetworkError};
use crate::objective::{Objective, ObjectiveError};
use crate::updater::Updater;
use crate::cache::BackwardCache;

#[derive(Debug)]
pub enum TrainerError {
    BadForwardPass(String),
    BadObjectiveGradient(String),
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
    items: Vec<BackwardCache>
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

    pub fn run(&mut self, data: &DataSet, batch_size: usize, epochs: usize) -> Result<(), TrainerError> {
        // todo: learn rayon enough to figure out how to parallelize this part

        // TODO: check batch size isn't bigger than number of data samples or maybe just cap it?
        // TODO: handle if batch size doesn't divide data size

        let mut aggregated_gradients: BackwardCache;
        let data_size: usize = data.samples.len();
        for _ in 0..epochs {
            for i in (0..data_size).step_by(batch_size) {
                let mut batch = Batch::new();
                for j in 0..batch_size {
                    let network_out: NetworkOutput = self.network.forward_pass(&data.samples[i+j])?;
                    let gradient_objective: Array1<f32> = self.objective.gradient(&network_out.output, &data.labels[i+j])?;
                    
                    batch.items.push(self.network.backward_pass(&network_out, &gradient_objective));
                }

                aggregated_gradients = aggregate_batch(&batch);
                self.updater.update(&aggregated_gradients, &mut self.network.layers);
            }
        }

        Ok(())
    }
}

fn aggregate_batch(batch: &Batch) -> BackwardCache {
    // TODO: validate batch is not empty

    // clone first element which will be our accumulator
    let mut accumulator: BackwardCache = batch.items[0].clone();

    let num_items: usize = batch.items.len();
    let num_cache_entries: usize = accumulator.entries.len();
    for item in &batch.items[1..] {
        for i in 0..num_cache_entries {
            accumulator.entries[i].weight_gradient += &item.entries[i].weight_gradient;
            accumulator.entries[i].bias_gradient += &item.entries[i].bias_gradient;
        }
    }

    let scale = num_items as f32;
    for entry in &mut accumulator.entries {
        entry.weight_gradient /= scale;
        entry.bias_gradient /= scale;
    }

    accumulator
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};
    use crate::cache::BackwardEntry;

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_trainer_aggregate_batch() {
        let test_weight_gradient11: Array2<f32> = array![[1.0, 2.0], [-1.0, 0.0]];
        let test_bias_gradient11: Array1<f32> = array![3.2, -1.1];
        let test_backward_entry11 = BackwardEntry {
            weight_gradient: test_weight_gradient11,
            bias_gradient: test_bias_gradient11
        };

        let test_weight_gradient12: Array2<f32> = array![[0.0, 1.0], [2.0, -2.0]];
        let test_bias_gradient12: Array1<f32> = array![2.2, 4.1];
        let test_backward_entry12 = BackwardEntry {
            weight_gradient: test_weight_gradient12,
            bias_gradient: test_bias_gradient12
        };

        let test_backward_cache1 = BackwardCache {
            entries: vec![test_backward_entry11, test_backward_entry12]
        };

        let test_weight_gradient21: Array2<f32> = array![[3.0, -1.5], [4.1, 2.0]];
        let test_bias_gradient21: Array1<f32> = array![1.0, -0.6];
        let test_backward_entry21 = BackwardEntry {
            weight_gradient: test_weight_gradient21,
            bias_gradient: test_bias_gradient21
        };

        let test_weight_gradient22: Array2<f32> = array![[2.5, 0.0], [-1.1, 3.5]];
        let test_bias_gradient22: Array1<f32> = array![3.0, -2.2];
        let test_backward_entry22 = BackwardEntry {
            weight_gradient: test_weight_gradient22,
            bias_gradient: test_bias_gradient22
        };

        let test_backward_cache2 = BackwardCache {
            entries: vec![test_backward_entry21, test_backward_entry22],
        };

        let test_batch = Batch {
            items: vec![test_backward_cache1, test_backward_cache2]
        };

        let result: BackwardCache = aggregate_batch(&test_batch); 

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
}
