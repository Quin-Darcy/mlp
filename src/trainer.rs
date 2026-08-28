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
        
        let mut aggregated_gradients = BackwardCache::new();
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
    todo!()
}
