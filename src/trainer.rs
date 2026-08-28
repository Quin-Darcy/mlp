use crate::data_set::DataSet;
use crate::network::{Network, NetworkError};
use crate::objective::Objective;
use crate::updater::Updater;
use crate::cache::BackwardCache;

#[derive(Debug)]
pub enum TrainerError {
    UhOh(String), // get a little more creative with name here ...
}

impl From<NetworkError> for TrainerError {
    fn from(e: NetworkError) -> Self {
        TrainerError::UhOh(format!("{e:?}"))
    }
}

pub struct Batch {
    items: Vec<BackwardCache>
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

    pub fn run(&self, data: &DataSet, batch_size: usize, epochs: usize) -> Result<(), TrainerError> {
        // todo: learn rayon enough to figure out how to parallelize this part
        Ok(())
    }
}

fn aggregate_batch(batch: &Batch) -> BackwardCache {
    todo!()
}
