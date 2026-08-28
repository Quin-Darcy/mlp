use crate::cache::BackwardCache;
use crate::layer::Layer;

pub enum Optimizer {
    SGD_SINGLE { learning_rate: f32 }, // stochastic gradient descent for single input
}

impl Optimizer {
    pub fn update(&self, cache: &BackwardCache, layers: &mut [Layer]) {
        match self {
            Self::SGD_SINGLE { learning_rate } => {
                for (layer, entry) in layers.iter_mut().zip(cache.entries.iter()) {
                    layer
                        .weights
                        .scaled_add(-learning_rate, &entry.weight_gradient);
                    layer
                        .biases
                        .scaled_add(-learning_rate, &entry.bias_gradient);
                }
            }
        }
    }
}
