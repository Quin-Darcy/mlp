use crate::cache::BackwardCache;
use crate::layer::Layer;

pub enum Optimizer {
    SGD_SINGLE, // stochastic gradient descent for single input
}

impl Optimizer {
    pub fn update(&self, learning_rate: f32, cache: &BackwardCache, layers: &mut [Layer]) {
        match self {
            Self::SGD_SINGLE => {
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
