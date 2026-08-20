use ndarray::Array1;

use crate::cache::{Entry, ForwardCache};
use crate::layer::Layer;

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    #[must_use]
    pub fn new() -> Self {
        todo!()
    }

    #[must_use]
    pub fn forward_pass(&self, input: &Array1<f32>) -> ForwardCache {
        let mut layer_input: Array1<f32> = input.clone();
        let _pre_activations: Array1<f32>;
        let mut cache = ForwardCache::new();

        for layer in &self.layers {
            let pre_post: (Array1<f32>, Array1<f32>) = layer.forward_pass(&layer_input);
            cache.entries.push(Entry {
                input: layer_input.clone(),
                pre_activations: pre_post.0,
            });
            layer_input = pre_post.1;
        }
        cache
    }
}
