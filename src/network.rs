use ndarray::Array1;

use crate::cache::{Entry, ForwardCache};
use crate::layer::{LayerError, Layer};


#[derive(Debug)]
pub enum NetworkError {
    InvalidArgDimensions(String),
}

pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    #[must_use]
    pub fn new(layers: Vec<Layer>) -> Result<Self, NetworkError> {
        // Validate that the dimensions of consecutive layers
        for i in 0..layers.len() - 1 {
            if layers[i].weights.dim().0 != layers[i+1].weights.dim().1 {
                return Err(NetworkError::InvalidArgDimensions(
                    "The number of rows in one layer's weight matrix must equal the number of columns in the next layer's weight matrix.".to_string(),
                ));
            }
        }

        Ok(Self {
            layers
        })
    }

    #[must_use]
    pub fn forward_pass(&self, input: &Array1<f32>) -> Result<ForwardCache, LayerError> {
        let mut layer_input: Array1<f32> = input.clone();
        let _pre_activations: Array1<f32>;
        let mut cache = ForwardCache::new();

        for layer in &self.layers {
            let pre_post: (Array1<f32>, Array1<f32>) = layer.forward_pass(&layer_input)?;
            cache.entries.push(Entry {
                input: layer_input.clone(),
                pre_activations: pre_post.0,
            });
            layer_input = pre_post.1;
        }
        Ok(cache)
    }
}
