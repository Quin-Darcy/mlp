use ndarray::{Array1, Array2};

use crate::activation::Activation;

#[derive(Debug)]
pub enum LayerError {
    InvalidArgDimensions(String),
}

pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
}

impl Layer {
    #[must_use]
    pub fn new(
        weights: Array2<f32>,
        biases: Array1<f32>,
        activation: Activation,
    ) -> Result<Self, LayerError> {
        if biases.dim() != weights.dim().0 {
            return Err(LayerError::InvalidArgDimensions(
                "Weight matrix rows must equal number of biases".to_string(),
            ));
        }

        Ok(Self {
            weights,
            biases,
            activation,
        })
    }

    #[must_use]
    pub fn forward_pass(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let pre_activations: Array1<f32> = self.weights.dot(input) + &self.biases;
        let post_activations: Array1<f32> = self.activation.apply(&pre_activations);
        (pre_activations, post_activations)
    }
}
