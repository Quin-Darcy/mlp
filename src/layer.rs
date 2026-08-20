use ndarray::{Array1, Array2};

use crate::activation::Activation;

pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activation: Activation,
}

impl Default for Layer {
    fn default() -> Self {
        Self::new()
    }
}

impl Layer {
    #[must_use] 
    pub fn new() -> Self {
        todo!()
    }

    #[must_use] 
    pub fn forward_pass(&self, input: &Array1<f32>) -> (Array1<f32>, Array1<f32>) {
        let pre_activations: Array1<f32> = self.weights.dot(input) + &self.biases;
        let post_activations: Array1<f32> = self.activation.apply(&pre_activations);
        (pre_activations, post_activations)
    }
}
