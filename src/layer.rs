use ndarray::{Array1, Array2};
use rand::Rng;
use rand::distr::{Distribution, Uniform};

use crate::activation::Activation;

#[derive(Debug)]
pub enum LayerError {
    InvalidArgDimensions(String),
    InvalidRand(String),
}

#[derive(Debug, PartialEq)]
pub struct LayerOutput {
    pub pre_activation: Array1<f32>,
    pub post_activation: Array1<f32>,
}

pub struct Layer {
    pub weights: Array2<f32>,
    pub biases: Array1<f32>,
    pub activation: Activation,
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
    pub fn new_random(
        dims: [usize; 2],
        value_range: [f32; 2],
        activation: Activation,
        rng: &mut impl Rng,
    ) -> Result<Self, LayerError> {
        // Validate the random range
        if value_range[1] < value_range[0] {
            return Err(LayerError::InvalidRand(
                "Value range must be of form [lower, upper] with lower < upper".to_string(),
            ));
        }

        let dist = Uniform::new(value_range[0], value_range[1])
            .map_err(|e| LayerError::InvalidRand(e.to_string()))?;
        let biases = Array1::from_shape_simple_fn(dims[0], || dist.sample(rng));
        let weights = Array2::from_shape_simple_fn((dims[0], dims[1]), || dist.sample(rng));

        Ok(Layer {
            weights,
            biases,
            activation,
        })
    }

    #[must_use]
    pub fn forward_pass(&self, input: &Array1<f32>) -> Result<LayerOutput, LayerError> {
        // Validate input against dimensions of layer's weights
        if input.dim() != self.weights.dim().1 {
            return Err(LayerError::InvalidArgDimensions(
                "Length of input must equal number of columns in weight matrix".to_string(),
            ));
        }
        let pre_activation: Array1<f32> = self.weights.dot(input) + &self.biases;
        let post_activation: Array1<f32> = self.activation.apply(&pre_activation);
        Ok(LayerOutput {
            pre_activation,
            post_activation,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    const EPSILON: f32 = 0.0001;

    #[test]
    fn test_layer_new_valid_args() {
        let test_weights: Array2<f32> = array![[1.0, 0.3, 1.4], [0.0, 1.0, 3.2]];
        let test_biases: Array1<f32> = array![1.0, 1.0];
        let test_activation = Activation::RELU;
        let test_layer = Layer::new(test_weights, test_biases, test_activation);

        assert!(test_layer.is_ok());
    }

    #[test]
    fn test_layer_new_invalid_args() {
        let test_weights: Array2<f32> = array![[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]];
        let test_biases: Array1<f32> = array![1.0, 1.0, 1.0];
        let test_activation = Activation::RELU;
        let test_layer = Layer::new(test_weights, test_biases, test_activation);

        assert!(test_layer.is_err());
    }

    #[test]
    fn test_layer_new_random_valid_args() {
        let seed: u64 = 48;
        let mut rng = StdRng::seed_from_u64(seed);

        let test_dims: [usize; 2] = [3, 4];
        let test_range: [f32; 2] = [-1.0, 1.0];
        let test_activation = Activation::RELU;

        let result = Layer::new_random(test_dims, test_range, test_activation, &mut rng);
        assert!(result.is_ok());
    }

    #[test]
    fn test_layer_new_random_invalid_args() {
        let seed: u64 = 48;
        let mut rng = StdRng::seed_from_u64(seed);

        let test_dims: [usize; 2] = [2, 3];
        let test_range: [f32; 2] = [3.0, 1.0];
        let test_activation = Activation::RELU;

        let result = Layer::new_random(test_dims, test_range, test_activation, &mut rng);
        assert!(result.is_err());
    }

    #[test]
    fn test_layer_new_random_dims() {
        let seed: u64 = 48;
        let mut rng = StdRng::seed_from_u64(seed);

        let test_dims: [usize; 2] = [2, 4];
        let test_range: [f32; 2] = [-1.0, 1.0];
        let test_activation = Activation::RELU;

        let test_layer =
            Layer::new_random(test_dims, test_range, test_activation, &mut rng).unwrap();
        assert_eq!(test_layer.weights.dim().0, test_dims[0]);
        assert_eq!(test_layer.weights.dim().1, test_dims[1]);
    }

    #[test]
    fn test_layer_new_random_forward_pass() {
        let seed: u64 = 48;
        let mut rng = StdRng::seed_from_u64(seed);

        let test_dims: [usize; 2] = [2, 4];
        let test_range: [f32; 2] = [-1.0, 1.0];
        let test_activation = Activation::RELU;
        let test_layer =
            Layer::new_random(test_dims, test_range, test_activation, &mut rng).unwrap();

        // With seed = 48
        // biases = [-0.12484574 -0.38281488]
        // weights = [[-0.821337, -0.24388695, 0.35480237, 0.9600971],
        //              [0.41448832, 0.29030514, 0.68937206, -0.30939674]]

        let test_input: Array1<f32> = array![1.0, 0.0, 0.0, 0.0];
        let expected_output: Array1<f32> = array![0.0, 0.03167343];
        let test_result = test_layer.forward_pass(&test_input).unwrap();
        assert!(
            (&test_result.post_activation - &expected_output)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }

    #[test]
    fn test_layer_forward_pass_valid_args() {
        let test_weights: Array2<f32> = array![
            [1.0, 0.5, -1.0, -0.5],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, -1.0]
        ];
        let test_biases: Array1<f32> = array![-3.0, 0.5, 1.0];
        let test_activation = Activation::RELU;
        let test_layer = Layer::new(test_weights, test_biases, test_activation).unwrap();

        let test_input: Array1<f32> = array![1.0, 0.0, 0.0, 0.0];
        let test_result = test_layer.forward_pass(&test_input);
        assert!(test_result.is_ok());
    }

    #[test]
    fn test_layer_forward_pass_invalid_args() {
        let test_weights: Array2<f32> = array![
            [1.0, 0.5, -1.0, -0.5],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, -1.0]
        ];
        let test_biases: Array1<f32> = array![-3.0, 0.5, 1.0];
        let test_activation = Activation::RELU;
        let test_layer = Layer::new(test_weights, test_biases, test_activation).unwrap();

        let test_input: Array1<f32> = array![1.0, 0.0];
        let test_result = test_layer.forward_pass(&test_input);
        assert!(test_result.is_err());
    }

    #[test]
    fn test_layer_forward_pass() {
        // Test 1
        let test_weights1: Array2<f32> = array![
            [1.0, 0.5, -1.0, -0.5],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, -1.0]
        ];
        let test_biases1: Array1<f32> = array![-3.0, 0.5, 1.0];
        let test_activation1 = Activation::RELU;
        let test_layer1 = Layer::new(test_weights1, test_biases1, test_activation1).unwrap();

        let test_input1: Array1<f32> = array![1.0, 0.0, 0.0, 0.0];
        let test_expected_pre1: Array1<f32> = array![-2.0, 0.5, 2.0];
        let test_expected_post1: Array1<f32> = array![0.0, 0.5, 2.0];
        let test_expected_output1 = LayerOutput {
            pre_activation: test_expected_pre1,
            post_activation: test_expected_post1,
        };
        let test_result1 = test_layer1.forward_pass(&test_input1).unwrap();
        assert!(
            (&test_result1.pre_activation - &test_expected_output1.pre_activation)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_result1.post_activation - &test_expected_output1.post_activation)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );

        // Test 2
        let test_weights2: Array2<f32> = array![[1.0, -1.0, 2.0, -2.0], [-2.0, 2.0, -1.0, 1.0]];
        let test_biases2: Array1<f32> = array![-2.4, 2.7];
        let test_activation2 = Activation::IDENTITY;
        let test_layer2 = Layer::new(test_weights2, test_biases2, test_activation2).unwrap();

        let test_input2: Array1<f32> = array![0.0, -1.0, 1.0, 0.3];
        let test_expected_pre2: Array1<f32> = array![0.0, 0.0];
        let test_expected_post2: Array1<f32> = array![0.0, 0.0];
        let test_expected_output2 = LayerOutput {
            pre_activation: test_expected_pre2,
            post_activation: test_expected_post2,
        };
        let test_result2 = test_layer2.forward_pass(&test_input2).unwrap();
        assert!(
            (&test_result2.pre_activation - &test_expected_output2.pre_activation)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_result2.post_activation - &test_expected_output2.post_activation)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );

        // Test 3
        let test_weights3: Array2<f32> = array![
            [-1.0, 2.0, -3.0, 4.0, -5.0],
            [1.0, -2.0, 3.0, -4.0, 5.0],
            [1.0, 0.0, 1.0, 0.0, 1.0]
        ];
        let test_biases3: Array1<f32> = array![26.0, -26.0, 1.0];
        let test_activation3 = Activation::RELU;
        let test_layer3 = Layer::new(test_weights3, test_biases3, test_activation3).unwrap();

        let test_input3: Array1<f32> = array![2.5, 0.0, 0.5, 1.0, 5.0];
        let test_expected_pre3: Array1<f32> = array![1.0, -1.0, 9.0];
        let test_expected_post3: Array1<f32> = array![1.0, 0.0, 9.0];
        let test_expected_output3 = LayerOutput {
            pre_activation: test_expected_pre3,
            post_activation: test_expected_post3,
        };
        let test_result3 = test_layer3.forward_pass(&test_input3).unwrap();
        assert!(
            (&test_result3.pre_activation - &test_expected_output3.pre_activation)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
        assert!(
            (&test_result3.post_activation - &test_expected_output3.post_activation)
                .iter()
                .all(|d| d.abs() < EPSILON)
        );
    }
}
