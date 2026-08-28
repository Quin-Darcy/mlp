use rand::SeedableRng;
use rand::rngs::StdRng;

use mlp::activation::Activation;
use mlp::layer::Layer;

fn main() {
    let seed: u64 = 48;
    let mut rng = StdRng::seed_from_u64(seed);

    let test_dims: [usize; 2] = [2, 4];
    let test_range: [f32; 2] = [-1.0, 1.0];
    let test_activation = Activation::RELU;

    let test_layer = Layer::new_random(test_dims, test_range, test_activation, &mut rng).unwrap();
    println!("Biases: {:?}", test_layer.biases);
    println!("Weights: {:?}", test_layer.weights);
}
