// Compare how the loss falls under different updaters and hyperparameters.
// Run with: cargo run --release --example grapher
mod data;
mod grapher;

use rand::SeedableRng;
use rand::rngs::StdRng;

use mlp::activation::Activation;
use mlp::cache::BackwardCache;
use mlp::data_set::DataSet;
use mlp::layer::Layer;
use mlp::network::Network;
use mlp::objective::Objective;
use mlp::trainer::Trainer;
use mlp::updater::Updater;

use grapher::{GrapherError, LossCurve, plot_loss_curves};

// Every configuration starts from the same random weights so the curves
// differ only by updater and hyperparameters. Input and output sizes come
// from the data set; every hidden layer has `hidden` RELU units
fn new_trainer(
    seed: u64,
    data: &DataSet,
    hidden_layers: usize,
    hidden: usize,
    updater: Updater,
) -> Trainer {
    let mut rng = StdRng::seed_from_u64(seed);
    let range: [f32; 2] = [-1.0, 1.0];
    let input_dim: usize = data.samples[0].dim();
    let output_dim: usize = data.labels[0].dim();

    let mut layers: Vec<Layer> = Vec::with_capacity(hidden_layers + 1);
    layers.push(Layer::new_random([hidden, input_dim], range, Activation::RELU, &mut rng).unwrap());
    for _ in 1..hidden_layers {
        layers
            .push(Layer::new_random([hidden, hidden], range, Activation::RELU, &mut rng).unwrap());
    }
    layers.push(
        Layer::new_random([output_dim, hidden], range, Activation::IDENTITY, &mut rng).unwrap(),
    );

    let network = Network::new(layers).unwrap();
    Trainer::new(network, Objective::MSE, updater)
}

// The configurations to compare, in matched pairs: with gamma = 0.9 the
// momentum step settles at 10x its learning rate, so each momentum entry
// has the same effective step as the plain SGD entry above it. Learning
// rates are powers of ten, the largest being 10^exponent. Built fresh per
// experiment because the momentum updater carries state
fn configs(exponent: i32) -> Vec<(String, Updater)> {
    let gamma: f32 = 0.9;
    let lr_large: f32 = 10f32.powi(exponent);
    let lr_medium: f32 = 10f32.powi(exponent - 1);
    let lr_small: f32 = 10f32.powi(exponent - 2);
    vec![
        (
            format!("SGD_SIMPLE lr={lr_large}"),
            Updater::SGD_SIMPLE {
                learning_rate: lr_large,
            },
        ),
        (
            format!("SGD_SIMPLE lr={lr_medium}"),
            Updater::SGD_SIMPLE {
                learning_rate: lr_medium,
            },
        ),
        (
            format!("SGD_MOMENTUM lr={lr_medium} gamma={gamma}"),
            Updater::SGD_MOMENTUM {
                learning_rate: lr_medium,
                gamma,
                update_vector: BackwardCache::new(),
            },
        ),
        (
            format!("SGD_MOMENTUM lr={lr_small} gamma={gamma}"),
            Updater::SGD_MOMENTUM {
                learning_rate: lr_small,
                gamma,
                update_vector: BackwardCache::new(),
            },
        ),
    ]
}

fn main() -> Result<(), GrapherError> {
    let seed: u64 = 48;
    let epochs: usize = 300;

    // (title, data set, hidden layers, hidden units, batch size, exponent of
    // the largest learning rate, output file). The deeper network diverges
    // at lr 0.1, so its pairs start one decade lower
    let experiments: Vec<(&str, DataSet, usize, usize, usize, i32, &str)> = vec![
        (
            "Easy: y = x^2",
            data::easy(),
            1,
            16,
            3,
            -1,
            "examples/grapher/images/loss_curves_easy.png",
        ),
        (
            "Hard: (sin(pi x) cos(pi y) z, x y z + x^2 - y^2)",
            data::hard(),
            1,
            32,
            8,
            -1,
            "examples/grapher/images/loss_curves_hard.png",
        ),
        (
            "Hard, 3 hidden layers of 32",
            data::hard(),
            3,
            32,
            8,
            -2,
            "examples/grapher/images/loss_curves_hard_deep.png",
        ),
    ];

    for (title, data, hidden_layers, hidden, batch_size, exponent, path) in &experiments {
        println!("{title}");
        let mut curves: Vec<LossCurve> = Vec::new();
        for (name, updater) in configs(*exponent) {
            let mut trainer = new_trainer(seed, data, *hidden_layers, *hidden, updater);
            let curve = LossCurve::record(&name, &mut trainer, data, *batch_size, epochs)?;
            println!("  {name}: final loss {}", curve.losses[epochs - 1]);
            curves.push(curve);
        }
        plot_loss_curves(&curves, title, path)?;
        println!("  written to {path}\n");
    }

    Ok(())
}
