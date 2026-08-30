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
// from the data set
fn new_trainer(seed: u64, data: &DataSet, hidden: usize, updater: Updater) -> Trainer {
    let mut rng = StdRng::seed_from_u64(seed);
    let range: [f32; 2] = [-1.0, 1.0];
    let input_dim: usize = data.samples[0].dim();
    let output_dim: usize = data.labels[0].dim();
    let layer1 = Layer::new_random([hidden, input_dim], range, Activation::RELU, &mut rng).unwrap();
    let layer2 =
        Layer::new_random([output_dim, hidden], range, Activation::IDENTITY, &mut rng).unwrap();
    let network = Network::new(vec![layer1, layer2]).unwrap();
    Trainer::new(network, Objective::MSE, updater)
}

// The configurations to compare. Built fresh per data set because the
// momentum updater carries state
fn configs() -> Vec<(&'static str, Updater)> {
    vec![
        (
            "SGD_SIMPLE lr=0.1",
            Updater::SGD_SIMPLE { learning_rate: 0.1 },
        ),
        (
            "SGD_SIMPLE lr=0.01",
            Updater::SGD_SIMPLE {
                learning_rate: 0.01,
            },
        ),
        (
            "SGD_MOMENTUM lr=0.01 gamma=0.9",
            Updater::SGD_MOMENTUM {
                learning_rate: 0.01,
                gamma: 0.9,
                update_vector: BackwardCache::new(),
            },
        ),
        (
            "SGD_MOMENTUM lr=0.001 gamma=0.9",
            Updater::SGD_MOMENTUM {
                learning_rate: 0.001,
                gamma: 0.9,
                update_vector: BackwardCache::new(),
            },
        ),
    ]
}

fn main() -> Result<(), GrapherError> {
    let seed: u64 = 48;
    let epochs: usize = 300;

    // (title, data set, hidden units, batch size, output file)
    let experiments: Vec<(&str, DataSet, usize, usize, &str)> = vec![
        (
            "Easy: y = x^2",
            data::easy(),
            16,
            3,
            "examples/grapher/images/loss_curves_easy.png",
        ),
        (
            "Hard: (sin(pi x) cos(pi y) z, x y z + x^2 - y^2)",
            data::hard(),
            32,
            8,
            "examples/grapher/images/loss_curves_hard.png",
        ),
    ];

    for (title, data, hidden, batch_size, path) in &experiments {
        println!("{title}");
        let mut curves: Vec<LossCurve> = Vec::new();
        for (name, updater) in configs() {
            let mut trainer = new_trainer(seed, data, *hidden, updater);
            let curve = LossCurve::record(name, &mut trainer, data, *batch_size, epochs)?;
            println!("  {name}: final loss {}", curve.losses[epochs - 1]);
            curves.push(curve);
        }
        plot_loss_curves(&curves, title, path)?;
        println!("  written to {path}\n");
    }

    Ok(())
}
