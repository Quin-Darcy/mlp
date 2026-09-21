use ndarray::Array1;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use rand::seq::IndexedRandom;

use mlp::activation::Activation;
use mlp::data_set::DataSet;
use mlp::layer::Layer;
use mlp::network::Network;
use mlp::objective::Objective;
use mlp::trainer::Trainer;
use mlp::updater::Updater;

enum SeqIn {
    SI0 = 0,   // 22
    SI1 = 1,   // 23
    SI2 = 2,   // 32
    SI3 = 3,   // 33
    SI4 = 4,   // 28
    SI5 = 5,   // 29
    SI6 = 6,   // 38
    SI7 = 7,   // 39
    SI8 = 8,   // 82
    SI9 = 9,   // 83
    SI10 = 10, // 92
    SI11 = 11, // 93
}

impl SeqIn {
    fn get(window: &[usize]) -> Self {
        match window {
            [2, 2] => SeqIn::SI0,
            [2, 3] => SeqIn::SI1,
            [3, 2] => SeqIn::SI2,
            [3, 3] => SeqIn::SI3,
            [2, 8] => SeqIn::SI4,
            [2, 9] => SeqIn::SI5,
            [3, 8] => SeqIn::SI6,
            [3, 9] => SeqIn::SI7,
            [8, 2] => SeqIn::SI8,
            [8, 3] => SeqIn::SI9,
            [9, 2] => SeqIn::SI10,
            [9, 3] => SeqIn::SI11,
            _ => todo!(),
        }
    }

    pub const fn one_hot(self) -> [f32; 12] {
        let mut v = [0.0; 12];
        v[self as usize] = 1.0;
        v
    }

    pub fn get_vec(self) -> Array1<f32> {
        Array1::from_vec(self.one_hot().to_vec())
    }
}

enum SeqOut {
   SO0 = 0, // 2
   SO1 = 1, // 3
   SO2 = 2, // 8
   SO3 = 3, // 9
}

impl SeqOut {
    fn get(expected_out: usize) -> Self {
        match expected_out {
            2 => SeqOut::SO0,
            3 => SeqOut::SO1,
            8 => SeqOut::SO2,
            9 => SeqOut::SO3,
            _ => todo!(),
        }
    }

    pub fn get_vec(self) -> Array1<f32> {
        let s: usize = self as usize;
        Array1::from_vec(vec![s as f32])
    }
}

fn create_sequence(seq_len: usize, cycle_len: usize) -> Vec<usize> {
    let mut rng = rand::rng();

    let mut phase_index: usize = rng.random_range(0..3);
    let mut sequence: Vec<usize> = Vec::with_capacity(seq_len);

    let set1: Vec<usize> = vec![2, 3];
    let set2: Vec<usize> = vec![8, 9];

    for _ in 0..seq_len {
        match phase_index {
            0 | 1 => {
                sequence.push(set1.choose(&mut rng).unwrap().clone());
            }
            2 => {
                sequence.push(set2.choose(&mut rng).unwrap().clone());
            }
            _ => {}
        }
        phase_index += 1;
        phase_index = phase_index % cycle_len;
    }
    sequence
}

fn parse_sequence(seq: &[usize], window_size: usize) -> DataSet {
    let mut samples: Vec<Array1<f32>> = Vec::new();
    let mut labels: Vec<Array1<f32>> = Vec::new();

    for i in 0..seq.len() {
        if i + window_size + 1 <= seq.len() - (window_size + 1) {
            let mut window: Vec<usize> = Vec::with_capacity(window_size);
            for j in 0..window_size {
                window.push(seq[i + j].clone());
            }
            samples.push(SeqIn::get(&window).get_vec());
            labels.push(SeqOut::get(seq[i + window_size + 1]).get_vec());
        }
    }
    DataSet::from_data(samples, labels).unwrap()
}

fn main() {
    let seed: u64 = 49;
    let mut rng = StdRng::seed_from_u64(seed);

    let l1_dims: [usize; 2] = [3, 12];
    let l1_vrange: [f32; 2] = [-1.0, 1.0];
    let l1_activation = Activation::RELU;
    let l1 = Layer::new_random(l1_dims, l1_vrange, l1_activation, &mut rng).unwrap();

    let l2_dims: [usize; 2] = [4, 3];
    let l2_vrange: [f32; 2] = [-1.0, 1.0];
    let l2_activation = Activation::IDENTITY;
    let l2 = Layer::new_random(l2_dims, l2_vrange, l2_activation, &mut rng).unwrap();

    let layers: Vec<Layer> = vec![l1, l2];
    let network = Network::new(layers).unwrap();
    let updater = Updater::SGD_SIMPLE {
        learning_rate: 0.01,
    };
    let objective = Objective::NLL;
    let trainer = Trainer::new(network, objective, updater);

    let sequence_len: usize = 17;
    let cycle_len: usize = 3;
    let sequence: Vec<usize> = create_sequence(sequence_len, cycle_len);
    let data: DataSet = parse_sequence(&sequence, cycle_len - 1);

    // Get baseline evaluation
    let pre_trained_loss: f32 = trainer.evaluate(&data).unwrap();
    println!("Pre-Trained Loss: {}", pre_trained_loss);

    // todo: now all I need to do is train network on dataset!
}
