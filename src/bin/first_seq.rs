use rand::RngExt;
use rand::SeedableRng;
use rand::seq::IndexedRandom;
use rand::rngs::StdRng;
use ndarray::Array1;

use mlp::activation::Activation;
use mlp::layer::Layer;
use mlp::network::Network;
use mlp::updater::Updater;
use mlp::objective::Objective;
use mlp::trainer::Trainer;
use mlp::data_set::DataSet;


enum SeqIn {
    C0 = 0, // 22
    C1 = 1, // 23
    C2 = 2, // 32
    C3 = 3, // 33
    C4 = 4, // 28
    C5 = 5, // 29
    C6 = 6, // 38
    C7 = 7, // 39
    C8 = 8, // 82
    C9 = 9, // 83
    C10 = 10, // 92
    C11 = 11, // 93
}

impl SeqIn {
    fn get(window: &[usize]) -> Self {
        match window {
            [2, 2] => SeqIn::C0,
            [2, 3] => SeqIn::C1,
            [3, 2] => SeqIn::C2,
            [3, 3] => SeqIn::C3,
            [2, 8] => SeqIn::C4,
            [2, 9] => SeqIn::C5,
            [3, 8] => SeqIn::C6,
            [3, 9] => SeqIn::C7,
            [8, 2] => SeqIn::C8,
            [8, 3] => SeqIn::C9,
            [9, 2] => SeqIn::C10,
            [9, 3] => SeqIn::C11,
            _ => todo!()
        }
    }
}

impl SeqIn {
    pub const fn one_hot(self) -> [f32; 12] {
        let mut v = [0.0; 12];
        v[self as usize] = 1.0;
        v
    }

    pub fn get_vec(self) -> Array1<f32> {
        Array1::from_vec(self.one_hot().to_vec())
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
            },
            2 => {
                sequence.push(set2.choose(&mut rng).unwrap().clone());
            },
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
            labels.push(Array1::from(vec![seq[i + window_size + 1] as f32]));
        }
    }
    DataSet::from_data(samples, labels).unwrap()
}

fn main() {
    let seed: u64 = 49;
    let mut rng = StdRng::seed_from_u64(seed);

    let l1_dims: [usize; 2] = [12, 3];
    let l1_vrange: [f32; 2] = [-1.0, 1.0];
    let l1_activation = Activation::RELU;
    let l1 = Layer::new_random(
        l1_dims,
        l1_vrange,
        l1_activation,
        &mut rng
    ).unwrap();

    let l2_dims: [usize; 2] = [4, 12];
    let l2_vrange: [f32; 2] = [-1.0, 1.0];
    let l2_activation = Activation::IDENTITY;
    let l2 = Layer::new_random(
        l2_dims,
        l2_vrange,
        l2_activation,
        &mut rng
    ).unwrap();

    let layers: Vec<Layer> = vec![l1, l2];
    let network = Network::new(layers).unwrap();
    let updater = Updater::SGD_SIMPLE { learning_rate: 0.01 };
    let objective = Objective::NLL;
    let trainer = Trainer::new(network, objective, updater);

    let sequence_len: usize = 17;
    let cycle_len: usize = 3;
    let sequence: Vec<usize> = create_sequence(sequence_len, cycle_len);
    let data: DataSet = parse_sequence(&sequence, cycle_len - 1);

    // todo: now all I need to do is train network on dataset!
}
