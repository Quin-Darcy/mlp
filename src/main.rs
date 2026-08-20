use ndarray::{Array1, Array2, array};

fn main() {
    let test1: Array1<f32> = array![0.0, 1.0, 2.0];
    let test2: Array2<f32> = array![[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]];

    println!("{:?}", test1.dim());
    println!("{:?}", test2.dim().0);
}
