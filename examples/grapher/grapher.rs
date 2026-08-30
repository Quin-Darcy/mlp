use plotters::prelude::*;

use mlp::data_set::DataSet;
use mlp::network::NetworkError;
use mlp::objective::ObjectiveError;
use mlp::trainer::{Trainer, TrainerError};

#[allow(dead_code)]
#[derive(Debug)]
pub enum GrapherError {
    BadTraining(String),
    BadForwardPass(String),
    BadObjective(String),
    BadPlot(String),
    EmptyCurves(String),
}

impl From<TrainerError> for GrapherError {
    fn from(e: TrainerError) -> Self {
        GrapherError::BadTraining(format!("{e:?}"))
    }
}

impl From<NetworkError> for GrapherError {
    fn from(e: NetworkError) -> Self {
        GrapherError::BadForwardPass(format!("{e:?}"))
    }
}

impl From<ObjectiveError> for GrapherError {
    fn from(e: ObjectiveError) -> Self {
        GrapherError::BadObjective(format!("{e:?}"))
    }
}

// Mean loss over a data set after each epoch of training, for one
// trainer configuration
pub struct LossCurve {
    pub name: String,
    pub losses: Vec<f32>,
}

impl LossCurve {
    // Train one epoch at a time so the loss can be measured after each
    pub fn record(
        name: &str,
        trainer: &mut Trainer,
        data: &DataSet,
        batch_size: usize,
        epochs: usize,
    ) -> Result<Self, GrapherError> {
        let mut losses: Vec<f32> = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            trainer.run(data, batch_size, 1)?;
            losses.push(mean_loss(trainer, data)?);
        }

        Ok(Self {
            name: name.to_string(),
            losses,
        })
    }
}

// Objective averaged over every sample in the data set
fn mean_loss(trainer: &Trainer, data: &DataSet) -> Result<f32, GrapherError> {
    let mut total: f32 = 0.0;
    for (sample, label) in data.samples.iter().zip(data.labels.iter()) {
        let output = trainer.network.forward_pass(sample)?;
        total += trainer.objective.compute(&output.output, label)?;
    }
    Ok(total / (data.samples.len() as f32))
}

// Draw every curve on one chart, loss against epoch, and save it as a PNG
pub fn plot_loss_curves(curves: &[LossCurve], title: &str, path: &str) -> Result<(), GrapherError> {
    if curves.is_empty() {
        return Err(GrapherError::EmptyCurves(
            "At least one loss curve is needed to plot".to_string(),
        ));
    }

    if curves.iter().any(|curve| curve.losses.is_empty()) {
        return Err(GrapherError::EmptyCurves(
            "Every loss curve must have at least one epoch".to_string(),
        ));
    }

    draw_loss_curves(curves, title, path).map_err(|e| GrapherError::BadPlot(e.to_string()))
}

// The plotters calls, separated so their errors can be converted in one place
fn draw_loss_curves(
    curves: &[LossCurve],
    title: &str,
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new(path, (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    // Axis ranges: longest curve on x, log-scaled loss on y so the tail of
    // the curves stays visible. The y range is widened to at least one
    // decade because a log axis over a narrower range gets no tick labels
    let max_epochs: usize = curves.iter().map(|c| c.losses.len()).max().unwrap_or(0);
    let all_losses = curves.iter().flat_map(|c| c.losses.iter().copied());
    let max_loss: f32 = all_losses.clone().fold(f32::MIN, f32::max);
    let min_loss: f32 = all_losses
        .fold(f32::MAX, f32::min)
        .min(max_loss / 10.0)
        .max(1e-10);

    let mut chart = ChartBuilder::on(&root)
        .caption(title, ("sans-serif", 30))
        .margin(20)
        .x_label_area_size(40)
        .y_label_area_size(60)
        .build_cartesian_2d(0..max_epochs, (min_loss..max_loss).log_scale())?;

    chart
        .configure_mesh()
        .x_desc("Epoch")
        .y_desc("Loss")
        .draw()?;

    // One line per curve, epochs numbered from 1, with a legend entry each
    for (i, curve) in curves.iter().enumerate() {
        let color = Palette99::pick(i);
        chart
            .draw_series(LineSeries::new(
                curve
                    .losses
                    .iter()
                    .enumerate()
                    .map(|(epoch, &loss)| (epoch + 1, loss)),
                color.stroke_width(2),
            ))?
            .label(&curve.name)
            .legend(move |(x, y)| PathElement::new([(x, y), (x + 20, y)], color.stroke_width(2)));
    }

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperRight)
        .background_style(WHITE.mix(0.8))
        .border_style(BLACK)
        .draw()?;

    root.present()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mlp::activation::Activation;
    use mlp::layer::Layer;
    use mlp::network::Network;
    use mlp::objective::Objective;
    use mlp::updater::Updater;
    use ndarray::{Array1, Array2, array};

    #[test]
    fn test_grapher_loss_curve_record() {
        let test_weights: Array2<f32> = array![[0.5, 0.0], [0.0, 0.5]];
        let test_biases: Array1<f32> = array![0.0, 0.0];
        let test_activation = Activation::IDENTITY;
        let test_layer = Layer::new(test_weights, test_biases, test_activation).unwrap();
        let test_network = Network::new(vec![test_layer]).unwrap();

        let test_updater = Updater::SGD_SIMPLE { learning_rate: 0.1 };
        let mut test_trainer = Trainer::new(test_network, Objective::MSE, test_updater);

        // Learn the identity map on 2 samples
        let test_data = DataSet::from_data(
            vec![array![1.0, 0.0], array![0.0, 1.0]],
            vec![array![1.0, 0.0], array![0.0, 1.0]],
        )
        .unwrap();

        let test_epochs: usize = 10;
        let test_curve =
            LossCurve::record("test", &mut test_trainer, &test_data, 1, test_epochs).unwrap();

        assert_eq!(test_curve.name, "test");
        assert_eq!(test_curve.losses.len(), test_epochs);
        assert!(test_curve.losses[test_epochs - 1] < test_curve.losses[0]);
    }

    #[test]
    fn test_grapher_plot_loss_curves_invalid_args() {
        let test_curves: Vec<LossCurve> = Vec::new();
        let result = plot_loss_curves(&test_curves, "test", "target/test_loss_curves.png");
        assert!(result.is_err());

        let test_curves: Vec<LossCurve> = vec![LossCurve {
            name: "empty".to_string(),
            losses: Vec::new(),
        }];
        let result = plot_loss_curves(&test_curves, "test", "target/test_loss_curves.png");
        assert!(result.is_err());
    }

    #[test]
    fn test_grapher_plot_loss_curves_valid_args() {
        let test_curves: Vec<LossCurve> = vec![
            LossCurve {
                name: "a".to_string(),
                losses: vec![1.0, 0.5, 0.25, 0.125],
            },
            LossCurve {
                name: "b".to_string(),
                losses: vec![1.0, 0.8, 0.6],
            },
        ];

        // Written under target/ so it is discarded with the build artifacts
        let result = plot_loss_curves(&test_curves, "test", "target/test_loss_curves.png");
        assert!(result.is_ok());
    }
}
