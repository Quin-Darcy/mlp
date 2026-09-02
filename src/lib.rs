#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::double_must_use)]
#![allow(clippy::cast_precision_loss)]
#![allow(non_camel_case_types)]
#![allow(clippy::similar_names)]

pub mod activation;
pub mod cache;
pub mod data_set;
pub mod layer;
pub mod network;
pub mod objective;
pub mod trainer;
pub mod updater;
