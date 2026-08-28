#![warn(clippy::pedantic)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::double_must_use)]
#![allow(clippy::cast_precision_loss)]
#![allow(non_camel_case_types)]

pub mod activation;
pub mod cache;
pub mod data_set;
pub mod layer;
pub mod network;
pub mod objective;
pub mod updater;
pub mod trainer;
