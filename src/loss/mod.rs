pub mod mse;
pub mod softmax_cross_entropy;

pub use mse::MSE;
pub use softmax_cross_entropy::SoftmaxCrossEntropyLoss;

use crate::autograd::Autograd;

pub enum Reduction {
    Mean,
    Sum,
    None,
}

pub trait Loss {
    fn forward(&self, pred: &Vec<Autograd>, target_index: usize) -> Autograd;
}
