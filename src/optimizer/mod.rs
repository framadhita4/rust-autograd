use crate::autograd::Autograd;

pub mod adamw;
pub mod sgd;

pub use adamw::AdamW;
pub use sgd::SGD;

pub trait Optimizer {
    fn step(&mut self, parameters: &[Autograd]);
    fn zero_grad(&self, parameters: &[Autograd]) {
        for p in parameters {
            p.zero_grad();
        }
    }
}
