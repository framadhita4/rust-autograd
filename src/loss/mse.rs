use ndarray::Array2;

use crate::{autograd::Autograd, loss::Loss};

pub struct MSE {}

impl MSE {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for MSE {
    fn forward(&self, pred: &Vec<Autograd>, target_index: usize) -> Autograd {
        let mut total_loss = Autograd::new(Array2::zeros((1, 1)));

        for (i, p) in pred.iter().enumerate() {
            let target_val = if i == target_index { 1.0 } else { 0.0 };
            let target = Autograd::new(Array2::from_elem((1, 1), target_val));
            let diff = p.sub(&target).pow(2.0);
            total_loss = total_loss.add(&diff);
        }

        total_loss.div(&Autograd::new(Array2::from_elem((1, 1), pred.len() as f64)))
    }
}
