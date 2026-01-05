use crate::autograd::Autograd;
use crate::loss::Loss;

pub struct SoftmaxCrossEntropyLoss {}

impl SoftmaxCrossEntropyLoss {
    pub fn new() -> Self {
        Self {}
    }
}

impl Loss for SoftmaxCrossEntropyLoss {
    fn forward(&self, pred: &Vec<Autograd>, target_index: usize) -> Autograd {
        let log_prob = pred[target_index].log();

        log_prob.neg()
    }
}
