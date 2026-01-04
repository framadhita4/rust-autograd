use crate::autograd::Autograd;

pub fn cross_entropy_loss(pred: &Vec<Autograd>, target_index: usize) -> Autograd {
    let log_prob = pred[target_index].log();

    log_prob.neg()
}
