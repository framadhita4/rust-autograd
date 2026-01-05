use crate::autograd::Autograd;
use crate::optimizer::Optimizer;

pub struct SGD {
    pub learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn step(&mut self, parameters: &[Autograd]) {
        for p in parameters {
            // w = w - lr * g
            let value = p.value();
            let grad = p.grad();
            let new_val = value - self.learning_rate * grad;

            p.set_value(new_val);
        }
    }
}
