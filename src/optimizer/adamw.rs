use crate::autograd::Autograd;
use crate::optimizer::Optimizer;
use ndarray::Array2;
use std::collections::HashMap;

pub struct AdamW {
    pub learning_rate: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub t: u32,
    m: HashMap<*const (), Array2<f64>>,
    v: HashMap<*const (), Array2<f64>>,
}

impl AdamW {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.0,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn with_params(
        learning_rate: f64,
        beta1: f64,
        beta2: f64,
        epsilon: f64,
        weight_decay: f64,
    ) -> Self {
        Self {
            learning_rate,
            beta1,
            beta2,
            epsilon,
            weight_decay,
            t: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }
}

impl Optimizer for AdamW {
    fn step(&mut self, parameters: &[Autograd]) {
        self.t += 1;
        let t = self.t as f64;

        for p in parameters {
            let ptr = p.as_ptr();
            let grad = p.grad();
            let value = p.value();

            // Get or initialize first moment (Mean)
            let m = self
                .m
                .entry(ptr)
                .or_insert_with(|| Array2::zeros(grad.raw_dim()));

            // b_1 * m_{t-1} + (1 - b_1) * g_t
            *m = self.beta1 * &*m + (1.0 - self.beta1) * &grad;

            // Get or initialize second moment (Variance)
            let v = self
                .v
                .entry(ptr)
                .or_insert_with(|| Array2::zeros(grad.raw_dim()));

            // b_2 * v_{t-1} + (1 - b_2) * g_t^2
            *v = self.beta2 * &*v + (1.0 - self.beta2) * &grad * &grad;

            // Bias correction
            // m_hat = m / (1 - b_1^t)
            let m_hat = m.clone() / (1.0 - self.beta1.powf(t));
            // v_hat = v / (1 - b_2^t)
            let v_hat = v.clone() / (1.0 - self.beta2.powf(t));

            // AdamW update: decouple weight decay
            // w = w - lr * (m_hat / (sqrt(v_hat) + eps) + wd * w)
            let update =
                m_hat / (v_hat.mapv(|x| x.sqrt()) + self.epsilon) + &value * self.weight_decay;
            let new_val = value - update * self.learning_rate;

            p.set_value(new_val);
        }
    }
}
