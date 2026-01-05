use ndarray::Array2;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::autograd::Autograd;

#[derive(Debug, Clone, Copy)]
pub enum Activation {
    ReLU,
    Tanh,
    Softmax,
    None,
}

#[derive(Debug, Clone)]
pub struct Neuron {
    weights: Vec<Autograd>,
    bias: Autograd,
}

impl Neuron {
    pub fn new(nin: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);

        let scale = (2.0 / nin as f64).sqrt();
        let weights = (0..nin)
            .map(|_| Autograd::new(Array2::from_elem((1, 1), rng.gen_range(-scale..scale))))
            .collect();
        let bias = Autograd::new(Array2::from_elem((1, 1), 0.0));

        Self { weights, bias }
    }

    pub fn call(&self, x: &[Autograd], activation: Activation) -> Autograd {
        let mut sum = self.bias.clone();
        for (w, xi) in self.weights.iter().zip(x.iter()) {
            // sum = sum + w * xi
            sum = sum.add(&w.mul(xi));
        }

        match activation {
            Activation::ReLU => sum.relu(),
            Activation::Tanh => sum.tanh(),
            Activation::Softmax => sum,
            Activation::None => sum,
        }
    }

    pub fn parameters(&self) -> Vec<Autograd> {
        let mut params = self.weights.clone();
        params.push(self.bias.clone());
        params
    }
}

pub struct Layer {
    neurons: Vec<Neuron>,
    activation: Activation,
}

impl Layer {
    pub fn new(nin: usize, nout: usize, activation: Activation, seed: u64) -> Self {
        let neurons = (0..nout)
            .map(|i| Neuron::new(nin, seed + i as u64))
            .collect();
        Self {
            neurons,
            activation,
        }
    }

    pub fn call(&self, x: &[Autograd]) -> Vec<Autograd> {
        let outputs = self
            .neurons
            .iter()
            .map(|n| n.call(x, self.activation))
            .collect::<Vec<Autograd>>();

        if let Activation::Softmax = self.activation {
            return self.softmax_layer(&outputs);
        }
        outputs
    }

    pub fn parameters(&self) -> Vec<Autograd> {
        self.neurons.iter().flat_map(|n| n.parameters()).collect()
    }

    fn softmax_layer(&self, logits: &Vec<Autograd>) -> Vec<Autograd> {
        let exps: Vec<Autograd> = logits.iter().map(|x| x.exp()).collect();

        let mut sum_exps = exps[0].clone();

        for i in 1..exps.len() {
            sum_exps = sum_exps.add(&exps[i]);
        }

        exps.into_iter().map(|x| x.div(&sum_exps)).collect()
    }
}

pub struct MLP {
    layers: Vec<Layer>,
}

impl MLP {
    pub fn new(nin: usize, nouts: &[usize], seed: u64) -> Self {
        let mut sizes = vec![nin];
        sizes.extend_from_slice(nouts);

        let layers = (0..nouts.len())
            .map(|i| {
                let activation = if i < nouts.len() - 1 {
                    Activation::ReLU
                } else {
                    Activation::Softmax
                };
                Layer::new(sizes[i], sizes[i + 1], activation, seed + i as u64)
            })
            .collect();

        Self { layers }
    }

    pub fn call(&self, x: &[Autograd]) -> Vec<Autograd> {
        let mut current = x.to_vec();
        for layer in &self.layers {
            current = layer.call(&current);
        }
        current
    }

    pub fn parameters(&self) -> Vec<Autograd> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }

    pub fn zero_grad(&self) {
        for p in self.parameters() {
            p.zero_grad();
        }
    }
}
