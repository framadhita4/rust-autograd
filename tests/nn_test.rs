use ndarray::array;
use rust_autograd::autograd::Autograd;
use rust_autograd::nn::{Activation, Layer, MLP, Neuron};

#[test]
fn test_neuron() {
    let n = Neuron::new(2, 42);
    let x = vec![Autograd::new(array![[1.0]]), Autograd::new(array![[-2.0]])];
    let y = n.call(&x, Activation::None);
    assert_eq!(y.value().shape(), &[1, 1]);

    let params = n.parameters();
    assert_eq!(params.len(), 3); // 2 weights + 1 bias
}

#[test]
fn test_layer() {
    let l = Layer::new(2, 3, Activation::ReLU, 42);
    let x = vec![Autograd::new(array![[1.0]]), Autograd::new(array![[-2.0]])];
    let y = l.call(&x);
    assert_eq!(y.len(), 3);
    assert_eq!(y[0].value().shape(), &[1, 1]);

    let params = l.parameters();
    assert_eq!(params.len(), 3 * (2 + 1)); // 3 neurons * (2 weights + 1 bias)
}

#[test]
fn test_layer_softmax_deterministic() {
    let l = Layer::new(2, 2, Activation::Softmax, 42);

    for p in l.parameters() {
        p.set_value(array![[0.0]]);
    }

    let x = vec![Autograd::new(array![[1.0]]), Autograd::new(array![[-2.0]])];
    let y = l.call(&x);

    // If logits are [0.0, 0.0], then:
    // exp(0.0) = 1.0
    // sum(exps) = 2.0
    // softmax = [1/2, 1/2] = [0.5, 0.5]

    assert_eq!(y.len(), 2);
    assert!((y[0].value()[[0, 0]] - 0.5).abs() < 1e-10);
    assert!((y[1].value()[[0, 0]] - 0.5).abs() < 1e-10);

    // Also verify that they sum to exactly 1.0
    let sum = y[0].value()[[0, 0]] + y[1].value()[[0, 0]];
    assert!((sum - 1.0).abs() < 1e-10);
}

#[test]
fn test_mlp() {
    let mlp = MLP::new(2, &[4, 1], 42);
    let x = vec![Autograd::new(array![[1.0]]), Autograd::new(array![[-2.0]])];
    let y = mlp.call(&x);
    assert_eq!(y.len(), 1);
    assert_eq!(y[0].value().shape(), &[1, 1]);

    let params = mlp.parameters();
    // Layer 1: 4 neurons * (2 + 1) = 12 params
    // Layer 2: 1 neuron * (4 + 1) = 5 params
    // Total: 17
    assert_eq!(params.len(), 17);
}
