use ndarray::array;
use simple_mlp::autograd::Autograd;
use simple_mlp::optimizer::{AdamW, Optimizer, SGD};

#[test]
fn test_sgd_optimizer() {
    let p = Autograd::new(array![[10.0]]);
    let params = vec![p.clone()];
    let mut optim = SGD::new(0.1);

    p.set_grad(array![[2.0]]);
    optim.step(&params);

    // w = w - lr * grad = 10.0 - 0.1 * 2.0 = 9.8
    assert!((p.value()[[0, 0]] - 9.8).abs() < 1e-10);
}

#[test]
fn test_adamw_optimizer() {
    let p = Autograd::new(array![[1.0]]);
    let params = vec![p.clone()];
    // lr=0.1, beta1=0.9, beta2=0.999, eps=1e-8, wd=0.0
    let mut optim = AdamW::new(0.1);

    p.set_grad(array![[0.1]]);
    optim.step(&params);

    // First step of Adam:
    // t = 1
    // m = 0.9 * 0 + (1-0.9) * 0.1 = 0.01
    // v = 0.999 * 0 + (1-0.999) * 0.1^2 = 0.001 * 0.01 = 0.00001
    // m_hat = 0.01 / (1 - 0.9^1) = 0.01 / 0.1 = 0.1
    // v_hat = 0.00001 / (1 - 0.999^1) = 0.00001 / 0.001 = 0.01
    // update = m_hat / (sqrt(v_hat) + eps) = 0.1 / (0.1 + 1e-8) ≈ 1.0
    // w = w - lr * update = 1.0 - 0.1 * 1.0 = 0.9
    assert!((p.value()[[0, 0]] - 0.9).abs() < 1e-7);
}
