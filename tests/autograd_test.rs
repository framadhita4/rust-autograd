use ndarray::array;
use simple_mlp::autograd::Autograd;

#[test]
fn test_add() {
    let a = Autograd::new(array![[1.0, 2.0], [3.0, 4.0]]);
    let b = Autograd::new(array![[5.0, 6.0], [7.0, 8.0]]);
    let c = a.add(&b);
    assert_eq!(c.value(), array![[6.0, 8.0], [10.0, 12.0]]);

    c.set_grad(array![[1.0, 1.0], [1.0, 1.0]]);
    c.backward();
    assert_eq!(a.grad(), array![[1.0, 1.0], [1.0, 1.0]]);
    assert_eq!(b.grad(), array![[1.0, 1.0], [1.0, 1.0]]);
}

#[test]
fn test_mul() {
    let a = Autograd::new(array![[1.0, 2.0], [3.0, 4.0]]);
    let b = Autograd::new(array![[5.0, 6.0], [7.0, 8.0]]);
    let c = a.mul(&b);
    assert_eq!(c.value(), array![[19.0, 22.0], [43.0, 50.0]]);

    c.set_grad(array![[1.0, 0.0], [0.0, 1.0]]);
    c.backward();
    assert_eq!(a.grad(), array![[5.0, 7.0], [6.0, 8.0]]);
    assert_eq!(b.grad(), array![[1.0, 3.0], [2.0, 4.0]]);
}

#[test]
fn test_div() {
    let a = Autograd::new(array![[10.0, 20.0]]);
    let b = Autograd::new(array![[2.0, 4.0]]);
    let c = a.div(&b);
    assert_eq!(c.value(), array![[5.0, 5.0]]);

    c.set_grad(array![[1.0, 1.0]]);
    c.backward();
    assert_eq!(a.grad(), array![[0.5, 0.25]]);
    assert_eq!(b.grad(), array![[-2.5, -1.25]]);
}

#[test]
fn test_sub() {
    let a = Autograd::new(array![[10.0, 5.0]]);
    let b = Autograd::new(array![[3.0, 2.0]]);
    let c = a.sub(&b);
    assert_eq!(c.value(), array![[7.0, 3.0]]);

    c.set_grad(array![[1.0, 1.0]]);
    c.backward();
    assert_eq!(a.grad(), array![[1.0, 1.0]]);
    assert_eq!(b.grad(), array![[-1.0, -1.0]]);
}

#[test]
fn test_pow() {
    let a = Autograd::new(array![[2.0, 3.0]]);
    let b = a.pow(2.0);
    assert_eq!(b.value(), array![[4.0, 9.0]]);

    b.set_grad(array![[1.0, 1.0]]);
    b.backward();
    // d(x^2)/dx = 2*x
    assert_eq!(a.grad(), array![[4.0, 6.0]]);
}

#[test]
fn test_log() {
    let a = Autograd::new(array![[10.0, 20.0]]);
    let b = a.log();
    assert!((b.value()[[0, 0]] - 2.302585092994046).abs() < 1e-10);
    assert!((b.value()[[0, 1]] - 2.995732273553991).abs() < 1e-10);

    // dlog(x)/dx = 1/x
    b.set_grad(array![[1.0, 1.0]]);
    b.backward();
    // Use epsilon for float comparison
    assert!((a.grad()[[0, 0]] - 0.1).abs() < 1e-7);
    assert!((a.grad()[[0, 1]] - 0.05).abs() < 1e-7);
}

#[test]
fn test_neg() {
    let a = Autograd::new(array![[1.0, -2.0]]);
    let b = a.neg();
    assert_eq!(b.value(), array![[-1.0, 2.0]]);

    b.set_grad(array![[1.0, 1.0]]);
    b.backward();
    assert_eq!(a.grad(), array![[-1.0, -1.0]]);
}

#[test]
fn test_exp() {
    let a = Autograd::new(array![[0.0, 1.0]]);
    let b = a.exp();
    assert!((b.value()[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((b.value()[[0, 1]] - std::f64::consts::E).abs() < 1e-10);

    b.set_grad(array![[1.0, 1.0]]);
    b.backward();
    assert!((a.grad()[[0, 0]] - 1.0).abs() < 1e-10);
    assert!((a.grad()[[0, 1]] - std::f64::consts::E).abs() < 1e-10);
}

#[test]
fn test_relu() {
    let a = Autograd::new(array![[-1.0, 2.0]]);
    let b = a.relu();
    assert_eq!(b.value(), array![[0.0, 2.0]]);

    b.set_grad(array![[1.0, 1.0]]);
    b.backward();
    assert_eq!(a.grad(), array![[0.0, 1.0]]);
}

#[test]
fn test_tanh() {
    let a = Autograd::new(array![[0.0]]);
    let b = a.tanh();
    assert_eq!(b.value()[[0, 0]], 0.0);

    b.set_grad(array![[1.0]]);
    b.backward();
    assert_eq!(a.grad()[[0, 0]], 1.0);
}
