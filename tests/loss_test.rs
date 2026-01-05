use ndarray::array;
use simple_mlp::autograd::Autograd;
use simple_mlp::loss::{Loss, MSE, SoftmaxCrossEntropyLoss};

#[test]
fn test_mse_loss() {
    let pred = vec![Autograd::new(array![[0.1]]), Autograd::new(array![[0.9]])];
    let target_index = 1;
    let loss_fn = MSE::new();
    let loss = loss_fn.forward(&pred, target_index);

    // target = [0.0, 1.0]
    // MSE = ((0.1-0.0)^2 + (0.9-1.0)^2) / 2
    //     = (0.01 + 0.01) / 2 = 0.01
    assert!((loss.value()[[0, 0]] - 0.01).abs() < 1e-7);

    loss.set_grad(array![[1.0]]);
    loss.backward();

    // dL/dp0 = (2/N) * (p0 - t0) = (2/2) * (0.1 - 0.0) = 0.1
    // dL/dp1 = (2/N) * (p1 - t1) = (2/2) * (0.9 - 1.0) = -0.1
    assert!((pred[0].grad()[[0, 0]] - 0.1).abs() < 1e-7);
    assert!((pred[1].grad()[[0, 0]] - (-0.1)).abs() < 1e-7);
}

#[test]
fn test_softmax_cross_entropy_loss() {
    let pred = vec![Autograd::new(array![[0.1]]), Autograd::new(array![[0.9]])];
    let target_index = 1;
    let loss_fn = SoftmaxCrossEntropyLoss::new();
    let loss = loss_fn.forward(&pred, target_index);

    // loss = -ln(pred[target_index]) = -ln(0.9)
    assert!((loss.value()[[0, 0]] - (-0.9f64.ln())).abs() < 1e-10);

    loss.set_grad(array![[1.0]]);
    loss.backward();

    // dL/dp[target] = -1/p[target] = -1/0.9
    assert!((pred[target_index].grad()[[0, 0]] - (-1.0 / 0.9)).abs() < 1e-7);
}
