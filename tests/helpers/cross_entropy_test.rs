#[test]
fn test_cross_entropy_loss() {
    let pred = vec![
        Autograd::new(array![[0.1, 0.2, 0.3]]),
        Autograd::new(array![[0.4, 0.5, 0.6]]),
        Autograd::new(array![[0.7, 0.8, 0.9]]),
    ];
    let target = 1;
    let loss = cross_entropy_loss(&pred, target);
    assert_eq!(loss, 0.2);
}
