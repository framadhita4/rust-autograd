use ndarray::Array2;
use rust_autograd::autograd::Autograd;
use rust_autograd::loss::{Loss, MSE, SoftmaxCrossEntropyLoss};
use rust_autograd::nn::MLP;
use rust_autograd::optimizer::{AdamW, Optimizer, SGD};

fn main() {
    // 2 -> 4 -> 2
    let mlp = MLP::new(2, &[4, 2], 42);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![0.0, 1.0, 1.0, 0.0];

    let epochs = 1000;
    let learning_rate = 0.1;

    // let loss_fn = MSE::new();
    let loss_fn = SoftmaxCrossEntropyLoss::new();

    // let mut optimizer = AdamW::new(learning_rate);
    let mut optimizer = SGD::new(learning_rate);

    println!("Starting training...");

    let parameters = mlp.parameters();

    for epoch in 1..=epochs {
        let mut total_loss = Autograd::new(Array2::zeros((1, 1)));

        for (x_data, &y_target) in inputs.iter().zip(targets.iter()) {
            let x: Vec<Autograd> = x_data
                .iter()
                .map(|&v| Autograd::new(Array2::from_elem((1, 1), v)))
                .collect();

            let outputs = mlp.call(&x);
            let loss = loss_fn.forward(&outputs, y_target as usize);

            total_loss = total_loss.add(&loss);
        }

        optimizer.zero_grad(&parameters);
        total_loss.set_grad(Array2::from_elem((1, 1), 1.0));
        total_loss.backward();

        optimizer.step(&parameters);

        if epoch % 50 == 0 || epoch == 1 {
            let loss_val = total_loss.value()[[0, 0]];
            println!("Epoch {:3} | Loss: {:.6}", epoch, loss_val);
        }
    }

    println!("\nTesting predictions:");
    for x_data in &inputs {
        let x: Vec<Autograd> = x_data
            .iter()
            .map(|&v| Autograd::new(Array2::from_elem((1, 1), v)))
            .collect();
        let outputs = mlp.call(&x);
        println!(
            "Input: {:?} | P(class=0): {:.4} | P(class=1): {:.4}",
            x_data,
            outputs[0].value()[[0, 0]],
            outputs[1].value()[[0, 0]]
        );
    }
}
