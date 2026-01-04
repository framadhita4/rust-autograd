use ndarray::Array2;
use simple_mlp::autograd::Autograd;
use simple_mlp::helpers::cross_entropy::cross_entropy_loss;
use simple_mlp::mlp::MLP;

fn main() {
    // 2 -> 4 -> 1
    let mlp = MLP::new(2, &[4, 2]);

    let inputs = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let targets = vec![0.0, 1.0, 1.0, 0.0];

    let epochs = 1000;
    let learning_rate = 0.1;

    println!("Starting training...");

    for epoch in 1..=epochs {
        let mut total_loss = Autograd::new(Array2::zeros((1, 1)));

        for (x_data, &y_target) in inputs.iter().zip(targets.iter()) {
            let x: Vec<Autograd> = x_data
                .iter()
                .map(|&v| Autograd::new(Array2::from_elem((1, 1), v)))
                .collect();

            let outputs = mlp.call(&x);
            let loss = cross_entropy_loss(&outputs, y_target as usize);

            total_loss = total_loss.add(&loss);
        }

        mlp.zero_grad();
        total_loss.set_grad(Array2::from_elem((1, 1), 1.0));
        total_loss.backward();

        // Update parameters (SGD)
        for p in mlp.parameters() {
            let grad = p.grad();
            let current_val = p.value();
            let new_val = current_val - grad * learning_rate;
            p.set_value(new_val);
        }

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
            "Input: {:?} | Pred: {:.4}",
            x_data,
            outputs[0].value()[[0, 0]]
        );
    }
}
