use ndarray::array;
use rust_autograd::autograd::Autograd;
use rust_autograd::helpers::visualization::Visualizer;
use std::io::Cursor;

#[test]
fn test_visualization() {
    let a = Autograd::new(array![[1.0]]);
    a.set_name("a");
    let b = Autograd::new(array![[2.0]]);
    b.set_name("b");
    let c = a.mul(&b); // c = a * b
    c.set_name("c");
    let d = c.add(&a); // d = c + a
    d.set_name("d");

    let mut buffer = Cursor::new(Vec::new());
    let visualizer = Visualizer::new()
        .vertical(false)
        .show_values(true)
        .precision(2)
        .add_output(d.clone(), "result".to_string());

    visualizer.draw(&d, &mut buffer).unwrap();

    // Also test saving to file
    visualizer.save(&d, "graphs/test_graph.dot").unwrap();
    assert!(std::path::Path::new("graphs/test_graph.dot").exists());
    std::fs::remove_file("graphs/test_graph.dot").unwrap();

    let dot = String::from_utf8(buffer.into_inner()).unwrap();

    assert!(dot.contains("digraph G"));
    assert!(dot.contains("rankdir=\"LR\""));
    // Use more flexible pattern matching for labels since node IDs (a0, a1...) can change depending on topo order
    assert!(dot.contains("label=\"a\\ndata: 1.00, grad: 0.00\""));
    assert!(dot.contains("label=\"b\\ndata: 2.00, grad: 0.00\""));
    assert!(dot.contains("Mul"));
    assert!(dot.contains("Add"));
    assert!(dot.contains("output0 [label=\"result\" shape=oval]"));
}
