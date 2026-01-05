use crate::autograd::Autograd;
use std::collections::HashMap;
use std::io::Write;

pub struct Visualizer {
    pub vertical: bool,
    pub show_values: bool,
    pub precision: usize,
    pub output_nodes: Vec<(Autograd, String)>,
}

impl Default for Visualizer {
    fn default() -> Self {
        Self {
            vertical: false,
            show_values: true,
            precision: 4,
            output_nodes: Vec::new(),
        }
    }
}

impl Visualizer {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn vertical(mut self, vertical: bool) -> Self {
        self.vertical = vertical;
        self
    }

    pub fn show_values(mut self, show_values: bool) -> Self {
        self.show_values = show_values;
        self
    }

    pub fn precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    pub fn add_output(mut self, node: Autograd, name: String) -> Self {
        self.output_nodes.push((node, name));
        self
    }

    pub fn save(&self, root: &Autograd, path: &str) -> std::io::Result<()> {
        let mut file = std::fs::File::create(path)?;
        self.draw(root, &mut file)
    }

    pub fn draw(&self, root: &Autograd, writer: &mut impl Write) -> std::io::Result<()> {
        let nodes = root.get_topo();

        writeln!(
            writer,
            "digraph G {{\nrankdir=\"{}\";\nnewrank=true;",
            if self.vertical { "TB" } else { "LR" }
        )?;

        let mut node_to_id = HashMap::new();
        for (id, node) in nodes.iter().enumerate() {
            node_to_id.insert(node.as_ptr(), id);
        }

        for (id, node) in nodes.iter().enumerate() {
            let name = node.name();
            let op = node.op();

            let display_name = if op == "None" {
                if name.is_empty() {
                    "Value".to_string()
                } else {
                    name
                }
            } else {
                op
            };

            let label = if self.show_values {
                let value = node.value();
                let grad = node.grad();

                if value.len() == 1 {
                    format!(
                        "{}\\ndata: {:.prec$}, grad: {:.prec$}",
                        display_name,
                        value[[0, 0]],
                        grad[[0, 0]],
                        prec = self.precision
                    )
                } else {
                    format!(
                        "{}\\nval shape: {:?}\\ngrad shape: {:?}",
                        display_name,
                        value.shape(),
                        grad.shape()
                    )
                }
            } else {
                display_name
            };

            let color = if node.op() == "None" {
                "style=filled fillcolor=\"#7fff7f\"" // leaf node: green
            } else {
                "style=filled fillcolor=\"#ffff7f\"" // op node: yellow
            };

            writeln!(
                writer,
                "a{} [label=\"{}\" shape=rect {}];",
                id, label, color
            )?;

            for child in node.children() {
                if let Some(&child_id) = node_to_id.get(&child.as_ptr()) {
                    writeln!(writer, "a{} -> a{};", child_id, id)?;
                }
            }
        }

        // Add output nodes
        for (i, (node, name)) in self.output_nodes.iter().enumerate() {
            if let Some(&node_id) = node_to_id.get(&node.as_ptr()) {
                writeln!(writer, "output{} [label=\"{}\" shape=oval];", i, name)?;
                writeln!(writer, "a{} -> output{} [ style=\"bold\" ];", node_id, i)?;
            }
        }

        writeln!(writer, "}}")?;
        Ok(())
    }
}
