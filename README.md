# micrograd-from-scratch

A tiny autograd engine and neural network library built from scratch in Python — inspired by [Andrej Karpathy's "The spelled-out intro to neural networks and backpropagation: building micrograd"](https://www.youtube.com/watch?v=VMj-3S1tku0&list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ).

This repo walks through implementing reverse-mode automatic differentiation (autograd) from first principles, then uses it to build a Multi-Layer Perceptron (MLP) — no PyTorch, no TensorFlow, just Python. A PyTorch-equivalent implementation is also included for comparison.

## Why build this?

Every layer of a neural network is really just a composite function `f(g(...(x)))`. Deriving gradients by hand for every parameter doesn't scale as networks get deeper. Autograd solves this generally: it builds a computation graph as operations happen, then walks it backward using the chain rule to compute every gradient automatically.

## What's inside

- **`Value` class** — a scalar wrapper that tracks its computation history (`_prev`, `_op`), gradient (`grad`), and the local backward function (`_backward`) needed to propagate gradients to its parents.
- **Operator overloading** — `__add__`, `__mul__`, etc., each defining both the forward computation and its corresponding local gradient rule.
- **Topological sort + backpropagation** — orders all nodes in the graph parent-before-child, then walks them in reverse, calling each node's `_backward()` to propagate gradients from output back to inputs.
- **Graph visualization** — a `draw_dot` helper using `graphviz.Digraph` to visualize the computation graph (nodes = values, edges = operations).
- **Neuron, Layer, MLP** — built on top of `Value`, composed from raw weights/biases up to a full multi-layer perceptron.
- **Training loop** — forward pass → loss → zero gradients → backward pass → parameter update, from scratch.
- **PyTorch equivalent** — the same `Neuron` / `Layer` / `MLP` / training loop reimplemented with `torch.Tensor` and `requires_grad=True`, showing how the same ideas map onto a real framework.

## Repository structure

```
.
├── Autograd.py       # Value class + autograd engine + graphviz visualization helper
├── Neural_Network.py # Neuron, Layer, MLP
├── blog.md           # Explanation of the entire code
└── README.md
```

## Usage

```python
from Autograd import Value
from Neural_Network import MLP

# a tiny 3-input, 2-hidden-layer (4,4), 1-output MLP
model = MLP(3, [4, 4, 1])

x = [2.0, 3.0, -1.0]
y_hat = model(x)

y_hat.backward()  # populates .grad on every Value in the graph

```

## Key ideas covered

- Reverse-mode automatic differentiation and the chain rule
- Building a custom autograd-tracking data structure from scratch
- Topological sort as the backbone of correct backpropagation ordering
- Translating a scalar-based autograd engine into a PyTorch tensor-based one

## Background

I wrote up the full walkthrough — the reasoning behind each design decision, the chain rule derivation, and common gotchas — as a blog post. [Link coming once it's published / see `blog.md` in this repo for now.]

## Acknowledgements

This project follows Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) lecture. All the core ideas belong to that lecture series — this repo is my from-scratch reimplementation while learning it, plus notes.
