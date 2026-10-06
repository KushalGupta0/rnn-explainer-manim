# Module 03: Unrolling Through Time — Loops, DAGs & Equivalence

[Back to Master Index](../index.md) | [Back to Module 02](02_the_rnn_cell.md) | [Next to Module 04](04_forward_pass_numerical.md)

---

## 1. The Recurrent Loop vs. The Unrolled Graph

An RNN is fundamentally defined by a recursive update operator:
$$h_t = \phi(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$

This computation can be drawn in two visually distinct, but **mathematically identical** ways:

```
REPRESENTATION 1: Cyclical Computational Loop
                 y_t
                  ▲
                  │ W_{hy}
             ┌────┴────┐
    ┌───────►│   h_t   │───────┐
    │        └────┬────┘       │
    │             ▲            │
    │ W_{hh}      │ W_{xh}     │ (Delay line: t-1)
    └─────────────┼────────────┘
                 x_t

REPRESENTATION 2: Unrolled Directed Acyclic Graph (DAG)
         y₁                   y₂                   y₃                   y_T
         ▲                    ▲                    ▲                    ▲
         │ W_{hy}             │ W_{hy}             │ W_{hy}             │ W_{hy}
       [ h₁ ] ─────────────►[ h₂ ] ─────────────►[ h₃ ] ─── ··· ──────►[ h_T ]
         ▲         W_{hh}     ▲         W_{hh}     ▲         W_{hh}     ▲
         │ W_{xh}             │ W_{xh}             │ W_{xh}             │ W_{xh}
        x₁                   x₂                   x₃                   x_T
```

### Why are they mathematically identical?
* **Representation 1** illustrates the **static physical architecture**: a single set of shared weight matrices acting recursively.
* **Representation 2** illustrates the **execution trace over time**: the computational graph obtained by expanding the loop for a sequence of length $T$.
* In automatic differentiation frameworks (PyTorch, TensorFlow, JAX), computers cannot backpropagate through cyclic graphs. The runtime **must unroll the loop into a Directed Acyclic Graph (DAG)** to establish the causal topological order of operations.

---

## 2. Explicit Algebraic Unrolling (Expanding the Recurrence)

Let us explicitly expand $h_t$ for the first three timesteps with initial state $h_0$:

### Timestep 1:
$$h_1 = \phi(W_{xh} x_1 + W_{hh} h_0 + b_h)$$

### Timestep 2:
$$h_2 = \phi(W_{xh} x_2 + W_{hh} h_1 + b_h)$$
Substitute the expression for $h_1$:
$$h_2 = \phi\Big(W_{xh} x_2 + W_{hh} \phi(W_{xh} x_1 + W_{hh} h_0 + b_h) + b_h\Big)$$

### Timestep 3:
$$h_3 = \phi(W_{xh} x_3 + W_{hh} h_2 + b_h)$$
Substitute the expression for $h_2$:
$$h_3 = \phi\Big(W_{xh} x_3 + W_{hh} \phi\big(W_{xh} x_2 + W_{hh} \phi(W_{xh} x_1 + W_{hh} h_0 + b_h) + b_h\big) + b_h\Big)$$

### The General Case: Deep Nested Composition
For any timestep $T$:
$$h_T = \phi\Big(W_{xh} x_T + W_{hh} \phi\big(W_{xh} x_{T-1} + \dots + W_{hh} \phi(W_{xh} x_1 + W_{hh} h_0 + b_h) \dots + b_h\big) + b_h\Big)$$

### The Core Pedagogical Insight:
> [!IMPORTANT]
> **An RNN of sequence length $T$ is mathematically equivalent to a $T$-layer deep feed-forward neural network!**
> * Depth is created not by stacking physical layers vertically, but by **propagating through time horizontally**.
> * The crucial difference from a standard $T$-layer MLP is that in an RNN, **every single layer shares the exact same weight matrix $W_{hh}$ and $W_{xh}$**.

---

## 3. Explicit Conceptual Taxonomy: Clarifying Terminology

Before deriving backpropagation, we must avoid conflating five core concepts:

| Term | Exact Definition | Changes Over Time? |
| :--- | :--- | :--- |
| **Recurrent Cell** | The parameterized hardware/mathematical operator $\mathcal{F}_\Theta(x_t, h_{t-1})$. | **No** (Static definition) |
| **Parameters ($\Theta$)** | The learned tensors $\{W_{xh}, W_{hh}, b_h, W_{hy}, b_y\}$. | **No** (Stationary across timesteps) |
| **Activations / States** | The forward evaluation values $\{a_t, h_t, o_t, y_t\}$. | **Yes** (Unique value for each $t$) |
| **Unrolled Network** | The $T$-step acyclic execution chain. | **Dynamic** (Length matches sequence $T$) |
| **Computational Graph** | The complete bipartite graph of operational nodes and tensor values used for reverse-mode automatic differentiation. | **Constructed per sample** |

---

## 4. Visual Semantics for Manim: Transitioning from Loop to DAG

How should a 3Blue1Brown-style video show the relationship between the loop and the unrolled graph?

### 1. The "Unspooling" Motion (Camera & Geometry)
* **Start**: A single central recurrent cell with an arrow emerging from its right side and looping back into its left side. A circular icon glows periodically as tokens $x_1, x_2, x_3$ enter from the bottom.
* **Transition**: The camera pulls back. The cyclical loop arrow snaps open, straightening into a horizontal directed vector.
* **Duplication with Ghosting**: As the loop straightens, duplicate nodes materialize across the horizontal axis: $t=1, t=2, t=3, \dots, t=T$.
* **Visual Anchor for Parameter Sharing**:
  * Color-code $W_{hh}$ with a vibrant primary hue (e.g., `#FF5555` Coral Red).
  * As the duplicate cells appear, draw a single glowing master badge labeled $\mathbf{W_{hh}}$ above the canvas, casting projected light rays onto each horizontal connection. This instantly conveys to the viewer: *"These are not new parameters; this is the same matrix evaluated repeatedly."*

---

## 📝 Checkpoint 03 Exercises

### Level 1 — Conceptual
Explain why deep learning frameworks like PyTorch must unroll an RNN into a Directed Acyclic Graph (DAG) before executing the backward pass, even though the forward code is written as a simple `for` loop.

### Level 2 — Mathematical
Consider a linear RNN ($h_t = W_{hh} h_{t-1} + W_{xh} x_t$) with $h_0 = \mathbf{0}$.
By induction, write the closed-form expression for $h_T$ as a sum involving powers of $W_{hh}$:
$$h_T = \sum_{k=1}^T (\dots)$$
What power of $W_{hh}$ acts on the first input $x_1$?

### Level 3 — Debugging / Misconception
A beginner states:
*"In an unrolled RNN of 100 timesteps, there are 100 times more weights to train than in an RNN of 1 timestep, which is why long sequences overfit."*
Correct this statement mathematically.

### Level 4 — Visualization (Manim Translation)
If you had to animate the influence of $x_1$ propagating across the unrolled chain to $y_4$, describe how you would use color, opacity, or particle flow to show how $x_1$ gets transformed at each intermediate node.

### Level 5 — Advanced (Computational Complexity)
For a sequence of length $T$, hidden dimension $d_h$, and input dimension $d_x$:
1. What is the time complexity of the forward pass?
2. What is the memory (space) complexity required to store intermediate activations for backpropagation? Why cannot the intermediate states $h_t$ be discarded during the forward pass?
