# Module 02: The RNN Cell — First Principles, Algebra & Dimensions

[Back to Master Index](../index.md) | [Back to Module 01](01_why_rnns_exist.md) | [Next to Module 03](03_unrolling_the_rnn.md)

---

## 1. First-Principles Derivation of the Recurrent Cell

In Module 01, we established the abstract recurrence relation:
$$h_t = f(x_t, h_{t-1})$$
Now, we derive the standard **vanilla Recurrent Neural Network (Elman RNN)** cell equations from elementary neural building blocks.

In a standard artificial neuron layer, an affine transformation is applied to an input vector, followed by a point-wise non-linear activation.
In an RNN, the cell receives **two distinct sources of information** at timestep $t$:
1. The external input vector $x_t \in \mathbb{R}^{d_x}$ (the current stimulus).
2. The past latent state vector $h_{t-1} \in \mathbb{R}^{d_h}$ (the accumulated temporal context).

To synthesize both inputs linearly before passing them through a non-linearity, we construct the **pre-activation vector** $a_t$:
$$\mathbf{a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h}$$

The updated hidden state is obtained by applying an element-wise activation function $\phi$:
$$\mathbf{h_t = \phi(a_t)}$$

Finally, to make a task-specific prediction at timestep $t$, the hidden state is projected to the output space via an affine transformation followed by output activation $g$:
$$\mathbf{o_t = W_{hy} h_t + b_y}$$
$$\mathbf{y_t = g(o_t)}$$

```
                   y_t  (Task Prediction ∈ ℝ^{d_y})
                    ▲
                    │ g(·)
                   o_t  (Output Pre-activation)
                    ▲
                    │ W_{hy} · h_t + b_y
                    │
                 [ h_t ] ◄── φ(·) ── [ a_t ]
                    ▲                  ▲
                    │ W_{hh}           │ W_{xh}
                    │                  │
                 h_{t-1}              x_t
            (Previous State)     (Current Input)
```

---

## 2. Rigorous Dimensional Analysis

To prevent dimension mismatches in neural implementations and mathematical derivations, every quantity must have a precisely defined vector space:

| Symbol | Mathematical Object | Dimensionality | Description |
| :--- | :--- | :--- | :--- |
| $x_t$ | Input vector | $\mathbb{R}^{d_x \times 1}$ | Observation at timestep $t$ |
| $h_{t-1}$ | Hidden state vector | $\mathbb{R}^{d_h \times 1}$ | Latent summary from timestep $t-1$ |
| $W_{xh}$ | Input-to-hidden weight matrix | $\mathbb{R}^{d_h \times d_x}$ | Maps input space into latent space |
| $W_{hh}$ | Hidden-to-hidden weight matrix | $\mathbb{R}^{d_h \times d_h}$ | Linear transition operator on latent space |
| $b_h$ | Hidden bias vector | $\mathbb{R}^{d_h \times 1}$ | Shifts the origin of the latent pre-activation |
| $a_t$ | Latent pre-activation vector | $\mathbb{R}^{d_h \times 1}$ | Affine combination of inputs and past state |
| $\phi(\cdot)$ | Activation function | $\mathbb{R}^{d_h} \to \mathbb{R}^{d_h}$ | Element-wise squashing function (typically $\tanh$) |
| $h_t$ | Updated hidden state vector | $\mathbb{R}^{d_h \times 1}$ | Synthesized latent representation at step $t$ |
| $W_{hy}$ | Hidden-to-output weight matrix | $\mathbb{R}^{d_y \times d_h}$ | Maps latent space to output space |
| $b_y$ | Output bias vector | $\mathbb{R}^{d_y \times 1}$ | Shifts the origin of the output pre-activation |
| $o_t$ | Output pre-activation vector | $\mathbb{R}^{d_y \times 1}$ | Unnormalized score / logit vector |
| $g(\cdot)$ | Output activation function | $\mathbb{R}^{d_y} \to \mathbb{R}^{d_y}$ | Softmax, Sigmoid, or Identity |
| $y_t$ | Output / prediction vector | $\mathbb{R}^{d_y \times 1}$ | Model emission at step $t$ |

### Why can $W_{xh} x_t$ and $W_{hh} h_{t-1}$ be added together?
Notice that $x_t \in \mathbb{R}^{d_x}$ and $h_{t-1} \in \mathbb{R}^{d_h}$ live in completely different vector spaces if $d_x \ne d_h$. They cannot be directly added.
However:
* $W_{xh}$ is a linear map $\mathbb{R}^{d_x} \to \mathbb{R}^{d_h}$. Therefore, $W_{xh} x_t \in \mathbb{R}^{d_h}$.
* $W_{hh}$ is an endomorphism $\mathbb{R}^{d_h} \to \mathbb{R}^{d_h}$. Therefore, $W_{hh} h_{t-1} \in \mathbb{R}^{d_h}$.
Because both linear transformations project their respective domains into the **common target space $\mathbb{R}^{d_h}$**, their vector addition:
$$(W_{xh} x_t) + (W_{hh} h_{t-1}) \in \mathbb{R}^{d_h}$$
is algebraically well-defined.

---

## 3. The Concatenation Representation (Compact Formulation)

In modern deep learning libraries (PyTorch, JAX), the two matrix multiplications are implemented as a single operation via block concatenation:

Let the combined weight matrix be:
$$W_h = \begin{bmatrix} W_{xh} & W_{hh} \end{bmatrix} \in \mathbb{R}^{d_h \times (d_x + d_h)}$$

Let the concatenated input-state vector be:
$$z_t = \begin{bmatrix} x_t \\ h_{t-1} \end{bmatrix} \in \mathbb{R}^{(d_x + d_h) \times 1}$$

Then:
$$a_t = W_h z_t + b_h = \begin{bmatrix} W_{xh} & W_{hh} \end{bmatrix} \begin{bmatrix} x_t \\ h_{t-1} \end{bmatrix} + b_h = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$

This equivalence proves that recurrent processing is mathematically identical to an MLP layer acting on the joint concatenation of $[x_t, h_{t-1}]^\top$.

---

## 4. Parameter Sharing Across Time

A cornerstone of recurrent models is that **the parameter tuple $\Theta = \{W_{xh}, W_{hh}, b_h, W_{hy}, b_y\}$ is strictly invariant with respect to time $t$**.

$$\Theta_1 = \Theta_2 = \dots = \Theta_T = \Theta$$

```
t=1:  a₁ = W_{xh} x₁ + W_{hh} h₀ + b_h   ──►  h₁ = φ(a₁)   ──►  y₁ = g(W_{hy} h₁ + b_y)
t=2:  a₂ = W_{xh} x₂ + W_{hh} h₁ + b_h   ──►  h₂ = φ(a₂)   ──►  y₂ = g(W_{hy} h₂ + b_y)
t=3:  a₃ = W_{xh} x₃ + W_{hh} h₂ + b_h   ──►  h₃ = φ(a₃)   ──►  y₃ = g(W_{hy} h₃ + b_y)
```

### Why is Parameter Sharing Crucial?
1. **Statistical Efficiency**: Instead of learning a separate set of weights for every timestep, the network learns a single, stationary **dynamical law** governing state transitions.
2. **Temporal Translation Equivariance**: If an event pattern shifts by $\Delta t$ timesteps, the exact same transformation operator is applied to it.
3. **Generalization to Arbitrary Sequences**: Because the model parameters do not depend on $T$, an RNN trained on sequences of length $T=20$ can immediately run on sequences of length $T=200$.

---

## 5. Concrete Numerical Example (Hand Calculation)

Let us define a miniature RNN with dimensions $d_x = 2$, $d_h = 2$, and $d_y = 1$:

Let:
$$W_{xh} = \begin{bmatrix} 0.5 & -0.2 \\ 0.1 & 0.4 \end{bmatrix}, \quad W_{hh} = \begin{bmatrix} 0.8 & 0.1 \\ -0.3 & 0.6 \end{bmatrix}, \quad b_h = \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix}$$
$$W_{hy} = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix}, \quad b_y = [0.0]$$
Activation functions: $\phi(z) = \tanh(z)$, $g(o) = o$ (linear output).

Initial boundary condition: $h_0 = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$.

Input at $t=1$: $x_1 = \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix}$.

### Step 1: Compute Pre-Activation $a_1$
$$W_{xh} x_1 = \begin{bmatrix} 0.5 & -0.2 \\ 0.1 & 0.4 \end{bmatrix} \begin{bmatrix} 1.0 \\ 0.5 \end{bmatrix} = \begin{bmatrix} 0.5(1.0) + (-0.2)(0.5) \\ 0.1(1.0) + 0.4(0.5) \end{bmatrix} = \begin{bmatrix} 0.5 - 0.1 \\ 0.1 + 0.2 \end{bmatrix} = \begin{bmatrix} 0.4 \\ 0.3 \end{bmatrix}$$

$$W_{hh} h_0 = \begin{bmatrix} 0.8 & 0.1 \\ -0.3 & 0.6 \end{bmatrix} \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} = \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix}$$

$$a_1 = W_{xh} x_1 + W_{hh} h_0 + b_h = \begin{bmatrix} 0.4 \\ 0.3 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.0 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 0.4 \\ 0.4 \end{bmatrix}$$

### Step 2: Compute Latent State $h_1$
$$h_1 = \tanh(a_1) = \begin{bmatrix} \tanh(0.4) \\ \tanh(0.4) \end{bmatrix} \approx \begin{bmatrix} 0.3799 \\ 0.3799 \end{bmatrix}$$

### Step 3: Compute Output $y_1$
$$o_1 = W_{hy} h_1 + b_y = \begin{bmatrix} 1.0 & -0.5 \end{bmatrix} \begin{bmatrix} 0.3799 \\ 0.3799 \end{bmatrix} + 0.0 = 1.0(0.3799) - 0.5(0.3799) = 0.1900$$
$$y_1 = g(o_1) = 0.1900$$

Now, observe what happens at $t=2$ with input $x_2 = \begin{bmatrix} 0.0 \\ 1.0 \end{bmatrix}$:
$$W_{xh} x_2 = \begin{bmatrix} -0.2 \\ 0.4 \end{bmatrix}$$
$$W_{hh} h_1 = \begin{bmatrix} 0.8 & 0.1 \\ -0.3 & 0.6 \end{bmatrix} \begin{bmatrix} 0.3799 \\ 0.3799 \end{bmatrix} = \begin{bmatrix} (0.8+0.1)(0.3799) \\ (-0.3+0.6)(0.3799) \end{bmatrix} = \begin{bmatrix} 0.9(0.3799) \\ 0.3(0.3799) \end{bmatrix} = \begin{bmatrix} 0.3419 \\ 0.1140 \end{bmatrix}$$
$$a_2 = \begin{bmatrix} -0.2 \\ 0.4 \end{bmatrix} + \begin{bmatrix} 0.3419 \\ 0.1140 \end{bmatrix} + \begin{bmatrix} 0.0 \\ 0.1 \end{bmatrix} = \begin{bmatrix} 0.1419 \\ 0.6140 \end{bmatrix}$$
$$h_2 = \tanh(a_2) = \begin{bmatrix} \tanh(0.1419) \\ \tanh(0.6140) \end{bmatrix} \approx \begin{bmatrix} 0.1410 \\ 0.5470 \end{bmatrix}$$
$$y_2 = W_{hy} h_2 = 1.0(0.1410) - 0.5(0.5470) = 0.1410 - 0.2735 = -0.1325$$

---

## 6. Visual Semantics for Manim: Part II Mental Models

| Mathematical Object | Geometric Interpretation | 2D Manim Visual Language |
| :--- | :--- | :--- |
| **$W_{xh} x_t$** | Linear mapping from $\mathbb{R}^{d_x} \to \mathbb{R}^{d_h}$. | External vector projected into the latent coordinate frame. |
| **$W_{hh} h_{t-1}$** | Linear transformation (rotation, shear, scaling) of $\mathbb{R}^{d_h}$. | The coordinate grid of the state space distorts; the old arrow $h_{t-1}$ transforms into $W_{hh} h_{t-1}$. |
| **$a_t$ (Addition)** | Vector sum of the transformed past and present. | Parallelogram law: arrow $W_{xh} x_t$ chained to the tip of $W_{hh} h_{t-1}$, plus bias displacement. |
| **$\phi(a_t)$ (Squashing)** | Non-linear contraction mapping into hypercube $[-1, 1]^{d_h}$. | The entire unbounded grid is warped non-linearly; any vectors stretching past $\pm 1$ get compressed toward the bounding box. |

---

## 📝 Checkpoint 02 Exercises

### Level 1 — Conceptual
Explain the physical and computational role of the matrix $W_{hh}$. How does its role differ from $W_{xh}$?

### Level 2 — Mathematical
Given $d_x = 100, d_h = 256, d_y = 50$:
1. Calculate the exact total number of trainable parameters in this vanilla RNN cell (including all weights and biases).
2. If the sequence length increases from $T=10$ to $T=1000$, does the parameter count change? Why or why not?

### Level 3 — Debugging / Misconception
A student writes the pre-activation equation as:
$$a_t = W_{xh} x_t \cdot W_{hh} h_{t-1} + b_h$$
using element-wise multiplication between the transformed vectors.
Assuming $d_x = 4$ and $d_h = 4$, why is element-wise interaction structurally different from matrix addition, and what happens to the gradient pathways if $W_{hh} h_{t-1} \approx 0$?

### Level 4 — Visualization (Manim Translation)
In Manim, how would you visually communicate that the matrix $W_{hh}$ is being reused identically at step $t=1, t=2, \dots, t=T$, rather than being a new matrix at each step?

### Level 5 — Advanced (Linear Algebra)
Consider the linear recurrence $h_t = W_{hh} h_{t-1}$ with zero inputs ($x_t = 0$). Under what condition on the eigenvalues $\lambda_i$ of $W_{hh}$ will $\|h_t\| \to 0$ as $t \to \infty$ for all initial states $h_0$?
