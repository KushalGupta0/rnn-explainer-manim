# Module 06: Backpropagation Through Time — Deep Mathematical Derivation

[Back to Master Index](../index.md) | [Back to Module 05](05_loss_and_bptt_foundations.md) | [Next to Module 07](07_scalar_rnn_gradient_dynamics.md) | [Deep Dive: BPTT Derivation](../deep_dives/bptt_derivation/README.md)

---

## 1. The Core Dilemma: Shared Parameters Across an Unrolled Chain

In a feed-forward network, layer $l$ has its own private weight matrix $W^{[l]}$. Computing $\frac{\partial L}{\partial W^{[l]}}$ requires applying the chain rule up to layer $l$.
In an unrolled RNN, the recurrent matrix $W_{hh}$ appears at **every single timestep**:
$$a_1 = W_{xh} x_1 + W_{hh} h_0 + b_h$$
$$a_2 = W_{xh} x_2 + W_{hh} h_1 + b_h$$
$$\dots$$
$$a_T = W_{xh} x_T + W_{hh} h_{T-1} + b_h$$

To find the gradient with respect to $W_{hh}$, we apply the **multivariable chain rule on shared variables**:
$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L}{\partial a_t} \frac{\partial a_t}{\partial W_{hh}}$$

Let us define the pre-activation error adjoint:
$$\delta_t \equiv \frac{\partial L}{\partial a_t} \in \mathbb{R}^{d_h \times 1}$$

Because $a_t = W_{hh} h_{t-1} + W_{xh} x_t + b_h$, taking the derivative of $a_t$ with respect to $W_{hh}$ treating $h_{t-1}$ as fixed yields:
$$\frac{\partial a_t}{\partial W_{hh}} = h_{t-1}^\top \implies \delta_t h_{t-1}^\top \in \mathbb{R}^{d_h \times d_h}$$

Therefore, the total gradient for $W_{hh}$ is the **sum of outer products across all timesteps**:
$$\mathbf{\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \delta_t h_{t-1}^\top}$$

---

## 2. Unraveling the Adjoint Vector $\delta_t$

How is $\delta_t$ computed?
By definition:
$$\delta_t = \frac{\partial L}{\partial a_t} = \left(\frac{\partial h_t}{\partial a_t}\right)^\top \frac{\partial L}{\partial h_t}$$
Since $h_t = \phi(a_t)$ is an element-wise activation:
$$\frac{\partial h_t}{\partial a_t} = \operatorname{diag}\big(\phi'(a_t)\big) \equiv D_t \in \mathbb{R}^{d_h \times d_h}$$
Thus:
$$\delta_t = D_t \frac{\partial L}{\partial h_t}$$

From Module 05, the hidden state gradient satisfies the backward recurrence:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \left(\frac{\partial h_{t+1}}{\partial h_t}\right)^\top \frac{\partial L}{\partial h_{t+1}}$$

Notice that:
$$\frac{\partial h_{t+1}}{\partial h_t} = \frac{\partial h_{t+1}}{\partial a_{t+1}} \frac{\partial a_{t+1}}{\partial h_t} = D_{t+1} W_{hh}$$
Substituting this back in:
$$\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \left( D_{t+1} W_{hh} \right)^\top \frac{\partial L}{\partial h_{t+1}} = \frac{\partial L_t}{\partial h_t} + W_{hh}^\top D_{t+1} \frac{\partial L}{\partial h_{t+1}}$$

Because $\delta_{t+1} = D_{t+1} \frac{\partial L}{\partial h_{t+1}}$, we discover the elegant recurrence relation for $\delta_t$:
$$\mathbf{\delta_t = D_t \left( \frac{\partial L_t}{\partial h_t} + W_{hh}^\top \delta_{t+1} \right)}$$

---

## 3. The Long-Range Temporal Jacobian Product

Let us trace how a loss at the distant future step $T$ propagates backward to an early step $t$:
$$\frac{\partial L_T}{\partial h_t} = \left(\frac{\partial h_T}{\partial h_t}\right)^\top \frac{\partial L_T}{\partial h_T}$$

By the multivariate chain rule across intermediate states $h_T, h_{T-1}, \dots, h_t$:
$$\frac{\partial h_T}{\partial h_t} = \frac{\partial h_T}{\partial h_{T-1}} \frac{\partial h_{T-1}}{\partial h_{T-2}} \cdots \frac{\partial h_{t+1}}{\partial h_t} = \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}}$$

Substitute the single-step Jacobian $\frac{\partial h_k}{\partial h_{k-1}} = D_k W_{hh}$:
$$\mathbf{\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \left( D_k W_{hh} \right) = (D_T W_{hh})(D_{T-1} W_{hh})\dots(D_{t+1} W_{hh})}$$

```
Forward Flow:
h_t ──► [ D_{t+1} W_{hh} ] ──► h_{t+1} ──► ... ──► [ D_T W_{hh} ] ──► h_T ──► L_T

Backward Gradient Flow:
∂L_T/∂h_t ◄── [ W_{hh}ᵀ D_{t+1} ] ◄── ... ◄── [ W_{hh}ᵀ D_T ] ◄── ∂L_T/∂h_T
```

### The Architectural Consequence:
This repeated multiplication of $(T - t)$ matrices is the **single mathematical root cause** of both:
1. **Vanishing Gradients**: If the transformations repeatedly contract vectors.
2. **Exploding Gradients**: If the transformations repeatedly expand vectors.

---

## 4. Visual Semantics for Manim: Backward Matrix Flow

### How to Animate BPTT in Manim:
* **The Dual Direction**: Show forward activation arrows in cyan moving left-to-right, then pause at $T$, light up the loss nodes, and trigger backward orange arrows travelling right-to-left.
* **The Weight Accumulator**:
  * Position a central repository box labeled $\frac{\partial L}{\partial W_{hh}}$ at the bottom of the screen.
  * As the backward pulse hits each node $t$, show an outer product flash $\delta_t \otimes h_{t-1}$, generating an orange matrix tile.
  * These tiles physically fly down into the central repository box and stack together, visually demonstrating the summation:
    $$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \delta_t h_{t-1}^\top$$

---

## 📝 Checkpoint 06 Exercises

### Level 1 — Conceptual
Why does the formula for $\frac{\partial L}{\partial W_{hh}}$ require summing over all timesteps $t=1, \dots, T$, whereas in an MLP with 5 layers, the gradient for layer 1's weight does not sum over layers 2, 3, 4, 5?

### Level 2 — Mathematical
Given a 2-step RNN with $h_0 = 0$, linear activations ($\phi(z) = z \implies D_t = I$), and scalar parameters:
$h_1 = w x_1$, $h_2 = u h_1 + w x_2$, with loss $L = \frac{1}{2}(h_2 - y^*)^2$.
Derive $\frac{\partial L}{\partial u}$ directly using the multivariable chain rule.

### Level 3 — Debugging / Misconception
A student calculates the gradient for $W_{hh}$ by taking:
$\frac{\partial L}{\partial W_{hh}} = \frac{\partial L}{\partial h_T} \frac{\partial h_T}{\partial W_{hh}}$
and claims this captures all necessary parameter updates.
What enormous portion of the computational graph did this calculation completely omit?

### Level 4 — Visualization (Manim Translation)
Design a visual metaphor to represent the adjoint $\delta_t = D_t (\frac{\partial L_t}{\partial h_t} + W_{hh}^\top \delta_{t+1})$. How would you visually illustrate the component coming from the future ($\delta_{t+1}$) versus the immediate error ($\frac{\partial L_t}{\partial h_t}$)?

### Level 5 — Advanced (Jacobian Transpose Ordering)
Carefully prove that if $\frac{\partial h_k}{\partial h_t} = \prod_{j=t+1}^k (D_j W_{hh})$, then the backward gradient propagation satisfies:
$$\left(\frac{\partial h_k}{\partial h_t}\right)^\top = \prod_{j=k}^{t+1} (W_{hh}^\top D_j)$$
Why does the order of matrix multiplication reverse upon transposition?
