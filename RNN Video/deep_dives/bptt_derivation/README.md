# Deep Dive: Full Mathematical Derivation of Backpropagation Through Time (BPTT)

[Back to Master Index](../../index.md) | [Back to Module 06](../../modules/06_bptt_deep_derivation.md)

---

## 1. The Total Derivative on Shared Parameters

Let the objective function be the cumulative sequence loss:
$$L = \sum_{t=1}^T L_t(\hat{y}_t, y_t^*)$$
where:
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$
$$h_t = \phi(a_t)$$
$$o_t = W_{hy} h_t + b_y$$
$$\hat{y}_t = g(o_t)$$

In this system, the recurrent weight matrix $W_{hh}$ is **shared across all timesteps $t \in \{1, \dots, T\}$**.

### The Shared Parameter Theorem
When a parameter tensor $\Theta$ appears at multiple nodes in a computational graph, the total derivative of the scalar objective $L$ with respect to $\Theta$ is the **sum of the partial contributions across all points of occurrence**:
$$\frac{\partial L}{\partial \Theta} = \sum_{t=1}^T \left. \frac{\partial L}{\partial \Theta_{(t)}} \right|_{\Theta_{(t)} = \Theta}$$
where $\Theta_{(t)}$ denotes the fictitious copy of the parameter operating at timestep $t$.

---

## 2. Derivation of Parameter Gradients

### A. Output Weight Matrix $W_{hy}$
The matrix $W_{hy}$ affects the loss at timestep $t$ directly through the logit vector $o_t = W_{hy} h_t + b_y$:
$$\frac{\partial L_t}{\partial o_t} \in \mathbb{R}^{d_y \times 1}$$
Since $o_t$ is a linear function of $W_{hy}$:
$$\frac{\partial L_t}{\partial W_{hy}} = \frac{\partial L_t}{\partial o_t} h_t^\top \in \mathbb{R}^{d_y \times d_h}$$
Summing across all timesteps:
$$\mathbf{\frac{\partial L}{\partial W_{hy}} = \sum_{t=1}^T \frac{\partial L_t}{\partial o_t} h_t^\top}$$
Similarly, for the output bias $b_y$:
$$\mathbf{\frac{\partial L}{\partial b_y} = \sum_{t=1}^T \frac{\partial L_t}{\partial o_t}}$$

---

### B. Recurrent Weight Matrix $W_{hh}$
This is the central derivation.
At timestep $t$, $W_{hh}$ acts on $h_{t-1}$ to form the pre-activation $a_t$:
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$

Define the adjoint (error gradient) of the pre-activation vector $a_t$ as:
$$\delta_t \equiv \frac{\partial L}{\partial a_t} \in \mathbb{R}^{d_h \times 1}$$
By the chain rule:
$$\delta_t = \frac{\partial h_t}{\partial a_t}^\top \frac{\partial L}{\partial h_t} = \operatorname{diag}\big(\phi'(a_t)\big) \frac{\partial L}{\partial h_t}$$

Since $a_t = W_{hh} h_{t-1} + \dots$, the instantaneous gradient of the pre-activation with respect to $W_{hh}$ is:
$$\frac{\partial a_t}{\partial W_{hh}} \implies \delta_t h_{t-1}^\top \in \mathbb{R}^{d_h \times d_h}$$

Summing over all occurrences of $W_{hh}$ from $t=1$ to $T$:
$$\mathbf{\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \delta_t h_{t-1}^\top = \sum_{t=1}^T \operatorname{diag}\big(\phi'(a_t)\big) \frac{\partial L}{\partial h_t} h_{t-1}^\top}$$

---

### C. Input Weight Matrix $W_{xh}$ and Bias $b_h$
By exact algebraic symmetry:
$$\mathbf{\frac{\partial L}{\partial W_{xh}} = \sum_{t=1}^T \delta_t x_t^\top \in \mathbb{R}^{d_h \times d_x}}$$
$$\mathbf{\frac{\partial L}{\partial b_h} = \sum_{t=1}^T \delta_t \in \mathbb{R}^{d_h \times 1}}$$

---

## 3. Expanding the Temporal Recurrence: The Jacobian Product Chain

To compute $\delta_t = \operatorname{diag}(\phi'(a_t))\frac{\partial L}{\partial h_t}$, we require $\frac{\partial L}{\partial h_t}$.
As established in Module 05:
$$\frac{\partial L}{\partial h_t} = \sum_{k=t}^T \left(\frac{\partial h_k}{\partial h_t}\right)^\top \frac{\partial L_k}{\partial h_k}$$

Now examine the transition Jacobian $\frac{\partial h_k}{\partial h_t}$ for $k > t$.
By the chain rule of multivariate calculus:
$$\frac{\partial h_k}{\partial h_t} = \frac{\partial h_k}{\partial h_{k-1}} \frac{\partial h_{k-1}}{\partial h_{k-2}} \dots \frac{\partial h_{t+1}}{\partial h_t} = \prod_{j=t+1}^k \frac{\partial h_j}{\partial h_{j-1}}$$

### What is the single-step Jacobian $\frac{\partial h_j}{\partial h_{j-1}}$?
Since $h_j = \phi(a_j)$ and $a_j = W_{hh} h_{j-1} + W_{xh} x_j + b_h$:
$$\frac{\partial h_j}{\partial h_{j-1}} = \frac{\partial h_j}{\partial a_j} \frac{\partial a_j}{\partial h_{j-1}} = \operatorname{diag}\big(\phi'(a_j)\big) W_{hh}$$

Let:
$$D_j \equiv \operatorname{diag}\big(\phi'(a_j)\big) \in \mathbb{R}^{d_h \times d_h}$$
Then:
$$\mathbf{\frac{\partial h_j}{\partial h_{j-1}} = D_j W_{hh}}$$

Therefore, the multi-step temporal transition Jacobian is the **ordered product of matrix-vector operators**:
$$\mathbf{\frac{\partial h_k}{\partial h_t} = \prod_{j=t+1}^k \left( D_j W_{hh} \right) = (D_k W_{hh})(D_{k-1} W_{hh})\dots(D_{t+1} W_{hh})}$$

---

## 4. The Complete Master BPTT Equation

Substituting this chain into the gradient of the recurrent weight $W_{hh}$:
$$\mathbf{\frac{\partial L}{\partial W_{hh}} = \sum_{k=1}^T \sum_{t=1}^k \operatorname{diag}\big(\phi'(a_k)\big) \left[ \prod_{j=t+1}^k D_j W_{hh} \right]^\top \frac{\partial L_k}{\partial h_k} h_{t-1}^\top}$$

This single equation exposes the entire theoretical landscape of Recurrent Neural Networks:
1. Every gradient computation across temporal span $(k - t)$ requires multiplying $(k - t)$ matrices together.
2. If this matrix product collapses to zero, the learning signal from step $k$ **vanishes** before reaching step $t$.
3. If this matrix product grows unboundedly, the learning signal **explodes**, destroying numerical stability.
