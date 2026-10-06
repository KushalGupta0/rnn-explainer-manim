# Deep Dive: The Recurrent Transition Jacobian & Product Chains

[Back to Master Index](../../index.md) | [Back to Module 09](../../modules/09_matrix_jacobian_view.md)

---

## 1. Vector Calculus Foundations: The Jacobian Matrix

Let $f: \mathbb{R}^n \to \mathbb{R}^m$ be a differentiable vector-valued function.
The **Jacobian matrix** $\frac{\partial f}{\partial x} \in \mathbb{R}^{m \times n}$ contains all first-order partial derivatives:
$$\left[ \frac{\partial f}{\partial x} \right]_{ij} = \frac{\partial f_i}{\partial x_j}$$

In our standard numerator-layout / vector calculus convention:
If $y = f(x)$, a small perturbation $dx \in \mathbb{R}^n$ produces a linear change $dy \in \mathbb{R}^m$:
$$dy = \left(\frac{\partial f}{\partial x}\right) dx$$

---

## 2. Derivation of the Single-Step Recurrent Jacobian

Consider the recurrent state update:
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h \in \mathbb{R}^{d_h}$$
$$h_t = \phi(a_t) \in \mathbb{R}^{d_h}$$

We seek the Jacobian matrix of $h_t$ with respect to the predecessor state $h_{t-1}$:
$$J_t \equiv \frac{\partial h_t}{\partial h_{t-1}} \in \mathbb{R}^{d_h \times d_h}$$

By the multivariate chain rule:
$$\frac{\partial h_t}{\partial h_{t-1}} = \frac{\partial h_t}{\partial a_t} \frac{\partial a_t}{\partial h_{t-1}}$$

### Step A: Derivative of Element-Wise Activation $\frac{\partial h_t}{\partial a_t}$
Because $\phi$ acts element-wise, the $i$-th component of $h_t$ depends **exclusively** on the $i$-th component of $a_t$:
$$[h_t]_i = \phi([a_t]_i) \implies \frac{\partial [h_t]_i}{\partial [a_t]_j} = \begin{cases} \phi'([a_t]_i) & \text{if } i = j \\ 0 & \text{if } i \ne j \end{cases}$$

Therefore, the Jacobian $\frac{\partial h_t}{\partial a_t}$ is a **pure diagonal matrix**:
$$D_t \equiv \frac{\partial h_t}{\partial a_t} = \operatorname{diag}\big(\phi'(a_t)\big) = \begin{bmatrix} \phi'([a_t]_1) & 0 & \dots & 0 \\ 0 & \phi'([a_t]_2) & \dots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \dots & \phi'([a_t]_{d_h}) \end{bmatrix}$$

### Step B: Derivative of Affine Map $\frac{\partial a_t}{\partial h_{t-1}}$
Since $a_t = W_{hh} h_{t-1} + \text{constant}$:
$$\frac{\partial a_t}{\partial h_{t-1}} = W_{hh} \in \mathbb{R}^{d_h \times d_h}$$

### Step C: The Combined Single-Step Jacobian
Combining both steps:
$$\mathbf{J_t = \frac{\partial h_t}{\partial h_{t-1}} = D_t W_{hh}}$$

---

## 3. The Multi-Step Temporal Jacobian Product

To evaluate how early representations influence future predictions, we examine the sensitivity of $h_T$ to $h_t$ ($T > t$):
$$\frac{\partial h_T}{\partial h_t} = \frac{\partial h_T}{\partial h_{T-1}} \frac{\partial h_{T-1}}{\partial h_{T-2}} \cdots \frac{\partial h_{t+1}}{\partial h_t}$$

Substituting the single-step Jacobians:
$$\mathbf{\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T (D_k W_{hh}) = (D_T W_{hh})(D_{T-1} W_{hh})\dots(D_{t+1} W_{hh})}$$

---

## 4. Transpose Ordering in Gradient Backpropagation

When propagating a scalar loss gradient $\nabla_{h_T} L \equiv \frac{\partial L}{\partial h_T} \in \mathbb{R}^{1 \times d_h}$ backwards to $h_t$:
$$\nabla_{h_t} L = \nabla_{h_T} L \cdot \left(\frac{\partial h_T}{\partial h_t}\right)$$

In column-vector gradient convention:
$$\frac{\partial L}{\partial h_t} = \left(\frac{\partial h_T}{\partial h_t}\right)^\top \frac{\partial L}{\partial h_T}$$

Recall the matrix transpose property: $(A B)^\top = B^\top A^\top$.
Applying this to the entire product chain:
$$\left( (D_T W_{hh})(D_{T-1} W_{hh})\dots(D_{t+1} W_{hh}) \right)^\top = (D_{t+1} W_{hh})^\top \dots (D_{T-1} W_{hh})^\top (D_T W_{hh})^\top$$

Since $D_k$ is diagonal, $D_k^\top = D_k$:
$$\mathbf{\left(\frac{\partial h_T}{\partial h_t}\right)^\top = \prod_{k=T}^{t+1} (W_{hh}^\top D_k) = (W_{hh}^\top D_{t+1})\dots(W_{hh}^\top D_{T-1})(W_{hh}^\top D_T)}$$

Notice the **anti-chronological reversal**: the backward gradient starts by multiplying with $W_{hh}^\top D_T$, and ends by multiplying with $W_{hh}^\top D_{t+1}$.
