# Deep Dive: Dynamical Systems, Fixed Points & Non-Normal Stability

[Back to Master Index](../../index.md) | [Back to Module 17](../../modules/17_research_level_precision.md)

---

## 1. Autonomous Recurrent Dynamics & Fixed Points

Consider an unforced (autonomous) Recurrent Neural Network where inputs are zero ($x_t = \mathbf{0}$):
$$h_t = \phi(W_{hh} h_{t-1} + b_h) \equiv F(h_{t-1})$$

A **fixed point** $h^* \in \mathbb{R}^{d_h}$ satisfies:
$$h^* = F(h^*) = \phi(W_{hh} h^* + b_h)$$

### Linear Stability Analysis:
Let $h_t = h^* + \epsilon_t$ be a small perturbation around the fixed point $h^*$.
Expanding by Taylor series to first order:
$$\epsilon_t \approx J(h^*) \epsilon_{t-1}$$
where $J(h^*) = \left. \frac{\partial F}{\partial h} \right|_{h^*} = \operatorname{diag}\big(\phi'(W_{hh} h^* + b_h)\big) W_{hh}$.

By Lyapunov's Indirect Method:
* The fixed point $h^*$ is **asymptotically stable** (an attractor) if all eigenvalues of $J(h^*)$ lie strictly inside the complex unit disk:
  $$\max_i |\lambda_i(J(h^*))| < 1$$
* Any perturbation decays to zero: $\lim_{t \to \infty} \|\epsilon_t\| = 0$.
* **The BPTT Implication**: If the forward system converges to a stable attractor, the backward gradient $\frac{\partial L}{\partial h_t}$ through that attractor **must vanish exponentially**!
  Information storage via stable fixed points intrinsically annihilates backward gradients.

---

## 2. Pseudospectra & Non-Normal Matrix Dynamics

In classical linear algebra, eigenvalues characterize normal matrices ($A A^\top = A^\top A$).
However, recurrent weight matrices $W_{hh}$ learned by gradient descent are almost always **strongly non-normal**.

### The $\epsilon$-Pseudospectrum Definition:
For any $\epsilon > 0$, the $\epsilon$-pseudospectrum $\Lambda_\epsilon(A)$ is the set of complex numbers:
$$\Lambda_\epsilon(A) \equiv \{ z \in \mathbb{C} : \|(z I - A)^{-1}\|_2 \ge \epsilon^{-1} \}$$
Or equivalently:
$$\Lambda_\epsilon(A) = \{ z \in \mathbb{C} : z \text{ is an eigenvalue of } (A + E) \text{ for some } \|E\|_2 < \epsilon \}$$

### Why Pseudospectra Matter in Deep Learning:
* In non-normal matrices, the eigenvalues can all be tiny ($|\lambda| \le 0.1$), yet the pseudospectrum can bulge significantly into the right half-plane.
* This causes **transient growth**: the norm of matrix powers $\|A^k\|$ can grow to astronomical values before eventually decaying to zero as $k \to \infty$.
* This explains why training vanilla RNNs with $\rho(W) < 1$ can still experience sudden gradient spikes and instability during intermediate epochs.

---

## 3. Unitary & Orthogonal Recurrent Networks

To prevent both vanishing and exploding gradients from the recurrent weights, modern research introduced **Orthogonal and Unitary RNNs** (Arjovsky et al., 2016):

$$W_{hh} \in \mathcal{O}(d_h) \implies W_{hh}^\top W_{hh} = I$$

### Mathematical Properties:
1. **Norm Preservation**: For any vector $v \in \mathbb{R}^{d_h}$:
   $$\|W_{hh} v\|_2 = \|v\|_2$$
2. **Singular Values are Identically 1**:
   $$\sigma_1(W_{hh}) = \sigma_2(W_{hh}) = \dots = \sigma_{d_h}(W_{hh}) = 1.0$$
3. **No Matrix Contraction/Expansion**: The recurrent weight matrix cannot independently shrink or magnify gradients across time:
   $$\|W_{hh}^{T-t}\|_2 = 1.0 \quad \text{for all } T, t$$

When coupled with activations with slope 1 (such as modReLU), Orthogonal RNNs maintain gradient flow across thousands of steps without requiring gating mechanisms.
