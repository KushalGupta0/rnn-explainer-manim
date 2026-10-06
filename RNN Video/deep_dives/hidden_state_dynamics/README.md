# Deep Dive: Hidden State Dynamics & Latent State Spaces

[Back to Master Index](../../index.md) | [Back to Module 01: Why RNNs Exist](../../modules/01_why_rnns_exist.md)

---

## 1. Formal Definition of the Hidden State

In sequence modeling, the sequence observed up to time $t$ is denoted by the prefix history tuple:
$$\mathcal{X}_{\le t} = (x_1, x_2, \dots, x_t) \in \mathcal{X}^t$$

The goal of any sequential model is to compute a conditional representation or predictive distribution:
$$P(y_t \mid x_1, x_2, \dots, x_t)$$

If modeled naively, the domain of the conditioning context grows with $t$:
$$\operatorname{dim}(\mathcal{X}^t) = t \cdot d_x$$
which cannot be processed by a static parameterized operator $g_\theta: \mathbb{R}^k \to \mathbb{R}^{d_y}$ without fixing $t$.

### The Latent State Mapping
A **Recurrent Neural Network** introduces a latent state space $\mathcal{H} \subseteq \mathbb{R}^{d_h}$ and an iterative transition operator:
$$F: \mathbb{R}^{d_x} \times \mathbb{R}^{d_h} \to \mathbb{R}^{d_h}$$
such that:
$$h_t = F(x_t, h_{t-1})$$
initialized with an initial state $h_0 \in \mathbb{R}^{d_h}$ (often $h_0 = \mathbf{0}$ or a learnable parameter vector).

---

## 2. Mathematical Interpretation: Discrete Dynamical Systems

Mathematically, the RNN is a **non-autonomous, non-linear discrete-time dynamical system**:
$$h_t = f_t(h_{t-1})$$
where the time-dependence of the transition operator is driven externally by the driving sequence (input force) $x_t$:
$$f_t(h) \equiv F(x_t, h)$$

### Phase Space & Trajectories
* The latent space $\mathbb{R}^{d_h}$ acts as the **phase space** of the system.
* At $t = 0$, the system begins at point $h_0 \in \mathbb{R}^{d_h}$.
* Each incoming token/vector $x_t$ applies a non-linear vector field transformation that transports the state from $h_{t-1}$ to $h_t$.
* Over a sequence of length $T$, the hidden states trace a **discrete trajectory** in $\mathbb{R}^{d_h}$:
  $$\gamma = \{h_0, h_1, h_2, \dots, h_T\}$$

---

## 3. The Compression Bottleneck: Information Theory Perspective

The hidden state $h_t$ is a **fixed-dimensional summary** of an arbitrarily long sequence:
$$h_t = \Phi(\mathcal{X}_{\le t})$$
where $\Phi: \bigcup_{k=1}^\infty \mathcal{X}^k \to \mathbb{R}^{d_h}$.

Because $\operatorname{dim}(h_t) = d_h$ is constant regardless of $t$:
* For $t \cdot d_x \le d_h$, the map $\Phi$ can theoretically be injective (lossless embedding).
* For $t \cdot d_x \gg d_h$, by Sard's Theorem and dimensionality constraints, $\Phi$ is strictly **non-injective and lossy**.
* The hidden state acts as an **information bottleneck**. It cannot retain all historical tokens; it must discard irrelevant details and retain only coordinates in $\mathbb{R}^{d_h}$ sufficient to minimize the downstream task loss.

---

## 4. Why "Memory" is a Metaphor, Not Literal Storage

In popular computer science, "memory" connotes a lookup table, RAM, or a disk drive:
$$\operatorname{Read}(\operatorname{Address}) \to \operatorname{Value}$$
where values are stored at distinct addresses and retrieved without mutual interference.

In a vanilla RNN:
1. **Superposition**: All contextual information is projected into the *same* continuous coordinates of vector $h_t$. New information $x_t$ is added to and warped with existing information $h_{t-1}$ via non-linear vector addition:
   $$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$
2. **Entanglement / Interference**: You cannot update one aspect of the "memory" without linearly transforming and squashing *all* previous coordinates via $W_h$ and the activation derivative.
3. Therefore, $h_t$ is not a computer memory bank; it is a **dynamic summary state** or **continuous contextual representation vector**.

---

## 5. Visual / Manim Geometric Intuition

In a 3Blue1Brown-style visual geometric representation:
* Imagine the latent space $\mathbb{R}^{d_h}$ (visualized as 2D or 3D).
* $h_{t-1}$ is a point or arrow originating from the origin.
* Applying $W_h$ rotates, stretches, or compresses the space.
* $W_x x_t$ is a translation vector injected into this transformed space.
* The bias $b$ adds an additional fixed displacement.
* The activation function $\tanh(\cdot)$ warps the unbounded space into a compact hypercube $[-1, 1]^{d_h}$.
* The new point is $h_t$.
