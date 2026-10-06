# RNN Animation Knowledge Specification: The Theoretical Source of Truth

[Back to Master Index](../index.md)

> **Target Medium**: 3Blue1Brown-style Mathematical & Geometric Animations in Manim  
> **Purpose**: Definitive theoretical, mathematical, computational, and geometric specification for designing an educational video on Recurrent Neural Networks, Backpropagation Through Time, and Gradient Stability.

---

## 1. Core RNN Governing Equations

Let $x_t \in \mathbb{R}^{d_x}$ be the input at timestep $t$, $h_t \in \mathbb{R}^{d_h}$ the latent hidden state, and $y_t \in \mathbb{R}^{d_y}$ the output.

### The Canonical Elman Formulation:
$$\begin{aligned}
a_t &= W_{xh} x_t + W_{hh} h_{t-1} + b_h \\
h_t &= \phi(a_t) \\
o_t &= W_{hy} h_t + b_y \\
\hat{y}_t &= g(o_t)
\end{aligned}$$

### Compact Concatenation Form:
$$a_t = \begin{bmatrix} W_{xh} & W_{hh} \end{bmatrix} \begin{bmatrix} x_t \\ h_{t-1} \end{bmatrix} + b_h$$

### Stationarity Principle (Parameter Sharing):
$$\{W_{xh}, W_{hh}, b_h, W_{hy}, b_y\}_{t=1} = \{W_{xh}, W_{hh}, b_h, W_{hy}, b_y\}_{t=2} = \dots = \{W_{xh}, W_{hh}, b_h, W_{hy}, b_y\}_{t=T}$$

---

## 2. Complete Forward-Pass Derivation & State Caching

For an input sequence $\mathcal{X} = (x_1, x_2, \dots, x_T)$ with initial state $h_0 = \mathbf{0}$:
$$\begin{aligned}
h_1 &= \phi(W_{xh} x_1 + W_{hh} h_0 + b_h) \\
h_2 &= \phi\big(W_{xh} x_2 + W_{hh} \phi(W_{xh} x_1 + W_{hh} h_0 + b_h) + b_h\big) \\
&\;\;\vdots \\
h_T &= \phi\Big(W_{xh} x_T + W_{hh} \phi\big(\dots \phi(W_{xh} x_1 + W_{hh} h_0 + b_h)\dots\big) + b_h\Big)
\end{aligned}$$

Every state vector $\{h_0, h_1, \dots, h_T\}$ and pre-activation $\{a_1, \dots, a_T\}$ must be stored in the forward activation cache to evaluate local derivatives during backpropagation.

---

## 3. Complete BPTT Derivation

Let the total scalar objective be:
$$L = \sum_{t=1}^T L_t(\hat{y}_t, y_t^*)$$

### Backward Hidden State Recurrence:
$$\mathbf{\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + W_{hh}^\top D_{t+1} \frac{\partial L}{\partial h_{t+1}}}$$
where:
$$D_t = \operatorname{diag}\big(\phi'(a_t)\big) \in \mathbb{R}^{d_h \times d_h}$$
Boundary condition: $\frac{\partial L}{\partial h_T} = \frac{\partial L_T}{\partial h_T} = W_{hy}^\top \frac{\partial L_T}{\partial o_T}$.

### Pre-activation Adjoint:
$$\delta_t \equiv \frac{\partial L}{\partial a_t} = D_t \frac{\partial L}{\partial h_t}$$

### Cumulative Parameter Gradients:
$$\begin{aligned}
\mathbf{\frac{\partial L}{\partial W_{hh}}} &= \sum_{t=1}^T \delta_t h_{t-1}^\top \in \mathbb{R}^{d_h \times d_h} \\
\mathbf{\frac{\partial L}{\partial W_{xh}}} &= \sum_{t=1}^T \delta_t x_t^\top \in \mathbb{R}^{d_h \times d_x} \\
\mathbf{\frac{\partial L}{\partial b_h}} &= \sum_{t=1}^T \delta_t \in \mathbb{R}^{d_h} \\
\mathbf{\frac{\partial L}{\partial W_{hy}}} &= \sum_{t=1}^T \frac{\partial L_t}{\partial o_t} h_t^\top \in \mathbb{R}^{d_y \times d_h}
\end{aligned}$$

---

## 4. Multi-Step Gradient Propagation Equations

The sensitivity of future hidden state $h_T$ to earlier hidden state $h_t$ is given by the product of intermediate Jacobians:
$$\mathbf{\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=t+1}^T (D_k W_{hh}) = (D_T W_{hh})(D_{T-1} W_{hh})\cdots(D_{t+1} W_{hh})}$$

Transposing for backward loss gradient propagation:
$$\mathbf{\frac{\partial L_T}{\partial h_t} = \left[ \prod_{k=T}^{t+1} (W_{hh}^\top D_k) \right] \frac{\partial L_T}{\partial h_T}}$$

---

## 5. Scalar Vanishing-Gradient Derivation

In a 1D scalar RNN ($h_t = \phi(w h_{t-1} + u x_t + b)$):
$$\frac{\partial h_T}{\partial h_t} = w^{T-t} \prod_{k=t+1}^T \phi'(a_k)$$
Assuming an effective constant factor $c = w \cdot \overline{\phi'}$:
$$\mathbf{\frac{\partial h_T}{\partial h_t} \approx c^{T-t}}$$

* If $|c| < 1.0$: Gradients decay exponentially to zero ($\lim_{\Delta t \to \infty} c^{\Delta t} = 0$).
* If $|c| > 1.0$: Gradients expand exponentially to infinity ($\lim_{\Delta t \to \infty} |c|^{\Delta t} = \infty$).

---

## 6. Matrix / Jacobian Spectral Interpretation

In multidimensional spaces ($h_t \in \mathbb{R}^{d_h}$):
$$\left\| \frac{\partial h_T}{\partial h_t} \right\|_2 \le \prod_{k=t+1}^T \|D_k W_{hh}\|_2 \le \prod_{k=t+1}^T \|D_k\|_2 \|W_{hh}\|_2 \le \big(\sigma_{\max}(W_{hh})\big)^{T-t}$$

### Four Spectral Measures:
1. **Eigenvalues $\lambda_i(W_{hh})$**: Scaling along invariant 1D directions ($W v = \lambda v$).
2. **Spectral Radius $\rho(W_{hh}) = \max |\lambda_i|$**: Asymptotic growth rate $\lim_{k \to \infty} \|W^k\|^{1/k}$.
3. **Singular Values $\sigma_i(W_{hh}) = \sqrt{\lambda_i(W^\top W)}$**: Semi-axes of transformed unit hypersphere.
4. **Operator 2-Norm $\|W_{hh}\|_2 = \sigma_{\max}(W_{hh})$**: Worst-case single-step vector amplification factor.

---

## 7. Exploding Gradients & Norm Clipping

### Origin:
When $\|W_{hh}^\top D_k\|_2 \ge \rho > 1$ along the backward path, gradient norm expands as $\rho^{T-t}$, causing arithmetic overflow (`Inf`/`NaN`) and destruction of learned weights.

### The Solution: Global Gradient Norm Clipping (Pascanu et al., 2013)
$$\mathbf{g \leftarrow \begin{cases} g & \text{if } \|g\|_2 \le \tau \\ g \cdot \frac{\tau}{\|g\|_2} & \text{if } \|g\|_2 > \tau \end{cases}}$$

* **Direction Preservation**: $\frac{g_{\text{clipped}}}{\|g_{\text{clipped}}\|} = \frac{g}{\|g\|}$ is strictly maintained.
* **Bounded Step**: $\|\Delta \Theta\|_2 \le \eta \tau$.
* Prevents catapulting across steep loss cliffs while preserving descent orientation.

---

## 8. Long-Term Dependency Optimization Barrier

* **Representational Power**: RNNs are Turing complete (Siegelmann & Sontag, 1995).
* **Optimization Barrier**: Storing information stably requires attractive fixed points (contractive dynamics in forward pass). Contractive forward dynamics force contractive backward Jacobians, guaranteeing that $\frac{\partial L_T}{\partial h_1} \to 0$ (Bengio et al., 1994).
* The model defaults to short-term Markovian predictions (recency bias).

---

## 9. Architectural Evolution: LSTM & GRU

### The Additive Linear Gradient Highway:
LSTMs replace non-linear recurrent squashing with an additive cell state:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
$$\frac{\partial C_T}{\partial C_t} \approx \prod_{k=t+1}^T \operatorname{diag}(f_k)$$
If forget gates remain open ($f_k \approx 1$), gradients flow through time with **zero decay and zero matrix multiplication** (The Constant Error Carousel).

---

## 10. Key Diagrams for the Animation

1. **The Dual Representation**: Cyclical single-machine loop snapping open into an unrolled horizontal DAG.
2. **The Vector Sum & Squashing**: $W_{hh} h_{t-1} + W_{xh} x_t$ parallelogram addition in $\mathbb{R}^2$, bounded into a $[-1, 1]^2$ box by $\tanh$.
3. **The SVD Ellipsoid**: Unit sphere rotated, stretched by $\sigma_1, \sigma_2$, and compressed along axes by $\phi'(a_t)$.
4. **The Backward Watershed**: Error flames at $L_t$ sending orange gradient fluid flowing right-to-left, bifurcating at each state nexus.
5. **The Loss Cliff**: 3D non-convex terrain dropping vertically, showing unclipped particles flying off-screen vs clipped particles descending smoothly.
6. **The Parallel Conduits (LSTM)**: A lower turbulent channel ($h_t$) and an upper frictionless glass pipeline ($C_t$).

---

## 11. Recommended Animation Sequence (10-Minute Video)

```
00:00 - 01:30 | Act I: The Sequential Paradox (Why MLPs fail; variable length & translation)
01:30 - 03:15 | Act II: The Recurrent Machine (The loop, dimensions, and parameter sharing)
03:15 - 04:30 | Act III: Unspooling Time (Equivalence of recurrent loop and horizontal DAG)
04:30 - 05:45 | Act IV: Forward Numerical Walkthrough (Live state caching & loss formation)
05:45 - 07:15 | Act V: Backpropagation Through Time (Chain rule, backward flow, Jacobians)
07:15 - 08:45 | Act VI: The Geometry of Vanishing & Exploding Gradients (SVD & saturation trap)
08:45 - 10:00 | Act VII: Architectural Solutions (Norm clipping & the LSTM linear highway)
```

---

## 12. Important Notation & Conventions

| Symbol | Meaning | Standard Dimension |
| :--- | :--- | :--- |
| $x_t$ | Input vector at time $t$ | $\mathbb{R}^{d_x \times 1}$ |
| $h_t$ | Hidden state vector at time $t$ | $\mathbb{R}^{d_h \times 1}$ |
| $a_t$ | Hidden pre-activation vector | $\mathbb{R}^{d_h \times 1}$ |
| $y_t$ / $\hat{y}_t$ | Model prediction vector | $\mathbb{R}^{d_y \times 1}$ |
| $W_{hh}$ | Recurrent hidden-to-hidden weight | $\mathbb{R}^{d_h \times d_h}$ |
| $W_{xh}$ | Input-to-hidden weight | $\mathbb{R}^{d_h \times d_x}$ |
| $W_{hy}$ | Hidden-to-output weight | $\mathbb{R}^{d_y \times d_h}$ |
| $D_t$ | Diagonal activation derivative matrix $\operatorname{diag}(\phi'(a_t))$ | $\mathbb{R}^{d_h \times d_h}$ |
| $J_t$ | Single-step transition Jacobian $\frac{\partial h_t}{\partial h_{t-1}} = D_t W_{hh}$ | $\mathbb{R}^{d_h \times d_h}$ |
| $\delta_t$ | Pre-activation error adjoint $\frac{\partial L}{\partial a_t} = D_t \frac{\partial L}{\partial h_t}$ | $\mathbb{R}^{d_h \times 1}$ |
| $\sigma_{\max}(A)$ | Largest singular value / Operator 2-norm | $\mathbb{R}_{\ge 0}$ |
| $\rho(A)$ | Spectral radius $\max_i |\lambda_i(A)|$ | $\mathbb{R}_{\ge 0}$ |

---

## 13. Verification Checklist for Manim Implementation

Before rendering final animation frames in Manim, verify each mathematical property:

- [ ] **Dimension Integrity**: Are input vectors, state vectors, and output rays drawn with distinct, consistent visual styles?
- [ ] **Parameter Sharing Indicator**: Is there a persistent visual anchor proving that $W_{hh}$ at $t=1$ is identical to $W_{hh}$ at $t=10$?
- [ ] **Unrolled Equivalence**: Does the transition from the single recurrent loop to the unrolled DAG preserve node connections without adding spurious parameters?
- [ ] **Forward-Backward Arrow Duality**: Do forward activations flow strictly chronologically ($1 \to T$, cool colors), and backward gradients flow strictly anti-chronologically ($T \to 1$, warm colors)?
- [ ] **Gradient Accumulation at Nexus**: Does each hidden state node $h_t$ receive two backward streams (vertical from $L_t$ and horizontal from $h_{t+1}$)?
- [ ] **SVD Transformation Sequence**: Does the matrix transformation animate in correct geometric order: rotate $V^\top \to$ stretch $\Sigma \to$ rotate $U \to$ squash $D_t$?
- [ ] **Saturation Visual**: Does the slope indicator on the $\tanh$ curve visibly flatten to horizontal as pre-activation values $|a_t|$ exceed $2.5$?
- [ ] **Clipping Geometry**: Does the norm clipping animation strictly preserve the angular orientation of the gradient vector while scaling its length to the circle boundary $\tau$?
- [ ] **LSTM Additive Highway**: Is the cell state $C_t$ visually isolated from matrix multiplications, showing an unhindered linear pipeline?
