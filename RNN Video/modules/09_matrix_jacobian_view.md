# Module 09: The Matrix / Jacobian View — Spectra, Norms & Geometry

[Back to Master Index](../index.md) | [Back to Module 08](08_activation_functions.md) | [Next to Module 10](10_vanishing_gradients.md) | [Deep Dive: Jacobians](../deep_dives/jacobian_matrices/README.md) | [Deep Dive: Spectral Theory](../deep_dives/spectral_theory_and_norms/README.md)

---

## 1. Upgrading from Scalars to Vectors

In Module 07, we saw how repeated multiplication by a scalar $c$ produces exponential scaling $c^{T-t}$.
Now, we return to the real multidimensional world where $h_t \in \mathbb{R}^{d_h}$ and $W_{hh} \in \mathbb{R}^{d_h \times d_h}$.

The recurrent state update is:
$$a_t = W_{xh} x_t + W_{hh} h_{t-1} + b_h$$
$$h_t = \phi(a_t)$$

The single-step transition sensitivity is the **Jacobian matrix**:
$$\mathbf{J_t \equiv \frac{\partial h_t}{\partial h_{t-1}} = D_t W_{hh}}$$
where $D_t = \operatorname{diag}(\phi'(a_t)) \in \mathbb{R}^{d_h \times d_h}$ is the diagonal activation derivative matrix.

---

## 2. The Chain of Jacobians Across Time

Tracing sensitivity across $N = T - t$ timesteps yields the chained matrix product:
$$\mathbf{\frac{\partial h_T}{\partial h_t} = J_T J_{T-1} \cdots J_{t+1} = \prod_{k=t+1}^T (D_k W_{hh})}$$

```
Unit Sphere in ℝ^{d_h}                                   Squashed Hyper-ellipsoid
       ┌───┐             J_{t+1}            J_T                  ┌─┐
      │  ●  │       ─────────────►  ··· ─────────────►          │   │
       └───┘                                                     └─┘
     (Volume = 1)                                            (Volume → 0)
```

In the scalar case, the single number $c$ determines growth ($|c| > 1$) or decay ($|c| < 1$).
In the vector case, **what mathematical property of these matrices determines whether gradients explode or vanish?**

---

## 3. Four Distinct Spectral Quantities

To analyze the behavior of repeated matrix transformations, mathematicians use four distinct tools:

```
+-----------------------------------------------------------------------------------------+
|                                The Four Spectral Pillars                                |
+------------------------+-------------------------------+--------------------------------+
| Metric                 | Mathematical Definition       | Physical / Geometric Meaning   |
+------------------------+-------------------------------+--------------------------------+
| Eigenvalues            | Av = λv                       | Stretch factor along invariant |
| λ_i(A)                 |                               | direction v (if one exists).   |
+------------------------+-------------------------------+--------------------------------+
| Spectral Radius        | ρ(A) = max_i |λ_i(A)|          | Asymptotic growth rate of      |
| ρ(A)                   |                               | pure matrix power A^k as k→∞.  |
+------------------------+-------------------------------+--------------------------------+
| Singular Values        | σ_i(A) = √(λ_i(Aᵀ A))         | Lengths of the semi-axes of the|
| σ_i(A)                 |                               | transformed unit hypersphere.  |
+------------------------+-------------------------------+--------------------------------+
| Operator 2-Norm        | ||A||_2 = σ_max(A)            | Absolute maximum expansion     |
| ||A||_2                |                               | factor of any unit vector.     |
+------------------------+-------------------------------+--------------------------------+
```

---

## 4. Why the Popular "Eigenvalue Intuition" Is Incomplete

A ubiquitous claim in machine learning literature is:
> *"If the eigenvalues of $W_{hh}$ are less than 1, gradients vanish. If greater than 1, gradients explode."*

As research mentors, we must clarify why this statement is **pedagogically useful, but mathematically insufficient**:

### 1. The Non-Normal Transient Expansion Problem
The spectral radius $\rho(W_{hh})$ dictates asymptotic growth ($k \to \infty$). However, if $W_{hh}$ is **non-normal** ($W_{hh} W_{hh}^\top \ne W_{hh}^\top W_{hh}$), its eigenvectors are not orthogonal.
A non-normal matrix can have $\rho(W_{hh}) = 0.5 < 1$, yet its largest singular value could be $\sigma_{\max}(W_{hh}) = 50.0$.
In short sequences ($T = 10, 20$), gradients can **violently explode** due to transient shear before the asymptotic eigenvalue decay takes effect!

### 2. Time-Varying Modulation ($D_k$)
The product is **not** $W_{hh}^k$. It is an alternating product of different matrices:
$$(D_T W_{hh})(D_{T-1} W_{hh}) \cdots (D_{t+1} W_{hh})$$
Because each $D_k$ depends on the dynamic input $x_k$, the directions that get stretched and squashed change at every single timestep.

### 3. The True Upper Bound (Singular Values)
By the submultiplicative property of induced matrix norms:
$$\left\| \frac{\partial h_T}{\partial h_t} \right\|_2 \le \prod_{k=t+1}^T \|D_k W_{hh}\|_2 \le \prod_{k=t+1}^T \|D_k\|_2 \|W_{hh}\|_2$$
Since $|\tanh'(z)| \le 1.0$, $\|D_k\|_2 \le 1.0$.
Therefore:
$$\mathbf{\left\| \frac{\partial h_T}{\partial h_t} \right\|_2 \le \big(\sigma_{\max}(W_{hh})\big)^{T-t}}$$
* **Sufficient Condition for Vanishing Gradients**: If $\sigma_{\max}(W_{hh}) < 1.0$, the gradient is guaranteed to vanish exponentially, regardless of activations or inputs.
* **Necessary Condition for Exploding Gradients**: For gradients to explode, the operator norm $\|D_k W_{hh}\|_2$ must exceed 1.0 along the backpropagation trajectory.

---

## 5. Visual Semantics for Manim: SVD & Geometric Space Deformation

### How to Animate Matrix Jacobians in Manim:
* **The Unit Hypersphere (2D/3D Circle/Sphere)**:
  * In the latent space $\mathbb{R}^2$ or $\mathbb{R}^3$, draw a glowing wireframe unit circle representing all possible gradient directions with norm $\|v\| = 1$.
* **The SVD Step-by-Step Transformation**:
  1. **Rotation $V^\top$**: The circle rotates smoothly.
  2. **Singular Value Stretch $\Sigma$**: The circle distorts into an ellipse. The major axis stretches by $\sigma_1$, and the minor axis contracts by $\sigma_2$.
  3. **Rotation $U$**: The ellipse rotates to its final orientation.
  4. **Diagonal Modulation $D_t$**: The ellipse is compressed along the standard coordinate axes by the activation slopes $\phi'(a_t)_i$.
* **Repeated Application**:
  * Apply this transformation repeatedly across 10 unrolled steps.
  * In the contractive regime, show the ellipse collapsing into a microscopic, razor-thin line segment of virtually zero area.
  * In the expansive regime, show the ellipse expanding explosively until its boundaries shatter outside the camera view.

---

## 📝 Checkpoint 09 Exercises

### Level 1 — Conceptual
Explain the geometric difference between an eigenvalue and a singular value when transforming a sphere of vectors.

### Level 2 — Mathematical
Let $W_{hh} = \begin{bmatrix} 0 & 2 \\ 0 & 0 \end{bmatrix}$.
1. Compute the eigenvalues of $W_{hh}$ and its spectral radius $\rho(W_{hh})$.
2. Compute $W_{hh}^\top W_{hh}$ and find the singular values $\sigma_1, \sigma_2$ of $W_{hh}$.
3. What is the operator 2-norm $\|W_{hh}\|_2$?
4. What is $W_{hh}^2$? Explain the apparent paradox between the large singular value and the power $W_{hh}^2$.

### Level 3 — Debugging / Misconception
An engineer checks their RNN weights and notes: *"All eigenvalues of $W_{hh}$ have magnitude $0.8 < 1$, so it is mathematically impossible for gradients to explode."*
Show why this conclusion is flawed.

### Level 4 — Visualization (Manim Translation)
Describe how you would animate Singular Value Decomposition (SVD) applied to a gradient vector field to visually prove that repeated matrix multiplications squash the space along certain dimensions while expanding it along others.

### Level 5 — Advanced (Orthogonal Initialization)
Why does initializing $W_{hh}$ as an orthogonal matrix ($W_{hh}^\top W_{hh} = I$) ensure that all singular values are exactly $\sigma_i = 1.0$? How does this initialization stabilize gradient flow at the start of training?
