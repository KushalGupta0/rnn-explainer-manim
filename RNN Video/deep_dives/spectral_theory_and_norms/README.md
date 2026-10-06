# Deep Dive: Spectral Theory, Matrix Norms & Singular Values

[Back to Master Index](../../index.md) | [Back to Module 09](../../modules/09_matrix_jacobian_view.md)

---

## 1. Matrix Norms & Operator Bounds

To quantify whether a vector expands or contracts when transformed by a matrix $A \in \mathbb{R}^{n \times n}$, we use the **induced matrix $2$-norm (operator norm)**:
$$\|A\|_2 \equiv \sup_{x \ne 0} \frac{\|A x\|_2}{\|x\|_2} = \max_{\|x\|_2 = 1} \|A x\|_2$$

### Submultiplicative Property:
For any sequence of matrices $A_1, A_2, \dots, A_m$:
$$\|A_1 A_2 \cdots A_m\|_2 \le \|A_1\|_2 \|A_2\|_2 \cdots \|A_m\|_2 = \prod_{i=1}^m \|A_i\|_2$$

If each individual factor has an operator norm strictly bounded by $\gamma < 1$:
$$\|A_1 A_2 \cdots A_m\|_2 \le \gamma^m \xrightarrow{m \to \infty} 0 \quad \text{(Guaranteed Vanishing)}$$

---

## 2. Singular Values vs. Eigenvalues

Two spectral concepts are frequently conflated in informal machine learning literature: **Eigenvalues** and **Singular Values**.

```
EIGENVALUE DECOMPOSITION (A = V Λ V⁻¹):
* Applies to square matrices.
* V may be non-orthogonal.
* Represents scaling along invariant 1D subspaces (eigenvectors).

SINGULAR VALUE DECOMPOSITION (A = U Σ Vᵀ):
* Applies to ANY matrix.
* U and V are strictly orthonormal (pure rotations/reflections).
* Σ = diag(σ₁, ..., σ_n) with σ₁ ≥ σ₂ ≥ ... ≥ σ_n ≥ 0.
* Geometrically maps a unit sphere into a hyper-ellipsoid.
```

### The Exact Mathematical Relationship:
The induced 2-norm of a matrix is identical to its **largest singular value**:
$$\mathbf{\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^\top A)}}$$

---

## 3. The Spectral Radius $\rho(A)$ & Gelfand's Formula

The **spectral radius** of a square matrix $A$ is the maximum absolute value of its eigenvalues:
$$\rho(A) \equiv \max_{i} |\lambda_i(A)|$$

### The Universal Inequality:
For any induced matrix norm:
$$\mathbf{\rho(A) \le \|A\|}$$

### Gelfand's Formula (Asymptotic Equivalence):
For any matrix $A$:
$$\lim_{k \to \infty} \|A^k\|^{1/k} = \rho(A)$$
This proves that the long-term asymptotic growth rate of pure powers $A^k$ is governed by $\rho(A)$.
* If $\rho(A) < 1$, then $\lim_{k \to \infty} A^k = 0$.
* If $\rho(A) > 1$, then $\lim_{k \to \infty} \|A^k\| = \infty$.

---

## 4. The Pitfall: Non-Normal Matrices & Transient Amplification

> [!CAUTION]
> **Why "$\rho(W_{hh}) < 1$ prevents exploding gradients" is false:**
> Gelfand's formula holds only as $k \to \infty$. For finite sequences ($T = 20, 50$), the behavior can be completely dominated by **transient amplification** if $W_{hh}$ is non-normal.

A matrix is **normal** if $A A^\top = A^\top A$ (e.g. symmetric or orthogonal matrices).
For normal matrices, $\rho(A) = \|A\|_2$.

For **non-normal matrices**, the eigenvectors are not orthogonal.
Consider the Jordan block / shear matrix:
$$A = \begin{bmatrix} 0.5 & 100 \\ 0 & 0.5 \end{bmatrix}$$
The eigenvalues are $\lambda_1 = 0.5, \lambda_2 = 0.5$.
Hence, the spectral radius is $\rho(A) = 0.5 < 1$.
Yet, let us compute $A^2$:
$$A^2 = \begin{bmatrix} 0.25 & 100 \\ 0 & 0.25 \end{bmatrix}$$
For vector $x = [0, 1]^\top$, $A x = [100, 0.5]^\top$, an amplification of **100 times**!
Before the asymptotic decay $0.5^k$ dominates (which requires hundreds of steps), the gradient can explode by orders of magnitude in the transient window.

---

## 5. Application to the RNN Jacobian Product

In an RNN, the gradient backpropagates via:
$$J_{\text{chain}} = \prod_{k=t+1}^T (D_k W_{hh})$$

### Why Simple Matrix Powers Do Not Apply:
1. **Time-Varying Diagonal Multipliers**: The matrices $D_k = \operatorname{diag}(\phi'(a_k))$ change at every step $k$ because the forward activations $a_k$ vary dynamically with input $x_k$.
2. **Non-Commutativity**: In general, $D_i W_{hh} D_j W_{hh} \ne (D_i D_j) W_{hh}^2$.
3. **Sufficient Condition for Vanishing Gradients (Pascanu et al., 2013)**:
   By the submultiplicative property:
   $$\|J_{\text{chain}}\|_2 \le \prod_{k=t+1}^T \|D_k W_{hh}\|_2 \le \prod_{k=t+1}^T \|D_k\|_2 \|W_{hh}\|_2$$
   Since $\tanh'(z) \le 1.0$, $\|D_k\|_2 \le 1.0$.
   Therefore:
   $$\mathbf{\|J_{\text{chain}}\|_2 \le \big(\sigma_{\max}(W_{hh})\big)^{T-t}}$$
   **Theorem**: If the largest singular value of the recurrent weight matrix satisfies $\sigma_{\max}(W_{hh}) < 1$, then the backpropagated gradient norm decays to zero exponentially fast with sequence distance $(T - t)$, guaranteeing vanishing gradients regardless of the input data!
