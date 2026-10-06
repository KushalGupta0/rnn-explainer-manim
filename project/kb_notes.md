# Knowledge Base Mathematical Audit & Notation Harmonization

*A rigorous spot-check of theoretical formulations in the RNN Knowledge Base, documenting resolved ambiguities, index conventions, and mathematical proofs prior to visual scene implementation.*

---

## 1. Spot-Check: BPTT Adjoint Recurrence

### Knowledge Base Formulation:
In `RNN Video/spec/rnn_animation_spec.md` (Section 3) and `modules/06_bptt_deep_derivation.md`:
$$\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L_t}{\partial \mathbf{h}_t} + \mathbf{W}_{hh}^\top \mathbf{D}_{t+1} \frac{\partial L}{\partial \mathbf{h}_{t+1}}$$
where $\mathbf{D}_{t+1} = \operatorname{diag}\big(\phi'(\mathbf{a}_{t+1})\big) \in \mathbb{R}^{d_h \times d_h}$.

### Rigorous First-Principles Verification:
Let $L = \sum_{\tau=1}^T L_\tau$ be the total scalar sequence loss. By the multivariable chain rule, state $\mathbf{h}_t$ influences the loss through two distinct pathways:
1. **Direct immediate influence**: $\mathbf{h}_t \to \hat{\mathbf{y}}_t \to L_t$.
2. **Indirect temporal influence**: $\mathbf{h}_t \to \mathbf{a}_{t+1} \to \mathbf{h}_{t+1} \to \sum_{\tau=t+1}^T L_\tau$.

Under standard gradient conventions where gradients of a scalar with respect to a column vector are column vectors (denominator layout):
$$\nabla_{\mathbf{h}_t} L = \nabla_{\mathbf{h}_t} L_t + \left( \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)^\top \nabla_{\mathbf{h}_{t+1}} L$$

The forward step transition is:
$$\mathbf{h}_{t+1} = \phi(\mathbf{a}_{t+1}) = \phi(\mathbf{W}_{hh} \mathbf{h}_t + \mathbf{W}_{xh} \mathbf{x}_{t+1} + \mathbf{b}_h)$$

The forward Fréchet Jacobian of $\mathbf{h}_{t+1}$ with respect to $\mathbf{h}_t$ is:
$$\frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} = \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{a}_{t+1}} \frac{\partial \mathbf{a}_{t+1}}{\partial \mathbf{h}_t} = \mathbf{D}_{t+1} \mathbf{W}_{hh}$$

Taking the transpose:
$$\left( \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)^\top = (\mathbf{D}_{t+1} \mathbf{W}_{hh})^\top = \mathbf{W}_{hh}^\top \mathbf{D}_{t+1}^\top$$

Because $\mathbf{D}_{t+1}$ is a diagonal matrix containing the element-wise derivatives $\phi'(a_{t+1, i})$, it is symmetric: $\mathbf{D}_{t+1}^\top = \mathbf{D}_{t+1}$. Therefore:
$$\left( \frac{\partial \mathbf{h}_{t+1}}{\partial \mathbf{h}_t} \right)^\top = \mathbf{W}_{hh}^\top \mathbf{D}_{t+1}$$

Substituting back yields:
$$\frac{\partial L}{\partial \mathbf{h}_t} = \frac{\partial L_t}{\partial \mathbf{h}_t} + \mathbf{W}_{hh}^\top \mathbf{D}_{t+1} \frac{\partial L}{\partial \mathbf{h}_{t+1}}$$

*Verdict*: **Mathematically exact.** The knowledge base equation is completely sound.

---

## 2. Spot-Check: The Vanishing-Gradient Multi-Step Bound

### Knowledge Base Formulation:
In `spec/rnn_animation_spec.md` (Section 6) and `modules/10_vanishing_gradients.md`:
$$\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\|_2 \le \prod_{k=t+1}^T \|\mathbf{D}_k \mathbf{W}_{hh}\|_2 \le \big(\sigma_{\max}(\mathbf{W}_{hh})\big)^{T-t}$$

### Rigorous Verification:
By the chain rule across unrolled steps:
$$\frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} = \prod_{k=t+1}^T \frac{\partial \mathbf{h}_k}{\partial \mathbf{h}_{k-1}} = \prod_{k=t+1}^T (\mathbf{D}_k \mathbf{W}_{hh})$$

Applying the submultiplicative property of the matrix operator 2-norm ($\|\mathbf{A} \mathbf{B}\|_2 \le \|\mathbf{A}\|_2 \|\mathbf{B}\|_2$):
$$\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\|_2 \le \prod_{k=t+1}^T \|\mathbf{D}_k\|_2 \|\mathbf{W}_{hh}\|_2$$

For the standard hyperbolic tangent activation function $\phi(z) = \tanh(z)$:
$$\phi'(z) = 1 - \tanh^2(z)$$
Since $\tanh^2(z) \ge 0$ for all real $z$, $\phi'(z) \in (0, 1]$.
Because $\mathbf{D}_k$ is diagonal:
$$\|\mathbf{D}_k\|_2 = \max_{1 \le i \le d_h} |\phi'(a_{k, i})| \le 1.0$$

Therefore, for all timesteps $k$:
$$\|\mathbf{D}_k\|_2 \|\mathbf{W}_{hh}\|_2 \le 1.0 \cdot \|\mathbf{W}_{hh}\|_2 = \sigma_{\max}(\mathbf{W}_{hh})$$

Multiplying across all $T - t$ intermediate steps:
$$\left\| \frac{\partial \mathbf{h}_T}{\partial \mathbf{h}_t} \right\|_2 \le \big(\sigma_{\max}(\mathbf{W}_{hh})\big)^{T-t}$$

*Pedagogical Insight for Animation*: While $\sigma_{\max}(\mathbf{W}_{hh}) < 1$ guarantees vanishing gradients, in practical networks the activation derivative actively accelerates this decay. When pre-activations $|a_{k, i}| \ge 2.0$, the derivative drops below $0.07$. Thus, even if $\sigma_{\max}(\mathbf{W}_{hh}) \approx 1.0$, the compound contraction factor is typically $\approx 0.3 - 0.5$, causing gradients to vanish within 5 to 10 steps.

*Verdict*: **Mathematically exact.**

---

## 3. Notation Inconsistencies & Pedagogical Resolutions

1. **Parameter Weight Matrix Notation**:
   * *Inconsistency in Literature & Notes*: Some textbooks and early course sections use $U$ for input weights and $W$ for hidden weights ($h_t = \tanh(U x_t + W h_{t-1})$), while others use double-subscript notation ($W_{xh}, W_{hh}$).
   * *Resolution for Video*: Standardize strictly on **double-subscript notation** ($\mathbf{W}_{xh}, \mathbf{W}_{hh}, \mathbf{W}_{hy}$). This immediately informs viewers of the source and target spaces without requiring mental lookups.

2. **Pre-activation Adjoint Time Indexing**:
   * *Potential Confusion*: In adjoint equations, $\delta_t$ is defined as $\frac{\partial L}{\partial \mathbf{a}_t} = \mathbf{D}_t \frac{\partial L}{\partial \mathbf{h}_t}$. Notice the index on $\mathbf{D}$ is $t$, matching $\mathbf{a}_t$. However, in the temporal recurrence $\frac{\partial L}{\partial \mathbf{h}_t} = \dots + \mathbf{W}_{hh}^\top \mathbf{D}_{t+1} \frac{\partial L}{\partial \mathbf{h}_{t+1}}$, the index is $t+1$.
   * *Resolution for Video*: Explicitly highlight in Scene 05 that the derivative $\mathbf{D}_{t+1}$ belongs to the receiving forward node at step $t+1$, before its signal transfers across $\mathbf{W}_{hh}^\top$ back into step $t$.

3. **Loss Notation ($L$ vs $L_t$)**:
   * *Resolution for Video*: Consistently render $L_t$ (with subscript) for local scalar prediction errors at individual timesteps, and bold scalar $L$ (without subscript) for the sequence total $\sum_{t=1}^T L_t$.
