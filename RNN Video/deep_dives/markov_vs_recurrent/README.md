# Deep Dive: Markovian Models vs. Latent Autoregressive Recurrence

[Back to Master Index](../../index.md) | [Back to Module 01: Why RNNs Exist](../../modules/01_why_rnns_exist.md)

---

## 1. The Classical Probabilistic Formulation of Sequences

By the chain rule of probability, any joint distribution over a sequence $X = (x_1, x_2, \dots, x_T)$ can be factored without loss of generality as:
$$P(x_1, x_2, \dots, x_T) = \prod_{t=1}^T P(x_t \mid x_1, x_2, \dots, x_{t-1}) = \prod_{t=1}^T P(x_t \mid x_{<t})$$

The conditioning set $x_{<t} = (x_1, \dots, x_{t-1})$ expands with every timestep.

---

## 2. The $k$-th Order Markov Approximation

To make sequence modeling tractable in classical statistical theory, the **$k$-th order Markov assumption** truncates the conditioning history to the immediate past $k$ tokens:
$$P(x_t \mid x_{<t}) \approx P(x_t \mid x_{t-1}, x_{t-2}, \dots, x_{t-k})$$

### The Fatal Limitations of Markov Models:
1. **Curse of Dimensionality / State Explosion**:
   If inputs are discrete tokens from vocabulary $\mathcal{V}$ of size $|\mathcal{V}| = V$, the number of transition probabilities to estimate or store is:
   $$\mathcal{O}(V^{k+1})$$
   For language models ($V \approx 50{,}000$), even $k=3$ (a 4-gram model) requires $50{,}000^4 \approx 6.25 \times 10^{18}$ parameters.
2. **Strict Horizon Blindness**:
   Any dependency that spans $k+1$ tokens is strictly inaccessible to the model.
   * Example: `"The code written in Python by several engineers who met in Berlin [ ... 30 words ... ] was executed."`
   * To predict `"was"` vs `"were"`, the model must maintain the singular subject `"code"` across 30+ tokens. A Markov model with $k < 30$ is structurally incapable of seeing the subject.

---

## 3. The Recurrent Alternative: Latent State Compression

Instead of truncating the history $x_{<t}$, an RNN maintains a **continuous latent representation vector** $h_{t-1} \in \mathbb{R}^{d_h}$:
$$P(x_t \mid x_{<t}) \approx P(x_t \mid h_{t-1})$$
where:
$$h_{t-1} = f(x_{t-1}, h_{t-2})$$

### Why Recurrence Breaks the Markov Ceiling:
* **Theoretically Infinite Memory Depth**: Because $h_{t-1}$ is computed recursively from $h_{t-2}$, which was computed from $h_{t-3}$, down to $h_0$, $h_{t-1}$ contains recursive functional dependencies on *all* previous inputs:
  $$h_{t-1} = f(x_{t-1}, f(x_{t-2}, f(x_{t-3}, \dots f(x_1, h_0)\dots)))$$
* **Constant Parameter Complexity**: The parameter count of $f$ is completely independent of the sequence length $T$ and history depth $k$. The number of parameters depends solely on the vector dimensions:
  $$\Theta = \{W_x, W_h, b\} \implies \mathcal{O}(d_h^2 + d_h d_x)$$

---

## 4. The Distinction Between Theoretical Capacity and Trainability

> [!IMPORTANT]
> **Theoretical Expressivity vs. Numerical Trainability:**
> Mathematically, an RNN can represent arbitrary Turing machines (Siegelmann & Sontag, 1995) and retain unbounded temporal dependencies in $h_t$.
> However, as we will derive in Modules 05 through 10, **training** such dependencies with gradient descent is violently constrained by the exponential decay or explosion of backpropagated gradients through the chain rule.
