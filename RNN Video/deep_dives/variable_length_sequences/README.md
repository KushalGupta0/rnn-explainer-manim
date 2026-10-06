# Deep Dive: Variable-Length Sequences & Weight Invariance

[Back to Master Index](../../index.md) | [Back to Module 01: Why RNNs Exist](../../modules/01_why_rnns_exist.md)

---

## 1. The Algebraic Rigidity of Standard Feed-Forward Neural Networks (MLPs)

A standard linear layer in a Feed-Forward Neural Network (FFNN / MLP) is defined as:
$$z = W x + b$$
where:
* $x \in \mathbb{R}^{D_{\text{in}}}$
* $W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}$
* $b \in \mathbb{R}^{D_{\text{out}}}$

Matrix multiplication requires strict dimensional compatibility:
$$\dim(\text{columns of } W) = \dim(x) = D_{\text{in}}$$

### The Concatenation Failure
If an input sequence consists of $T$ vectors of feature dimension $d_x$:
$$X = [x_1, x_2, \dots, x_T]$$
and we attempt to feed this into an MLP by flattening/concatenating:
$$x_{\text{flat}} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_T \end{bmatrix} \in \mathbb{R}^{T \cdot d_x}$$
then the weight matrix $W$ must have shape:
$$W \in \mathbb{R}^{D_{\text{out}} \times (T \cdot d_x)}$$

### Immediate Structural Flaws:
1. **Variable Length Incompatibility**: If an inference sample has $T' \ne T$ steps (e.g., a sentence of 5 words instead of 20 words), the linear operator $W x_{\text{flat}}$ is **undefined** because dimension mismatch errors occur:
   $$(D_{\text{out}} \times T \cdot d_x) \times (T' \cdot d_x \times 1) \implies \text{Dimension mismatch if } T' \ne T$$
2. **Padding is an Artificial Hack**: Padded zero vectors alter statistical distributions, waste computational memory ($O(T_{\max} \cdot d_x)$), and still cannot generalize to sequences where $T > T_{\max}$.

---

## 2. Lack of Temporal Translation Invariance

Consider language, audio, or financial data:
* Sequence A: `"The dog chased the cat"`
* Sequence B: `"Yesterday, the dog chased the cat"`

In Sequence A, the phrase `"the dog chased"` occupies positions $t=1, 2, 3$.
In Sequence B, the exact same semantic phrase occupies positions $t=2, 3, 4$.

In an MLP where inputs are concatenated into $x_{\text{flat}}$:
$$W = \begin{bmatrix} W_{:, 1:d_x} & W_{:, d_x+1:2d_x} & \dots & W_{:, (T-1)d_x+1:Td_x} \end{bmatrix}$$
The weights that interact with `"the dog"` at index 1 are *completely independent parameters* from the weights that interact with `"the dog"` at index 2!

The model has **zero temporal translation invariance**:
* It must independently learn the meaning of `"the dog"` at position 1, position 2, position 3, ..., position $T$.
* Parameter count scales linearly with sequence length:
  $$N_{\text{parameters}} = \mathcal{O}(T \cdot d_x \cdot D_{\text{out}})$$

---

## 3. Why Global Pooling (Averaging) Fails

One naive proposal to handle arbitrary $T$ with fixed parameters is:
$$x_{\text{pooled}} = \frac{1}{T} \sum_{t=1}^T x_t \in \mathbb{R}^{d_x}$$
followed by an MLP: $y = \operatorname{MLP}(x_{\text{pooled}})$.

### Why this is mathematically catastrophic for sequences:
* **Loss of Permutation Sensitivity**: Addition is commutative:
  $$\frac{1}{T}(x_1 + x_2 + x_3) = \frac{1}{T}(x_3 + x_1 + x_2)$$
* Sentences with opposite meanings yield identical representations:
  * `"dog bites man"` $\implies x_{\text{dog}} + x_{\text{bites}} + x_{\text{man}}$
  * `"man bites dog"` $\implies x_{\text{man}} + x_{\text{bites}} + x_{\text{dog}}$
* The temporal order (the *topology of the sequence*) is completely annihilated.

---

## 4. The Recurrent Solution: Shared Transition Operators

Recurrent models solve both problems simultaneously by parameterizing a single transition operator $f_\theta$ that is **applied recursively**:
$$h_t = f_\theta(x_t, h_{t-1})$$

1. **Arbitrary $T$**: The operator operates on $(x_t, h_{t-1}) \in \mathbb{R}^{d_x + d_h} \to \mathbb{R}^{d_h}$. It can be evaluated for 1 step, 10 steps, or $10^6$ steps without altering parameter dimensions.
2. **Temporal Translation Equivariance**: The exact same weight parameters are evaluated at every time step $t$. An extracted feature at $t=1$ undergoes the exact same mathematical transformation rules as one occurring at $t=50$.
3. **Order Preserved**: Because $h_t$ is updated sequentially, $(x_1 \to x_2) \ne (x_2 \to x_1)$.
