# Module 16: Architectural Solutions — From Vanilla RNN to LSTM and GRU

[Back to Master Index](../index.md) | [Back to Module 15](15_common_misconceptions.md) | [Next to Module 17](17_research_level_precision.md)

---

## 1. The Causal Lineage of Gated Recurrent Architectures

Modern gated recurrent architectures did not emerge by coincidence. They are the direct mathematical response to the vanishing gradient failure of the vanilla RNN:

```
Vanilla RNN Recurrence:
h_t = tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
         │
         ▼
Multiplicative Jacobian Chain in BPTT:
∂h_T / ∂h_t = ∏_{k=t+1}^T (D_k W_{hh})
         │
         ▼
Exponential Decay / Instability: ||∂L / ∂h_t|| → 0
         │
         ▼
ARCHITECTURAL HYPOTHESIS (Hochreiter & Schmidhuber, 1997):
Replace the repeated non-linear matrix multiplication with an ADDITIVE linear state highway:
C_t = C_{t-1} + ΔC_t  ⟹  ∂C_t / ∂C_{t-1} ≈ I  (Identity Gradient Flow)
         │
         ▼
Long Short-Term Memory (LSTM) & Gated Recurrent Unit (GRU)
```

---

## 2. Long Short-Term Memory (LSTM) Cell Equations

An LSTM separates memory into two distinct vectors:
1. **$C_t \in \mathbb{R}^{d_h}$ (The Cell State)**: The protected, linear internal conveyor belt.
2. **$h_t \in \mathbb{R}^{d_h}$ (The Hidden State)**: The filtered, task-visible working memory.

Information flow is controlled by three non-linear gates $\in (0, 1)^{d_h}$:

```
                              C_{t-1} ──────[ × ]─────────────────────[ + ]──────► C_t
                                              ▲                         ▲
                                              │ f_t (Forget)            │ i_t ⊙ C̃_t (Input)
                                              │                         │
                                         ┌────┴────┐               ┌────┴────┐
                                         │  σ(·)   │               │   *     │
                                         └────┬────┘               └────┬────┘
                                              ▲                         ▲
                                              │                         │
h_{t-1}, x_t ─────────────────────────────────┴─────────────────────────┴────────► h_t
```

### The Six Governing Equations:
Let $[h_{t-1}, x_t] \in \mathbb{R}^{d_h + d_x}$ denote concatenation:

1. **Forget Gate Vector $f_t$** (Decides what fraction of old memory to erase):
   $$f_t = \sigma\big(W_f [h_{t-1}, x_t] + b_f\big) \in (0, 1)^{d_h}$$
2. **Input Gate Vector $i_t$** (Decides which coordinates of new information to write):
   $$i_t = \sigma\big(W_i [h_{t-1}, x_t] + b_i\big) \in (0, 1)^{d_h}$$
3. **Candidate Memory Vector $\tilde{C}_t$** (The new prospective information):
   $$\tilde{C}_t = \tanh\big(W_c [h_{t-1}, x_t] + b_c\big) \in (-1, 1)^{d_h}$$
4. **Cell State Update $C_t$** (The additive conveyor belt):
   $$\mathbf{C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t}$$
5. **Output Gate Vector $o_t$** (Decides what parts of the cell state to reveal to the task):
   $$o_t = \sigma\big(W_o [h_{t-1}, x_t] + b_o\big) \in (0, 1)^{d_h}$$
6. **Hidden State Readout $h_t$**:
   $$\mathbf{h_t = o_t \odot \tanh(C_t)}$$

---

## 3. Why the Cell State Fixes Vanishing Gradients: The Additive Highway

Examine the partial derivative of the current cell state $C_t$ with respect to the previous cell state $C_{t-1}$:
$$\frac{\partial C_t}{\partial C_{t-1}} = \operatorname{diag}(f_t) + \underbrace{\frac{\partial f_t}{\partial C_{t-1}} \odot C_{t-1} + \frac{\partial (i_t \odot \tilde{C}_t)}{\partial C_{t-1}}}_{\text{Indirect terms through } h_{t-1}}$$

Along the direct linear shortcut, the transition Jacobian is simply:
$$\mathbf{\frac{\partial C_t}{\partial C_{t-1}} \approx \operatorname{diag}(f_t)}$$

### The Constant Error Carousel (CEC):
Across $T - t$ timesteps, the gradient backpropagates via:
$$\frac{\partial C_T}{\partial C_t} \approx \prod_{k=t+1}^T \operatorname{diag}(f_k)$$
* **Notice the fundamental difference from vanilla RNNs**:
  There is **no recurrent weight matrix $W_{hh}$** in this product!
  There is **no activation derivative $\phi'(a_k)$** in this product!
* If the forget gate is learned to be open ($f_k \approx 1.0$), then:
  $$\frac{\partial C_T}{\partial C_t} \approx \prod_{k=t+1}^T I = I$$
* **The gradient flows across 100 timesteps completely unattenuated**!
* The network can choose to remember a piece of information indefinitely by keeping $f_k \approx 1$ and $i_k \approx 0$, protecting the gradient pathway from vanishing.

---

## 4. The Gated Recurrent Unit (GRU)

Introduced by Cho et al. (2014), the GRU simplifies the LSTM by merging the cell state and hidden state, and coupling the forget and input gates into a single **update gate** $z_t$:

```
               (1 - z_t)
h_{t-1} ───────[ × ]─────────────────────[ + ]──────► h_t
                 ▲                         ▲
                 │                         │ z_t
                 │ update                  │
            ┌────┴────┐               ┌────┴────┐
            │  σ(·)   │               │  tanh   │  <-- Candidate h̃_t
            └────┬────┘               └────┬────┘      (reset by r_t)
                 ▲                         ▲
                 │                         │
              h_{t-1}, x_t              r_t ⊙ h_{t-1}, x_t
```

### The GRU Governing Equations:
1. **Reset Gate $r_t$**:
   $$r_t = \sigma\big(W_r [h_{t-1}, x_t] + b_r\big)$$
2. **Update Gate $z_t$**:
   $$z_t = \sigma\big(W_z [h_{t-1}, x_t] + b_z\big)$$
3. **Candidate State $\tilde{h}_t$**:
   $$\tilde{h}_t = \tanh\big(W [r_t \odot h_{t-1}, x_t] + b\big)$$
4. **Interpolated State Update $h_t$**:
   $$\mathbf{h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t}$$

### Key Property of GRU:
* If $z_t = 0$: $h_t = h_{t-1}$ (State is completely copied forward; gradient flows perfectly: $\frac{\partial h_t}{\partial h_{t-1}} = I$).
* If $z_t = 1$: $h_t = \tilde{h}_t$ (Old state is completely wiped and overwritten by current input).
* Has $25\%$ fewer parameters than LSTM ($\approx 3 d_h^2$ vs $4 d_h^2$), making it faster to train on smaller datasets.

---

## 5. Architectural Comparison Matrix

| Property | Vanilla RNN (Elman) | LSTM (Hochreiter & Schmidhuber) | GRU (Cho et al.) |
| :--- | :--- | :--- | :--- |
| **Number of Weight Matrices** | 1 hidden transition ($W_{hh}$) | 4 gate transformations ($W_f, W_i, W_c, W_o$) | 3 gate transformations ($W_r, W_z, W$) |
| **State Vectors** | 1 ($h_t$) | 2 ($C_t$ and $h_t$) | 1 ($h_t$) |
| **Recurrence Mechanism** | Non-linear affine squashing | **Additive linear conveyor belt** | **Linear convex interpolation** |
| **Gradient Highway** | No (decays as $\prod D_k W_{hh}$) | **Yes** ($\prod f_k$, can equal $I$) | **Yes** ($\prod (1 - z_k)$, can equal $I$) |
| **Susceptibility to Vanishing** | Extreme | Very low (only if $f \to 0$) | Very low (only if $z \to 1$) |
| **Parameter Complexity** | $\mathcal{O}(d_h^2 + d_h d_x)$ | $\mathcal{O}(4(d_h^2 + d_h d_x))$ | $\mathcal{O}(3(d_h^2 + d_h d_x))$ |

---

## 6. Visual Semantics for Manim: The Protected Conveyor Belt

### How to Animate LSTM Gradient Flow in Manim:
* **The Two Parallel Rivers**:
  * Render two horizontal conduits:
    * The **lower conduit** ($h_t$): Turbulent, filtered water with spinning waterwheels (gates).
    * The **upper conduit** ($C_t$): A smooth, illuminated glass pipeline running straight across all timesteps with zero friction.
* **The Forget Valve**:
  * Animate the forget gate $f_t$ as an iris valve along the upper glass pipeline.
  * When $f_t = 1.0$, the valve is wide open; a glowing blue sphere passes through without losing speed.
* **The Backward Gradient Laser**:
  * When backpropagating, send a red laser through the upper pipeline $C_T \to C_0$.
  * Because the pipeline has no matrix distortions ($W_{hh}$), the laser shoots through 50 timesteps with 100% brightness, reaching $t=1$ instantly.
  * Overlay the text: *"The Additive Gradient Highway: No matrix multiplication. No exponential decay."*

---

## 📝 Checkpoint 16 Exercises

### Level 1 — Conceptual
Explain why replacing the multiplicative update $h_t = \phi(W h_{t-1})$ with the additive update $C_t = C_{t-1} + \Delta C_t$ fundamentally changes gradient propagation.

### Level 2 — Mathematical
Given hidden dimension $d_h = 512$ and input dimension $d_x = 128$:
Compute the exact parameter count (weights and biases) for:
1. A Vanilla RNN cell.
2. A GRU cell.
3. An LSTM cell.

### Level 3 — Debugging / Misconception
A researcher trains an LSTM and notices that the forget gate bias was initialized to $b_f = -5.0$.
What will happen to the forget gate activations $f_t = \sigma(W [h, x] - 5.0)$ at the beginning of training? How will this impact gradient flow across time? (Why does modern practice initialize $b_f = +1.0$ or $+2.0$?).

### Level 4 — Visualization (Manim Translation)
Design a visual storyboard showing the interaction between the forget gate $f_t$, input gate $i_t$, and candidate $\tilde{C}_t$ during the forward update of the cell state $C_t$.

### Level 5 — Advanced (Peephole Connections)
In Gers & Schmidhuber's "peephole LSTM", the gates receive direct connections from the previous cell state: $f_t = \sigma(W_f [h_{t-1}, x_t] + P_f C_{t-1} + b_f)$.
How does this modify the Jacobian $\frac{\partial C_t}{\partial C_{t-1}}$?
