# Module 05: Loss & Backpropagation Through Time (Foundations)

[Back to Master Index](../index.md) | [Back to Module 04](04_forward_pass_numerical.md) | [Next to Module 06](06_bptt_deep_derivation.md)

---

## 1. Sequence Loss Formulation

In a sequential task, predictions are evaluated against ground-truth targets. The total scalar objective function $L$ over a sequence of length $T$ is the sum (or average) of per-timestep loss terms:
$$\mathbf{L = \sum_{t=1}^T L_t}$$
where each $L_t = \ell(\hat{y}_t, y_t^*)$ evaluates the prediction $\hat{y}_t = g(W_{hy} h_t + b_y)$ at that exact moment.

For Sequence-to-One classification (e.g. document classification), the loss exists only at the terminal step:
$$L = L_T = \ell(\hat{y}_T, y_T^*)$$
which is simply a special case where $L_t = 0$ for all $t < T$.

---

## 2. The Chain Rule on a Branching Computational Graph

In standard backpropagation for feed-forward networks, each layer outputs to exactly one downstream layer.
In an unrolled RNN, notice the critical structural property of the hidden state $h_t$:

```
                       L_t  (Immediate Step Loss)
                        ▲
                        │
                       y_t
                        ▲
                        │
h_{t-1} ─────────────►[ h_t ] ─────────────► h_{t+1} ──► ... ──► L_{t+1}, ..., L_T
                     (Nexus)                  ▲
                                              │ (Future Pathway)
```

At timestep $t$, the hidden state vector $h_t$ influences the total loss $L$ along **two distinct pathways**:
1. **The Direct Vertical Pathway**:
   $$h_t \longrightarrow y_t \longrightarrow L_t$$
   This immediately affects the loss at the current timestep.
2. **The Indirect Horizontal Pathway**:
   $$h_t \longrightarrow h_{t+1} \longrightarrow h_{t+2} \longrightarrow \dots \longrightarrow h_T \longrightarrow \{L_{t+1}, \dots, L_T\}$$
   This propagates forward in time, influencing all future hidden states and all future losses.

### The Fundamental BPTT Recurrence for Hidden States
By the multivariate chain rule for functions with multiple downstream paths, the total derivative of $L$ with respect to $h_t$ is the sum of both contributions:
$$\mathbf{\frac{\partial L}{\partial h_t} = \frac{\partial L_t}{\partial h_t} + \left(\frac{\partial h_{t+1}}{\partial h_t}\right)^\top \frac{\partial L}{\partial h_{t+1}}}$$

*(Note: In row-vector gradient notation, this is $\nabla_{h_t} L = \nabla_{h_t} L_t + \nabla_{h_{t+1}} L \cdot \frac{\partial h_{t+1}}{\partial h_t}$.)*

---

## 3. Progressive Step-by-Step Derivation for $T = 3$

To see gradient accumulation in action, let us trace backwards from $t=3$ down to $t=1$ in a 3-step sequence.

The total loss is:
$$L = L_1 + L_2 + L_3$$

### Step A: Terminal Step $t = 3$
At the final timestep $T=3$, there is no future hidden state $h_4$. Therefore, the horizontal pathway does not exist:
$$\frac{\partial L}{\partial h_3} = \frac{\partial L_3}{\partial h_3}$$

The direct gradient from the output layer is simply:
$$\frac{\partial L_3}{\partial h_3} = \frac{\partial o_3}{\partial h_3}^\top \frac{\partial L_3}{\partial o_3} = W_{hy}^\top \frac{\partial L_3}{\partial o_3}$$

---

### Step B: Intermediate Step $t = 2$
The hidden state $h_2$ directly affects $L_2$, and indirectly affects $L_3$ via $h_3$:
$$\frac{\partial L}{\partial h_2} = \frac{\partial L_2}{\partial h_2} + \left(\frac{\partial h_3}{\partial h_2}\right)^\top \frac{\partial L}{\partial h_3}$$

Substitute the result from Step A:
$$\frac{\partial L}{\partial h_2} = \frac{\partial L_2}{\partial h_2} + \left(\frac{\partial h_3}{\partial h_2}\right)^\top \frac{\partial L_3}{\partial h_3}$$

---

### Step C: Initial Step $t = 1$
The hidden state $h_1$ directly affects $L_1$, and indirectly affects the future via $h_2$:
$$\frac{\partial L}{\partial h_1} = \frac{\partial L_1}{\partial h_1} + \left(\frac{\partial h_2}{\partial h_1}\right)^\top \frac{\partial L}{\partial h_2}$$

Now, substitute the expanded expression for $\frac{\partial L}{\partial h_2}$:
$$\frac{\partial L}{\partial h_1} = \frac{\partial L_1}{\partial h_1} + \left(\frac{\partial h_2}{\partial h_1}\right)^\top \left[ \frac{\partial L_2}{\partial h_2} + \left(\frac{\partial h_3}{\partial h_2}\right)^\top \frac{\partial L_3}{\partial h_3} \right]$$

Distribute the transpose product:
$$\mathbf{\frac{\partial L}{\partial h_1} = \frac{\partial L_1}{\partial h_1} + \left(\frac{\partial h_2}{\partial h_1}\right)^\top \frac{\partial L_2}{\partial h_2} + \left(\frac{\partial h_2}{\partial h_1}\right)^\top \left(\frac{\partial h_3}{\partial h_2}\right)^\top \frac{\partial L_3}{\partial h_3}}$$

---

## 4. The Anatomy of Gradient Accumulation

Look closely at the three terms contributing to $\frac{\partial L}{\partial h_1}$:

$$\frac{\partial L}{\partial h_1} = \underbrace{\frac{\partial L_1}{\partial h_1}}_{\text{Lag } 0\text{ (instantaneous)}} + \underbrace{\left(\frac{\partial h_2}{\partial h_1}\right)^\top \frac{\partial L_2}{\partial h_2}}_{\text{Lag } 1\text{ (1-step history)}} + \underbrace{\left(\frac{\partial h_3}{\partial h_1}\right)^\top \frac{\partial L_3}{\partial h_3}}_{\text{Lag } 2\text{ (2-step history)}}$$
where $\frac{\partial h_3}{\partial h_1} = \frac{\partial h_3}{\partial h_2} \frac{\partial h_2}{\partial h_1}$ by the chain rule!

### Generalization to $T$ Timesteps:
For any timestep $t$:
$$\mathbf{\frac{\partial L}{\partial h_t} = \sum_{k=t}^T \left(\frac{\partial h_k}{\partial h_t}\right)^\top \frac{\partial L_k}{\partial h_k}}$$
where by definition $\frac{\partial h_t}{\partial h_t} = I$ (the identity matrix).

```
Future Losses:        L₁            L₂            L₃            ...            L_T
                      │             │             │                            │
                      ▼             ▼             ▼                            ▼
                 (Lag 0)       (Lag 1)       (Lag 2)                      (Lag T-t)
                      │             │             │                            │
                      │       (∂h₂/∂h_t)ᵀ   (∂h₃/∂h_t)ᵀ                  (∂h_T/∂h_t)ᵀ
                      │             │             │                            │
                      ▼             ▼             ▼                            ▼
Accumulated at h_t:  [ + ] ◄───────[ + ] ◄───────[ + ] ◄──────────────────────[ + ] ◄── ∂L/∂h_t
```

---

## 5. Visual Semantics for Manim: Reverse Inundation

### Visualizing Backward Flow:
* **The Watershed Metaphor**: Think of each loss $L_t$ as a mountain spring releasing a surge of colored fluid (e.g. glowing orange/red gradient plasma) backwards through the graph.
* **The Junction at $h_t$**:
  * An immediate drop pours straight down from $L_t$.
  * Simultaneously, a horizontal river flows from the right ($h_{t+1}$).
  * At the node $h_t$, both streams merge into a single, thicker stream that continues flowing leftward into $h_{t-1}$.
* **The Color Attenuation Cue**:
  * As the plasma river passes through each horizontal transition edge, if the transmission factor (Jacobian) is less than 1, the river becomes dimmer, thinner, and translucent.
  * If the Jacobian is greater than 1, the river violently surges, boils, and turns blinding white.

---

## 📝 Checkpoint 05 Exercises

### Level 1 — Conceptual
Why does the gradient $\frac{\partial L}{\partial h_t}$ contain a sum over future timesteps $k \ge t$, but contains **zero** terms from past timesteps $k < t$?

### Level 2 — Mathematical
Suppose for a scalar RNN that:
$\frac{\partial L_1}{\partial h_1} = 0.5$, $\frac{\partial L_2}{\partial h_2} = 0.4$, $\frac{\partial L_3}{\partial h_3} = 0.3$.
Suppose the transition derivatives are constants:
$\frac{\partial h_2}{\partial h_1} = 0.6$ and $\frac{\partial h_3}{\partial h_2} = 0.6$.
Compute the numerical value of $\frac{\partial L}{\partial h_3}$, $\frac{\partial L}{\partial h_2}$, and $\frac{\partial L}{\partial h_1}$.

### Level 3 — Debugging / Misconception
A researcher writes:
*"In Sequence-to-One classification, because $L_1 = L_2 = \dots = L_{T-1} = 0$, the gradient at $h_1$ is simply zero because there are no intermediate losses to backpropagate."*
Spot the mathematical flaw.

### Level 4 — Visualization (Manim Translation)
How would you visually represent the fact that $h_t$ receives gradient contributions from both its immediate vertical output $y_t$ and the horizontal future state $h_{t+1}$?

### Level 5 — Advanced (Memory vs Recomputation)
In Backpropagation Through Time across $T = 10{,}000$ steps, standard reverse-mode automatic differentiation requires $\mathcal{O}(T)$ memory storage. How does "Gradient Checkpointing" trade extra forward computation to reduce this memory footprint to $\mathcal{O}(\sqrt{T})$?
