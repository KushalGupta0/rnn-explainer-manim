# Module 04: The Forward Pass — Step-by-Step Numerical Walkthrough

[Back to Master Index](../index.md) | [Back to Module 03](03_unrolling_the_rnn.md) | [Next to Module 05](05_loss_and_bptt_foundations.md)

---

## 1. System Specification: A Miniature Scalar RNN

To eliminate dimensional clutter and inspect the numerical mechanics with absolute transparency, we specify a 1-dimensional scalar RNN ($d_x = 1, d_h = 1, d_y = 1$) executed across $T = 3$ timesteps.

### Governing Scalar Equations:
* **Pre-activation**: $a_t = w \cdot x_t + u \cdot h_{t-1} + b$
* **Hidden State**: $h_t = \tanh(a_t)$
* **Output / Logit**: $o_t = v \cdot h_t + c$
* **Prediction**: $\hat{y}_t = o_t$ (linear identity read-out)
* **Step Loss**: $L_t = \frac{1}{2}(\hat{y}_t - y_t^*)^2$ (mean squared error)
* **Total Sequence Loss**: $L = \sum_{t=1}^3 L_t$

### Parameter Values:
* Input weight: $w = 0.50$
* Recurrent hidden weight: $u = 0.80$
* Hidden bias: $b = 0.10$
* Output weight: $v = 1.20$
* Output bias: $c = -0.20$
* Initial boundary condition: $h_0 = 0.00$

### Input Sequence & Ground Truth Targets:
* Timestep 1: $x_1 = 1.00$, Target $y_1^* = 0.50$
* Timestep 2: $x_2 = -0.50$, Target $y_2^* = -0.20$
* Timestep 3: $x_3 = 0.20$, Target $y_3^* = 0.80$

---

## 2. Manual Step-by-Step Numerical Execution

```
                     t = 1                  t = 2                  t = 3
Inputs:              x₁ = 1.00              x₂ = -0.50             x₃ = 0.20
Targets:             y₁* = 0.50             y₂* = -0.20            y₃* = 0.80
```

### Timestep $t = 1$:
1. **Pre-activation $a_1$**:
   $$a_1 = w \cdot x_1 + u \cdot h_0 + b = (0.50)(1.00) + (0.80)(0.00) + 0.10 = 0.50 + 0.00 + 0.10 = \mathbf{0.6000}$$
2. **Hidden State $h_1$**:
   $$h_1 = \tanh(a_1) = \tanh(0.6000) = \frac{e^{0.60} - e^{-0.60}}{e^{0.60} + e^{-0.60}} = \mathbf{0.5370}$$
3. **Output Score $o_1$ & Prediction $\hat{y}_1$**:
   $$\hat{y}_1 = o_1 = v \cdot h_1 + c = (1.20)(0.5370) - 0.20 = 0.6444 - 0.20 = \mathbf{0.4444}$$
4. **Step Loss $L_1$**:
   $$L_1 = \frac{1}{2}(\hat{y}_1 - y_1^*)^2 = \frac{1}{2}(0.4444 - 0.5000)^2 = \frac{1}{2}(-0.0556)^2 = \mathbf{0.00155}$$

---

### Timestep $t = 2$:
1. **Pre-activation $a_2$**:
   Notice that $h_1 = 0.5370$ is now carried into the recurrence:
   $$a_2 = w \cdot x_2 + u \cdot h_1 + b = (0.50)(-0.50) + (0.80)(0.5370) + 0.10$$
   $$a_2 = -0.2500 + 0.4296 + 0.1000 = \mathbf{0.2796}$$
2. **Hidden State $h_2$**:
   $$h_2 = \tanh(0.2796) = \mathbf{0.2725}$$
3. **Output Score $o_2$ & Prediction $\hat{y}_2$**:
   $$\hat{y}_2 = o_2 = v \cdot h_2 + c = (1.20)(0.2725) - 0.20 = 0.3270 - 0.20 = \mathbf{0.1270}$$
4. **Step Loss $L_2$**:
   $$L_2 = \frac{1}{2}(\hat{y}_2 - y_2^*)^2 = \frac{1}{2}(0.1270 - (-0.2000))^2 = \frac{1}{2}(0.3270)^2 = \mathbf{0.05347}$$

---

### Timestep $t = 3$:
1. **Pre-activation $a_3$**:
   Here $h_2 = 0.2725$ enters the recurrence:
   $$a_3 = w \cdot x_3 + u \cdot h_2 + b = (0.50)(0.20) + (0.80)(0.2725) + 0.10$$
   $$a_3 = 0.1000 + 0.2180 + 0.1000 = \mathbf{0.4180}$$
2. **Hidden State $h_3$**:
   $$h_3 = \tanh(0.4180) = \mathbf{0.3951}$$
3. **Output Score $o_3$ & Prediction $\hat{y}_3$**:
   $$\hat{y}_3 = o_3 = v \cdot h_3 + c = (1.20)(0.3951) - 0.20 = 0.4741 - 0.20 = \mathbf{0.2741}$$
4. **Step Loss $L_3$**:
   $$L_3 = \frac{1}{2}(\hat{y}_3 - y_3^*)^2 = \frac{1}{2}(0.2741 - 0.8000)^2 = \frac{1}{2}(-0.5259)^2 = \mathbf{0.13828}$$

---

## 3. Total Sequence Loss Accumulation

The total scalar objective that will be minimized during training is the sum of step-wise losses:
$$L = L_1 + L_2 + L_3 = 0.00155 + 0.05347 + 0.13828 = \mathbf{0.19330}$$

### Complete Activation Trace Table (Cached for Backpropagation):

| Quantity | $t = 1$ | $t = 2$ | $t = 3$ |
| :--- | :--- | :--- | :--- |
| **Input $x_t$** | $1.0000$ | $-0.5000$ | $0.2000$ |
| **Past State $h_{t-1}$** | $0.0000$ | $0.5370$ | $0.2725$ |
| **Pre-activation $a_t$** | $0.6000$ | $0.2796$ | $0.4180$ |
| **Hidden State $h_t$** | $0.5370$ | $0.2725$ | $0.3951$ |
| **Prediction $\hat{y}_t$** | $0.4444$ | $0.1270$ | $0.2741$ |
| **Target $y_t^*$** | $0.5000$ | $-0.2000$ | $0.8000$ |
| **Residual $(\hat{y}_t - y_t^*)$**| $-0.0556$ | $0.3270$ | $-0.5259$ |
| **Loss $L_t$** | $0.00155$ | $0.05347$ | $0.13828$ |

> [!IMPORTANT]
> **Why the Forward Pass Must Cache Every State:**
> Notice that to calculate the derivatives later during BPTT (Modules 05 & 06), we will need $\tanh'(a_t) = 1 - h_t^2$ and $h_{t-1}$ at every timestep.
> If an RNN runs for $T$ timesteps, **all $T$ hidden state vectors must be held in RAM/VRAM** until the backward pass finishes!

---

## 4. Visual Semantics for Manim: Forward Numerical Pulse

### How to Animate the Numerical Flow in Manim:
* **The Pulse Particle**: A bright glowing dot travels along the directed edges of the unrolled graph.
* **Live HUD Overlay**: Next to each node ($a_t$, $h_t$, $\hat{y}_t$), display a digital number display:
  * When the pulse hits $a_1$, `0.600` appears.
  * When it passes through the $\tanh$ gate, the number morphs smoothly into `0.537`.
  * The arrow branches: one stream flows up to produce prediction `0.444` and computes error `0.0015`, while the other stream flows rightward into the $t=2$ cell.
* **Color Temperature**:
  * Positive activations $\to$ Cool cyan / bright green.
  * Negative activations $\to$ Muted amber / crimson.
  * Error magnitude $(\hat{y}_t - y_t^*)^2$ represented by the size and glow of the loss node $L_t$.

---

## 📝 Checkpoint 04 Exercises

### Level 1 — Conceptual
If $h_0$ were initialized to $10.0$ instead of $0.0$, how would the pre-activation $a_1$ and hidden state $h_1$ change? What property of the $\tanh$ activation prevents $h_1$ from growing to infinity?

### Level 2 — Mathematical
Using the exact parameters from this lesson:
Suppose we add a 4th timestep with $x_4 = 0.00$ and $y_4^* = 0.00$.
Compute $a_4, h_4, \hat{y}_4,$ and $L_4$ to 4 decimal places.

### Level 3 — Debugging / Misconception
A software engineer writes an RNN inference loop and frees memory by executing:
`del h_prev` inside the loop to save RAM during the forward pass of training.
Why will the backward automatic differentiation engine crash during training if this is done?

### Level 4 — Visualization (Manim Translation)
How would you visually animate the difference between the forward pass (which flows chronologically $t=1 \to T$) and the backward pass (which flows anti-chronologically $T \to 1$)?

### Level 5 — Advanced (Nonlinear Saturation Detection)
Looking at the table of pre-activations:
$a_1 = 0.6000, a_2 = 0.2796, a_3 = 0.4180$.
Calculate the local derivative $\frac{dh_t}{da_t} = 1 - h_t^2$ for each step.
Which timestep has the largest local derivative? Which has the smallest?
