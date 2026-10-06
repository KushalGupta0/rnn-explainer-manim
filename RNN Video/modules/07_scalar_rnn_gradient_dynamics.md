# Module 07: The Scalar RNN — Geometric Scaling & Toy Dynamics

[Back to Master Index](../index.md) | [Back to Module 06](06_bptt_deep_derivation.md) | [Next to Module 08](08_activation_functions.md)

---

## 1. Why Start with a Scalar Recurrence?

In multi-dimensional RNNs, gradient propagation involves matrix multiplications, eigenvalue decompositions, and singular value rotations. These algebraic details can obscure the underlying geometric phenomenon.

By reducing the hidden state, weight, and input to 1-dimensional scalars, the matrix algebra collapses into elementary calculus. This reveals the core engine of gradient instability in its purest form: **repeated scalar exponentiation**.

Consider the scalar recurrence:
$$a_t = w \cdot h_{t-1} + u \cdot x_t + b$$
$$h_t = \phi(a_t)$$
where $w, u, b, h_t, x_t, a_t \in \mathbb{R}$.

---

## 2. Deriving the Scalar Temporal Derivative

Let us compute the local derivative of state $h_t$ with respect to its predecessor $h_{t-1}$:
$$\frac{\partial h_t}{\partial h_{t-1}} = \frac{dh_t}{da_t} \cdot \frac{\partial a_t}{\partial h_{t-1}} = \phi'(a_t) \cdot w$$

Now consider the sensitivity of a distant future state $h_T$ to an early state $h_t$ (where $T > t$).
By the scalar chain rule:
$$\frac{\partial h_T}{\partial h_t} = \frac{\partial h_T}{\partial h_{T-1}} \cdot \frac{\partial h_{T-1}}{\partial h_{T-2}} \cdots \frac{\partial h_{t+1}}{\partial h_t} = \prod_{i=t+1}^T \Big(\phi'(a_i) \cdot w\Big)$$

Factoring out the constant weight $w$, which appears $(T - t)$ times:
$$\mathbf{\frac{\partial h_T}{\partial h_t} = w^{T-t} \prod_{i=t+1}^T \phi'(a_i)}$$

---

## 3. The Power Law: When the Derivative is Approximately Constant

Suppose for simplicity that along the forward trajectory, the activation derivative fluctuates around an effective average value $\overline{\phi'}$.
Define the **effective per-step contraction/expansion factor**:
$$c \equiv \overline{\phi'} \cdot w$$

Then the cumulative sensitivity across $(T - t)$ timesteps scales as an exponential function:
$$\mathbf{\frac{\partial h_T}{\partial h_t} \approx c^{T-t}}$$

This single expression reveals why recurrent optimization is fundamentally different from feed-forward networks with shallow depth. The gradient signal does not decay or grow linearly; it follows an **exponential curve governed by base $c$ and exponent $(T - t)$**.

---

## 4. Concrete Numerical Experiments

Let us compute exact numbers across three realistic sequence lengths:
* $N = 10$ timesteps (a short phrase or audio frame chunk)
* $N = 50$ timesteps (a typical sentence)
* $N = 100$ timesteps (a short paragraph or time-series window)

### Case A: Contractive Dynamics ($|c| < 1$) — Vanishing Gradients
Let $c = 0.50$ (e.g., $w = 1.0$ and $\phi' = 0.5$, or $w = 0.5$ and $\phi' = 1.0$):

$$\begin{aligned}
c^{10} &= 0.5^{10} \approx \mathbf{9.7656 \times 10^{-4}} \quad (\approx 0.00098) \\
c^{50} &= 0.5^{50} \approx \mathbf{8.8818 \times 10^{-16}} \\
c^{100} &= 0.5^{100} \approx \mathbf{7.8886 \times 10^{-31}}
\end{aligned}$$

#### What this means physically:
* At $N = 10$, the gradient signal has already lost over $99.9\%$ of its strength.
* At $N = 50$, the gradient is smaller than standard 64-bit double precision machine epsilon ($\epsilon_{\text{mach}} \approx 2.22 \times 10^{-16}$). It underflows to zero!
* At $N = 100$, the learning signal has completely evaporated into digital noise. The parameter $w$ receives **zero numerical update** from the dependency between step 1 and step 100.

---

### Case B: Expansive Dynamics ($|c| > 1$) — Exploding Gradients
Let $c = 1.50$ (e.g., $w = 2.0$ and $\phi' = 0.75$):

$$\begin{aligned}
c^{10} &= 1.5^{10} \approx \mathbf{57.665} \\
c^{50} &= 1.5^{50} \approx \mathbf{6.3762 \times 10^{8}} \quad (\approx 637 \text{ million}) \\
c^{100} &= 1.5^{100} \approx \mathbf{4.0656 \times 10^{17}}
\end{aligned}$$

#### What this means physically:
* At $N = 10$, a unit perturbation at $h_1$ creates an amplified disturbance of nearly $60$ units at $h_{10}$.
* At $N = 50$, a backpropagated gradient is scaled by over $600$ million. A standard gradient descent step:
  $$w \leftarrow w - \eta \frac{\partial L}{\partial w}$$
  takes an astronomical jump across the loss landscape, destroying all learned weights and causing arithmetic overflow (`NaN` / `Inf`).
* At $N = 100$, the system experiences catastrophic numerical divergence.

---

## 5. Visual Semantics for Manim: The Shrinking & Expanding Arrow

### How to Animate the Scalar Dynamics in Manim:
* **The Sensitivity Vector**:
  * Represent the gradient $\frac{\partial h_k}{\partial h_t}$ as a vertical arrow at each timestep $k$.
  * At $k = T$, initialize the arrow with height $H = 2.0$ (representing the loss gradient).
* **The Step-by-Step Multiplier Ring**:
  * As the arrow travels backwards from $k \to k-1$, animate it passing through a semi-transparent ring labeled with the multiplier `× 0.5`.
  * **When $c = 0.5$**:
    * At step 1 back: Arrow shrinks to $1.0$.
    * At step 2 back: Arrow shrinks to $0.5$.
    * At step 5 back: Arrow shrinks to $0.0625$ (a tiny stub).
    * At step 10 back: The arrow is an invisible microscopic point.
    * Display an accompanying plot of $f(t) = c^t$ collapsing onto the horizontal asymptote $y=0$.
  * **When $c = 1.5$**:
    * At step 1 back: Arrow grows to $3.0$.
    * At step 3 back: Arrow bursts outside the camera view frustum ($H = 6.75$).
    * Trigger a camera zoom-out effect, showing the arrow towering like a skyscraper, flashing bright warning red as `NaN` appears on screen.

---

## 📝 Checkpoint 07 Exercises

### Level 1 — Conceptual
If $|c| = 1.000$ exactly, does the gradient vanish or explode? Why is maintaining $|c| = 1.000$ throughout training extraordinarily difficult in practice with vanilla RNNs?

### Level 2 — Mathematical
Suppose $w = 2.0$, and the activation function is $\phi(z) = \tanh(z)$.
If the pre-activation $a_t = 1.5$ at every timestep, compute:
1. The exact activation derivative $\phi'(1.5) = 1 - \tanh^2(1.5)$.
2. The exact value of $c = w \cdot \phi'(1.5)$.
3. Does the gradient vanish or explode across 20 timesteps under these conditions?

### Level 3 — Debugging / Misconception
A programmer argues:
*"If we initialize $w = 0.999$, we are safe from exploding gradients, and because $0.999 \approx 1$, gradients will not vanish either."*
Calculate $0.999^{1000}$ and explain what happens over long sequences.

### Level 4 — Visualization (Manim Translation)
How would you visually design the transition from the contractive regime ($c=0.5$) to the expansive regime ($c=1.5$) using an interactive dial or slider in Manim?

### Level 5 — Advanced (Log-Space Gradients)
In statistical physics and numerical analysis, products of many numbers are often computed in log-space:
$$\log \left| \frac{\partial h_T}{\partial h_t} \right| = \sum_{i=t+1}^T \log |\phi'(a_i) w|$$
Explain how this logarithmic formulation connects BPTT to **Lyapunov exponents** in dynamical systems theory.
