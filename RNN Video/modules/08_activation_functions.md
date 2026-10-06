# Module 08: Activation Functions — Sigmoid, Tanh, ReLU & Saturation

[Back to Master Index](../index.md) | [Back to Module 07](07_scalar_rnn_gradient_dynamics.md) | [Next to Module 09](09_matrix_jacobian_view.md) | [Deep Dive: Activation Saturation](../deep_dives/activation_saturation/README.md)

---

## 1. The Triad of Non-Linearities in Neural Networks

To understand why gradients vanish, we must examine the derivative properties of the three most influential activation functions: **Sigmoid**, **Tanh**, and **ReLU**.

```
    SIGMOID: σ(x)                TANH: tanh(x)               RELU: max(0, x)
       1 ┌───────                 1 ┌───────                 3 ┌       /
         │      /                   │      /                   │      /
     0.5 ├───-─/──-             0.0 ├───-─/──-               0 └─────/──────
         │    /                     │    /                     -2 -1 0 1 2 3
       0 └───/────               -1 └───/────
       -4 -2 0 2 4                -3 -1 0 1 3

   DERIVATIVE: σ'(x)           DERIVATIVE: tanh'(x)         DERIVATIVE: ReLU'(x)
    0.25 ┌───/\───              1.0 ┌───/\───              1.0 ┌     ┌───────
         │  /  \                    │  /  \                    │     │
       0 └─/────\─                0 └─/────\─                0 └─────┴───────
        Max = 0.25                 Max = 1.00               Active = 1, Dead = 0
```

---

## 2. Comparative Mathematical Profile

| Metric | Sigmoid $\sigma(x)$ | Hyperbolic Tangent $\tanh(x)$ | Rectified Linear Unit $\operatorname{ReLU}(x)$ |
| :--- | :--- | :--- | :--- |
| **Formula** | $\frac{1}{1 + e^{-x}}$ | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $\max(0, x)$ |
| **Range** | $(0, 1)$ | $(-1, 1)$ | $[0, \infty)$ |
| **Zero-Centered?** | No ($> 0$ always) | **Yes** ($\tanh(0) = 0$) | No ($\ge 0$) |
| **Derivative** | $\sigma(x)(1 - \sigma(x))$ | $1 - \tanh^2(x)$ | $1$ if $x > 0$, else $0$ |
| **Max Derivative** | **$0.25$** (at $x=0$) | **$1.00$** (at $x=0$) | **$1.00$** (for all $x > 0$) |
| **Saturation Zones** | Both tails ($x \to \pm \infty$) | Both tails ($x \to \pm \infty$) | Negative region only ($x < 0$) |
| **RNN Behavior** | Extreme gradient vanishing | Standard, but saturates | Risk of exploding activations |

---

## 3. The Mechanism of Saturation

### What is Saturation?
A non-linear function is **saturated** when small changes in its input produce negligible changes in its output:
$$\left| \frac{d\phi(a)}{da} \right| \approx 0$$
For both sigmoid and tanh:
* When pre-activations $|a|$ grow large (due to accumulated states or large incoming weights), the function flattens against its asymptotic bounds ($\pm 1$ for tanh, $0$ or $1$ for sigmoid).
* In this regime, the slope of the tangent line is virtually horizontal.
* Any backpropagated gradient passing through this neuron is multiplied by this near-zero slope, acting as a **gradient sink**.

### Why Tanh is Preferred over Sigmoid in Vanilla RNNs:
1. **Higher Peak Slope**: $\tanh'(0) = 1.0$ compared to $\sigma'(0) = 0.25$. Tanh allows up to 4 times better gradient transmission in the linear regime.
2. **Zero-Centered States**: Tanh produces outputs in $(-1, 1)$ with mean centered near zero, preventing directional bias accumulation during repeated recurrent updates.

---

## 4. Deconstructing the Myth: "The Activation Function Causes Vanishing Gradients"

> [!WARNING]
> **Common Inaccuracy in Popular Deep Learning Tutorials:**
> Many introductory tutorials state: *"Vanishing gradients happen because the derivative of tanh is at most 1, and usually less than 1, so multiplying numbers less than 1 makes them vanish."*
> **This explanation is technically incomplete and misleading.**

### Why It Is Incomplete:
The Jacobian of the recurrent transition is:
$$J_t = D_t W_{hh} = \operatorname{diag}(\phi'(a_t)) W_{hh}$$
* The derivative of the activation function $D_t$ does **not** act in isolation; it is coupled directly with the weight matrix $W_{hh}$.
* If $W_{hh}$ has singular values significantly larger than 1 (e.g. $\sigma_{\max} = 10.0$), the product $D_t W_{hh}$ can easily have norm $> 1$ even if $\phi'(a_t) = 0.20$ ($0.20 \times 10.0 = 2.0$).
* **The real cause of vanishing gradients is the dual interaction**:
  If weights are small ($\|W_{hh}\| < 1$), gradients vanish regardless of activation. If weights are large ($\|W_{hh}\| > 1$), activations push into the saturation zone ($|a_t| \gg 0$), which forces $\phi'(a_t) \to 0$, causing gradients to vanish anyway!

---

## 5. Visual Semantics for Manim: The Tangent Slope Gauge

### How to Animate Saturation in Manim:
* **The Dynamic Tangent Line**:
  * Plot the $\tanh(x)$ curve prominently in the center.
  * Animate a glowing tracer point moving along the curve corresponding to the current pre-activation $a_t$.
  * Render the tangent line at that point, with a floating numerical slope readout: $m = 1 - \tanh^2(a_t)$.
  * When $a_t = 0$, show the tangent line angled at $45^\circ$ ($m = 1.0$, bright green).
  * As $a_t$ slides out to $x = 2.5$, animate the tangent line flattening into a horizontal bar ($m = 0.07$, fading to dark red).
* **The Transmission Filter**:
  * Represent the backward gradient pulse as a beam of light trying to pass through a glass prism.
  * When the neuron is unsaturated ($m \approx 1$), the prism is 100% transparent; the gradient passes unattenuated.
  * When the neuron is saturated ($m \approx 0$), the prism becomes an opaque dark mirror; the gradient pulse is absorbed and extinguished.

---

## 📝 Checkpoint 08 Exercises

### Level 1 — Conceptual
Explain why initializing the recurrent weights with very large values often triggers vanishing gradients rather than exploding gradients when using $\tanh$ activations.

### Level 2 — Mathematical
Let an RNN use the Sigmoid activation $\phi(z) = \sigma(z)$.
If $W_{hh} = 2.0$, what is the absolute theoretical maximum possible value of the single-step derivative $\frac{\partial h_t}{\partial h_{t-1}} = \sigma'(a_t) W_{hh}$? Can gradients ever explode in this specific network?

### Level 3 — Debugging / Misconception
A team replaces $\tanh$ with $\operatorname{ReLU}$ in a vanilla RNN to prevent vanishing gradients, but finds that training immediately crashes with `Loss: NaN` after 3 iterations.
Explain the mathematical reason why unconstrained ReLU activations can be catastrophic in recurrent networks.

### Level 4 — Visualization (Manim Translation)
How would you visually animate the difference in gradient transmission between an unsaturated state ($a_t = 0.1$) and a saturated state ($a_t = 3.5$) in an unrolled computational graph?

### Level 5 — Advanced (Leaky ReLU in RNNs)
Consider $\operatorname{LeakyReLU}(x) = \max(\alpha x, x)$ with $\alpha = 0.01$.
Does Leaky ReLU resolve the vanishing gradient problem in vanilla RNNs over sequences of length $T = 500$? Analyze both regimes ($x > 0$ and $x < 0$).
