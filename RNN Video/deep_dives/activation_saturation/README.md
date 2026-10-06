# Deep Dive: Activation Function Saturation & Derivative Collapse

[Back to Master Index](../../index.md) | [Back to Module 08](../../modules/08_activation_functions.md)

---

## 1. Complete Derivations of Classic Activation Derivatives

### A. The Logistic Sigmoid Function
The sigmoid function is defined as:
$$\sigma(x) = \frac{1}{1 + e^{-x}} = (1 + e^{-x})^{-1}$$

Taking the first derivative with respect to $x$:
$$\begin{aligned}
\sigma'(x) &= \frac{d}{dx} (1 + e^{-x})^{-1} = -(1 + e^{-x})^{-2} \cdot (-e^{-x}) \\
&= \frac{e^{-x}}{(1 + e^{-x})^2} = \frac{1}{1 + e^{-x}} \cdot \frac{e^{-x}}{1 + e^{-x}} \\
&= \sigma(x) \cdot \frac{(1 + e^{-x}) - 1}{1 + e^{-x}} \\
&= \sigma(x) \cdot (1 - \sigma(x))
\end{aligned}$$

#### Finding the Global Maximum:
To find the maximum of $f(s) = s(1 - s)$ where $s \in (0, 1)$:
$$f'(s) = 1 - 2s = 0 \implies s = \frac{1}{2}$$
Since $\sigma(0) = \frac{1}{2}$:
$$\max_{x \in \mathbb{R}} \sigma'(x) = \sigma'(0) = \frac{1}{2}\left(1 - \frac{1}{2}\right) = \mathbf{\frac{1}{4} = 0.25}$$

> [!CAUTION]
> **The Sigmoid Curse in Recurrent Networks:**
> Because $\sigma'(x) \le 0.25$ for all $x \in \mathbb{R}$, in a recurrent chain $\prod_{i=t+1}^T \sigma'(a_i) w$, the activation derivative alone imposes a decay factor of at least $0.25^{T-t}$.
> Even if $w = 1.0$, across just 10 timesteps:
> $$\sigma'(a)^{10} \le (0.25)^{10} \approx 9.54 \times 10^{-7}$$
> This guarantees catastrophic gradient vanishing unless $w \ge 4.0$, which then creates extreme instability.

---

### B. The Hyperbolic Tangent Function ($\tanh$)
The hyperbolic tangent is defined as:
$$\tanh(x) = \frac{\sinh(x)}{\cosh(x)} = \frac{e^x - e^{-x}}{e^x + e^{-x}}$$

Using the quotient rule:
$$\begin{aligned}
\frac{d}{dx} \tanh(x) &= \frac{\frac{d}{dx}(e^x - e^{-x})(e^x + e^{-x}) - (e^x - e^{-x})\frac{d}{dx}(e^x + e^{-x})}{(e^x + e^{-x})^2} \\
&= \frac{(e^x + e^{-x})^2 - (e^x - e^{-x})^2}{(e^x + e^{-x})^2} \\
&= 1 - \left(\frac{e^x - e^{-x}}{e^x + e^{-x}}\right)^2 \\
&= \mathbf{1 - \tanh^2(x)}
\end{aligned}$$

#### Finding the Global Maximum:
Since $\tanh(x) \in (-1, 1)$, $\tanh^2(x) \ge 0$:
$$\max_{x \in \mathbb{R}} \tanh'(x) = 1 - 0 = \mathbf{1.0} \quad (\text{attained at } x = 0)$$
Everywhere else, $0 < \tanh'(x) < 1.0$.

---

## 2. The Saturation Regime

A neuron is defined to be **saturated** when its input pre-activation $|a|$ is sufficiently large that the activation function operates on its nearly flat asymptotic plateaus:

$$\lim_{x \to \infty} \tanh'(x) = 0 \quad \text{and} \quad \lim_{x \to -\infty} \tanh'(x) = 0$$

### Numerical Derivative Profile for $\tanh(x)$:

| Pre-activation $x$ | $\tanh(x)$ | Derivative $\tanh'(x) = 1 - \tanh^2(x)$ | Attenuation per step |
| :--- | :--- | :--- | :--- |
| **$0.0$** | $0.0000$ | **$1.0000$** | $100\%$ transmission |
| **$0.5$** | $0.4621$ | **$0.7864$** | $21.4\%$ loss |
| **$1.0$** | $0.7616$ | **$0.4200$** | $58.0\%$ loss |
| **$2.0$** | $0.9640$ | **$0.0707$** | $92.9\%$ loss |
| **$3.0$** | $0.9951$ | **$0.0099$** | $99.0\%$ loss |
| **$4.0$** | $0.9993$ | **$0.0013$** | $99.87\%$ loss |

---

## 3. The Causal Chain of Vanishing Gradients

The relationship between non-linearities and vanishing gradients is a strict deterministic chain:

```
Large Weights / Accumulated State History
                    │
                    ▼
Pre-activations |a_t| drift into saturation (|a_t| > 2.5)
                    │
                    ▼
Local derivatives collapse:  φ'(a_t) → 0
                    │
                    ▼
Diagonal Jacobian operator D_t = diag(φ'(a_t)) has microscopic entries
                    │
                    ▼
Repeated product:  ∏_{k=t+1}^T (D_k W_{hh}) collapses exponentially
                    │
                    ▼
∂L_T / ∂h_t ≈ 0  (Early parameters receive zero learning signal)
```

---

## 4. Why ReLU Does Not Trivially Fix RNNs

In deep feed-forward networks, the Rectified Linear Unit ($\operatorname{ReLU}(x) = \max(0, x)$) largely solved vanishing gradients because its derivative in the active regime is constant:
$$\operatorname{ReLU}'(x) = 1.0 \quad \text{for } x > 0$$

### Why ReLU is Dangerous in Vanilla RNNs:
In an RNN, the transition matrix $W_{hh}$ is applied repeatedly across time.
If an RNN uses ReLU activations and the network operates in the active regime ($a_t > 0$):
$$\frac{\partial h_T}{\partial h_t} = \prod_{k=t+1}^T (I \cdot W_{hh}) = W_{hh}^{T-t}$$
1. If the largest eigenvalue of $W_{hh}$ is even slightly greater than 1 ($\lambda > 1.01$), activations and gradients **grow unboundedly to infinity** because ReLU is completely unbounded from above ($[0, \infty)$).
2. If $\lambda < 1.0$, gradients vanish exponentially.
3. If pre-activations fall below zero ($a_t < 0$), $\operatorname{ReLU}'(a_t) = 0$, causing "dead neurons" that permanently sever the temporal gradient pathway.

Therefore, $\tanh$ remains the historical standard for vanilla RNNs because its $[-1, 1]$ bounding prevents forward activations from exploding, even though its derivative contributes to vanishing gradients.
