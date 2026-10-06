# Spoken Narration: Scene 04 — The Forward Pass: Numerical State Caching

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~45 seconds.

---

### [Beat 1: The Initial Spark] (0:00 - 0:12 | 12s)
Let us trace the forward pass through a concrete sequence. We begin with a neutral initial memory vector, $h_0$, set to zero. As the first input vector $x_1$ enters, the cell computes its first hidden state, $h_1$.

### [Beat 2: Compounding Context] (0:12 - 0:24 | 12s)
At timestep two, $h_1$ is transferred along the horizontal connection. It combines with $x_2$, producing an updated state $h_2$. Notice that $h_2$ now contains compressed mathematical information from both $x_1$ and $x_2$.

### [Beat 3: Freezing the Activation Cache] (0:24 - 0:35 | 11s)
As this process repeats, an essential computational requirement emerges: during training, every single intermediate hidden state vector must be frozen and stored in memory. We cannot discard them, because they will be indispensable during backpropagation.

### [Beat 4: The Accumulation of Error] (0:35 - 0:45 | 10s)
At each step, the model's predictions are compared against ground-truth targets, igniting local loss values. The total objective for the sequence is simply the sum of these errors across all timesteps.
