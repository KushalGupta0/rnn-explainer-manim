# Spoken Narration: Scene 06 — The Vanishing Gradient: Exponential Collapse

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~45 seconds.

---

### [Beat 1: The Cascading Shrinkage] (0:00 - 0:12 | 12s)
Everything seems orderly until our sequences become long. Watch what happens when an error signal at step six attempts to reach the earliest state at step one. With each temporal hop backward, the gradient pulse visibly dims and shrinks.

### [Beat 2: Total Evaporation] (0:12 - 0:23 | 11s)
By the time the signal reaches the first timestep, it has collapsed into an imperceptible whisper. The earliest state receives virtually zero training signal. As far as the optimization algorithm is concerned, the opening tokens of the sequence might as well never have happened.

### [Beat 3: The Mathematical Culprit] (0:23 - 0:34 | 11s)
Why does this happen? The sensitivity of future states to past states is governed by a product of transition Jacobians. Each term is a product of the weight matrix and the activation derivative. But the derivative of the hyperbolic tangent is strictly at most one, and in saturated regions, it drops precipitously toward zero.

### [Beat 4: Bengio's Dilemma] (0:34 - 0:45 | 11s)
If the effective multiplier across steps is even slightly less than one—say zero point eight—over twenty steps, the gradient is multiplied by zero point eight to the power of twenty. That is less than four hundredths of its original strength. The network is struck by amnesia.
