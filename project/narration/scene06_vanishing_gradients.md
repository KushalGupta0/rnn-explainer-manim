# Spoken Narration: Scene 06 — The Vanishing Gradient: Exponential Collapse

**Pacing**: Calm, deliberate, conversational. Target duration: 65s (144 words | 132.9 WPM).

---

### [Beat 1: Cascading Shrinkage] (0:00 - 0:16 | 16s)
As sequences grow longer, a fundamental flaw appears. Watch what happens when an error signal at step six travels backward toward step one. At each temporal hop, the gradient pulse visibly dims and contracts.

### [Beat 2: Total Evaporation] (0:16 - 0:32 | 16s)
By the time the signal reaches the first step, it has collapsed into an imperceptible whisper. State $h_1$ receives almost zero learning signal. The opening tokens might as well never have existed.

### [Beat 3: The Mathematical Culprit] (0:32 - 0:49 | 17s)
This collapse happens because the multi-step sensitivity is a product of transition Jacobians. Each term contains the activation derivative. But the derivative of tanh is at most one, and in saturated regions, it drops near zero.

### [Beat 4: Bengio's Dilemma] (0:49 - 1:05 | 16s)
If the effective multiplier is even slightly below one—say zero point eight—over twenty steps, the gradient shrinks to zero point eight to the twentieth power. That is less than four percent of its original magnitude.
