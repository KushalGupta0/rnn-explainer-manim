# Spoken Narration: Scene 07 — Exploding Gradients & Norm Clipping Geometry

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~45 seconds.

---

### [Beat 1: The Exponential Detonation] (0:00 - 0:12 | 12s)
What happens if the weights are large and the spectral norm exceeds one? The exact opposite catastrophe occurs. Instead of decaying, the gradient vector multiplies exponentially at every backward step, ballooning into millions, overflowing arithmetic limits, and producing catastrophic NaNs.

### [Beat 2: Catapulting Off the Cliff] (0:12 - 0:22 | 10s)
On the loss surface, this creates extreme instability. A massive gradient vector takes a gigantic parameter update step, instantly catapulting our weights out of well-behaved valleys and into numerical oblivion.

### [Beat 3: The Geometric Solution: Norm Clipping] (0:22 - 0:34 | 12s)
Fortunately, unlike vanishing gradients, exploding gradients have a remarkably clean geometric remedy: global gradient norm clipping. Imagine a hypersphere of threshold radius tau centered at the origin. If our gradient vector stays inside the sphere, we leave it untouched.

### [Beat 4: Direction Invariance] (0:34 - 0:45 | 11s)
If the gradient vector breaches the boundary, we do not distort its orientation. Instead, we project its tip directly back to the surface of the sphere. The direction of steepest descent is perfectly preserved; only the velocity of the step is tamed.
