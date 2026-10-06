# Spoken Narration: Scene 13 — The Gated Recurrent Unit (GRU): Streamlined Gating

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~45 seconds.

---

### [Beat 1: Architectural Simplification] (0:00 - 0:11 | 11s)
While the LSTM effectively cures vanishing gradients, its dual-channel architecture requires four separate affine projections. In 2014, Cho et al. introduced the Gated Recurrent Unit, or GRU, collapsing the cell state and hidden state into a single unified vector.

### [Beat 2: The Reset Gate & Candidate State] (0:11 - 0:23 | 12s)
The GRU uses a reset gate $r_t$ to determine how much of past history to expose when computing candidate memory $\tilde{h}_t$. When $r_t$ is near zero, the cell ignores historical context entirely, allowing the model to forget irrelevant prior phrases and start fresh.

### [Beat 3: The Convex Combination Update] (0:23 - 0:34 | 11s)
Instead of independent forget and input gates, the GRU couples them into a single update gate $z_t$. The new hidden state is an exact convex combination: $h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$. When $z_t$ is near zero, past memory flows straight through untouched.

### [Beat 4: The Practical Tradeoff] (0:34 - 0:45 | 11s)
With 25% fewer parameters and faster training times, the GRU preserves the crucial additive gradient highway while streamlining computation. Both LSTM and GRU prove that gating is the fundamental antidote to exponential gradient decay.
