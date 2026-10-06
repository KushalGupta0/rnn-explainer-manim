# Spoken Narration: Scene 02 — The RNN Cell: First Principles & Dimensions

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~45 seconds.

---

### [Beat 1: The Two Inflows] (0:00 - 0:12 | 12s)
Here is the core engine that makes sequence processing possible: the recurrent cell. At any given moment, the cell receives two distinct signals: fresh external evidence, vector $x_t$, and a summary of everything that came before, the hidden state $h_{t-1}$.

### [Beat 2: Linear Superposition] (0:12 - 0:25 | 13s)
Both vectors are transformed by their respective weight matrices: $W_{xh}$ projects the input into the latent space, while $W_{hh}$ rotates and scales the past state. Inside the cell, these two transformed vectors simply add together, alongside a bias vector, to produce the pre-activation $a_t$.

### [Beat 3: The Hyperbolic Squashing] (0:25 - 0:35 | 10s)
Before this combined vector can represent the updated state, it passes through an activation function—typically the hyperbolic tangent. This bounds every single component within the clean range of minus one to one, preventing forward values from blowing up into infinity.

### [Beat 4: Readout & State Emission] (0:35 - 0:45 | 10s)
The result is $h_t$: a newly minted vector of memory. It serves two masters simultaneously: it emits an outward prediction $\hat{y}_t$, and it stands ready to be passed forward as the memory for the next moment in time.
