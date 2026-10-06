# Spoken Narration: Scene 11 — State-Space Interpretation: Latent Phase Trajectories

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~50 seconds.

---

### [Beat 1: The Dynamical System Lens] (0:00 - 0:12 | 12s)
Step back from equations and computational graphs. Viewed geometrically, a recurrent neural network is a discrete nonlinear dynamical system. Its internal memory is not a static memory tape, but a point traveling through a compact latent state space.

### [Beat 2: Tracing the Latent Trajectory] (0:12 - 0:25 | 13s)
Because tanh squashes all activations, the entire state space is bounded inside the hypercube from minus one to plus one. Starting from the origin at $h_0$, each incoming token applies an external vector force, steering the state along a continuous path: $h_1$, $h_2$, $h_3$, tracing a distinct trajectory for every unique sequence.

### [Beat 3: Contractive Basins & Information Collapse] (0:25 - 0:38 | 13s)
Under stable weights, the transition mapping contracts the space. Distinct starting points and distant historical perturbations get sucked into the same contractive basin. Over time, trajectory variations collapse into a single point—erasing the memory of which path brought the network here.

### [Beat 4: The Need for Architectural Gating] (0:38 - 0:50 | 12s)
In an un-gated recurrent network, you cannot escape this dilemma: either trajectories collapse into attractors and forget, or they diverge into chaotic instability. To maintain persistent memory without chaos, we need an entirely new architectural mechanism: controllable gating.
