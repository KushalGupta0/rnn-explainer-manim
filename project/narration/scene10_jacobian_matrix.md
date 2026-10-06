# Spoken Narration: Scene 10 — The Jacobian Matrix: Geometry of High-Dimensional Gradient Flow

**Pacing**: Calm, deliberate, conversational yet precise. Target time: ~50 seconds.

---

### [Beat 1: Upgrading to Multidimensional Geometry] (0:00 - 0:12 | 12s)
In full-scale neural networks, the hidden state is a high-dimensional vector. Taking the derivative of an updated state vector $\mathbf{h}_t$ with respect to its predecessor $\mathbf{h}_{t-1}$ produces a full matrix of partial derivatives: the transition Jacobian, $\mathbf{J}_t$.

### [Beat 2: Factoring the Jacobian] (0:12 - 0:24 | 12s)
By the multivariate chain rule, this Jacobian factors cleanly into two linear operations: the recurrent weight matrix $\mathbf{W}_{hh}$, followed by a diagonal matrix $\mathbf{D}_t$ of element-wise activation derivatives. Together, they rotate, shear, and squash our gradient space.

### [Beat 3: The Geometric SVD: Transforming the Unit Sphere] (0:24 - 0:37 | 13s)
Consider a unit circle representing all possible gradient directions. At each timestep, the Jacobian stretches it along its principal singular axes and squashes it along others. Across many timesteps, chained Jacobians repeatedly flatten the sphere into an ultra-thin ellipse—diminishing volume and extinguishing gradient components.

### [Beat 4: The Spectral Norm Criterion] (0:37 - 0:50 | 13s)
By norm submultiplicity, the total gradient is strictly bounded by the maximum singular value: $\sigma_{\max}(\mathbf{W}_{hh})^{T-t}$. If this spectral norm is below one, gradients inevitably collapse to zero across all directions. If above one, components aligned with dominant singular vectors can detonate exponentially.
