# Recurrent Neural Networks (RNN) — Mathematical Explainer Video

An 8-minute, 3Blue1Brown-quality educational animation explaining **Recurrent Neural Networks (RNNs)** from first principles to advanced gating mechanisms (LSTMs & GRUs), built with [Manim Community Edition](https://www.manim.community/).

All 15 chapters are unified into a single production script: [`rnn_explainer.py`](rnn_explainer.py).

---

## 🚀 Quickstart: Clone & Render

Anyone can clone this repository, run the script for their desired quality tier, and immediately get the complete video.

### 1. Prerequisites
- **Python**: 3.10+
- **FFmpeg**: Installed and on system PATH
- **LaTeX** (e.g. MiKTeX, TeX Live) with `dvisvgm`

### 2. Setup
```bash
git clone https://github.com/KushalGupta0/rnn-explainer-manim.git
cd rnn-explainer-manim
pip install -r requirements.txt
```

---

## 🎬 1-Click Rendering Scripts

Convenient, native rendering scripts are provided for **Windows**, **macOS**, and **Linux** for each resolution tier:

| Quality | Resolution & Framerate | Windows | macOS / Linux | Direct Python CLI |
| :--- | :--- | :--- | :--- | :--- |
| **Preview** | 480p @ 15 fps (`-ql`) | `render_low.bat` | `./render_low.sh` | `python render.py -q low` |
| **Standard** | 720p @ 30 fps (`-qm`) | `render_medium.bat` | `./render_medium.sh` | `python render.py -q medium` |
| **Full HD** | 1080p @ 60 fps (`-qh`) | `render_high.bat` | `./render_high.sh` | `python render.py -q high` |
| **4K Ultra HD** | 2160p @ 60 fps (`-qk`) | `render_4k.bat` | `./render_4k.sh` | `python render.py -q 4k` |

Completed videos are automatically placed into the `output/` directory (e.g., `output/rnnexplainer_low.mp4`).

---

## 📚 Video Chapters (15-Part Curriculum)

The complete video is sequenced across 15 rigorous mathematical chapters:

| # | Chapter Title | Core Mathematical / Visual Concepts |
| :-: | :--- | :--- |
| **01** | **Why RNNs Exist** | The sequence length dilemma; fixed-width matrix bottlenecks vs recurrent temporal sharing. |
| **02** | **The RNN Cell** | First principles, weight matrices ($W_{xh}, W_{hh}, W_{hy}$), vector addition & tanh nonlinearity. |
| **03** | **Unrolling Through Time** | Temporal loop representation vs. Directed Acyclic Graph (DAG) with physical parameter sharing. |
| **04** | **Forward Pass & Loss** | Multi-step unrolling, forward state caching, sequential readouts, and composite loss accumulation. |
| **05** | **Backpropagation Through Time (BPTT)** | Reverse gradient fluid flow, adjoint state equations $\partial L / \partial h_t$, and temporal chain rule. |
| **06** | **Vanishing Gradients** | Repeated scalar multiplication cascade ($\gamma^k$), exponential decay horizon ($k \to 50$). |
| **07** | **Exploding Gradients** | Exponential norm blowup ($\rho > 1$), gradient cliffs, and geometric Euclidean norm clipping. |
| **08** | **Long-Term Dependencies** | Bengio's dilemma: gradient extinction vs. temporal memory retention over long context windows. |
| **09** | **The Scalar RNN Microscope** | Numerical sensitivity derivation, derivative bounds $|\phi'| \le 1.0$, and product decay. |
| **10** | **The Jacobian Matrix View** | Transition Jacobian $J_t = D_t W_{hh}^\top$, SVD singular value collapse, and spectral norm upper bounds. |
| **11** | **Discrete Dynamical Systems** | Phase space trajectories, contractive fixed-point attractors, and loss of initial condition information. |
| **12** | **Long Short-Term Memory (LSTM)** | Additive cell state highway ($C_t$), forget gate, input gate, candidate memory, and output gate. |
| **13** | **Gated Recurrent Units (GRU)** | Parameter-efficient gating, reset gate ($r_t$), update gate ($z_t$), and convex state interpolation. |
| **14** | **Why Gated RNNs Help** | Error carousel: additive gradient propagation preventing exponential gradient vanishing. |
| **15** | **Final Architectural Synthesis** | Comparative taxonomy, stationary operators, and the unified theory of recurrent dynamics. |

---

## 🛠️ Advanced Usage

### Render a Single Chapter
You can also render any individual chapter directly via `render.py`:
```bash
python render.py --quality low --scene Scene01WhyRNNsExist
```
Or directly with Manim:
```bash
manim -qm rnn_explainer.py Scene10JacobianMatrix
```

---

## 📁 Repository Structure

```text
├── lib/
│   ├── components.py        # Reusable visual components (LabeledBox, Node, CaptionBar, RecurrentLoopCell)
│   ├── palette.py           # 3Blue1Brown-inspired color system and typography constants
│   └── video_scene.py       # Base scene with automated layout & margin verification
├── manim.cfg                # Manim configuration (cache settings, default background)
├── rnn_explainer.py         # Complete unified codebase: all 15 scenes + master video class
├── render.py                # Cross-platform CLI rendering pipeline
├── render_low.bat           # Windows 480p preview runner
├── render_medium.bat        # Windows 720p standard runner
├── render_high.bat          # Windows 1080p Full HD runner
├── render_4k.bat            # Windows 4K Ultra HD runner
├── render_low.sh            # Linux/macOS 480p preview runner
├── render_medium.sh         # Linux/macOS 720p standard runner
├── render_high.sh           # Linux/macOS 1080p Full HD runner
├── render_4k.sh             # Linux/macOS 4K Ultra HD runner
├── requirements.txt         # Python dependencies
└── README.md
```

---

## 📄 License
MIT License. Free for educational and research use.
