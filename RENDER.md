# Master Rendering & Assembly Guide

This document contains the exact commands used to render all 15 scenes of the RNN Explainer Video, generate review contact sheets, and assemble the final master video using FFmpeg.

---

## 1. Environment Activation

Ensure you are using the project virtual environment where Manim CE v0.21.0 is installed:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Set the layout check environment variable to guarantee zero visual collisions:

```powershell
$env:MANIM_LAYOUT_CHECK="1"
```

---

## 2. Rendering All 15 Scenes

### Batch PowerShell Command (Recommended)

To render all 15 scenes sequentially at preview quality (`480p15`, `-ql`):

```powershell
$scenes = @(
    @("scene01_why_rnns_exist.py", "Scene01WhyRNNsExist"),
    @("scene02_the_rnn_cell.py", "Scene02TheRNNCell"),
    @("scene03_unrolling_dag.py", "Scene03UnrollingDAG"),
    @("scene04_forward_pass.py", "Scene04ForwardPass"),
    @("scene05_bptt_flow.py", "Scene05BPTTFlow"),
    @("scene06_vanishing_gradients.py", "Scene06VanishingGradients"),
    @("scene07_exploding_gradients.py", "Scene07ExplodingGradients"),
    @("scene08_long_term_dependencies.py", "Scene08LongTermDependencies"),
    @("scene09_scalar_rnn.py", "Scene09ScalarRNN"),
    @("scene10_jacobian_matrix.py", "Scene10JacobianMatrix"),
    @("scene11_state_space.py", "Scene11StateSpace"),
    @("scene12_lstm.py", "Scene12LSTM"),
    @("scene13_gru.py", "Scene13GRU"),
    @("scene14_why_gated_rnns_help.py", "Scene14WhyGatedRNNsHelp"),
    @("scene15_final_synthesis.py", "Scene15FinalSynthesis")
)

foreach ($item in $scenes) {
    Write-Host "Rendering $($item[1])..." -ForegroundColor Cyan
    .venv\Scripts\python.exe -m manim -ql "project/scenes/$($item[0])" $item[1]
}
```

### Individual Scene Commands

```powershell
# Scene 01: Why RNNs Exist (42s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene01_why_rnns_exist.py Scene01WhyRNNsExist

# Scene 02: The RNN Cell (48s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene02_the_rnn_cell.py Scene02TheRNNCell

# Scene 03: Unrolling DAG (40s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene03_unrolling_dag.py Scene03UnrollingDAG

# Scene 04: Forward Pass (40s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene04_forward_pass.py Scene04ForwardPass

# Scene 05: Backpropagation Through Time (65s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene05_bptt_flow.py Scene05BPTTFlow

# Scene 06: Vanishing Gradients (65s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene06_vanishing_gradients.py Scene06VanishingGradients

# Scene 07: Exploding Gradients (34s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene07_exploding_gradients.py Scene07ExplodingGradients

# Scene 08: Long-Term Dependencies (30s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene08_long_term_dependencies.py Scene08LongTermDependencies

# Scene 09: Scalar RNN (30s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene09_scalar_rnn.py Scene09ScalarRNN

# Scene 10: Jacobian Matrix View (30s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene10_jacobian_matrix.py Scene10JacobianMatrix

# Scene 11: State-Space Interpretation (28s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene11_state_space.py Scene11StateSpace

# Scene 12: LSTM Architecture (29s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene12_lstm.py Scene12LSTM

# Scene 13: GRU Simplification (26s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene13_gru.py Scene13GRU

# Scene 14: Vanilla vs. Gated RNNs (27s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene14_why_gated_rnns_help.py Scene14WhyGatedRNNsHelp

# Scene 15: Final Synthesis (28s)
.venv\Scripts\python.exe -m manim -ql project/scenes/scene15_final_synthesis.py Scene15FinalSynthesis
```

---

## 3. Review Contact Sheet Generation

To extract 16 evenly spaced frames from a rendered scene and tile them into a $4 \times 4$ visual review contact sheet:

```powershell
# Example for Scene 07:
ffmpeg -y -i media/videos/scene07_exploding_gradients/480p15/Scene07ExplodingGradients.mp4 -vf "fps=16/34.4,scale=320:180,tile=4x4" project/review/scene07_contact.png
```

---

## 4. Master Video Assembly (FFmpeg Concatenation)

### Step 4.1: Concat Manifest File

The manifest `renders/concat_list.txt` specifies all 15 scene video paths:

```text
file 'D:/Project/Manim-videos/media/videos/scene01_why_rnns_exist/480p15/Scene01WhyRNNsExist.mp4'
file 'D:/Project/Manim-videos/media/videos/scene02_the_rnn_cell/480p15/Scene02TheRNNCell.mp4'
file 'D:/Project/Manim-videos/media/videos/scene03_unrolling_dag/480p15/Scene03UnrollingDAG.mp4'
file 'D:/Project/Manim-videos/media/videos/scene04_forward_pass/480p15/Scene04ForwardPass.mp4'
file 'D:/Project/Manim-videos/media/videos/scene05_bptt_flow/480p15/Scene05BPTTFlow.mp4'
file 'D:/Project/Manim-videos/media/videos/scene06_vanishing_gradients/480p15/Scene06VanishingGradients.mp4'
file 'D:/Project/Manim-videos/media/videos/scene07_exploding_gradients/480p15/Scene07ExplodingGradients.mp4'
file 'D:/Project/Manim-videos/media/videos/scene08_long_term_dependencies/480p15/Scene08LongTermDependencies.mp4'
file 'D:/Project/Manim-videos/media/videos/scene09_scalar_rnn/480p15/Scene09ScalarRNN.mp4'
file 'D:/Project/Manim-videos/media/videos/scene10_jacobian_matrix/480p15/Scene10JacobianMatrix.mp4'
file 'D:/Project/Manim-videos/media/videos/scene11_state_space/480p15/Scene11StateSpace.mp4'
file 'D:/Project/Manim-videos/media/videos/scene12_lstm/480p15/Scene12LSTM.mp4'
file 'D:/Project/Manim-videos/media/videos/scene13_gru/480p15/Scene13GRU.mp4'
file 'D:/Project/Manim-videos/media/videos/scene14_why_gated_rnns_help/480p15/Scene14WhyGatedRNNsHelp.mp4'
file 'D:/Project/Manim-videos/media/videos/scene15_final_synthesis/480p15/Scene15FinalSynthesis.mp4'
```

### Step 4.2: Concat Command

```powershell
ffmpeg -y -f concat -safe 0 -i renders/concat_list.txt -c copy renders/final/rnn_explainer_full.mp4
```

### Step 4.3: Probe Output Verification

```powershell
ffprobe -v error -show_entries format=duration,size,bit_rate -show_entries stream=width,height,r_frame_rate,codec_name -of default=noprint_wrappers=1 renders/final/rnn_explainer_full.mp4
```

Expected output:
- `codec_name=h264`
- `width=854`
- `height=480`
- `r_frame_rate=15/1`
- `duration=472.323242` (~7m 52s)
- `size=9124507` bytes (~8.7 MB)
