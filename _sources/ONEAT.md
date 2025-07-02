## 🧠 Action Classification using Oneat

![Oneat Architecture](demoimages/FigS7.png)

### 🚀 Architecture Overview

Oneat’s core network, **DenseVollNet**, processes short 4D image crops by treating **time frames as input channels** and applying **only spatial convolutions (Z, Y, X)**:

- **Input:**  
  Patches of size `(Z, Y, X)` with **T timepoints folded as C = T channels** (shape `(Z, Y, X, T)`).

- **DenseVollNet Backbone:**  
  1. **Initial 3D Conv**  
     - `Conv3D` with `startfilter` filters, kernel `(k_z, k_y, k_x)` (e.g. `7×7×7`), `padding='same'`  
     - **BatchNorm → ReLU**  
  2. **Three Dense Block Stages**  
     - Each stage *i* has `depth_i` layers:  
       - **Bottleneck**: `Conv3D(1×1×1)`, reducing channels (`4·F`)  
       - **Feature**:    `Conv3D(mid_kernel, mid_kernel, mid_kernel)`, growth rate `F`  
       - **Concat** the new features with previous tensor  
     - **Transition layers** between stages (except after the last):  
       - `Conv3D(1×1×1)` to compress channels (`reduction` factor)  
       - `MaxPool3D(2×2×2)` to downsample spatial dims  
         - **Downsampling factor per pool:** 2  
         - **Total downsampling factor:** 4 (after two pools), e.g. `(8,64,64) → (2,16,16)`  
  3. **BN → ReLU** after final dense block

- **Fully-Convolutional Head:**  
  - A **single large `Conv3D`** (kernel `mid_kernel³`, padding='valid') replaces FC layers, outputting `categories + nboxes·box_vector` channels.  
    - Kernel size = `(Z/4, Y/4, X/4)` = `(2,16,16)` for input `(8,64,64)` and `last_conv_factor=4`.  
    - This **collapses** the spatial map to `1×1×1` per channel.  
  - **Split** these channels into:  
    - **Classification map** (`categories` channels) → Softmax  
    - **Regression map** (`nboxes·box_vector` channels) → Sigmoid  
  - **Concat** classification & regression outputs.

This design lets Oneat scan any `(Z, Y, X)` volume with **T** frames in one pass, yielding per-voxel mitosis predictions.

---

### ⚙️ Oneat Mitosis Detector

- **Input crops:** `64×64×8` voxels over `T` timepoints (folded into channels).  
- **Training samples:**  
  - **Positive:** Napari-clicked mitotic events  
  - **Negative:** Random non-dividing crops  
- **Loss:** Binary cross-entropy

After training, Oneat predicts mitosis coordinates `(t, z, y, x)` in whole 4D stacks, which the **TrackMate-Oneat** plugin uses to:

1. **Insert trajectory branches** at predicted mitoses  
2. **Link daughter cells** within a 16.5 µm radius (Jaqaman linker)  
3. **Optionally apply MARI** to enforce perpendicular daughter positioning, reducing false positives

**Performance:**  
Integrating Oneat cuts false branching by > 60 % (with MARI) vs. native TrackMate, while keeping high recall for biologically faithful lineage trees.


![TrackMate-Oneat Accuracy](demoimages/Fig1.png)