## 🧠 Action Classification using Oneat

![Oneat Architecture](demoimages/FigS7.png)

### 🚀 Architecture Overview

- **Modified DenseNet Backbone**  
  Oneat extends a 3-stage DenseNet-style feature extractor. Each stage comprises dense blocks (BN → ReLU → Conv3D, repeated twice; growth rate = 8) and transition layers (1×1×1 conv + 2×2×2 avg-pool).  
  Instead of fully-connected layers, Oneat uses a **large 3D convolution** (encompassing multiple time and spatial voxels), making it a **fully convolutional network** that accepts inputs of arbitrary size.

- **🎯 Dual Detection Head**  
  - **Action Classification**  
    - Conv3D(1×1×3) → BatchNorm → ReLU → Conv3D(1×1×1) + Softmax (mitosis vs. non-mitosis)  
  - **Regression Output**  
    - Conv3D(1×1×3) → BatchNorm → ReLU → Conv3D(1×1×1) (event confidence / spatio-temporal co-ordinates)

- **🛠️ Post-processing**  
  Voxel-wise predictions are refined via non-maximal suppression and optional **MARI (Mitosis Angular Region of Interest)** filtering to yield final mitotic event coordinates.

---

### ⚙️ Oneat Mitosis Detector

- **Input:** 64×64×8 voxel crops over 3 time points, centered on each nucleus centroid.  
- **Training Samples:**  
  - **Positive:** Manual Napari clicks on mitotic nuclei  
  - **Negative:** Random crops from non-dividing nuclei  
- **Loss:** Binary cross-entropy for mitosis vs. non-mitosis classification.

Once trained, Oneat processes entire 4D stacks (T, Z, Y, X), outputting predicted mitosis coordinates (TZYX).

These coordinates feed into the **TrackMate-Oneat** plugin:

1. **Branch insertion:** Insert trajectory splits at predicted mitosis points.  
2. **Daughter linking:** Associate daughter cells within a 16.5 µm radius using a Jaqaman linker.  
3. **MARI filtering:** Optionally constrain daughter assignments to be perpendicular to the mother’s major axis, minimizing false positives.

**Performance:**  
Integrating Oneat reduces false branching by > 60 % compared to TrackMate’s native splitter (with MARI) while preserving high detection recall, yielding more biologically realistic lineage reconstructions.
"""