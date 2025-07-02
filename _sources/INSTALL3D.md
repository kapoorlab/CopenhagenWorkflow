"""## 🐍 Creating a Virtual Environment

Download the Anaconda package for your OS. The instructions below are tailored for Windows users but can also be applied on Mac or Linux. Once Anaconda is installed, open an Anaconda Prompt or a Windows PowerShell. If the installation was system-wide, you should see `(base)` before your command prompt.

To create a new conda environment (skip this step if you already have one for this project):

```bash
conda create -n analysisenv python=3.10
```

![Step 1: Create conda environment](demoimages/1_conda_install.png)

After pressing Enter, you should see output similar to:

![Step 2: Environment creation output](demoimages/2_conda_install.png)

Activate your new environment:

```bash
conda activate analysisenv
```

![Step 3: Activate environment](demoimages/3_conda_install.png)

---

## 🛠️ Installing Mamba

Mamba is a fast package manager that helps resolve dependencies efficiently. Install it into your environment:

```bash
conda install mamba -c conda-forge
```

![Step 4: Install mamba](demoimages/4_conda_install.png)

---

## 📦 Installing Main Packages

Install the core packages required for analysis:

```bash
pip install caped-ai ultralytics
```

![Step 5: Install core packages](demoimages/5_conda_install.png)

Wait for the installation to complete:

![Step 6: Packages downloading](demoimages/6_conda_install.png)
![Step 7: Packages installed](demoimages/7_conda_install.png)

---

## 🚀 Installing CUDA Toolkit

Even if your HPC has system-wide CUDA, it's recommended to include it in your environment:

```bash
mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
```

![Step 8: CUDA toolkit install](demoimages/8_conda_install.png)

Then install the NVIDIA compiler:

```bash
mamba install -c nvidia cuda-nvcc=11.3.58
```

![Step 9: NVIDIA nvcc install](demoimages/9_conda_install.png)

---

## 🔒 Locking TensorFlow & NumPy Versions

To ensure compatibility with our models (trained on TensorFlow 2.x):

```bash
python3 -m pip install tensorflow-gpu==2.10.*
pip uninstall numpy
pip install numpy==1.26.4
```

---

## ✅ Successful Installation

Verify everything by entering a Python REPL:

```bash
python
```

Then run:

```python
import vollseg, oneat, csbdeep, numpy, tensorflow, torch, lightning, ultralytics
```

You should see no errors, similar to:

![Step 10: Verification output](demoimages/10_conda_install.png)

Exit the shell:

```bash
quit()
```

Congratulations! Your environment is now ready to run all our scripts, notebooks, and Napari plugins.
"""