Workflow Website: https://kapoorlab.github.io/CopenhagenWorkflow/

# The CopenhagenWorkflow

Developed by Mari Tolonen &amp; Varun Kapoor for Dr. Jakub Sedzinski's Lab.

## Installation


You can follow a step by step installation guide for installing the codes required for 3D cell tracking by following [these instrctions](INSTALL3D.md).


For quick installation you can also copy pase the lines below in a Powershell/Linux terminal:
   
      

      conda create -n capedenv python=3.10
      conda activate capedenv
      conda install mamba -c conda-forge
      pip install caped-ai ultralytics napari_fast4dreg
      mamba install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
      mamba install -c nvidia cuda-nvcc=11.3.58
      python3 -m pip install tensorflow-gpu==2.10.*
      pip uninstall numpy
      pip install numpy==1.26.4
     
      
      
