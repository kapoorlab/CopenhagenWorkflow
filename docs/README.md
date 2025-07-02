Workflow Website: https://kapoorlab.github.io/CopenhagenWorkflow/

# Copenhagen Tracking Workflow

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
     
      
      



# Workflow Steps

This workflow is for segmentation and tracking of Xenopus cells in 3D and contains the following steps in order:

1) Segmentation of nuclei and membrane channels, details in this [section](SEGMENTATION.md).
2) Converting the segmentation labels to point cloud representations for both the channels, details in this [section](POINTCLOUDS.md).
3) Tracking the nuclei channel and transfer the nuclei tracks to membrane channel, details in this [section](TRACKING.md).
4) Using cellshape autoencoder models/ marching cubes to compute additional shape and dynamic features, details in this [section](TRACKING.md).
5) Cell Fate computation using the computed shape and dynamic feature vectors, classification of trajectories into pre-defined cluster classes, obtaining the nearness score of cell tracks over time, obtaining statistics of mitotic trajectories in terms of the shape and dynamic feature vectors using this [notebook](10_show_global_dynamic_dataframe.ipynb) 



# File Structure
We do everything with hydra config files. The config files are partitioned into several scenarios: segmentation scenario, oneat prediction scenario, tracking scenario, cellshape training/ training oneat/ training segmentation scenario. Each scenario has pre set links to yaml files that contain the required model paths, dataset paths, parameters. As an example see our [scenario_track](conf/scenario_track.yaml), here the dataset to be tracked is chosen along with model paths and model parameters.  

You would have to modify the parent paths (mentioned in the scenario yaml files) according to your workstation/HPC paths & then the workflow described here should work flawless, without bugs and as described here.


## Experimental Data

Your experimental data folders have this structure:

- region_of_interest: contains 2D region of interest tha contains the sample
- membrane_timelapses: contains timelapse image for the membrane channel
- seg_membrane_timelapses: contains timelapse segmentation image for the membrane channel
- seg_nuclei_timelapses: contains timelapse segmenation image for the nuclei channel
- oneat_detections: csv file of oneat detected locations for mitosis
- split_nuclei_membrane_raw: dual channel images (CZYX) containing the nuclei and the membrane channel
- nuclei_timelapses: contains timelapse image for the nuclei channel
-  nuclei_membrane_tracking: directory containing hyperstack of both the nuclei and the membrane channel, csv file of mistosis locations, xml and csv files for both the channels coming from TrackMate and the master xml file created by NapaTrackMater.

Any new dataset that we get from you will be organized in the same structure as pointed above.

