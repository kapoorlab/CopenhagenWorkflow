from napatrackmater.homology import  save_barcodes_and_stats, diagrams_over_time
from pathlib import Path 
import os
import matplotlib
matplotlib.use('Agg')
import pandas as pd


dataset_name = 'Fifth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
save_dir = os.path.join(tracking_directory, f'{channel}checkerboard')
Path(save_dir).mkdir(exist_ok=True)

print('Reading dataframe')
tracks_dataframe_path = os.path.join(data_frames_dir, f'results_dataframe_normalized_{channel}.csv')
tracks_dataframe = pd.read_csv(tracks_dataframe_path)

print(f'Computing diagrams over time')
diagrams = diagrams_over_time(tracks_dataframe, spatial_cols=('z','y','x'), max_dim=1)
save_barcodes_and_stats(
    diagrams_by_time=diagrams,
   
    save_dir=save_dir,
)