from napatrackmater.Trackvector import TrackVector
from pathlib import Path 
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from napatrackmater.Trackvector import (SHAPE_FEATURES, 
                                        DYNAMIC_FEATURES, 
                                        SHAPE_DYNAMIC_FEATURES,
                                        
                                        )
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# %%
dataset_name = 'Second'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'

master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))

save_dir = os.path.join(tracking_directory, f'msd_plots_{channel}predicted_morpho_feature_attention_shallowest_litest/')
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

Path(save_dir).mkdir(exist_ok=True, parents=True) 


dataframe_file = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}predicted_morpho_feature_attention_shallowest_litest.csv')
 

shape_cols = SHAPE_FEATURES
dynamic_cols = DYNAMIC_FEATURES
feature_cols = SHAPE_DYNAMIC_FEATURES



# %%
track_vectors = TrackVector(master_xml_path=xml_path)
track_vectors.t_minus = 0
track_vectors.t_plus = track_vectors.tend
track_vectors.y_start = 0
track_vectors.y_end = track_vectors.ymax
track_vectors.x_start = 0
track_vectors.x_end = track_vectors.xmax

tracks_goblet_basal_radial_dataframe = pd.read_csv(dataframe_file)
cell_type_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]

cell_types = cell_type_dataframe['Cell_Type'].unique()


for cell_type in cell_types:
    # Filter DataFrame by Cell Type
    filtered_tracks = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == cell_type]
    
    # Get unique Track IDs for this cell type
    track_ids = filtered_tracks['Track ID'].unique()
    
    # Create a new figure
    plt.figure(figsize=(12, 6))
    
    # Iterate over each Track ID and plot a line for each track
    for track_id in track_ids:
        # Filter the DataFrame for this specific track
        track_data = filtered_tracks[filtered_tracks['Track ID'] == track_id].copy()
        
        # Normalize the time for this track (set the first time point to t = 0)
        track_data['t_normalized'] = track_data['t'] - track_data['t'].min()
        
        # Plot the MSD values over the normalized time
        plt.plot(track_data['t_normalized'], track_data['MSD'], label=f'Track ID {track_id}')
    
    # Set plot titles and labels
    plt.title(f'Mean Square Displacement (MSD) over Normalized Time by Track ID for Cell Type {cell_type}')
    plt.xlabel('Normalized Time (t)')
    plt.ylabel('Mean Square Displacement (MSD)')
    
    # Save the plot as an image
    plt.savefig(os.path.join(save_dir, f'MSD_Cell_Type_{cell_type}.png'))
    
    # Adjust layout and display/save
    plt.tight_layout()
    plt.close()    
