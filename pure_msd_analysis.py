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
                                        populate_zero_gen_tracklets, 
                                        get_zero_gen_daughter_generations,
                                        populate_daughter_tracklets,
                                        plot_at_mitosis_time, 
                                        plot_histograms_for_groups,
                                        create_video,
                                        normalize_list
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

save_dir = os.path.join(tracking_directory, f'msd_plots_{channel}/')
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

Path(save_dir).mkdir(exist_ok=True, parents=True) 


dataframe_file = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}predicted.csv')
 

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


cell_types = tracks_goblet_basal_radial_dataframe['Cell_Type'].unique()

for cell_type in cell_types:
    filtered_tracks = tracks_goblet_basal_radial_dataframe[tracks_goblet_basal_radial_dataframe['Cell_Type'] == cell_type]
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=filtered_tracks, x='t', y='MSD', hue='Cell_Type', alpha=0.7)
    plt.title('Mean Square Displacement (MSD) over Time by Cell Type')
    plt.xlabel('Time')
    plt.ylabel('Mean Square Displacement (MSD)')
    plt.legend(title='Cell Type')
    plt.savefig(os.path.join(save_dir, f'MSD_Cell_Type_{cell_type}'))
    plt.tight_layout()

