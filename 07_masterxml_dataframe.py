# %%
# %%
from napatrackmater.Trackvector import TrackVector
from pathlib import Path 
import os
import numpy as np
import pandas as pd
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
dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'

master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
oneat_detections = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/oneat_detections/non_maximal_oneat_mitosis_locations_{channel}timelapse_{dataset_name.lower()}_dataset.csv'


save_dir = os.path.join(tracking_directory, f'distribution_plots/')
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

Path(save_dir).mkdir(exist_ok=True, parents=True) 
Path(data_frames_dir).mkdir(exist_ok=True, parents=True) 

save_file_normalized = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
save_file_unnormalized = os.path.join(data_frames_dir , f'results_dataframe_{channel}.csv')  
save_file = os.path.join(data_frames_dir , f'results_dataframe_{channel}.csv')   

block_size = 100
overlap = 50
plot_at_mitosis_time_distribution = False
verbose_generation_plots = False
shape_cols = SHAPE_FEATURES
dynamic_cols = DYNAMIC_FEATURES
feature_cols = SHAPE_DYNAMIC_FEATURES


# %%
track_vectors = TrackVector(master_xml_path=xml_path)
track_vectors.oneat_csv_file = oneat_detections
track_vectors.oneat_threshold_cutoff = 0.9999
track_vectors.t_minus = 0
track_vectors.t_plus = track_vectors.tend
track_vectors.y_start = 0
track_vectors.y_end = track_vectors.ymax
track_vectors.x_start = 0
track_vectors.x_end = track_vectors.xmax
if plot_at_mitosis_time_distribution:
  track_vectors._interactive_function()

if os.path.exists(save_file_unnormalized):
    print(f'reading data from {save_file_unnormalized}')
    global_shape_dynamic_dataframe = pd.read_csv(save_file_unnormalized)
    
else:    
    print('no saved un-normalized dataframe found, computing ...')
    track_vectors._interactive_function()
    global_shape_dynamic_dataframe = track_vectors.get_shape_dynamic_feature_dataframe()
    copy_dataframe = global_shape_dynamic_dataframe.copy(deep = True)
    global_shape_dynamic_dataframe.to_csv(save_file_unnormalized)
if os.path.exists(save_file_normalized):
    print(f'reading data from {save_file_normalized}')      
    normalized_global_shape_dynamic_dataframe = pd.read_csv(save_file_normalized)
else:
    normalized_global_shape_dynamic_dataframe = copy_dataframe    
    for col in feature_cols:
                 normalized_global_shape_dynamic_dataframe[col] = normalize_list(copy_dataframe[col])
    normalized_global_shape_dynamic_dataframe.to_csv(save_file_normalized)

# %%

if not os.path.exists(save_file_unnormalized):
    global_shape_dynamic_dataframe.to_csv(save_file_unnormalized)

if not os.path.exists(save_file_normalized):
    normalized_global_shape_dynamic_dataframe.to_csv(save_file_normalized)
