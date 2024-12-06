# %%
# %%
from napatrackmater.Trackvector import TrackVector
from pathlib import Path 
import os

from scipy.stats import norm
from napatrackmater.Trackvector import (SHAPE_FEATURES, 
                                        DYNAMIC_FEATURES, 
                                        SHAPE_DYNAMIC_FEATURES,
                                        normalize_list
                                        )
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# %%
dataset_name = 'Fifth_Extra_Radial'
base_dataset_name = 'Fifth'
cell_type = 'Radial'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{base_dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}/nuclei_membrane_tracking/'
channel = 'membrane_'

master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
oneat_detections = f'{home_folder}Mari_Data_Oneat/Mari_{base_dataset_name}_Dataset_Analysis/oneat_detections/non_maximal_oneat_mitosis_locations_{channel}timelapse_{base_dataset_name.lower()}_dataset.csv'


save_dir = os.path.join(tracking_directory, f'distribution_plots/')
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

Path(save_dir).mkdir(exist_ok=True, parents=True) 
Path(data_frames_dir).mkdir(exist_ok=True, parents=True) 

save_file_normalized = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}.csv')
save_file_unnormalized = os.path.join(data_frames_dir , f'goblet_basal_dataframe_{channel}.csv')  

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


track_vectors._interactive_function()
global_shape_dynamic_dataframe = track_vectors.get_shape_dynamic_feature_dataframe()
global_shape_dynamic_dataframe['Cell_Type'] = cell_type
copy_dataframe = global_shape_dynamic_dataframe.copy(deep = True)
global_shape_dynamic_dataframe.to_csv(save_file_unnormalized)
normalized_global_shape_dynamic_dataframe = copy_dataframe    
for col in feature_cols:
                normalized_global_shape_dynamic_dataframe[col] = normalize_list(copy_dataframe[col])
normalized_global_shape_dynamic_dataframe.to_csv(save_file_normalized)

# %%
