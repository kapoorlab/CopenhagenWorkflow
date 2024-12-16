# %%
# %%
from pathlib import Path 
import os
import pandas as pd

from napatrackmater.Trackvector import TrackVector
                                        
                                        

# %%
dataset_name = 'Fifth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'membrane_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
oneat_detections = f'/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/oneat_detections/non_maximal_oneat_mitosis_locations_{channel}timelapse_{dataset_name.lower()}_dataset.csv'


goblet_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_corrected/goblet_cells_nuclei_annotations_inception.csv'
basal_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_corrected/basal_cells_nuclei_annotations_inception.csv'
radial_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_corrected/radially_intercalating_cells_nuclei_annotations_inception.csv'


goblet_cells_dataframe = pd.read_csv(goblet_cells_file)
basal_cells_dataframe = pd.read_csv(basal_cells_file)
radial_cells_dataframe = pd.read_csv(radial_cells_file)

normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
goblet_basal_radial_dataframe = os.path.join(data_frames_dir , f'goblet_basal_dataframe_{channel}.csv')





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

print(f'reading data from {normalized_dataframe}')
tracks_dataframe = pd.read_csv(normalized_dataframe)
track_vectors._interactive_function()
tracks_goblet_basal_radial_dataframe = tracks_dataframe
globlet_track_ids = track_vectors._get_trackmate_ids_by_location(goblet_cells_dataframe)
print(f'Total Trackmate IDs for globlet cells {len(globlet_track_ids)}')
basal_track_ids = track_vectors._get_trackmate_ids_by_location(basal_cells_dataframe)
print(f'Total Trackmate IDs for basal cells {len(basal_track_ids)}')
radial_track_ids = track_vectors._get_trackmate_ids_by_location(radial_cells_dataframe)
print(f'Total Trackmate IDs for radial cells {len(radial_track_ids)}')
goblet_df = pd.DataFrame({'TrackMate Track ID': globlet_track_ids, 'Cell_Type': 'Goblet'})
basal_df = pd.DataFrame({'TrackMate Track ID': basal_track_ids, 'Cell_Type': 'Basal'})
radial_df = pd.DataFrame({'TrackMate Track ID': radial_track_ids, 'Cell_Type': 'Radial'})

basal_radial_dataframe = pd.concat([goblet_df, basal_df, radial_df], ignore_index=True)
basal_radial_dataframe['TrackMate Track ID'] = basal_radial_dataframe['TrackMate Track ID'].astype(str)
tracks_goblet_basal_radial_dataframe['TrackMate Track ID'] = tracks_goblet_basal_radial_dataframe['TrackMate Track ID'].astype(str)

merged_dataframe = tracks_goblet_basal_radial_dataframe.merge(basal_radial_dataframe, on='TrackMate Track ID', how='left')

merged_dataframe.to_csv(goblet_basal_radial_dataframe, index=False)


