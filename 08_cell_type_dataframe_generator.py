# %%
# %%
from pathlib import Path 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from napatrackmater import  create_analysis_cell_type_tracklets, convert_pseudo_tracks_to_simple_arrays
from napatrackmater.Trackvector import (TrackVector,
                                        create_cluster_plot,
                                        cross_correlation_class,
                                        SHAPE_FEATURES, 
                                        DYNAMIC_FEATURES, 
                                        SHAPE_DYNAMIC_FEATURES,
                                        plot_histograms_for_cell_type_groups
                                        
                                        )

# %%
dataset_name = 'Second'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
oneat_detections = f'/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/oneat_detections/non_maximal_oneat_mitosis_locations_{channel}timelapse_{dataset_name.lower()}_dataset.csv'
     
goblet_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400_{channel}/goblet_cells_{channel}annotations_inception.csv'
basal_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400_{channel}/basal_cells_{channel}annotations_inception.csv'
radial_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400_{channel}/radially_intercalating_cells_{channel}annotations_inception.csv'


goblet_cells_dataframe = pd.read_csv(goblet_cells_file)
basal_cells_dataframe = pd.read_csv(basal_cells_file)
radial_cells_dataframe = pd.read_csv(radial_cells_file)

normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
goblet_basal_radial_dataframe = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}predicted.csv')

time_delta = 2
block_size = 100
overlap = 50
verbose_generation_plots = False
method="ward"
criterion="distance"
metric="euclidean" 

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

print(f'reading data from {normalized_dataframe}')
tracks_dataframe = pd.read_csv(normalized_dataframe)
if os.path.exists(goblet_basal_radial_dataframe):
    tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
else:

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


    for index, row in tracks_goblet_basal_radial_dataframe.iterrows():
            track_id = row['TrackMate Track ID']
            match_row = basal_radial_dataframe[basal_radial_dataframe['TrackMate Track ID'] == track_id]
            if not match_row.empty:
                cell_type = match_row.iloc[0]['Cell_Type']
                tracks_goblet_basal_radial_dataframe.at[index, 'Cell_Type'] = cell_type

    tracks_goblet_basal_radial_dataframe.to_csv(goblet_basal_radial_dataframe, index=False)
        


# %%
len(tracks_goblet_basal_radial_dataframe['TrackMate Track ID'].unique())

# %%
unique_cell_types = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]['Cell_Type'].unique()

print("Number of unique cell types:", len(unique_cell_types))
print("Unique cell types:", unique_cell_types)
cell_type_label_mapping = {
    "Basal": 1,
    "Radial":2, 
    "Goblet":3
}
label_cell_type_mapping = {v: k for k, v in cell_type_label_mapping.items()}
print("Cell type counts, unique TrackMate Track IDs, and count of mitotic tracks:")
for cell_type in unique_cell_types:
    cell_type_df = tracks_goblet_basal_radial_dataframe[tracks_goblet_basal_radial_dataframe['Cell_Type'] == cell_type]
    unique_track_ids = cell_type_df['TrackMate Track ID'].unique()
    
    dividing_count = 0
    for track_id in unique_track_ids:
        track_df = cell_type_df[cell_type_df['TrackMate Track ID'] == track_id]
        if track_df['Dividing'].iloc[0] == 1:
            dividing_count += 1
    
    count = len(cell_type_df)
    print(f"{cell_type}: {count} rows, unique TrackMate Track IDs: {len(unique_track_ids)}, mitotic tracks: {dividing_count}")

cell_type_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]
cell_type_dataframe['Cell_Type'].unique()
print(cell_type_label_mapping)
cell_type_dataframe.loc[:, 'Cell_Type_Label'] = cell_type_dataframe['Cell_Type'].map(cell_type_label_mapping)

correlation_dataframe = cell_type_dataframe.copy()

# %%
correlation_dataframe.head()
