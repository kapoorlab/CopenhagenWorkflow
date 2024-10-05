# %%
from pathlib import Path 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from napatrackmater.Trackvector import (TrackVector,
                                        SHAPE_FEATURES, 
                                        DYNAMIC_FEATURES, 
                                        SHAPE_DYNAMIC_FEATURES,
                                        
                                        )


dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'membrane_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
save_dir = os.path.join(tracking_directory, f'cell_fate_accuracy/')
Path(save_dir).mkdir(exist_ok=True)
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
oneat_detections = f'/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/oneat_detections/non_maximal_oneat_mitosis_locations_{channel}timelapse_{dataset_name.lower()}_dataset.csv'

normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')

goblet_basal_radial_dataframe = os.path.join(data_frames_dir , f'train_test_goblet_basal_dataframe_normalized_{channel}.csv')

train_dataframe = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}.csv')
val_dataframe = os.path.join(data_frames_dir , f'val_goblet_basal_dataframe_normalized_{channel}.csv')


track_vectors = TrackVector(master_xml_path=xml_path)
track_vectors.oneat_csv_file = oneat_detections
track_vectors.oneat_threshold_cutoff = 0.9999
track_vectors.t_minus = 0
track_vectors.t_plus = track_vectors.tend
track_vectors.y_start = 0
track_vectors.y_end = track_vectors.ymax
track_vectors.x_start = 0
track_vectors.x_end = track_vectors.xmax


tracks_dataframe = pd.read_csv(normalized_dataframe)

if os.path.exists(goblet_basal_radial_dataframe):
    tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
else:
        track_vectors._interactive_function()
        tracks_goblet_basal_radial_dataframe = tracks_dataframe
        gt_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/manual_labeled_cells_sixth_dataset.csv'


        gt_cells_dataframe = pd.read_csv(gt_cells_file)
        gt_goblet_cells_dataframe = gt_cells_dataframe[gt_cells_dataframe['celltype_label'] == 'goblet'].copy()
        gt_basal_cells_dataframe = gt_cells_dataframe[gt_cells_dataframe['celltype_label'] == 'basal'].copy()
        gt_radial_cells_dataframe = gt_cells_dataframe[gt_cells_dataframe['celltype_label'] == 'radial'].copy()


        for d in [gt_goblet_cells_dataframe, gt_basal_cells_dataframe, gt_radial_cells_dataframe]:
            d.rename(columns={
                'Centroid.X': 'axis-2',
                'Centroid.Y': 'axis-1',
                'Centroid.Z': 'axis-0'
            }, inplace=True)


        gt_globlet_track_ids = track_vectors._get_trackmate_ids_by_location(gt_goblet_cells_dataframe)
        print(f'Total GT Trackmate IDs for globlet cells {len(gt_globlet_track_ids)}')
        gt_basal_track_ids = track_vectors._get_trackmate_ids_by_location(gt_basal_cells_dataframe)
        print(f'Total GT Trackmate IDs for basal cells {len(gt_basal_track_ids)}')
        gt_radial_track_ids = track_vectors._get_trackmate_ids_by_location(gt_radial_cells_dataframe)
        print(f'Total GT Trackmate IDs for radial cells {len(gt_radial_track_ids)}')

        goblet_df = pd.DataFrame({'TrackMate Track ID': gt_globlet_track_ids, 'Cell_Type': 'Goblet'})
        basal_df = pd.DataFrame({'TrackMate Track ID': gt_basal_track_ids, 'Cell_Type': 'Basal'})
        radial_df = pd.DataFrame({'TrackMate Track ID': gt_radial_track_ids, 'Cell_Type': 'Radial'})

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


print('Goblet', len(tracks_goblet_basal_radial_dataframe[tracks_goblet_basal_radial_dataframe['Cell_Type'] == 'Goblet']['TrackMate Track ID'].unique()))
print('Basal', len(tracks_goblet_basal_radial_dataframe[tracks_goblet_basal_radial_dataframe['Cell_Type'] == 'Basal']['TrackMate Track ID'].unique()))
print('Radial', len(tracks_goblet_basal_radial_dataframe[tracks_goblet_basal_radial_dataframe['Cell_Type'] == 'Radial']['TrackMate Track ID'].unique()))


tracks_goblet_basal_radial_dataframe_clean = tracks_goblet_basal_radial_dataframe.dropna(subset=['Cell_Type'])


goblet_ids = tracks_goblet_basal_radial_dataframe_clean[tracks_goblet_basal_radial_dataframe_clean['Cell_Type'] == 'Goblet']['TrackMate Track ID'].unique()
radial_ids = tracks_goblet_basal_radial_dataframe_clean[tracks_goblet_basal_radial_dataframe_clean['Cell_Type'] == 'Radial']['TrackMate Track ID'].unique()
basal_ids = tracks_goblet_basal_radial_dataframe_clean[tracks_goblet_basal_radial_dataframe_clean['Cell_Type'] == 'Basal']['TrackMate Track ID'].unique()

# Sample unique Track IDs
goblet_sample_ids = pd.Series(goblet_ids).sample(n=200, random_state=42).values
radial_sample_ids = pd.Series(radial_ids).sample(n=20, random_state=42).values
basal_sample_ids = pd.Series(basal_ids).sample(n=20, random_state=42).values

# Create test dataframe by selecting all rows corresponding to sampled IDs
goblet_sample = tracks_goblet_basal_radial_dataframe_clean[tracks_goblet_basal_radial_dataframe_clean['TrackMate Track ID'].isin(goblet_sample_ids)]
radial_sample = tracks_goblet_basal_radial_dataframe_clean[tracks_goblet_basal_radial_dataframe_clean['TrackMate Track ID'].isin(radial_sample_ids)]
basal_sample = tracks_goblet_basal_radial_dataframe_clean[tracks_goblet_basal_radial_dataframe_clean['TrackMate Track ID'].isin(basal_sample_ids)]

# Concatenate samples into test dataframe
test_dataframe = pd.concat([goblet_sample, radial_sample, basal_sample], ignore_index=True)

# Use the TrackMate Track IDs to create the remaining dataframe
remaining_dataframe = tracks_goblet_basal_radial_dataframe_clean[~tracks_goblet_basal_radial_dataframe_clean['TrackMate Track ID'].isin(test_dataframe['TrackMate Track ID'])]

# Save the dataframes
val_dataframe = os.path.join(data_frames_dir , f'val_goblet_basal_dataframe_normalized_{channel}.csv')
train_dataframe = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}.csv')

test_dataframe.to_csv(val_dataframe, index=False)
remaining_dataframe.to_csv(train_dataframe, index=False)

# Print summary of training and testing data
print('Going in training: ')
print('Goblet', len(remaining_dataframe[remaining_dataframe['Cell_Type'] == 'Goblet']['TrackMate Track ID'].unique()))
print('Basal', len(remaining_dataframe[remaining_dataframe['Cell_Type'] == 'Basal']['TrackMate Track ID'].unique()))
print('Radial', len(remaining_dataframe[remaining_dataframe['Cell_Type'] == 'Radial']['TrackMate Track ID'].unique()))

print('Separated for testing: ')
print('Goblet', len(test_dataframe[test_dataframe['Cell_Type'] == 'Goblet']['TrackMate Track ID'].unique()))
print('Basal', len(test_dataframe[test_dataframe['Cell_Type'] == 'Basal']['TrackMate Track ID'].unique()))
print('Radial', len(test_dataframe[test_dataframe['Cell_Type'] == 'Radial']['TrackMate Track ID'].unique()))
