 
import os
import pandas as pd

dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')


goblet_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_attention_morpho_feature_attention_shallowest_litest_nuclei_balanced_nuclei_morpho_dynamic_balanced/goblet_cells_nuclei_annotations_inception.csv'
basal_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_attention_morpho_feature_attention_shallowest_litest_nuclei_balanced_nuclei_morpho_dynamic_balanced/basal_cells_nuclei_annotations_inception.csv'
radial_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_attention_morpho_feature_attention_shallowest_litest_nuclei_balanced_nuclei_morpho_dynamic_balanced/radially_intercalating_cells_nuclei_annotations_inception.csv'


goblet_cells_dataframe = pd.read_csv(goblet_cells_file)
basal_cells_dataframe = pd.read_csv(basal_cells_file)
radial_cells_dataframe = pd.read_csv(radial_cells_file)

normalized_dataframe_file = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
goblet_basal_radial_dataframe_file = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_attention_shallowest_litest_nuclei_balanced_nuclei_morpho_dynamic_balanced.csv')



print(f'reading data from {normalized_dataframe_file}')
tracks_dataframe = pd.read_csv(normalized_dataframe_file)
tracks_goblet_basal_radial_dataframe = tracks_dataframe
globlet_track_ids = goblet_cells_dataframe['TrackMate Track ID']
print(f'Total Trackmate IDs for globlet cells {len(globlet_track_ids)}')
basal_track_ids = basal_cells_dataframe['TrackMate Track ID']
print(f'Total Trackmate IDs for basal cells {len(basal_track_ids)}')
radial_track_ids = radial_cells_dataframe['TrackMate Track ID']
print(f'Total Trackmate IDs for radial cells {len(radial_track_ids)}')
goblet_df = pd.DataFrame({'TrackMate Track ID': globlet_track_ids, 'Cell_Type': 'Goblet'})
basal_df = pd.DataFrame({'TrackMate Track ID': basal_track_ids, 'Cell_Type': 'Basal'})
radial_df = pd.DataFrame({'TrackMate Track ID': radial_track_ids, 'Cell_Type': 'Radial'})

basal_radial_dataframe = pd.concat([goblet_df, basal_df, radial_df], ignore_index=True)
basal_radial_dataframe['TrackMate Track ID'] = basal_radial_dataframe['TrackMate Track ID'].astype(str)
tracks_goblet_basal_radial_dataframe['TrackMate Track ID'] = tracks_goblet_basal_radial_dataframe['TrackMate Track ID'].astype(str)

merged_dataframe = tracks_goblet_basal_radial_dataframe.merge(basal_radial_dataframe, on='TrackMate Track ID', how='left')

merged_dataframe.to_csv(goblet_basal_radial_dataframe_file, index=False)

     
