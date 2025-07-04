
from pathlib import Path 
import os
import pandas as pd

from napatrackmater.Trackvector import (
                                        SHAPE_FEATURES, 
                                        DYNAMIC_FEATURES, 
                                        SHAPE_DYNAMIC_FEATURES,
                                        )

dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))

model_name = 'morphodynamic_features_mitosis_growth_rate_16'

mitosis_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/mitosis_predicted_attention_morphodynamic_feature_mitosis_25_growth_rate_16/Mitotic_inception.csv'
non_mitosis_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/mitosis_predicted_attention_morphodynamic_feature_mitosis_25_growth_rate_16/Non-Mitotic_inception.csv'


mitosis_cells_dataframe = pd.read_csv(mitosis_cells_file)
non_mitosis_cells_dataframe = pd.read_csv(non_mitosis_cells_file)

normalized_dataframe_file = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
mitosis_dataframe_file = os.path.join(data_frames_dir , f'mitosis_dataframe_normalized_{channel}.csv')



print(f'reading data from {normalized_dataframe_file}')
tracks_dataframe = pd.read_csv(normalized_dataframe_file)

tracks_mitosis_dataframe = tracks_dataframe
mitosis_track_ids = mitosis_cells_dataframe['TrackMate Track ID']
print(f'Total Trackmate IDs for mitosis cells {len(mitosis_track_ids)}')
non_mitosis_track_ids = non_mitosis_cells_dataframe['TrackMate Track ID']
print(f'Total Trackmate IDs for non mitosis cells {len(non_mitosis_track_ids)}')
mitosis_df = pd.DataFrame({'TrackMate Track ID': mitosis_track_ids, 'Cell_Type': 'Mitosis'})
non_mitosis_df = pd.DataFrame({'TrackMate Track ID': non_mitosis_track_ids, 'Cell_Type': 'Non Mitosis'})

mitosis_dataframe = pd.concat([mitosis_df, non_mitosis_df], ignore_index=True)
mitosis_dataframe['TrackMate Track ID'] = mitosis_dataframe['TrackMate Track ID'].astype(str)
tracks_mitosis_dataframe['TrackMate Track ID'] = tracks_mitosis_dataframe['TrackMate Track ID'].astype(str)

merged_dataframe = tracks_mitosis_dataframe.merge(mitosis_dataframe, on='TrackMate Track ID', how='left')

merged_dataframe.to_csv(mitosis_dataframe_file, index=False)



