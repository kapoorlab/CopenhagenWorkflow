# %%
# %%
from pathlib import Path 
import os
import pandas as pd

from napatrackmater.Trackvector import (TrackVector,
                                        SHAPE_FEATURES, 
                                        DYNAMIC_FEATURES, 
                                        SHAPE_DYNAMIC_FEATURES,
                                        
                                        )

# %%
dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))

model_name = 'morphodynamic_features_mitosis_gr32'

mitosis_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/mitosis_predicted_attention_morphodynamic_features_mitosis_gr32_morpho_dynamic/Mitotic_inception.csv'
non_mitosis_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/mitosis_predicted_attention_morphodynamic_features_mitosis_gr32_morpho_dynamic/Non-Mitotic_inception.csv'


mitosis_cells_dataframe = pd.read_csv(mitosis_cells_file)
non_mitosis_cells_dataframe = pd.read_csv(non_mitosis_cells_file)

normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
mitosis_dataframe = os.path.join(data_frames_dir , f'mitosis_dataframe_normalized_{channel}.csv')

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

track_vectors.t_minus = 0
track_vectors.t_plus = track_vectors.tend
track_vectors.y_start = 0
track_vectors.y_end = track_vectors.ymax
track_vectors.x_start = 0
track_vectors.x_end = track_vectors.xmax

print(f'reading data from {normalized_dataframe}')
tracks_dataframe = pd.read_csv(normalized_dataframe)

tracks_mitosis_dataframe = tracks_dataframe
mitosis_track_ids = mitosis_cells_dataframe['TrackMate Track ID']
print(f'Total Trackmate IDs for mitosis cells {len(mitosis_track_ids)}')
non_mitosis_track_ids = non_mitosis_cells_dataframe['TrackMate Track ID']
print(f'Total Trackmate IDs for basal cells {len(non_mitosis_track_ids)}')
mitosis_df = pd.DataFrame({'TrackMate Track ID': mitosis_track_ids, 'Cell_Type': 'Mitosis'})
non_mitosis_df = pd.DataFrame({'TrackMate Track ID': non_mitosis_track_ids, 'Cell_Type': 'Non Mitosis'})

tracks_mitosis_dataframe = pd.concat([mitosis_df, non_mitosis_df], ignore_index=True)
tracks_mitosis_dataframe['TrackMate Track ID'] = tracks_mitosis_dataframe['TrackMate Track ID'].astype(str)
tracks_mitosis_dataframe['TrackMate Track ID'] = tracks_mitosis_dataframe['TrackMate Track ID'].astype(str)


for index, row in tracks_mitosis_dataframe.iterrows():
            track_id = row['TrackMate Track ID']
            match_row = tracks_mitosis_dataframe[tracks_mitosis_dataframe['TrackMate Track ID'] == track_id]
            if not match_row.empty:
                cell_type = match_row.iloc[0]['Cell_Type']
                tracks_mitosis_dataframe.at[index, 'Cell_Type'] = cell_type

tracks_mitosis_dataframe.to_csv(mitosis_dataframe, index=False)
        


unique_cell_types = tracks_mitosis_dataframe[~tracks_mitosis_dataframe['Cell_Type'].isna()]['Cell_Type'].unique()

print("Number of unique cell types:", len(unique_cell_types))
print("Unique cell types:", unique_cell_types)
cell_type_label_mapping = {
    "Non Mitosis": 0,
    "Mitosis":1, 
}

print("Cell type counts, unique TrackMate Track IDs, and count of mitotic tracks:")
for cell_type in unique_cell_types:
    cell_type_df = tracks_mitosis_dataframe[tracks_mitosis_dataframe['Cell_Type'] == cell_type]
    unique_track_ids = cell_type_df['TrackMate Track ID'].unique()
    
    dividing_count = 0
    for track_id in unique_track_ids:
        track_df = cell_type_df[cell_type_df['TrackMate Track ID'] == track_id]
        if track_df['Mitosis'].iloc[0] == 1:
            dividing_count += 1
    
    count = len(cell_type_df)
    print(f"{cell_type}: {count} rows, unique TrackMate Track IDs: {len(unique_track_ids)}, mitotic tracks: {dividing_count}")

cell_type_dataframe = tracks_mitosis_dataframe[~tracks_mitosis_dataframe['Cell_Type'].isna()]
cell_type_dataframe['Cell_Type'].unique()
print(cell_type_label_mapping)
cell_type_dataframe.loc[:, 'Cell_Type_Label'] = cell_type_dataframe['Cell_Type'].map(cell_type_label_mapping)

correlation_dataframe = cell_type_dataframe.copy()

# %%
correlation_dataframe.head()