# %%
# %%
from pathlib import Path 
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
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
oneat_detections = f'/home/debian/jz/Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/oneat_detections/non_maximal_oneat_mitosis_locations_{channel}timelapse_{dataset_name.lower()}_dataset.csv'
     
goblet_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400_{channel}/goblet_cells_{channel}annotations_inception.csv'
basal_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400_{channel}/basal_cells_{channel}annotations_inception.csv'
radial_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400_{channel}/radially_intercalating_cells_{channel}annotations_inception.csv'


goblet_cells_dataframe = pd.read_csv(goblet_cells_file)
basal_cells_dataframe = pd.read_csv(basal_cells_file)
radial_cells_dataframe = pd.read_csv(radial_cells_file)

normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
goblet_basal_radial_dataframe = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}predicted.csv')

time_delta = 10
block_size = 100
overlap = 50
verbose_generation_plots = False
save_dir = os.path.join(tracking_directory, f'cell_type_clustering_plots_predicted_{channel}/')
distribution_dir = os.path.join(tracking_directory, f'cell_type_distribution_plots_predicted_{channel}/')
Path(save_dir).mkdir(exist_ok=True, parents=True)
Path(distribution_dir).mkdir(exist_ok=True, parents=True)
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
    print(f'Trackmate IDs for globlet cells {globlet_track_ids}')
    basal_track_ids = track_vectors._get_trackmate_ids_by_location(basal_cells_dataframe)
    print(f'Trackmate IDs for basal cells {basal_track_ids}')
    radial_track_ids = track_vectors._get_trackmate_ids_by_location(radial_cells_dataframe)
    print(f'Trackmate IDs for radial cells {radial_track_ids}')

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
print(correlation_dataframe.head())
print(correlation_dataframe.keys())

# %%
t_start = 0
t_delta = time_delta
t_total = max(cell_type_dataframe['t'])
starting_label_shape_dynamic = 0
starting_label_dynamic = 0
starting_label_shape = 0

copy_celltype_dataframe = cell_type_dataframe.copy()


local_cell_type_dataframe_dict = {}
all_new_columns = {}
for i in tqdm(np.arange(t_start, t_total, t_delta)):
                local_analysis_vectors, local_shape_dynamic_dataframe = create_analysis_cell_type_tracklets(copy_celltype_dataframe, t_minus=i, t_plus=i + t_delta, cell_type='Cell_Type_Label') 

               
                result =  convert_pseudo_tracks_to_simple_arrays(local_analysis_vectors,t_delta = t_delta)
                
               
                if result is not None:
                        ( local_shape_dynamic_covariance_2d,
                        local_shape_covariance_2d,
                        local_dynamic_covariance_2d, 
                        local_shape_dynamic_cluster_labels_dict, 
                        local_shape_cluster_labels_dict, 
                        local_dynamic_cluster_labels_dict,
                       
                        local_cluster_distance_map_shape_dynamic_dict,
                        local_cluster_distance_map_shape_dict,
                        local_cluster_distance_map_dynamic_dict,
                        local_cluster_eucledian_distance_map_shape_dynamic_dict,
                        local_cluster_eucledian_distance_map_shape_dict,
                        local_cluster_eucledian_distance_map_dynamic_dict,
                        local_analysis_track_ids) = result
                        local_shape_dynamic_dataframe = local_shape_dynamic_dataframe.copy()
                        all_new_columns.update({
                        f"Shape Dynamic Cluster_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_shape_dynamic_cluster_labels_dict),
                        f"Shape Cluster_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_shape_cluster_labels_dict),
                        f"Dynamic Cluster_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_dynamic_cluster_labels_dict),
                        f"Shape Dynamic CellType Distance_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_cluster_distance_map_shape_dynamic_dict),
                        f"Shape CellType Distance_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_cluster_distance_map_shape_dict),
                        f"Dynamic CellType Distance_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_cluster_distance_map_dynamic_dict),
                        f"Shape Dynamic CellType Eucledian Distance_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_cluster_eucledian_distance_map_shape_dynamic_dict),
                        f"Shape CellType Eucledian Distance_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_cluster_eucledian_distance_map_shape_dict),
                        f"Dynamic CellType Eucledian Distance_{i}": local_shape_dynamic_dataframe["Track ID"].map(local_cluster_eucledian_distance_map_dynamic_dict)
                        })
                        
                        
copy_celltype_dataframe = copy_celltype_dataframe.assign(**all_new_columns)

# %%
unique_col_names = ["Cluster_CellType_Distances", "Cluster_CellType_Eucledian_Distances", "Cluster_Label_Type"]
cluster_extended_dynamic_dataframe = create_cluster_plot(dataframe=copy_celltype_dataframe, cluster_type='Dynamic Cluster', cluster_distance_type='Dynamic CellType Distance',cluster_eucledian_distance_type='Dynamic CellType Eucledian Distance',  negate_cluster_distance_type='Shape Dynamic CellType Distance',negate_cluster_eucledian_distance_type='Shape Dynamic CellType Eucledian Distance',unique_col_names=unique_col_names )
cluster_extended_shape_dynamic_dataframe = create_cluster_plot(dataframe=copy_celltype_dataframe, cluster_type='Shape Dynamic Cluster', cluster_distance_type= 'Shape Dynamic CellType Distance', cluster_eucledian_distance_type= 'Shape Dynamic CellType Eucledian Distance',unique_col_names=unique_col_names)
cluster_extended_shape_dataframe = create_cluster_plot(dataframe=copy_celltype_dataframe, cluster_type='Shape Cluster',cluster_distance_type='Shape CellType Distance', cluster_eucledian_distance_type='Shape CellType Eucledian Distance', negate_cluster_distance_type='Shape Dynamic CellType Distance', negate_cluster_eucledian_distance_type='Shape Dynamic CellType Eucledian Distance',unique_col_names=unique_col_names)


# %%
print(cluster_extended_shape_dataframe.keys())


hue_options = ["Shape_Cluster_CellType_Distances"]
cluster_plots = ["MSD"] + SHAPE_DYNAMIC_FEATURES
shape_save_dir = os.path.join(save_dir, 'CellFate_Shape_Clustering/')
Path(shape_save_dir).mkdir(exist_ok=True, parents=True)
cell_type = list(map(int, cluster_extended_shape_dataframe["Shape_Cluster_Label_Type"].unique()))
key_list = [label_cell_type_mapping[value] for value in cell_type]

for index, cell_type_index in enumerate(cell_type):
        cell_type_label = key_list[index]
        for cluster_plot in cluster_plots:
                    filtered_tracks = cluster_extended_shape_dataframe[cluster_extended_shape_dataframe["Shape_Cluster_Label_Type"] == cell_type_index]
                    for hue_option in hue_options:  
                        plt.figure(figsize=(15, 6))
                        scatter = plt.scatter(filtered_tracks['t'],filtered_tracks[cluster_plot],c=filtered_tracks[hue_option],cmap='viridis')
                        plt.xlabel('Time (t)')
                        plt.ylabel(f'{cluster_plot}_{cell_type_label}')
                    
                        plt.colorbar(scatter, label = hue_option)
                        plt.title(f'Cell Type based clustering')
                        fig_name = f"{channel}{cluster_plot}_{hue_option}_{cell_type_label}_all.png"
                        plt.savefig(os.path.join(shape_save_dir, fig_name))

        for count, hue_option in enumerate(hue_options):    
                        unique_t_values = filtered_tracks['t'].unique()     
                        t_values = []
                        mean_update_column_values = []     
                        for t_value in unique_t_values:
                            filtered_data_t = filtered_tracks[filtered_tracks['t'] == t_value]
                            mean_update_column = filtered_data_t[hue_option].mean()
                            t_values.append(t_value)
                            mean_update_column_values.append(mean_update_column)    
                        plt.figure(figsize=(15, 6))
                        plt.plot(t_values, mean_update_column_values, marker='o')  
                        plt.xlabel('Time (t)')
                        plt.ylabel(f'Shape feature distance_{cell_type_label}')
                        plt.title(f'Pseudo_Shape feature {hue_option}')
                        fig_name = f"pseudo_{channel}{hue_option}_{cell_type_label}_line_plot_all.png"
                        plt.savefig(os.path.join(shape_save_dir, fig_name))






                


# %% [markdown]
# ### Lets get the distribution of shape and dynamic features for each cell type

# %%
t_max = correlation_dataframe['t'].max()
time_blocks = []
start = 0
while start < t_max:
    end = min(start + block_size, t_max)
    time_blocks.append((start, end))
    start += overlap
for start, end in time_blocks:
    cell_type_save_path = os.path.join(tracking_directory, f'cell_type_data_{start}_end_t_{end}/')
    Path(cell_type_save_path).mkdir(parents=True, exist_ok=True)
    sliced_df = correlation_dataframe[(correlation_dataframe['t'] >= start) & (correlation_dataframe['t'] < end)]

    track_vectors.plot_cell_type_times(sliced_df, save_path = cell_type_save_path)
    print(f'Start time {start}, End time {end}')
    name = f't_start_{start}_t_end_{end}'
    plot_histograms_for_cell_type_groups(cell_type_save_path, distribution_dir, dataset_name, channel, label_dict = label_cell_type_mapping, name = name, plot_show=False)


# %% [markdown]
# ### Cross Correlation Features

# %%
