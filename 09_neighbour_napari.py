# %%
from pathlib import Path 
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import imread
import matplotlib.pyplot as plt
import napari

# %%
dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, 'dataframes/')
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
goblet_basal_radial_dataframe = os.path.join(data_frames_dir, f'goblet_basal_dataframe_normalized_{channel}predicted_morpho_feature_attention_shallowest_litest.csv')
segmentation_img_path = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/seg_nuclei_timelapses/{timelapse_to_track}.tif'  

bonds_dir = os.path.join(tracking_directory, f'neighbour_plots_{channel}predicted_morpho_feature_attention_shallowest_litest/')
bonds_csv_path = os.path.join(bonds_dir, 'bonds.csv')
neighbour_radius_xy = 70 
partner_time = 20  
color_palette = {
    'Basal': '#1f77b4',  
    'Radial': '#ff7f0e',
    'Goblet': '#2ca02c',
}

tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
neighbour_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]
viewer = napari.Viewer()
segmentation_image = imread(segmentation_img_path)
viewer.add_labels(segmentation_image)


# %%
time_points = sorted(neighbour_dataframe['t'].unique())
bonds_df = pd.read_csv(bonds_csv_path)
bond_persistence = (
        bonds_df.groupby(['Track ID', 'Neighbor Track ID'])['Time']
        .nunique()
        .reset_index(name='Persistence')
    )

persistent_bonds_df = bond_persistence[bond_persistence['Persistence'] >= partner_time]
bond_persistence = (
        bonds_df.groupby(['Track ID', 'Neighbor Track ID'])['Time']
        .nunique()
        .reset_index(name='Persistence')
    )
persistent_bonds_df = bond_persistence[bond_persistence['Persistence'] >= partner_time]
max_persistence = persistent_bonds_df['Persistence'].max() if not persistent_bonds_df.empty else 1

for t in tqdm(time_points, desc='Plotting Bonds Spatially'):
    vectors = []
    colors = []
    time_df = tracks_goblet_basal_radial_dataframe[tracks_goblet_basal_radial_dataframe['t'] == t]
    bonds_at_time = bonds_df[bonds_df['Time'] == t]
    for _, row in bonds_at_time.iterrows():
            track_id = row['Track ID']
            neighbor_id = row['Neighbor Track ID']

            bond_persist_row = persistent_bonds_df[
                (persistent_bonds_df['Track ID'] == track_id) &
                (persistent_bonds_df['Neighbor Track ID'] == neighbor_id)
            ]

            if bond_persist_row.empty:
                # If no persistence data is found or it doesn't meet the threshold, skip
                continue

            # Get the persistence value for coloring
            persistence = bond_persist_row['Persistence'].values[0]
 
            # Get the coordinates for track and neighbor
            cell_coords = time_df[time_df['Track ID'] == track_id][['z', 'y', 'x']].values
            neighbor_coords = time_df[time_df['Track ID'] == neighbor_id][['z', 'y', 'x']].values
            persistence_norm = persistence / max_persistence if max_persistence > 0 else 0 
            bond_color = plt.cm.coolwarm(persistence_norm)
            colors.append(bond_color[:3])
            if cell_coords.size == 0 or neighbor_coords.size == 0:
                continue  # Skip if coordinates are missing

            # Add time dimension as the first coordinate
            cell_coords_4d = np.insert(cell_coords[0], 0, t)  # Add `t` as the first element of cell_coords
            neighbor_coords_4d = np.insert(neighbor_coords[0], 0, t)  # Add `t` as the first element of neighbor_coords

            # Convert to vectors format
            vector = np.array([cell_coords_4d, neighbor_coords_4d])  # Create a 4D vector from cell to neighbor
            vectors.append(vector)

            if vectors:
                viewer.add_vectors(
                    np.array(vectors),
                    edge_color=np.array(colors),
                    edge_width=1,
                    name=f'Bonds at t={t}'
                )
 
napari.run()


