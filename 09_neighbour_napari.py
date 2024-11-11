# %%
from pathlib import Path 
import os
import napari.viewer
import numpy as np
import pandas as pd
from tqdm import tqdm
from tifffile import imread
import matplotlib.pyplot as plt
import napari
import concurrent
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

cell_fate_img_path = os.path.join(tracking_directory, f'cell_fate_{channel}colored_segmentation/')
segmentation_img_path = os.path.join(cell_fate_img_path, f'{timelapse_to_track}_colored.tif')

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

segmentation_image = imread(segmentation_img_path)
print('Read Segmentation image')
viewer = napari.Viewer()
viewer.add_labels(segmentation_image,scale=[1, 1, 1, 1])
viewer.dims.ndisplay = 3
viewer.dims.set_current_step(0, 0)
print('Added image to Napari Viewer')


print(f'Reading bonds csv file')
time_points = sorted(neighbour_dataframe['t'].unique())
bonds_df = pd.read_csv(bonds_csv_path)
bond_persistence = (
        bonds_df.groupby(['Track ID', 'Neighbor Track ID'])['Time']
        .nunique()
        .reset_index(name='Persistence')
    )

persistent_bonds_df = bond_persistence[bond_persistence['Persistence'] >= partner_time]

max_persistence = persistent_bonds_df['Persistence'].max() if not persistent_bonds_df.empty else 1

print('bonds file read')
def plot_bonds_at_time(t):
    layer_name = f'Bonds at t={t}'
    if layer_name in [layer.name for layer in viewer.layers]:
        print(f"Layer for time {t} already exists, skipping computation.")
        return 
    lines = []
    colors = []
    time_df = tracks_goblet_basal_radial_dataframe[tracks_goblet_basal_radial_dataframe['t'] == t]
    bonds_at_time = bonds_df[bonds_df['Time'] == t]

    for _, row in tqdm(bonds_at_time.iterrows(), desc=f'Computing bonds view {t}'):
        track_id = row['Track ID']
        neighbor_id = row['Neighbor Track ID']

        bond_persist_row = persistent_bonds_df[
            (persistent_bonds_df['Track ID'] == track_id) &
            (persistent_bonds_df['Neighbor Track ID'] == neighbor_id)
        ]

        if bond_persist_row.empty:
            continue

        persistence = bond_persist_row['Persistence'].values[0]
        cell_coords = time_df[time_df['Track ID'] == track_id][['z', 'y', 'x']].values
        neighbor_coords = time_df[time_df['Track ID'] == neighbor_id][['z', 'y', 'x']].values
        if cell_coords.size == 0 or neighbor_coords.size == 0:
            continue

        persistence_norm = persistence / max_persistence if max_persistence > 0 else 0
        bond_color = plt.cm.coolwarm(persistence_norm)
        colors.append(bond_color[:3])

        for cell, neighbor in zip(cell_coords, neighbor_coords):
            print('cell',cell)
            print('neighbor', neighbor)
            line = np.array([cell, neighbor])
            lines.append(line)

    if lines:
        viewer.add_shapes(
            lines,
            shape_type='line',
            edge_color=np.array(colors),
            edge_width=1,
            name=layer_name
        )
        for layer in viewer.layers:
           layer.visible = (layer.name == layer_name)
           if isinstance(layer, napari.layers.Labels):
               layer.visible = True   


def parallel_plot_bonds():
    with concurrent.futures.ThreadPoolExecutor(os.cpu_count() - 1) as executor:
        executor.map(plot_bonds_at_time, time_points)

# Call the function to compute bonds in parallel
parallel_plot_bonds()

def update_view(event):
        print("The number of dims shown is now:", event.value)
        t = time_points[viewer.dims.current_step[0]] 
        plot_bonds_at_time(t)
    
viewer.dims.events.connect(update_view)

napari.run()


