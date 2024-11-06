from pathlib import Path 
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from concurrent.futures import ThreadPoolExecutor, as_completed

dataset_name = 'Fifth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, 'dataframes/')
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
goblet_basal_radial_dataframe = os.path.join(data_frames_dir, f'goblet_basal_dataframe_normalized_{channel}predicted_morpho_feature_attention_shallowest_litest.csv')
save_dir = os.path.join(tracking_directory, f'neighbour_plots_{channel}predicted_morpho_feature_attention_shallowest_litest/')
Path(save_dir).mkdir(exist_ok=True, parents=True)
bonds_csv_path = os.path.join(save_dir, 'bonds.csv')
neighbour_radius_xy = 70 
partner_time = 20  
color_palette = {
    'Basal': '#1f77b4',  
    'Radial': '#ff7f0e',
    'Goblet': '#2ca02c',
}

tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
neighbour_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]


def compute_bond_breaks_and_bonds(df, radius_xy):
    bonds = defaultdict(lambda: defaultdict(list))
    unique_time_points = sorted(df['t'].unique())
    trackmate_ids = df['TrackMate Track ID'].unique()

    def process_trackmate_id(trackmate_id):
        """Processes bond breaks for individual tracklets within a TrackMate track ID."""
        local_bonds = defaultdict(lambda: defaultdict(list))
        
        # Loop through each unique track ID for the given TrackMate ID
        for track_id in df[df['TrackMate Track ID'] == trackmate_id]['Track ID'].unique():

            for time_point in unique_time_points:
                time_df = df[df['t'] == time_point]
                
                if time_df.empty:
                    continue
                current_track_df = time_df[time_df['Track ID'] == track_id]
                if not current_track_df.empty:
                    current_coords = current_track_df.iloc[0][['z', 'y', 'x']].values

                    # Calculate distances within the same tracklet for current neighbors
                    distances = np.sqrt((time_df['y'] - current_coords[1])**2 +
                                        (time_df['x'] - current_coords[2])**2)
                    current_neighbors = set(time_df[(distances <= radius_xy) & 
                                            
                                            (time_df['Track ID'] != track_id)]['Track ID'])

                    for neighbor_id in current_neighbors:
                        local_bonds[track_id][time_point].append(neighbor_id)
                    
                    

        return local_bonds


    for trackmate_id in tqdm(trackmate_ids,  desc="Computing Bond Breaks"):    
            local_bonds = process_trackmate_id(trackmate_id)
            
            for track_id, bond_times in local_bonds.items():
                for time, neighbors in bond_times.items():
                    bonds[track_id][time].extend(neighbors)

    return  bonds




def get_total_bonds_at_time(bonds_df, time_point):
    """
    Computes the total number of unique bonds at a given time point from bonds_df.
    
    Args:
        bonds_df (pd.DataFrame): DataFrame with columns ['TrackMate Track ID', 'Time', 'Neighbor TrackMate Track ID']
                                 representing bonds between trackmate IDs at each time point.
        time_point (int): The time point at which to calculate the total number of bonds.

    Returns:
        int: Total number of bonds at the specified time point.
    """
    # Filter the DataFrame for the specified time point
    bonds_at_time = bonds_df[bonds_df['Time'] == time_point]

    total_bonds = bonds_at_time[['Track ID', 'Neighbor Track ID']].drop_duplicates().shape[0]
    
    return total_bonds



if  os.path.exists(bonds_csv_path):
    print("Loading bonds and bond_durations from CSV files.")
    bond_breaks_df = pd.read_csv(bond_breaks_csv_path)
    bonds_df = pd.read_csv(bonds_csv_path)
else:
    print("Calculating bonds and bond_durations.")
    bonds = compute_bond_breaks_and_bonds(neighbour_dataframe, neighbour_radius_xy)
    
    bonds_df = pd.DataFrame(
        [(track_id, time, neighbor_id) for track_id, time_dict in bonds.items() for time, neighbors in time_dict.items() for neighbor_id in neighbors],
        columns=['Track ID', 'Time', 'Neighbor Track ID']
    )
    bonds_df.to_csv(bonds_csv_path, index=False)
    print(f"Bond breaks data saved to {bond_breaks_csv_path}")

 


time_points = sorted(neighbour_dataframe['t'].unique())

def plot_bonds_spatially(df, bonds_df, color_palette, save_dir, time_points, partner_time):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Compute bond persistence directly from bonds_df
    # Group by Track ID and Neighbor Track ID and calculate the number of unique time points for each bond
    bond_persistence = (
        bonds_df.groupby(['Track ID', 'Neighbor Track ID'])['Time']
        .nunique()
        .reset_index(name='Persistence')
    )

    # Filter bonds that persist for at least partner_time frames
    persistent_bonds_df = bond_persistence[bond_persistence['Persistence'] >= partner_time]

    # Find the maximum persistence for color normalization
    max_persistence = persistent_bonds_df['Persistence'].max() if not persistent_bonds_df.empty else 1

    for t in tqdm(time_points, desc='Plotting Bonds Spatially'):
        fig, ax = plt.subplots(figsize=(18, 15))
        time_df = df[df['t'] == t]

        # Plot each cell type with unique color
        for cell_type, color in color_palette.items():
            cell_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_df['x'], cell_df['y'], color=color, label=cell_type, s=100, alpha=0.7)

        # Filter the bonds for the current time point
        bonds_at_time = bonds_df[bonds_df['Time'] == t]

        # Plot each bond that has enough persistence
        for _, row in bonds_at_time.iterrows():
            track_id = row['Track ID']
            neighbor_id = row['Neighbor Track ID']

            # Check if the bond meets the persistence threshold
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
            cell_coords = time_df[time_df['Track ID'] == track_id][['x', 'y']].values
            neighbor_coords = time_df[time_df['Track ID'] == neighbor_id][['x', 'y']].values

            if cell_coords.size == 0 or neighbor_coords.size == 0:
                continue

            # Normalize persistence for color mapping
            persistence_norm = persistence 
            bond_color = plt.cm.coolwarm(persistence_norm)

            # Plot the bond with a line between track and neighbor
            ax.plot([cell_coords[0][0], neighbor_coords[0][0]], 
                    [cell_coords[0][1], neighbor_coords[0][1]], 
                    color=bond_color, linewidth=3)

        # Add color bar for bond persistence
        norm = mcolors.Normalize(vmin=0, vmax=max_persistence)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Bond Persistence (Time Points)')

        ax.set_title(f"Bonds at Time Point {t} (XY Plane)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'bonds_time_{t}.png'), dpi=300)
        plt.close(fig)


plot_bonds_spatially(neighbour_dataframe, bonds_df, color_palette, save_dir, time_points, partner_time)


def plot_neighbour_time(df, bonds_df, color_palette, save_dir):
    timepoints = sorted(df['t'].unique())
    cell_types = df['Cell_Type'].unique()

    neighbor_counts = {cell_type: {neighbor_type: [0] * len(timepoints) for neighbor_type in cell_types} for cell_type in cell_types}

    for time_idx, t in enumerate(tqdm(timepoints, desc='Neighbour Time')):
        time_df = df[df['t'] == t]

        bonds_at_time = bonds_df[bonds_df['Time'] == t]

        for _, row in bonds_at_time.iterrows():
            track_id, neighbor_id = row['Track ID'], row['Neighbor Track ID']

            cell_type_row = time_df[time_df['Track ID'] == track_id]
            neighbor_type_row = time_df[time_df['Track ID'] == neighbor_id]

            if not cell_type_row.empty and not neighbor_type_row.empty:
                cell_type = cell_type_row['Cell_Type'].iloc[0]
                neighbor_type = neighbor_type_row['Cell_Type'].iloc[0]

                neighbor_counts[cell_type][neighbor_type][time_idx] += 1

    fig, axs = plt.subplots(len(cell_types), 1, figsize=(16, len(cell_types) * 5), sharex=True)

    if len(cell_types) == 1:
        axs = [axs]  

    for idx, cell_type in enumerate(cell_types):
        ax = axs[idx]
        for neighbor_type in cell_types:
            y_values = neighbor_counts[cell_type][neighbor_type]
            ax.plot(timepoints, y_values, label=f'{neighbor_type} as Neighbor', color=color_palette.get(neighbor_type, 'grey'), marker='o')

        ax.set_title(f'Neighbor Counts Over Time for {cell_type} Cells')
        ax.set_xlabel('Timepoint')
        ax.set_ylabel('Neighbor Count')
        ax.legend(loc='upper right')
        ax.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'neighbor_counts_over_time.png'))
    plt.close(fig)

plot_neighbour_time(neighbour_dataframe, bonds_df, color_palette, save_dir)
