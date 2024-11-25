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

dataset_name = 'Sixth'
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


def compute_bonds(df, radius_xy):
    bonds = defaultdict(lambda: defaultdict(list))
    unique_time_points = sorted(df['t'].unique())
    trackmate_ids = df['TrackMate Track ID'].unique()

    def process_trackmate_id(trackmate_id):
        local_bonds = defaultdict(lambda: defaultdict(list))
        
        for track_id in df[df['TrackMate Track ID'] == trackmate_id]['Track ID'].unique():

            for time_point in unique_time_points:
                time_df = df[df['t'] == time_point]
                
                if time_df.empty:
                    continue
                current_track_df = time_df[time_df['Track ID'] == track_id]
                if not current_track_df.empty:
                    current_coords = current_track_df.iloc[0][['z', 'y', 'x']].values

                    distances = np.sqrt((time_df['y'] - current_coords[1])**2 +
                                        (time_df['x'] - current_coords[2])**2)
                    current_neighbors = set(time_df[(distances <= radius_xy) & 
                                            
                                            (time_df['Track ID'] != track_id)]['Track ID'])

                    for neighbor_id in current_neighbors:
                        local_bonds[track_id][time_point].append(neighbor_id)
                    
                    

        return local_bonds


    for trackmate_id in tqdm(trackmate_ids,  desc="Computing Bonds"):    
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
    bonds_at_time = bonds_df[bonds_df['Time'] == time_point]

    total_bonds = bonds_at_time[['Track ID', 'Neighbor Track ID']].drop_duplicates().shape[0]
    
    return total_bonds



if  os.path.exists(bonds_csv_path):
    print("Loading bonds and bond_durations from CSV files.")
    bonds_df = pd.read_csv(bonds_csv_path)
else:
    print("Calculating bonds and bond_durations.")
    bonds = compute_bonds(neighbour_dataframe, neighbour_radius_xy)
    
    bonds_df = pd.DataFrame(
        [(track_id, time, neighbor_id) for track_id, time_dict in bonds.items() for time, neighbors in time_dict.items() for neighbor_id in neighbors],
        columns=['Track ID', 'Time', 'Neighbor Track ID']
    )
    bonds_df.to_csv(bonds_csv_path, index=False)
    print(f"Bond data saved to {bonds_csv_path}")

 


time_points = sorted(neighbour_dataframe['t'].unique())

def plot_bonds_spatially(df, bonds_df, color_palette, save_dir, time_points, partner_time):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    bond_persistence = (
        bonds_df.groupby(['Track ID', 'Neighbor Track ID'])['Time']
        .nunique()
        .reset_index(name='Persistence')
    )

    persistent_bonds_df = bond_persistence[bond_persistence['Persistence'] >= partner_time]

    max_persistence = persistent_bonds_df['Persistence'].max() if not persistent_bonds_df.empty else 1

    for t in tqdm(time_points, desc='Plotting Bonds Spatially'):
        fig, ax = plt.subplots(figsize=(18, 15))
        time_df = df[df['t'] == t]

        for cell_type, color in color_palette.items():
            cell_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_df['x'], cell_df['y'], color=color, label=cell_type, s=100, alpha=0.7)

        bonds_at_time = bonds_df[bonds_df['Time'] == t]

        for _, row in bonds_at_time.iterrows():
            track_id = row['Track ID']
            neighbor_id = row['Neighbor Track ID']

            bond_persist_row = persistent_bonds_df[
                (persistent_bonds_df['Track ID'] == track_id) &
                (persistent_bonds_df['Neighbor Track ID'] == neighbor_id)
            ]

            if bond_persist_row.empty:
                continue

            persistence = bond_persist_row['Persistence'].values[0]
 
            cell_coords = time_df[time_df['Track ID'] == track_id][['x', 'y']].values
            neighbor_coords = time_df[time_df['Track ID'] == neighbor_id][['x', 'y']].values

            if cell_coords.size == 0 or neighbor_coords.size == 0:
                continue
            persistence_norm = persistence / max_persistence if max_persistence > 0 else 0 
            bond_color = plt.cm.coolwarm(persistence_norm)

            for cell, neighbor in zip(cell_coords, neighbor_coords):

                ax.plot([cell[0], neighbor[0]], 
                        [cell[1], neighbor[1]], 
                        color=bond_color, linewidth=3)

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




def plot_persistent_times_over_time(bonds_df, neighbour_dataframe, color_palette, save_dir):
    """
    Plots the persistent times for each cell type over time and saves the plot as PNGs.

    Args:
        bonds_df (pd.DataFrame): DataFrame containing bond information.
        neighbour_dataframe (pd.DataFrame): DataFrame with cell type information.
        color_palette (dict): Dictionary mapping cell types to colors.
        save_dir (str): Directory to save the plots.
        time_points (list): List of time points for the dataset.
        partner_time (int): Minimum persistence duration to consider.
    """
    # Compute bond persistence
    bond_persistence = (
        bonds_df.groupby(['Track ID', 'Neighbor Track ID'])['Time']
        .nunique()
        .reset_index(name='Persistence')
    )

    # Filter for bonds persisting for at least `partner_time`
    persistent_bonds_df = bond_persistence[bond_persistence['Persistence'] >= partner_time]

    # Merge persistence with cell type data
    merged_df = persistent_bonds_df.merge(
        neighbour_dataframe[['Track ID', 'Cell_Type', 't']],
        how='left',
        left_on='Track ID',
        right_on='Track ID'
    )
    
    # Aggregate persistence data over time for each cell type
    persistence_over_time = merged_df.groupby(['t', 'Cell_Type'])['Persistence'].mean().reset_index()

    # Plot persistence over time for each cell type
    plt.figure(figsize=(12, 8))
    for cell_type, color in color_palette.items():
        cell_data = persistence_over_time[persistence_over_time['Cell_Type'] == cell_type]
        plt.plot(
            cell_data['t'],
            cell_data['Persistence'],
            label=cell_type,
            color=color,
            marker='o',
            linestyle='-'
        )
    
    # Add labels, legend, and title
    plt.title("Average Persistent Time of Bonds Over Time by Cell Type", fontsize=16)
    plt.xlabel("Time Point", fontsize=14)
    plt.ylabel("Average Persistence (Time Points)", fontsize=14)
    plt.legend(title="Cell Type", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Save the plot
    plot_path = os.path.join(save_dir, "persistent_time_over_time.png")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Persistent time over time plot saved at {plot_path}")


plot_bonds_spatially(neighbour_dataframe, bonds_df, color_palette, save_dir, time_points, partner_time)

plot_neighbour_time(neighbour_dataframe, bonds_df, color_palette, save_dir)

plot_persistent_times_over_time(bonds_df, neighbour_dataframe, color_palette, save_dir)
