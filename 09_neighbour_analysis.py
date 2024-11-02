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
bond_breaks_csv_path = os.path.join(save_dir, 'bond_breaks.csv')
neighbour_radius_xy = 70 
partner_time = 0  

color_palette = {
    'Basal': '#1f77b4',  
    'Radial': '#ff7f0e',
    'Goblet': '#2ca02c',
}

tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
neighbour_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]



def compute_bond_breaks(df, radius_xy, jump_time=1):
    """
    Computes bond breaks between consecutive time points, where a bond break is defined as a neighbor 
    that is present at the current time but not in the next time point.

    Args:
        df (pd.DataFrame): DataFrame containing tracking data with columns 'TrackMate Track ID', 't', 'x', 'y', 'z'.
        radius_xy (float): Distance threshold for considering a neighboring bond in the XY plane.
        jump_time (int): Time increment for checking bond breaks (default is 1 for consecutive time points).

    Returns:
        dict: Bond breaks with keys (trackmate_id, neighbor_id, time_point) and counts as values.
    """
    bond_breaks = defaultdict(int)
    total_bonds_at_time = defaultdict(int)
    unique_time_points = sorted(df['t'].unique())

    for trackmate_id in tqdm(df['TrackMate Track ID'].unique(), desc="Computing Bond Breaks"):
        for time_point in unique_time_points:
            # Get current neighbors
            current_df = df[(df['TrackMate Track ID'] == trackmate_id) & (df['t'] == time_point)]
            if current_df.empty:
                continue
            
            current_coords = current_df.iloc[0][['z', 'y', 'x']].values
            next_time = time_point + jump_time
            next_df = df[df['t'] == next_time]

            # Find current neighbors within radius
            distances = np.sqrt((next_df['y'] - current_coords[1])**2 + (next_df['x'] - current_coords[2])**2)
            current_neighbors = set(df[(distances <= radius_xy) & (df['TrackMate Track ID'] != trackmate_id)]['TrackMate Track ID'])
            total_bonds_at_time[time_point] += len(current_neighbors)
            # Find next neighbors within radius
            next_neighbors = set(next_df[(distances <= radius_xy) & (next_df['TrackMate Track ID'] != trackmate_id)]['TrackMate Track ID'])

            # Determine bond breaks by finding neighbors in current but not in the next time point
            broken_bonds = current_neighbors - next_neighbors
            for neighbor_id in broken_bonds:
                bond_breaks[(trackmate_id, neighbor_id, time_point)] += 1

    normalized_bond_breaks = {key: count / total_bonds_at_time[key[2]] for key, count in bond_breaks.items()}

    return normalized_bond_breaks




def plot_bond_breaks(df, bond_breaks_df, color_palette, save_dir, time_points):
    max_break_count = bond_breaks_df['Break Count'].max() if not bond_breaks_df.empty else 1

    for t in tqdm(time_points, desc='Plotting Bond Breaks'):
        fig, ax = plt.subplots(figsize=(18, 15))
        time_df = df[df['t'] == t]

        # Plot each cell type with unique color
        for cell_type, color in color_palette.items():
            cell_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_df['x'], cell_df['y'], color=color, label=cell_type, s=100, alpha=0.7)

        # Filter the bond breaks for the current time point
        bonds_at_time = bond_breaks_df[bond_breaks_df['Time'] == t]

        # Plot each bond with color based on break count
        for _, row in bonds_at_time.iterrows():
            trackmate_id = row['TrackMate Track ID']
            neighbor_id = row['Neighbor TrackMate Track ID']
            break_count = row['Break Count']

            cell_coords = time_df[time_df['TrackMate Track ID'] == trackmate_id][['x', 'y']].values
            neighbor_coords = time_df[time_df['TrackMate Track ID'] == neighbor_id][['x', 'y']].values

            if cell_coords.size == 0 or neighbor_coords.size == 0:
                continue

            # Set bond color based on break count
            bond_color = cm.get_cmap("coolwarm")(break_count / max_break_count)
            ax.plot([cell_coords[0][0], neighbor_coords[0][0]], [cell_coords[0][1], neighbor_coords[0][1]], color=bond_color, linewidth=3)

        # Add color bar for bond breaks
        norm = mcolors.Normalize(vmin=0, vmax=max_break_count)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Bond Break Count')

        ax.set_title(f"Bond Breaks at Time Point {t} (XY Plane)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'bond_breaks_time_{t}.png'), dpi=300)
        plt.close(fig)


if os.path.exists(bond_breaks_csv_path):
    print("Loading bonds and bond_durations from CSV files.")
    bond_breaks_df = pd.read_csv(bond_breaks_csv_path)
else:
    print("Calculating bonds and bond_durations.")
    bond_breaks = compute_bond_breaks(neighbour_dataframe, neighbour_radius_xy)
    bond_breaks_df = pd.DataFrame(
    [(trackmate_id, neighbor_id, time_point, count) for (trackmate_id, neighbor_id, time_point), count in bond_breaks.items()],
    columns=['TrackMate Track ID', 'Neighbor TrackMate Track ID', 'Time', 'Break Count']
)
    bond_breaks_df.to_csv(bond_breaks_csv_path, index=False)
    print(f"Bond breaks data saved to {bond_breaks_csv_path}")

 


time_points = sorted(neighbour_dataframe['t'].unique())

plot_bond_breaks(neighbour_dataframe, bond_breaks_df,  color_palette, save_dir, time_points)


def plot_neighbour_time(df, bonds_df, color_palette, save_dir):
    timepoints = sorted(df['t'].unique())
    cell_types = df['Cell_Type'].unique()

    neighbor_counts = {cell_type: {neighbor_type: [0] * len(timepoints) for neighbor_type in cell_types} for cell_type in cell_types}

    for time_idx, t in enumerate(tqdm(timepoints, desc='Neighbour Time')):
        time_df = df[df['t'] == t]

        bonds_at_time = bonds_df[bonds_df['Time'] == t]

        for _, row in bonds_at_time.iterrows():
            trackmate_id, neighbor_id = row['TrackMate Track ID'], row['Neighbor TrackMate Track ID']

            cell_type_row = time_df[time_df['TrackMate Track ID'] == trackmate_id]
            neighbor_type_row = time_df[time_df['TrackMate Track ID'] == neighbor_id]

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
