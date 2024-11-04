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
bonds_csv_path = os.path.join(save_dir, 'bonds.csv')
neighbour_radius_xy = 70 
partner_time = 0  
color_palette = {
    'Basal': '#1f77b4',  
    'Radial': '#ff7f0e',
    'Goblet': '#2ca02c',
}

tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
neighbour_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]






def compute_bond_breaks_and_bonds(df, radius_xy, max_separation_time=5):
    bond_breaks = defaultdict(int)
    bonds = defaultdict(lambda: defaultdict(list))
    unique_time_points = sorted(df['t'].unique())
    trackmate_ids = df['TrackMate Track ID'].unique()

    def process_trackmate_id(trackmate_id):
        """Processes bond breaks for individual tracklets within a TrackMate track ID."""
        local_bond_breaks = defaultdict(int)
        local_bonds = defaultdict(lambda: defaultdict(list))
        
        # Loop through each unique track ID for the given TrackMate ID
        for track_id in df[df['TrackMate Track ID'] == trackmate_id]['Track ID'].unique():
            track_df = df[(df['TrackMate Track ID'] == trackmate_id) & (df['Track ID'] == track_id)]

            for time_point in unique_time_points:
                time_df = track_df[track_df['t'] == time_point]
                
                if time_df.empty:
                    continue

                current_coords = time_df.iloc[0][['z', 'y', 'x']].values

                # Calculate distances within the same tracklet for current neighbors
                distances = np.sqrt((time_df['y'] - current_coords[1])**2 +
                                    (time_df['x'] - current_coords[2])**2)
                current_neighbors = set(df[(distances <= radius_xy) & 
                                        (df['TrackMate Track ID'] == trackmate_id) &
                                        (df['Track ID'] != track_id)]['Track ID'])

                for neighbor_id in current_neighbors:
                    local_bonds[track_id][time_point].append(neighbor_id)

                # Check if each neighbor bond persists across the next `max_separation_time` frames
                for neighbor_id in current_neighbors:
                    bond_persistent = False
                    for offset in range(1, max_separation_time + 1):
                        future_time = time_point + offset
                        future_df = df[(df['t'] == future_time) & 
                                    (df['TrackMate Track ID'] == trackmate_id) & 
                                    (df['Track ID'] == track_id)]
                        future_distances = np.sqrt((future_df['y'] - current_coords[1])**2 + 
                                                (future_df['x'] - current_coords[2])**2)
                        future_neighbors = set(future_df[(future_distances <= radius_xy) & 
                                                        (future_df['Track ID'] != track_id)]['Track ID'])
                        
                        if neighbor_id in future_neighbors:
                            bond_persistent = True
                            break

                    if not bond_persistent:
                        local_bond_breaks[(track_id, neighbor_id, time_point)] += 1

        return local_bond_breaks, local_bonds


    # Run in parallel over all trackmate_ids
    with ThreadPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        futures = [executor.submit(process_trackmate_id, trackmate_id) for trackmate_id in trackmate_ids]
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Bond Breaks"):
            local_bond_breaks, local_bonds = future.result()
            for k, v in local_bond_breaks.items():
                bond_breaks[k] += v
            for track_id, bond_times in local_bonds.items():
                for time, neighbors in bond_times.items():
                    bonds[track_id][time].extend(neighbors)

    return bond_breaks, bonds




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


def plot_bond_breaks(df, bond_breaks_df, bonds_df, color_palette, save_dir, time_points):

    total_bond_breaks_by_time = [
        bond_breaks_df[bond_breaks_df['Time'] == t]['Break Count'].sum()
        for t in time_points 
    ]
    max_total_bond_breaks = max(total_bond_breaks_by_time) if total_bond_breaks_by_time else 1

    max_bonds = 1

    for t in tqdm(time_points ) :
        total_bonds = get_total_bonds_at_time(bonds_df, t) 
        if total_bonds > max_bonds:
            max_bonds = total_bonds

    for t in tqdm(time_points, desc='Plotting Bond Breaks'):
        fig, ax = plt.subplots(figsize=(18, 15))
        time_df = df[df['t'] == t]

        total_bond_breaks_at_t = bond_breaks_df[bond_breaks_df['Time'] == t]['Break Count'].sum()
        total_bonds = get_total_bonds_at_time(bonds_df, t)
        total_bond_breaks_at_t = total_bond_breaks_at_t 
        for cell_type, color in color_palette.items():
            cell_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_df['x'], cell_df['y'], color=color, label=cell_type, s=100, alpha=0.7)

        bonds_at_time = bond_breaks_df[bond_breaks_df['Time'] == t]

        for _, row in bonds_at_time.iterrows():
            track_id = row['Track ID']
            neighbor_id = row['Neighbor Track ID']

            cell_coords = time_df[time_df['Track ID'] == track_id][['x', 'y']].values
            neighbor_coords = time_df[time_df['Track ID'] == neighbor_id][['x', 'y']].values

            if cell_coords.size == 0 or neighbor_coords.size == 0:
                continue

            bond_color = matplotlib.colormaps["coolwarm"](total_bond_breaks_at_t)
            ax.plot([cell_coords[0][0], neighbor_coords[0][0]], 
                    [cell_coords[0][1], neighbor_coords[0][1]], 
                    color=bond_color, linewidth=3)

        norm = mcolors.Normalize(vmin=0, vmax=max_total_bond_breaks)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label(f'Total Bond Breaks at Time (Max={max_total_bond_breaks})')

        ax.set_title(f"Bond Breaks at Time Point {t} (XY Plane)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')
        ax.grid(True)
        plt.savefig(os.path.join(save_dir, f'bond_break_{t}.png'))
        plt.close()


if os.path.exists(bond_breaks_csv_path) and os.path.exists(bonds_csv_path):
    print("Loading bonds and bond_durations from CSV files.")
    bond_breaks_df = pd.read_csv(bond_breaks_csv_path)
    bonds_df = pd.read_csv(bonds_csv_path)
else:
    print("Calculating bonds and bond_durations.")
    bond_breaks, bonds = compute_bond_breaks_and_bonds(neighbour_dataframe, neighbour_radius_xy)
    bond_breaks_df = pd.DataFrame(
    [(track_id, neighbor_id, time_point, count) for (track_id, neighbor_id, time_point), count in bond_breaks.items()],
    columns=['Track ID', 'Neighbor Track ID', 'Time', 'Break Count']
)
    bonds_df = pd.DataFrame(
        [(track_id, time, neighbor_id) for track_id, time_dict in bonds.items() for time, neighbors in time_dict.items() for neighbor_id in neighbors],
        columns=['Track ID', 'Time', 'Neighbor Track ID']
    )
    bond_breaks_df.to_csv(bond_breaks_csv_path, index=False)
    bonds_df.to_csv(bonds_csv_path, index=False)
    print(f"Bond breaks data saved to {bond_breaks_csv_path}")

 


time_points = sorted(neighbour_dataframe['t'].unique())

plot_bond_breaks(neighbour_dataframe, bond_breaks_df,bonds_df, color_palette, save_dir, time_points)

def plot_bonds_spatially(df, bonds_df, color_palette, save_dir, time_points):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    max_bond_count = bonds_df.groupby('Time').size().max()
    
    for t in tqdm(time_points, desc='Plotting Bonds Spatially'):
        fig, ax = plt.subplots(figsize=(18, 15))
        time_df = df[df['t'] == t]

        # Plot each cell type with unique color
        for cell_type, color in color_palette.items():
            cell_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_df['x'], cell_df['y'], color=color, label=cell_type, s=100, alpha=0.7)

        # Filter the bonds for the current time point
        bonds_at_time = bonds_df[bonds_df['Time'] == t]

        # Plot each bond with color based on the frequency of bonds
        for _, row in bonds_at_time.iterrows():
            track_id = row['Track ID']
            neighbor_id = row['Neighbor Track ID']
            
            cell_coords = time_df[time_df['Track ID'] == track_id][['x', 'y']].values
            neighbor_coords = time_df[time_df['Track ID'] == neighbor_id][['x', 'y']].values

            if cell_coords.size == 0 or neighbor_coords.size == 0:
                continue

            bond_color =  matplotlib.colormaps["coolwarm"](len(bonds_at_time))
            ax.plot([cell_coords[0][0], neighbor_coords[0][0]], [cell_coords[0][1], neighbor_coords[0][1]], color=bond_color, linewidth=3)

        # Add color bar for bond density
        norm = mcolors.Normalize(vmin=0, vmax=max_bond_count)
        sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Bond Frequency')

        ax.set_title(f"Bonds at Time Point {t} (XY Plane)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')
        ax.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'bonds_time_{t}.png'), dpi=300)
        plt.close(fig)

plot_bonds_spatially(neighbour_dataframe, bonds_df, color_palette, save_dir, time_points)


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
