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

neighbour_radius_xy = 70 
neighbour_radius_z = 5
color_palette = {
    'Basal': '#1f77b4',  
    'Radial': '#ff7f0e',
    'Goblet': '#2ca02c',
}

# Load data
tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
neighbour_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]

def find_and_track_bonds(df, radius_xy, height_z):
    bonds = defaultdict(lambda: defaultdict(list))
    bond_durations = bond_durations = defaultdict(int)  

    unique_trackmate_ids = df['TrackMate Track ID'].unique()
    unique_time_points = sorted(df['t'].unique())
    
    for trackmate_id in tqdm(unique_trackmate_ids):  
        for time_point in unique_time_points:
            current_track = df[(df['TrackMate Track ID'] == trackmate_id) & (df['t'] == time_point)]
            if current_track.empty:
                continue
            
            current_coords = current_track.iloc[0][['z', 'y', 'x']].values
            
            # Filter DataFrame for the current time point
            time_filtered_df = df[df['t'] == time_point]
            
            # Compute distances
            distances_xy = np.sqrt((time_filtered_df['y'] - current_coords[1])**2 + 
                                   (time_filtered_df['x'] - current_coords[2])**2)
            distances_z = np.abs(time_filtered_df['z'] - current_coords[0])
            
            # Determine neighbors within radius
            within_radius_xy = distances_xy <= radius_xy
            within_height_z = distances_z <= height_z
            valid_indices = within_radius_xy & within_height_z
            
            # Get valid TrackMate Track IDs (exclude self)
            valid_trackmate_ids = time_filtered_df[valid_indices]['TrackMate Track ID'].unique()
            valid_trackmate_ids = valid_trackmate_ids[valid_trackmate_ids != trackmate_id]
            
            # Store neighbors (bonds) and increment bond durations
            for neighbor_id in valid_trackmate_ids:
                bonds[trackmate_id][time_point].append(neighbor_id)
                bond_durations[(trackmate_id, neighbor_id)] += 1
                
    return bonds, bond_durations

# Function to map bond durations to colors
def get_bond_color(bond_time, max_bond_time):
    norm = mcolors.Normalize(vmin=0, vmax=max_bond_time)
    cmap = cm.get_cmap("coolwarm")  # Choose a colormap
    return cmap(norm(bond_time))

# Run neighbor and bond analysis
bonds, bond_durations = find_and_track_bonds(neighbour_dataframe, neighbour_radius_xy, neighbour_radius_z)
time_points = sorted(neighbour_dataframe['t'].unique())

# Plot spatial neighbors with bond colors based on bond_time
def plot_spatial_neighbors_with_bond_time(df, bonds, bond_durations, color_palette, save_dir, time_points):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    max_bond_time = max(bond_durations.values()) if bond_durations else 1
    
    for t in time_points:
        time_df = df[df['t'] == t]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot cells by type
        for cell_type, color in color_palette.items():
            cell_type_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_type_df['x'], cell_type_df['y'], cell_type_df['z'], 
                       color=color, label=cell_type, s=20, alpha=0.7)
        
        # Plot bonds with colors based on bond_time
        for trackmate_id, neighbors_at_t in bonds.items():
            if t in neighbors_at_t:
                cell_coords = time_df[time_df['TrackMate Track ID'] == trackmate_id][['x', 'y', 'z']].values
                if cell_coords.size == 0:
                    continue
                cell_coords = cell_coords[0]
                
                for neighbor_id in neighbors_at_t[t]:
                    neighbor_coords = time_df[time_df['TrackMate Track ID'] == neighbor_id][['x', 'y', 'z']].values
                    if neighbor_coords.size == 0:
                        continue
                    neighbor_coords = neighbor_coords[0]
                    
                    # Get the bond_time and corresponding color
                    bond_time = bond_durations[(trackmate_id, neighbor_id)]
                    bond_color = get_bond_color(bond_time, max_bond_time)
                    
                    # Plot bond with color based on bond_time
                    ax.plot([cell_coords[0], neighbor_coords[0]],
                            [cell_coords[1], neighbor_coords[1]],
                            [cell_coords[2], neighbor_coords[2]], color=bond_color, alpha=0.7)

        ax.set_title(f"Cell Neighbors at Time Point {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc='upper right')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'spatial_neighbors_time_{t}.png'))
        plt.close(fig)

# Run the plotting function
plot_spatial_neighbors_with_bond_time(neighbour_dataframe, bonds, bond_durations, color_palette, save_dir, time_points)

# Function to detect bond-breaking events over time
def detect_bond_breaks(bonds, time_points, threshold_xy, threshold_z, df):
    bond_breaks = defaultdict(list)  # Tracks broken bonds per cell over time
    
    # Check each track's bonds at each time point
    for trackmate_id, time_bonds in bonds.items():
        previous_neighbors = set()
        
        for t in time_points:
            current_neighbors = set(time_bonds.get(t, []))
            
            # Identify bonds that existed but are now missing (broken bonds)
            broken_bonds = previous_neighbors - current_neighbors
            
            # Check if each broken bond is outside the threshold distance at this time point
            current_track = df[(df['TrackMate Track ID'] == trackmate_id) & (df['t'] == t)]
            if not current_track.empty:
                current_coords = current_track.iloc[0][['z', 'y', 'x']].values
                
                for broken_neighbor in broken_bonds:
                    neighbor_track = df[(df['TrackMate Track ID'] == broken_neighbor) & (df['t'] == t)]
                    if neighbor_track.empty:
                        continue
                    
                    neighbor_coords = neighbor_track.iloc[0][['z', 'y', 'x']].values
                    distance_xy = np.sqrt((neighbor_coords[1] - current_coords[1])**2 + 
                                          (neighbor_coords[2] - current_coords[2])**2)
                    distance_z = np.abs(neighbor_coords[0] - current_coords[0])
                    
                    # Record broken bond if it exceeds the thresholds
                    if distance_xy > threshold_xy or distance_z > threshold_z:
                        bond_breaks[trackmate_id].append((t, broken_neighbor))
            
            # Update previous neighbors for next time point
            previous_neighbors = current_neighbors
            
    return bond_breaks

# Run neighbor and bond analysis
bonds = find_and_track_bonds(neighbour_dataframe, neighbour_radius_xy, neighbour_radius_z)
time_points = sorted(neighbour_dataframe['t'].unique())
bond_breaks = detect_bond_breaks(bonds, time_points, neighbour_radius_xy, neighbour_radius_z, neighbour_dataframe)

# Plot bond-breaking events over time


def plot_bond_breaks(df, bond_breaks, save_dir):
    timepoints = sorted(df['t'].unique())
    cell_types = df['Cell_Type'].unique()
    
    bond_break_count = {cell_type: [0] * len(timepoints) for cell_type in cell_types}
    
    # Count bond breaks per cell type at each time point
    for trackmate_id, breaks in bond_breaks.items():
        track_cell_type = df[df['TrackMate Track ID'] == trackmate_id]['Cell_Type'].iloc[0]
        
        for t, _ in breaks:
            time_idx = timepoints.index(t)
            bond_break_count[track_cell_type][time_idx] += 1
    
    # Plot bond break counts over time
    fig, ax = plt.subplots(figsize=(16, 8))
    
    for cell_type in cell_types:
        y_values = bond_break_count[cell_type]
        ax.plot(timepoints, y_values, label=f'{cell_type} Bond Breaks', color=color_palette.get(cell_type, 'grey'), marker='o')
    
    ax.set_title('Bond Breaks Over Time by Cell Type')
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Bond Break Count')
    ax.legend(loc='upper right')
    ax.grid(True)
    
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir, 'bond_break_counts.png'))

plot_bond_breaks(neighbour_dataframe, bond_breaks, save_dir)


def plot_spatial_neighbors(df, bonds, color_palette, save_dir, time_points):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    for t in time_points:
        time_df = df[df['t'] == t]
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        for cell_type, color in color_palette.items():
            cell_type_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_type_df['x'], cell_type_df['y'], cell_type_df['z'], 
                       color=color, label=cell_type, s=20, alpha=0.7)
        
        for trackmate_id, neighbors_at_t in bonds.items():
            if t in neighbors_at_t:
                cell_coords = time_df[time_df['TrackMate Track ID'] == trackmate_id][['x', 'y', 'z']].values
                if cell_coords.size == 0:
                    continue
                cell_coords = cell_coords[0]
                
                for neighbor_id in neighbors_at_t[t]:
                    neighbor_coords = time_df[time_df['TrackMate Track ID'] == neighbor_id][['x', 'y', 'z']].values
                    if neighbor_coords.size == 0:
                        continue
                    neighbor_coords = neighbor_coords[0]
                    ax.plot([cell_coords[0], neighbor_coords[0]],
                            [cell_coords[1], neighbor_coords[1]],
                            [cell_coords[2], neighbor_coords[2]], color='gray', alpha=0.4)

        ax.set_title(f"Cell Neighbors at Time Point {t}")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.legend(loc='upper right')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'spatial_neighbors_time_{t}.png'))
        plt.close(fig)

time_points = sorted(neighbour_dataframe['t'].unique())
plot_spatial_neighbors(neighbour_dataframe, bonds, color_palette, save_dir, time_points)

def plot_neighbour_time(df, bonds, color_palette, save_dir):
    timepoints = sorted(df['t'].unique())
    cell_types = df['Cell_Type'].unique()
    
    # Initialize dictionary to store neighbor counts per cell type over time
    neighbor_counts = {cell_type: {neighbor_type: [0] * len(timepoints) for neighbor_type in cell_types} for cell_type in cell_types}

    # Loop over each time point and count neighbors by cell type
    for time_idx, t in enumerate(timepoints):
        time_df = df[df['t'] == t]
        
        for trackmate_id, neighbors_at_t in bonds.items():
            # Check if trackmate_id exists in time_df
            cell_type_row = time_df[time_df['TrackMate Track ID'] == trackmate_id]
            if cell_type_row.empty:
                continue  # Skip if no data for this trackmate_id at this time point
            
            cell_type = cell_type_row['Cell_Type'].iloc[0]
            
            if t in neighbors_at_t:
                for neighbor_id in neighbors_at_t[t]:
                    # Check if neighbor_id exists in time_df
                    neighbor_type_row = time_df[time_df['TrackMate Track ID'] == neighbor_id]
                    if neighbor_type_row.empty:
                        continue  # Skip if no data for this neighbor_id at this time point
                    
                    neighbor_type = neighbor_type_row['Cell_Type'].iloc[0]
                    neighbor_counts[cell_type][neighbor_type][time_idx] += 1

    # Plot neighbor counts over time
    fig, axs = plt.subplots(len(cell_types), 1, figsize=(16, len(cell_types) * 5), sharex=True)

    if len(cell_types) == 1:
        axs = [axs]  # Ensure axs is always a list for uniform indexing

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

# Run the function
plot_neighbour_time(neighbour_dataframe, bonds, color_palette, save_dir)
