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

neighbour_radius_xy = 70 
partner_time = 150  # Bonds lasting longer than this will be specially plotted
color_palette = {
    'Basal': '#1f77b4',  
    'Radial': '#ff7f0e',
    'Goblet': '#2ca02c',
}

tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
neighbour_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]

bonds_csv_path = os.path.join(save_dir, 'bonds.csv')
bond_durations_csv_path = os.path.join(save_dir, 'bond_durations.csv')

def find_and_track_bonds(df, radius_xy):
    bonds = defaultdict(lambda: defaultdict(list))
    bond_durations = defaultdict(int)
    
    unique_trackmate_ids = df['TrackMate Track ID'].unique()
    unique_time_points = sorted(df['t'].unique())
    
    for trackmate_id in tqdm(unique_trackmate_ids):  
        for time_point in unique_time_points:
            current_track = df[(df['TrackMate Track ID'] == trackmate_id) & (df['t'] == time_point)]
            if current_track.empty:
                continue
            
            current_coords = current_track.iloc[0][['z', 'y', 'x']].values
            time_filtered_df = df[df['t'] == time_point]
            
            distances_xy = np.sqrt((time_filtered_df['y'] - current_coords[1])**2 + 
                                   (time_filtered_df['x'] - current_coords[2])**2)
           
            
            within_radius_xy = distances_xy <= radius_xy
            
            valid_indices = within_radius_xy 
            
            valid_trackmate_ids = time_filtered_df[valid_indices]['TrackMate Track ID'].unique()
            valid_trackmate_ids = valid_trackmate_ids[valid_trackmate_ids != trackmate_id]
            
            for neighbor_id in valid_trackmate_ids:
                bonds[trackmate_id][time_point].append(neighbor_id)
                bond_durations[(trackmate_id, neighbor_id)] += 1
    
    bonds_df = pd.DataFrame(
        [(trackmate_id, time, neighbor_id) for trackmate_id, time_dict in bonds.items() for time, neighbors in time_dict.items() for neighbor_id in neighbors],
        columns=['TrackMate Track ID', 'Time', 'Neighbor TrackMate Track ID']
    )
    
    bond_durations_df = pd.DataFrame(
        [(trackmate_id, neighbor_id, duration) for (trackmate_id, neighbor_id), duration in bond_durations.items()],
        columns=['TrackMate Track ID', 'Neighbor TrackMate Track ID', 'Duration']
    )
    
    return bonds_df, bond_durations_df

if os.path.exists(bonds_csv_path) and os.path.exists(bond_durations_csv_path):
    print("Loading bonds and bond_durations from CSV files.")
    bonds_df = pd.read_csv(bonds_csv_path)
    bond_durations_df = pd.read_csv(bond_durations_csv_path)
else:
    print("Calculating bonds and bond_durations.")
    bonds_df, bond_durations_df = find_and_track_bonds(neighbour_dataframe, neighbour_radius_xy)
    bonds_df.to_csv(bonds_csv_path, index=False)
    bond_durations_df.to_csv(bond_durations_csv_path, index=False)

bond_durations = {(row['TrackMate Track ID'], row['Neighbor TrackMate Track ID']): row['Duration'] for _, row in bond_durations_df.iterrows()}

def get_bond_color(bond_time, max_bond_time):
    norm = mcolors.Normalize(vmin=0, vmax=max_bond_time)
    cmap = matplotlib.colormaps.get_cmap("coolwarm")
    return cmap(norm(bond_time))

def plot_spatial_neighbors_with_bond_time_2D(df, bonds_df, bond_durations, color_palette, save_dir, time_points):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    max_bond_time = max(bond_durations.values()) if bond_durations else 1
    
    for t in time_points:
        time_df = df[df['t'] == t]
        
        fig, ax = plt.subplots(figsize=(18, 15))  # Enlarged plot size for better visibility
        
        # Plot cells by type in 2D XY plane with larger markers
        for cell_type, color in color_palette.items():
            cell_type_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_type_df['x'], cell_type_df['y'], color=color, label=cell_type, s=100, alpha=0.7)
        
        bonds_at_time = bonds_df[bonds_df['Time'] == t]
        
        for _, row in bonds_at_time.iterrows():
            trackmate_id, neighbor_id = row['TrackMate Track ID'], row['Neighbor TrackMate Track ID']
            cell_coords = time_df[time_df['TrackMate Track ID'] == trackmate_id][['x', 'y']].values
            neighbor_coords = time_df[time_df['TrackMate Track ID'] == neighbor_id][['x', 'y']].values
            
            if cell_coords.size == 0 or neighbor_coords.size == 0:
                continue
            
            bond_time = bond_durations.get((trackmate_id, neighbor_id), 0)
            bond_color = get_bond_color(bond_time, max_bond_time)
            
            ax.plot([cell_coords[0][0], neighbor_coords[0][0]], [cell_coords[0][1], neighbor_coords[0][1]], 
                    color=bond_color, alpha=0.2, linewidth=2)  # Increased line width for visibility
        
        # Color bar for bond durations
        norm = mcolors.Normalize(vmin=0, vmax=max_bond_time)
        cmap = matplotlib.colormaps.get_cmap("coolwarm")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Bond Duration (timepoints)')
        
        ax.set_title(f"Cell Neighbors at Time Point {t} (XY Plane)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'spatial_neighbors_time_{t}_2D.png'), dpi=300)  # High DPI for clarity
        plt.close(fig)

def plot_long_duration_bonds_2D(df, bonds_df, bond_durations, color_palette, save_dir, time_points):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    max_bond_time = max(bond_durations.values()) if bond_durations else 1
    
    for t in time_points:
        time_df = df[df['t'] == t]
        
        fig, ax = plt.subplots(figsize=(18, 15))  # Enlarged plot size for better visibility
        
        # Plot cells by type in 2D XY plane with larger markers
        for cell_type, color in color_palette.items():
            cell_type_df = time_df[time_df['Cell_Type'] == cell_type]
            ax.scatter(cell_type_df['x'], cell_type_df['y'], color=color, label=cell_type, s=100, alpha=0.7)
        
        bonds_at_time = bonds_df[bonds_df['Time'] == t]
        
        for _, row in bonds_at_time.iterrows():
            trackmate_id, neighbor_id = row['TrackMate Track ID'], row['Neighbor TrackMate Track ID']
            cell_coords = time_df[time_df['TrackMate Track ID'] == trackmate_id][['x', 'y']].values
            neighbor_coords = time_df[time_df['TrackMate Track ID'] == neighbor_id][['x', 'y']].values
            
            if cell_coords.size == 0 or neighbor_coords.size == 0:
                continue
            
            bond_time = bond_durations.get((trackmate_id, neighbor_id), 0)
            if bond_time > partner_time:
                bond_color = get_bond_color(bond_time, max_bond_time)
                ax.plot([cell_coords[0][0], neighbor_coords[0][0]], [cell_coords[0][1], neighbor_coords[0][1]], 
                        color=bond_color, alpha=0.7, linewidth=3)  # Thicker bonds for long duration

        # Color bar for bond durations
        norm = mcolors.Normalize(vmin=0, vmax=max_bond_time)
        cmap = matplotlib.colormaps.get_cmap("coolwarm")
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical')
        cbar.set_label('Bond Duration (timepoints)')

        ax.set_title(f"Long-duration Bonds at Time Point {t} (XY Plane)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc='upper right')
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'150_long_duration_bonds_time_{t}_2D.png'), dpi=300)  # High DPI for clarity
        plt.close(fig)

time_points = sorted(neighbour_dataframe['t'].unique())
#plot_spatial_neighbors_with_bond_time_2D(neighbour_dataframe, bonds_df, bond_durations, color_palette, save_dir, time_points)
plot_long_duration_bonds_2D(neighbour_dataframe, bonds_df, bond_durations, color_palette, save_dir, time_points)




def detect_bond_breaks(bonds_df, time_points, threshold_xy, df):
    bond_breaks = defaultdict(list)
    
    # Ensure bonds DataFrame is not empty
    if bonds_df.empty:
        print("No bonds available to analyze.")
        return bond_breaks

    # Iterate over each TrackMate Track ID in the bonds DataFrame
    trackmate_ids = bonds_df['TrackMate Track ID'].unique()
    
    for trackmate_id in trackmate_ids:
        previous_neighbors = set()  # Track neighbors from the previous time point

        # Filter bonds for the current TrackMate ID
        track_bonds_df = bonds_df[bonds_df['TrackMate Track ID'] == trackmate_id]
        
        for t in time_points:
            # Get current neighbors for this TrackMate ID and time point
            current_neighbors = set(track_bonds_df[track_bonds_df['Time'] == t]['Neighbor TrackMate Track ID'])
            
            # Determine broken bonds as those in the previous timepoint but missing in the current
            broken_bonds = previous_neighbors - current_neighbors
            
            # Retrieve current track's coordinates for distance checks
            current_track = df[(df['TrackMate Track ID'] == trackmate_id) & (df['t'] == t)]
            if not current_track.empty:
                current_coords = current_track.iloc[0][['z', 'y', 'x']].values

                # Check each broken bond for distance threshold
                for broken_neighbor in broken_bonds:
                    neighbor_track = df[(df['TrackMate Track ID'] == broken_neighbor) & (df['t'] == t)]
                    if neighbor_track.empty:
                        continue  # Skip if no data for this neighbor

                    neighbor_coords = neighbor_track.iloc[0][['z', 'y', 'x']].values
                    distance_xy = np.sqrt((neighbor_coords[1] - current_coords[1])**2 + 
                                          (neighbor_coords[2] - current_coords[2])**2)

                    # Record the bond break if the distance exceeds the threshold
                    if distance_xy > threshold_xy:
                        bond_breaks[trackmate_id].append((t, broken_neighbor))
            
            # Update previous neighbors for the next time point
            previous_neighbors = current_neighbors
            
    return bond_breaks

time_points = sorted(neighbour_dataframe['t'].unique())
bond_breaks = detect_bond_breaks(bonds_df, time_points, neighbour_radius_xy, neighbour_dataframe)


def plot_bond_breaks(df, bond_breaks, save_dir):
    timepoints = sorted(df['t'].unique())
    cell_types = df['Cell_Type'].unique()
    
    bond_break_count = {cell_type: [0] * len(timepoints) for cell_type in cell_types}
    
    for trackmate_id, breaks in bond_breaks.items():
        track_cell_type = df[df['TrackMate Track ID'] == trackmate_id]['Cell_Type'].iloc[0]
        
        for t, _ in breaks:
            time_idx = timepoints.index(t)
            bond_break_count[track_cell_type][time_idx] += 1
    
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



def plot_neighbour_time(df, bonds_df, color_palette, save_dir):
    timepoints = sorted(df['t'].unique())
    cell_types = df['Cell_Type'].unique()

    # Initialize the dictionary to store neighbor counts per cell type over time
    neighbor_counts = {cell_type: {neighbor_type: [0] * len(timepoints) for neighbor_type in cell_types} for cell_type in cell_types}

    for time_idx, t in enumerate(timepoints):
        time_df = df[df['t'] == t]

        # Filter bonds DataFrame for the current time point
        bonds_at_time = bonds_df[bonds_df['Time'] == t]

        # Count neighbors for each cell type at this time point
        for _, row in bonds_at_time.iterrows():
            trackmate_id, neighbor_id = row['TrackMate Track ID'], row['Neighbor TrackMate Track ID']

            # Get cell types of the current cell and its neighbor
            cell_type_row = time_df[time_df['TrackMate Track ID'] == trackmate_id]
            neighbor_type_row = time_df[time_df['TrackMate Track ID'] == neighbor_id]

            # Continue only if both cells exist in the time frame
            if not cell_type_row.empty and not neighbor_type_row.empty:
                cell_type = cell_type_row['Cell_Type'].iloc[0]
                neighbor_type = neighbor_type_row['Cell_Type'].iloc[0]

                # Increment neighbor count
                neighbor_counts[cell_type][neighbor_type][time_idx] += 1

    # Plotting
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

plot_neighbour_time(neighbour_dataframe, bonds_df, color_palette, save_dir)
