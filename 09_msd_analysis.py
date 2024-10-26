from napatrackmater.Trackvector import TrackVector
from pathlib import Path 
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'

master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))

save_dir = os.path.join(tracking_directory, f'msd_plots_{channel}predicted_morpho_feature_attention_shallowest_litest/')
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

Path(save_dir).mkdir(exist_ok=True, parents=True) 

dataframe_file = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}predicted_morpho_feature_attention_shallowest_litest.csv')

# MSD model function for fitting
def msd_model(t, D, alpha):
    return D * t ** alpha

# Load Data
track_vectors = TrackVector(master_xml_path=xml_path)
tracks_goblet_basal_radial_dataframe = pd.read_csv(dataframe_file)
cell_type_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]

cell_types = cell_type_dataframe['Cell_Type'].unique()

# Initialize dictionary to store motion type counts for each cell type
motion_stats = {cell_type: {"Directed": 0, "Brownian": 0, "Random": 0} for cell_type in cell_types}

# MSD Analysis and Plotting for Each Cell Type
for cell_type in cell_types:
    # Filter DataFrame by Cell Type
    filtered_tracks = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == cell_type]
    
    # Get unique TrackMate Track IDs for this cell type
    trackmate_track_ids = filtered_tracks['TrackMate Track ID'].unique()
    
    # Create a new figure for MSD plots for this cell type
    plt.figure(figsize=(12, 6))
    
    # Iterate over each TrackMate Track ID
    for trackmate_id in trackmate_track_ids:
        trackmate_data = filtered_tracks[filtered_tracks['TrackMate Track ID'] == trackmate_id]
        
        # Get unique Track IDs within this TrackMate Track ID
        track_ids = trackmate_data['Track ID'].unique()
        
        # Iterate over each Track ID within the TrackMate Track ID
        for track_id in track_ids:
            # Filter the DataFrame for this specific track
            track_data = trackmate_data[trackmate_data['Track ID'] == track_id].copy()
            
            # Normalize the time for this track (set the first time point to t = 0)
            track_data['t_normalized'] = track_data['t'] - track_data['t'].min()
            # Ensure there are enough data points (e.g., at least 3 points) for fitting
            if len(track_data['t_normalized']) < 3 or len(track_data['MSD']) < 3:
                continue

            # Fit MSD data to the msd_model to determine alpha
            try:
                popt, _ = curve_fit(msd_model, track_data['t_normalized'], track_data['MSD'], maxfev=5000)
                D, alpha = popt  # Extract diffusion coefficient and alpha
                
                # Classify track based on alpha
                if alpha > 1.2:
                    motion_type = "Directed"
                elif 0.8 <= alpha <= 1.2:
                    motion_type = "Brownian"
                else:
                    motion_type = "Random"
                
                # Update motion type count for the cell type
                motion_stats[cell_type][motion_type] += 1
                
                # Plot the MSD with the fitted curve
                plt.plot(track_data['t_normalized'], track_data['MSD'], label=f'TrackMate {trackmate_id} / Track {track_id}', alpha=0.5)
                plt.plot(track_data['t_normalized'], msd_model(track_data['t_normalized'], *popt), linestyle='--')
            
            except RuntimeError:
                print(f"Could not fit MSD for TrackMate Track ID {trackmate_id} / Track ID {track_id} in Cell Type {cell_type}.")
                continue
    
    # Set plot titles and labels
    plt.title(f'MSD over Normalized Time with Fitted Motion Type for Cell Type {cell_type}')
    plt.xlabel('Normalized Time (t)')
    plt.ylabel('Mean Square Displacement (MSD)')
    plt.legend()
    
    # Save the plot
    plt.savefig(os.path.join(save_dir, f'MSD_Fit_Cell_Type_{cell_type}.png'))
    plt.close()

# Convert motion_stats to DataFrame for summary and save
motion_stats_df = pd.DataFrame(motion_stats).T
motion_stats_df.to_csv(os.path.join(save_dir, 'msd_motion_type_statistics.csv'))

print("MSD analysis complete. Motion statistics saved as 'msd_motion_type_statistics.csv'.")

# 1. Bar Plot for Motion Types by Cell Type
plt.figure(figsize=(10, 6))
motion_stats_df.plot(kind='bar', stacked=True, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
plt.title('Distribution of Motion Types by Cell Type')
plt.xlabel('Cell Type')
plt.ylabel('Number of Tracks')
plt.legend(title='Motion Type')
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'motion_type_distribution_bar.png'))
plt.close()

# 2. Pie Charts for Each Cell Type Showing Proportion of Motion Types
for cell_type in motion_stats_df.index:
    plt.figure(figsize=(6, 6))
    plt.pie(motion_stats_df.loc[cell_type], labels=motion_stats_df.columns, 
            autopct='%1.1f%%', startangle=90, colors=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.title(f'Motion Type Proportion for Cell Type: {cell_type}')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'motion_type_proportion_{cell_type}.png'))
    plt.close()

print("Motion type distribution plots saved.")
