from napatrackmater.Trackvector import TrackVector
from pathlib import Path 
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, OptimizeWarning
import warnings
from sklearn.metrics import r2_score
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Configuration
dataset_name = 'Second'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'

master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))

save_dir = os.path.join(tracking_directory, f'msd_plots_{channel}predicted')
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

Path(save_dir).mkdir(exist_ok=True, parents=True) 

dataframe_file = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}predicted.csv')

def quadratic_msd(t, a, b, c):
    return a * t**2 + b * t + c

def linear_msd(t, m, c):
    return m * t + c

# Load Data
track_vectors = TrackVector(master_xml_path=xml_path)
tracks_goblet_basal_radial_dataframe = pd.read_csv(dataframe_file)
cell_type_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]
cell_types = cell_type_dataframe['Cell_Type'].unique()


RESIDUAL_THRESHOLD = 0.1
R2_THRESHOLD = 0.8

trend_stats = {cell_type: {"Valid Trend": 0, "Invalid Trend": 0} for cell_type in cell_types}

for cell_type in cell_types:
    filtered_tracks = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == cell_type]
    trackmate_track_ids = filtered_tracks['TrackMate Track ID'].unique()
    
    for count, trackmate_id in enumerate(trackmate_track_ids):
        trackmate_data = filtered_tracks[filtered_tracks['TrackMate Track ID'] == trackmate_id]
        track_ids = trackmate_data['Track ID'].unique()
        
        for track_id in track_ids:
            track_data = trackmate_data[trackmate_data['Track ID'] == track_id].copy()
            if len(track_data['t']) < 5 or len(track_data['MSD']) < 5:
                continue
            
            t = track_data['t'].values
            msd = track_data['MSD'].values
            
            t_threshold = t.max() / 4
            early_indices = t <= t_threshold
            late_indices = t > t_threshold
            t_early, msd_early = t[early_indices], msd[early_indices]
            t_late, msd_late = t[late_indices], msd[late_indices]
            if len(msd[early_indices]) < 5 or len(msd[late_indices]) < 5:
                continue
            try:
                popt_early, _ = curve_fit(quadratic_msd, t_early, msd_early, maxfev=10000)
                popt_late, _ = curve_fit(linear_msd, t_late, msd_late, maxfev=10000)
                
                msd_early_fit = quadratic_msd(t_early, *popt_early)
                msd_late_fit = linear_msd(t_late, *popt_late)
                
                early_residuals = msd_early - msd_early_fit
                late_residuals = msd_late - msd_late_fit
                
                r2_early = r2_score(msd_early, msd_early_fit)
                r2_late = r2_score(msd_late, msd_late_fit)
                
                # Criteria for Valid Trend
                is_valid_early = (popt_early[0] > 0) and (r2_early > R2_THRESHOLD)
                is_valid_late = (popt_late[0] > 0) and (r2_late > R2_THRESHOLD)
                
                if is_valid_early and is_valid_late:
                    trend_stats[cell_type]["Valid Trend"] += 1
                else:
                    trend_stats[cell_type]["Invalid Trend"] += 1
            
            except (RuntimeError, OptimizeWarning):
                trend_stats[cell_type]["Invalid Trend"] += 1

# Plot Summary Bar Chart
cell_types = list(trend_stats.keys())
valid_trends = [trend_stats[ct]["Valid Trend"] for ct in cell_types]
invalid_trends = [trend_stats[ct]["Invalid Trend"] for ct in cell_types]

plt.figure(figsize=(10, 6))
bar_width = 0.35
x = range(len(cell_types))

plt.bar(x, valid_trends, bar_width, label="Valid Trends", color='green')
plt.bar(x, invalid_trends, bar_width, bottom=valid_trends, label="Invalid Trends", color='red')

plt.xlabel("Cell Types")
plt.ylabel("Number of Tracks")
plt.title("Trend Validation: Quadratic-to-Linear MSD")
plt.xticks(x, cell_types, rotation=45)
plt.legend()
plt.tight_layout()

# Save the plot
summary_plot_path = os.path.join(save_dir, "msd_langevin_test.png")
plt.savefig(summary_plot_path)
plt.close()

# Optional: Print summary
for cell_type, stats in trend_stats.items():
    print(f"{cell_type}: {stats['Valid Trend']} valid, {stats['Invalid Trend']} invalid")
                  

   