from pathlib import Path 
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


# %%
dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'

master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))

save_dir = os.path.join(tracking_directory, f'dual_predicted_attention_morpho_nuclei_membrane_nuclei_morpho_dynamic/')
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

Path(save_dir).mkdir(exist_ok=True, parents=True) 


dataframe_file = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_dual_predicted_attention_morpho_nuclei_membrane_nuclei_morpho_dynamic.csv')
gt_dataframe_file = os.path.join(data_frames_dir , f'val_goblet_basal_dataframe_normalized_{channel}.csv') 

tracks_goblet_basal_radial_dataframe = pd.read_csv(dataframe_file)
gt_tracks_goblet_basal_radial_dataframe = pd.read_csv(gt_dataframe_file)



cell_type_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]

cell_types = cell_type_dataframe['Cell_Type'].unique()


for cell_type in cell_types:

    filtered_tracks = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == cell_type]

    gt_filtered_tracks = gt_tracks_goblet_basal_radial_dataframe[gt_tracks_goblet_basal_radial_dataframe['Cell_Type'] == cell_type]
    
    
    if cell_type == 'Goblet':
        goblet_track_ids = filtered_tracks['TrackMate Track ID'].unique()
        gt_goblet_track_ids = gt_filtered_tracks['TrackMate Track ID'].unique()
        print(f'GT tracks for {cell_type}: {len(gt_goblet_track_ids)} total predicted tracks {len(goblet_track_ids)}')
    elif cell_type == 'Basal':
        basal_track_ids = filtered_tracks['TrackMate Track ID'].unique()
        gt_basal_track_ids = gt_filtered_tracks['TrackMate Track ID'].unique()
        print(f'GT tracks for {cell_type}: {len(gt_basal_track_ids)} total predicted tracks {len(basal_track_ids)}')

    elif cell_type == 'Radial':
        radial_track_ids = filtered_tracks['TrackMate Track ID'].unique()
        gt_radial_track_ids = gt_filtered_tracks['TrackMate Track ID'].unique()
        print(f'GT tracks for {cell_type}: {len(gt_radial_track_ids)} total predicted tracks {len(radial_track_ids)}')
        print('Radial Track Ids', gt_radial_track_ids)

        matching_radial_track_ids = set(radial_track_ids) & set(gt_radial_track_ids)
        print('Matching Radial Track IDs:', matching_radial_track_ids)



# Define class map and class names
class_map_gbr = {
    0: "Basal",
    1: "Radial",
    2: "Goblet"
}
class_names = ['Basal', 'Radial', 'Goblet']

# Function to compute counts of TP and FN
def compute_counts(predicted_ids, gt_ids):
    tp = len(set(predicted_ids) & set(gt_ids))  # True Positives
    fn = len(set(gt_ids) - set(predicted_ids))  # False Negatives
    return tp, fn

# True Positives and False Negatives for each cell type
tp_goblet, fn_goblet = compute_counts(goblet_track_ids, gt_goblet_track_ids)
tp_basal, fn_basal = compute_counts(basal_track_ids, gt_basal_track_ids)
tp_radial, fn_radial = compute_counts(radial_track_ids, gt_radial_track_ids)

# Function to compute misclassifications
def compute_misclassifications(predicted_ids, gt_ids):
    return len(set(predicted_ids) & set(gt_ids))

# Misclassifications for each cell type
misclassifications_goblet_as_basal = compute_misclassifications(goblet_track_ids, gt_basal_track_ids)
misclassifications_goblet_as_radial = compute_misclassifications(goblet_track_ids, gt_radial_track_ids)

misclassifications_basal_as_goblet = compute_misclassifications(basal_track_ids, gt_goblet_track_ids)
misclassifications_basal_as_radial = compute_misclassifications(basal_track_ids, gt_radial_track_ids)

misclassifications_radial_as_goblet = compute_misclassifications(radial_track_ids, gt_goblet_track_ids)
misclassifications_radial_as_basal = compute_misclassifications(radial_track_ids, gt_basal_track_ids)

# Print true positives and misclassifications
print(f'Predicted Goblet and actually Goblet: {tp_goblet}')
print(f'Predicted Basal and actually Basal: {tp_basal}')
print(f'Predicted Radial and actually Radial: {tp_radial}')

print(f'Predicted Goblet but actually Basal: {misclassifications_goblet_as_basal}')
print(f'Predicted Goblet but actually Radial: {misclassifications_goblet_as_radial}')
print(f'Predicted Basal but actually Goblet: {misclassifications_basal_as_goblet}')
print(f'Predicted Basal but actually Radial: {misclassifications_basal_as_radial}')
print(f'Predicted Radial but actually Goblet: {misclassifications_radial_as_goblet}')
print(f'Predicted Radial but actually Basal: {misclassifications_radial_as_basal}')

# Create the confusion matrix
conf_matrix_array = np.array([
    [tp_basal, misclassifications_basal_as_radial, misclassifications_basal_as_goblet],  # Basal
    [misclassifications_radial_as_basal, tp_radial, misclassifications_radial_as_goblet],  # Radial
    [misclassifications_goblet_as_basal, misclassifications_goblet_as_radial, tp_goblet],  # Goblet
])

confusion_df = pd.DataFrame(conf_matrix_array, index=class_names, columns=class_names)

# Print the confusion matrix
print("\nConfusion Matrix Summary:")
print(confusion_df)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_array, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Actual Labels')
plt.ylabel('Predicted Labels')
plt.title('Cell Type Misclassification Matrix')
plt.savefig(f'{save_dir}/{channel}cell_type_misclassification_matrix.png')

# %%
# Compute accuracy
total_gt = len(gt_goblet_track_ids) + len(gt_basal_track_ids) + len(gt_radial_track_ids)
total_tp = tp_goblet + tp_basal + tp_radial
accuracy = total_tp / total_gt
print(total_tp, total_gt)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
