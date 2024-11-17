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

save_dir = os.path.join(tracking_directory, f'mitosis_{channel}morpho_dynamic/')
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

Path(save_dir).mkdir(exist_ok=True, parents=True) 


dataframe_file = os.path.join(data_frames_dir , f'mitosis_dataframe_normalized_{channel}.csv')
gt_dataframe_file = os.path.join(data_frames_dir , f'val_goblet_basal_dataframe_normalized_{channel}.csv') 

tracks_mitosis_dataframe = pd.read_csv(dataframe_file)
gt_tracks_mitosis_dataframe = pd.read_csv(gt_dataframe_file)



cell_type_dataframe = tracks_mitosis_dataframe[~tracks_mitosis_dataframe['Cell_Type'].isna()]

mitosis_types = cell_type_dataframe['Cell_Type'].unique()


for mitosis_type in mitosis_types:

    filtered_tracks = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == mitosis_type]
    

    if mitosis_type == 'Mitosis':
        gt_filtered_tracks = gt_tracks_mitosis_dataframe[gt_tracks_mitosis_dataframe['Dividing'] == 1]
        dividing_track_ids = filtered_tracks['TrackMate Track ID'].unique()
        gt_dividing_track_ids = gt_filtered_tracks['TrackMate Track ID'].unique()
        print(f'GT tracks for {mitosis_type}: {len(gt_dividing_track_ids)} total predicted tracks {len(dividing_track_ids)}')
    elif mitosis_type == 'Non Mitosis':
        gt_filtered_tracks = gt_tracks_mitosis_dataframe[gt_tracks_mitosis_dataframe['Dividing'] == 0]
        non_dividing_track_ids = filtered_tracks['TrackMate Track ID'].unique()
        gt_non_dividing_track_ids = gt_filtered_tracks['TrackMate Track ID'].unique()
        print(f'GT tracks for {mitosis_type}: {len(gt_non_dividing_track_ids)} total predicted tracks {len(non_dividing_track_ids)}')



class_map_gbr = {
    0: "Non Mitosis",
    1: "Mitosis",
}
class_names = ['Non Mitosis', 'Mitosis']


def compute_counts(predicted_ids, gt_ids):
    tp = len(set(predicted_ids) & set(gt_ids))  
    fn = len(set(gt_ids) - set(predicted_ids))  
    return tp, fn

tp_dividing, fn_dividing = compute_counts(dividing_track_ids, gt_dividing_track_ids)
tp_non_dividing, fn_non_dividing = compute_counts(non_dividing_track_ids, gt_non_dividing_track_ids)

def compute_misclassifications(predicted_ids, gt_ids):
    return len(set(predicted_ids) & set(gt_ids))

misclassifications_dividing_as_non_dividing = compute_misclassifications(dividing_track_ids, gt_non_dividing_track_ids)
misclassifications_non_dividing_as_dividing = compute_misclassifications(non_dividing_track_ids, gt_dividing_track_ids)


print(f'Predicted Dividing and actually Dividing: {tp_dividing}')
print(f'Predicted Non Dividing and actually non Dividing: {tp_non_dividing}')

print(f'Predicted Dividing but actually Non Dividing: {misclassifications_dividing_as_non_dividing}')
print(f'Predicted Non Dividing but actually Dividing: {misclassifications_non_dividing_as_dividing}')


# Create the confusion matrix
conf_matrix_array = np.array([
    [tp_non_dividing, misclassifications_non_dividing_as_dividing],  
    [misclassifications_dividing_as_non_dividing, tp_dividing],  
    
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
total_gt = len(gt_dividing_track_ids) + len(gt_non_dividing_track_ids) 
total_tp = tp_dividing + tp_non_dividing
accuracy = total_tp / total_gt
print(total_tp, total_gt)
print(f"Overall Accuracy: {accuracy * 100:.2f}%")
