# %%
from pathlib import Path 
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from napatrackmater.Trackvector import (TrackVector,
                                        SHAPE_FEATURES, 
                                        DYNAMIC_FEATURES, 
                                        SHAPE_DYNAMIC_FEATURES,
                                        
                                        )

# %%
dataset_name = 'Second'
home_folder = '/home/debian/jz/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
save_dir = os.path.join(tracking_directory, f'cell_fate_accuracy/')
Path(save_dir).mkdir(exist_ok=True)
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
  
goblet_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400/goblet_cells_{channel}annotations_inception.csv'
basal_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400/basal_cells_{channel}annotations_inception.csv'
radial_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_25_t_initial_50_t_final_400/radially_intercalating_cells_{channel}annotations_inception.csv'

goblet_cells_dataframe = pd.read_csv(goblet_cells_file)
basal_cells_dataframe = pd.read_csv(basal_cells_file)
radial_cells_dataframe = pd.read_csv(radial_cells_file)



gt_goblet_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations/goblet_cells_{channel}annotations.csv'
gt_basal_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations/basal_cells_{channel}annotations.csv'
gt_radial_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations/radially_intercalating_cells_{channel}annotations.csv'

gt_goblet_cells_dataframe = pd.read_csv(gt_goblet_cells_file)
gt_basal_cells_dataframe = pd.read_csv(gt_basal_cells_file)
gt_radial_cells_dataframe = pd.read_csv(gt_radial_cells_file)


# %%
track_vectors = TrackVector(master_xml_path=xml_path)
track_vectors.t_minus = 0
track_vectors.t_plus = track_vectors.tend
track_vectors.y_start = 0
track_vectors.y_end = track_vectors.ymax
track_vectors.x_start = 0
track_vectors.x_end = track_vectors.xmax

track_vectors._interactive_function()

# %%



goblet_track_ids = track_vectors._get_trackmate_ids_by_location(goblet_cells_dataframe)
print(f'Trackmate IDs for globlet cells {goblet_track_ids}')
basal_track_ids = track_vectors._get_trackmate_ids_by_location(basal_cells_dataframe)
print(f'Trackmate IDs for basal cells {basal_track_ids}')
radial_track_ids = track_vectors._get_trackmate_ids_by_location(radial_cells_dataframe)
print(f'Trackmate IDs for radial cells {radial_track_ids}')



# %%
gt_globlet_track_ids = track_vectors._get_trackmate_ids_by_location(gt_goblet_cells_dataframe)
print(f'GT Trackmate IDs for globlet cells {gt_globlet_track_ids}')
gt_basal_track_ids = track_vectors._get_trackmate_ids_by_location(gt_basal_cells_dataframe)
print(f'GT Trackmate IDs for basal cells {gt_basal_track_ids}')
gt_radial_track_ids = track_vectors._get_trackmate_ids_by_location(gt_radial_cells_dataframe)
print(f'GT Trackmate IDs for radial cells {gt_radial_track_ids}')

# %%

class_map_gbr = {
    0: "Basal",
    1: "Radial",
    2: "Goblet"
}

def compute_misclassifications(predicted_ids, gt_ids, predicted_class, actual_class):
    # Calculate misclassifications of a predicted class that are actually another class
    return len(set(predicted_ids) & set(gt_ids))

# Compute misclassifications
misclassifications_goblet_as_basal = compute_misclassifications(goblet_track_ids, gt_basal_track_ids, "Goblet", "Basal")
misclassifications_goblet_as_radial = compute_misclassifications(goblet_track_ids, gt_radial_track_ids, "Goblet", "Radial")

misclassifications_basal_as_goblet = compute_misclassifications(basal_track_ids, gt_globlet_track_ids, "Basal", "Goblet")
misclassifications_basal_as_radial = compute_misclassifications(basal_track_ids, gt_radial_track_ids, "Basal", "Radial")

misclassifications_radial_as_goblet = compute_misclassifications(radial_track_ids, gt_globlet_track_ids, "Radial", "Goblet")
misclassifications_radial_as_basal = compute_misclassifications(radial_track_ids, gt_basal_track_ids, "Radial", "Basal")

# Print results
print(f'Predicted Goblet but actually Basal: {misclassifications_goblet_as_basal}')
print(f'Predicted Goblet but actually Radial: {misclassifications_goblet_as_radial}')
print(f'Predicted Basal but actually Goblet: {misclassifications_basal_as_goblet}')
print(f'Predicted Basal but actually Radial: {misclassifications_basal_as_radial}')
print(f'Predicted Radial but actually Goblet: {misclassifications_radial_as_goblet}')
print(f'Predicted Radial but actually Basal: {misclassifications_radial_as_basal}')

# Create a summary confusion matrix
confusion_matrix_summary = {
    'Predicted': ['Goblet', 'Goblet', 'Basal', 'Basal', 'Radial', 'Radial'],
    'Actual': ['Basal', 'Radial', 'Goblet', 'Radial', 'Goblet', 'Basal'],
    'Count': [
        misclassifications_goblet_as_basal,
        misclassifications_goblet_as_radial,
        misclassifications_basal_as_goblet,
        misclassifications_basal_as_radial,
        misclassifications_radial_as_goblet,
        misclassifications_radial_as_basal,
    ]
}

# Print the confusion matrix summary
print("\nMisclassification Summary:")
print("{:<20} {:<15} {:<5}".format('Predicted', 'Actual', 'Count'))
for i in range(len(confusion_matrix_summary['Predicted'])):
    print("{:<20} {:<15} {:<5}".format(
        confusion_matrix_summary['Predicted'][i],
        confusion_matrix_summary['Actual'][i],
        confusion_matrix_summary['Count'][i],
    ))

# %%
# Total ground truth counts for each category
total_globlet = len(gt_globlet_track_ids)
total_basal = len(gt_basal_track_ids)
total_radial = len(gt_radial_track_ids)

# Calculate percentages
percent_tp_globlet = (tp_globlet / total_globlet) * 100 if total_globlet > 0 else 0
percent_fn_globlet = (fn_globlet / total_globlet) * 100 if total_globlet > 0 else 0

percent_tp_basal = (tp_basal / total_basal) * 100 if total_basal > 0 else 0
percent_fn_basal = (fn_basal / total_basal) * 100 if total_basal > 0 else 0

percent_tp_radial = (tp_radial / total_radial) * 100 if total_radial > 0 else 0
percent_fn_radial = (fn_radial / total_radial) * 100 if total_radial > 0 else 0

# Print results
print(f'Globlet Cells - TP: {percent_tp_globlet:.2f}%, FN: {percent_fn_globlet:.2f}%')
print(f'Basal Cells - TP: {percent_tp_basal:.2f}%, FN: {percent_fn_basal:.2f}%')
print(f'Radial Cells - TP: {percent_tp_radial:.2f}%, FN: {percent_fn_radial:.2f}%')

# Create a confusion matrix array
conf_matrix_array = np.array([
    [percent_tp_basal, percent_fn_basal],
    [percent_tp_radial, percent_fn_radial],
    [percent_tp_globlet, percent_fn_globlet]
])

save_path = Path(goblet_cells_file).parent.stem
# Plot the confusion matrix
plt.figure(figsize=(8, 6))
class_names = list(class_map_gbr.values())
sns.heatmap(conf_matrix_array, annot=True, fmt='.2f', cmap='Blues', xticklabels=['TP', 'FN'], yticklabels=class_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix (Percentages)')
plt.savefig(os.path.join(save_dir,save_path + '.png'))
plt.show()

# %%


# %%



