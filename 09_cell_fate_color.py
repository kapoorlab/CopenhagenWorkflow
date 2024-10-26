import pandas as pd
import numpy as np
import tifffile as tiff
from skimage.util import map_array
from tqdm import tqdm
import os

def color_segmentation_by_cell_type(segmentation_img_path, cell_data_csv, output_img_path):
    # Load segmentation image and cell data
    segmentation_img = tiff.imread(segmentation_img_path)  # Assumes TZYX format
    print(f'Segmentation image shape TZYX: {segmentation_img.shape}')
    cell_data = pd.read_csv(cell_data_csv)

    # Define color mappings for each cell type as unique labels
    cell_type_colors = {
        'Basal': 1,    # New label for Basal cells
        'Radial': 2,   # New label for Radial cells
        'Goblet': 3    # New label for Goblet cells
    }
    
    # Prepare the output image as a copy of the segmentation image
    colored_segmentation = segmentation_img.copy()
    
    # Iterate over each timepoint
    for t in tqdm(range(segmentation_img.shape[0]), desc="Processing Timepoints"):
        # Filter data for the current timepoint
        timepoint_data = cell_data[cell_data['t'] == t]

        # Lists to store original and new labels for this timepoint
        original_labels = []
        new_labels = []

        # Populate original and new labels based on cell type positions at this timepoint
        for cell_type, new_label in cell_type_colors.items():
            cell_type_data = timepoint_data[timepoint_data['Cell_Type'] == cell_type]
            
            # Extract unique segmentation labels at the specified positions
            for _, row in cell_type_data.iterrows():
                z, y, x = int(row['z']), int(row['y']), int(row['x'])
                
                # Ensure coordinates are within image bounds
                if 0 <= z < segmentation_img.shape[1] and \
                   0 <= y < segmentation_img.shape[2] and 0 <= x < segmentation_img.shape[3]:
                    label = segmentation_img[t, z, y, x]
                    original_labels.append(label)
                    new_labels.append(new_label)

        # Convert lists to numpy arrays for efficient mapping
        original_labels = np.array(original_labels)
        new_labels = np.array(new_labels)

        # Apply mapping for the current timepoint
        colored_segmentation[t] = map_array(segmentation_img[t], original_labels, new_labels)

    # Save the relabeled image
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    tiff.imwrite(output_img_path, colored_segmentation)
    print(f"Saved colored segmentation image to {output_img_path}")

# Usage
dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, 'dataframes/')
goblet_basal_radial_dataframe = os.path.join(data_frames_dir, f'goblet_basal_dataframe_normalized_{channel}predicted_morpho_feature_attention_shallowest_litest.csv')

save_dir = os.path.join(tracking_directory, f'cell_fate_{channel}colored_segmentation/')
segmentation_img_path = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/seg_nuclei_timelapses/{timelapse_to_track}.tif'  
output_img_path = os.path.join(save_dir, f'{timelapse_to_track}_colored.tif')

color_segmentation_by_cell_type(segmentation_img_path, goblet_basal_radial_dataframe, output_img_path)
