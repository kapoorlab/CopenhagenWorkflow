import pandas as pd
from pathlib import Path
import tifffile as tiff
import os

def color_segmentation_by_cell_type(segmentation_img_path, cell_data_csv, output_img_path):
    segmentation_img = tiff.imread(segmentation_img_path) 
    print(f'Segmntation image shape TZYX {segmentation_img.shape}')
    cell_data = pd.read_csv(cell_data_csv)

    cell_type_colors = {
        'Basal': 1,    
        'Radial': 2,
        'Goblet': 3,    
    }
    
    colored_segmentation = segmentation_img.copy()
    
    for cell_type, color in cell_type_colors.items():
        cell_type_data = cell_data[cell_data['Cell_Type'] == cell_type]
        
        for _, row in cell_type_data.iterrows():
            t, z, y, x = int(row['t']), int(row['z']), int(row['y']), int(row['x'])

            if 0 <= t < segmentation_img.shape[0] and 0 <= z < segmentation_img.shape[1] and \
               0 <= y < segmentation_img.shape[2] and 0 <= x < segmentation_img.shape[3]:
                label = segmentation_img[t, z, y, x]
                
                colored_segmentation[segmentation_img == label] = color

    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    tiff.imwrite(output_img_path, colored_segmentation)
    print(f"Saved colored segmentation image to {output_img_path}")

dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, 'dataframes/')
goblet_basal_radial_dataframe = os.path.join(data_frames_dir, f'goblet_basal_dataframe_normalized_{channel}predicted_morpho_feature_attention_shallowest_litest.csv')

save_dir = os.path.join(tracking_directory, f'cell_fate_{channel}colored_segmentation/')
Path(save_dir).mkdir(exist_ok=True)
segmentation_img_path = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/seg_nuclei_timelapses/{timelapse_to_track}.tif'  

output_img_path = os.path.join(save_dir, f'{timelapse_to_track}_colored.tif')

color_segmentation_by_cell_type(segmentation_img_path, goblet_basal_radial_dataframe, output_img_path)
