import os
from pathlib import Path
from napatrackmater import create_h5, filter_and_get_tracklets
import pandas as pd

def process_datasets(home_folder, dataset_names, image_dataset_names, image_folder_names,  channel, train_save_dir, tracking_directory_name='nuclei_membrane_tracking/', time_window = 10, crop_size = [256,256,8]):


    for idx, dataset_name in enumerate(dataset_names):
        image_dataset_name = image_dataset_names[idx]
        image_folder_name = image_folder_names[idx]
        tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}/{tracking_directory_name}'
        raw_image = f'{home_folder}Mari_Data_Oneat/Mari_{image_folder_name}/{channel}timelapses/timelapse_{image_dataset_name.lower()}_dataset.tif'
        segmentation_image = f'{home_folder}Mari_Data_Oneat/Mari_{image_folder_name}/seg_{channel}timelapses/timelapse_{image_dataset_name.lower()}_dataset.tif'
        data_frames_dir = os.path.join(tracking_directory, 'dataframes/')
        normalized_dataframe_file = os.path.join(data_frames_dir, f'goblet_basal_dataframe_normalized_{channel}.csv')
        dataset_dataframe = pd.read_csv(normalized_dataframe_file)
        
        cell_type_dataframe = dataset_dataframe[~dataset_dataframe['Cell_Type'].isna()]
        class_map_gbr = {
            0: "Basal",
            1: "Radial",
            2: "Goblet"
        }

        for train_label, cell_type in class_map_gbr.items(): 

            filter_and_get_tracklets(cell_type_dataframe, cell_type, time_window, raw_image, crop_size, segmentation_image, dataset_name, train_save_dir, 
                                train_label)
       
        create_h5(train_save_dir,train_size=0.95,save_name="cellfate_vision_training_data_gbr")
 
    

home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
dataset_name = [
    'Second_Dataset_Analysis', 'Fifth_Dataset_Analysis', 'Sixth_Dataset_Analysis', 
    'Fifth_Extra_Goblet', 'Fifth_Extra_Radial', 'Third_Extra_Goblet', 'Third_Extra_Radial']
image_folder_name = [
    'Second_Dataset_Analysis', 'Fifth_Dataset_Analysis', 'Sixth_Dataset_Analysis', 
    'Fifth_Dataset_Analysis', 'Fifth_Dataset_Analysis', 'Third_Dataset_Analysis', 'Third_Dataset_Analysis']
image_dataset_name = [
    'Second', 'Fifth', 'Sixth', 
    'Fifth', 'Fifth', 'Third', 'Third']
time_window = 25
crop_size = [128,128,8]
train_save_dir = f'{home_folder}Mari_Data_Training/vision_track_training_data/'
Path(train_save_dir).mkdir(exist_ok=True)
process_datasets(home_folder, dataset_name, image_dataset_name, image_folder_name,  channel='nuclei_', train_save_dir=train_save_dir, time_window=time_window, crop_size = crop_size)
