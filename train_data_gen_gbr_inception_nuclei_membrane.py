import os
import numpy as np
from napatrackmater import create_analysis_tracklets
import pandas as pd
from tqdm import tqdm 
from napatrackmater.Trackvector import (
    SHAPE_FEATURES,
    DYNAMIC_FEATURES,
    SHAPE_DYNAMIC_FEATURES
)
from sklearn.model_selection import train_test_split
import h5py

def process_datasets(home_folder, dataset_names, tracking_directory_name='nuclei_membrane_tracking/', tracklet_length=25, stride=4):
    training_arrays = []  
    labels = []
    channel='nuclei_'
    second_channel = 'membrane_'

    def create_training_arrays(array, tracklet_length, stride):
        processed_arrays = []
        N, F = array.shape
        if N > tracklet_length:
            num_subarrays = (N - tracklet_length) // stride + 1
            for i in range(num_subarrays):
                start_index = i * stride
                end_index = start_index + tracklet_length
                sub_array = array[start_index:end_index, :]
                processed_arrays.append(sub_array)
        return np.asarray(processed_arrays)

    for dataset_name in tqdm(dataset_names):
        tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}/{tracking_directory_name}'
        data_frames_dir = os.path.join(tracking_directory, 'dataframes/')
        normalized_dataframe_file = os.path.join(data_frames_dir, f'goblet_basal_dataframe_normalized_{channel}.csv')
        dataset_dataframe = pd.read_csv(normalized_dataframe_file)
        analysis_vectors, _ = create_analysis_tracklets(dataset_dataframe)
        cell_type_dataframe = dataset_dataframe[~dataset_dataframe['Cell_Type'].isna()]

        second_normalized_dataframe_file = os.path.join(data_frames_dir, f'goblet_basal_dataframe_normalized_{second_channel}.csv')
        second_dataset_dataframe = pd.read_csv(second_normalized_dataframe_file)
        second_analysis_vectors, _ = create_analysis_tracklets(second_dataset_dataframe)

        # Iterate over TrackMate Track IDs instead of individual Track IDs directly
        for trackmate_track_id, group_df in tqdm(cell_type_dataframe.groupby("TrackMate Track ID")):
            basal_tracklet_ids = group_df[group_df['Cell_Type'] == 'Basal']['Track ID']
            goblet_tracklet_ids = group_df[group_df['Cell_Type'] == 'Goblet']['Track ID']
            radial_tracklet_ids = group_df[group_df['Cell_Type'] == 'Radial']['Track ID']

            for tracklet_ids, cell_label in [
                (basal_tracklet_ids, 0),
                (goblet_tracklet_ids, 2),
                (radial_tracklet_ids, 1)
            ]:
                for track_id in tracklet_ids:
                    try:
                        (shape_dynamic_dataframe_list, shape_dataframe_list, dynamic_dataframe_list, full_dataframe_list) = analysis_vectors[track_id]
                        (second_shape_dynamic_dataframe_list, second_shape_dataframe_list, second_dynamic_dataframe_list, second_full_dataframe_list) = second_analysis_vectors[track_id]
                        shape_track_array = np.array([[item for item in record.values()] for record in shape_dataframe_list])
                        dynamic_track_array = np.array([[item for item in record.values()] for record in dynamic_dataframe_list])
                        second_shape_track_array = np.array([[item for item in record.values()] for record in second_shape_dataframe_list])
                        # Combine shape and dynamic features
                        print(shape_track_array.shape, dynamic_track_array.shape, second_shape_track_array.shape)
                        combined_track_array = np.concatenate((shape_track_array, dynamic_track_array, second_shape_track_array), axis=-1)

                        # Create training arrays
                        training_subarrays = create_training_arrays(combined_track_array, tracklet_length, stride)
                        training_arrays.extend(training_subarrays)
                        labels.extend([cell_label] * len(training_subarrays))
                    except KeyError:
                        print(f'key {track_id} not found, skipping')

    # Convert lists to numpy arrays
    training_arrays = np.asarray(training_arrays)
    labels = np.asarray(labels)

    # Split the combined data into training and validation sets
    train_arrays, val_arrays, train_labels, val_labels = train_test_split(
        training_arrays, labels, test_size=0.1, random_state=42
    )

    train_save_dir = f'{home_folder}Mari_Data_Training/track_training_data/'

    combined_h5_training_data = {
        'train_arrays': train_arrays,
        'train_labels': train_labels,
        'val_arrays': val_arrays,
        'val_labels': val_labels
    }

    # Save combined shape and dynamic features in a single H5 file
    with h5py.File(os.path.join(train_save_dir, f'morphodynamic_training_data_gbr_{tracklet_length}_{channel}{second_channel}.h5'), 'w') as hf:
        for key, value in combined_h5_training_data.items():
            hf.create_dataset(key, data=value)

home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
dataset_name = [
    'Second_Dataset_Analysis', 'Fifth_Dataset_Analysis', 'Sixth_Dataset_Analysis', 
    'Fifth_Extra_Goblet', 'Fifth_Extra_Radial']
 #, 'Third_Extra_Goblet', 'Third_Extra_Radial']
tracklet_lengths = [25]
strides = [4]
for index, tracklet_length in enumerate(tracklet_lengths):
    stride = strides[index]
    process_datasets(home_folder, dataset_name,  tracklet_length=tracklet_length, stride=stride)
