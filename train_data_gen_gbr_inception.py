#!/usr/bin/env python
# coding: utf-8




import os
import numpy as np
from napatrackmater import  create_analysis_tracklets
import pandas as pd
from napatrackmater.Trackvector import (
    SHAPE_FEATURES,
    DYNAMIC_FEATURES,
    SHAPE_DYNAMIC_FEATURES
)
from sklearn.model_selection import train_test_split
import h5py


def process_datasets(home_folder, dataset_names, channel='nuclei_', tracking_directory_name='nuclei_membrane_tracking/', tracklet_length=50, stride=4):
    shape_training_arrays_basal = []
    dynamic_training_arrays_basal = []
    shape_training_arrays_goblet = []
    dynamic_training_arrays_goblet = []
    shape_training_arrays_radial = []
    dynamic_training_arrays_radial = []

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

    for dataset_name in dataset_names:
        tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/{tracking_directory_name}'
        data_frames_dir = os.path.join(tracking_directory, 'dataframes/')
        normalized_dataframe_file = os.path.join(data_frames_dir, f'goblet_basal_dataframe_normalized_{channel}.csv')
        dataset_dataframe = pd.read_csv(normalized_dataframe_file)
        analysis_vectors, _ = create_analysis_tracklets(dataset_dataframe)
        
        cell_type_dataframe = dataset_dataframe[~dataset_dataframe['Cell_Type'].isna()]
        basal_tracklet_ids = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == 'Basal']['Track ID']
        goblet_tracklet_ids = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == 'Goblet']['Track ID']
        radial_tracklet_ids = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == 'Radial']['Track ID']

        for tracklet_ids, shape_training_arrays, dynamic_training_arrays in [
            (basal_tracklet_ids, shape_training_arrays_basal, dynamic_training_arrays_basal),
            (goblet_tracklet_ids, shape_training_arrays_goblet, dynamic_training_arrays_goblet),
            (radial_tracklet_ids, shape_training_arrays_radial, dynamic_training_arrays_radial)
        ]:
            for track_id in tracklet_ids:
                try:
                    (shape_dynamic_dataframe_list, shape_dataframe_list, dynamic_dataframe_list, full_dataframe_list) = analysis_vectors[track_id]
                    shape_track_array = np.array([[item for item in record.values()] for record in shape_dataframe_list])
                    dynamic_track_array = np.array([[item for item in record.values()] for record in dynamic_dataframe_list])
                    shape_training_arrays.extend(create_training_arrays(shape_track_array, tracklet_length, stride))
                    dynamic_training_arrays.extend(create_training_arrays(dynamic_track_array, tracklet_length, stride))
                except KeyError:
                    print(f'key {track_id} not found, skipping')

    shape_training_arrays_basal = np.asarray(shape_training_arrays_basal)
    dynamic_training_arrays_basal = np.asarray(dynamic_training_arrays_basal)
    shape_training_arrays_goblet = np.asarray(shape_training_arrays_goblet)
    dynamic_training_arrays_goblet = np.asarray(dynamic_training_arrays_goblet)
    shape_training_arrays_radial = np.asarray(shape_training_arrays_radial)
    dynamic_training_arrays_radial = np.asarray(dynamic_training_arrays_radial)

    shape_basal_labels = np.zeros(len(shape_training_arrays_basal), dtype=int)
    dynamic_basal_labels = np.zeros(len(dynamic_training_arrays_basal), dtype=int)
    shape_goblet_labels = np.full(len(shape_training_arrays_goblet), 2, dtype=int)
    dynamic_goblet_labels = np.full(len(dynamic_training_arrays_goblet), 2, dtype=int)
    shape_radial_labels = np.ones(len(shape_training_arrays_radial), dtype=int)
    dynamic_radial_labels = np.ones(len(dynamic_training_arrays_radial), dtype=int)

    train_save_dir = f'{home_folder}Mari_Data_Training/track_training_data/'
    
    shape_goblet_train_arrays, shape_goblet_val_arrays, shape_goblet_train_labels, shape_goblet_val_labels = train_test_split(
    shape_training_arrays_goblet, shape_goblet_labels, test_size=0.2, random_state=42)

    shape_basal_train_arrays, shape_basal_val_arrays, shape_basal_train_labels, shape_basal_val_labels = train_test_split(
        shape_training_arrays_basal, shape_basal_labels, test_size=0.2, random_state=42)

    shape_radial_train_arrays, shape_radial_val_arrays, shape_radial_train_labels, shape_radial_val_labels = train_test_split(
        shape_training_arrays_radial, shape_radial_labels, test_size=0.2, random_state=42)

    train_shape_arrays = np.concatenate((shape_basal_train_arrays, shape_radial_train_arrays, shape_goblet_train_arrays ))
    train_shape_labels = np.concatenate((shape_basal_train_labels, shape_radial_train_labels, shape_goblet_train_labels ))
    val_shape_arrays = np.concatenate((shape_basal_val_arrays, shape_radial_val_arrays, shape_goblet_val_arrays ))
    val_shape_labels = np.concatenate((shape_basal_val_labels, shape_radial_val_labels, shape_goblet_val_labels ))



    shape_h5_training_data = {
        'train_arrays': train_shape_arrays,
        'train_labels': train_shape_labels,
        'val_arrays': val_shape_arrays,
        'val_labels': val_shape_labels
    }


    with h5py.File(os.path.join(train_save_dir, f'shape_training_data_gbr_{tracklet_length}.h5'), 'w') as hf:
        for key, value in shape_h5_training_data.items():
            hf.create_dataset(key, data=value)

    dynamic_goblet_train_arrays, dynamic_goblet_val_arrays, dynamic_goblet_train_labels, dynamic_goblet_val_labels = train_test_split(
    dynamic_training_arrays_goblet, dynamic_goblet_labels, test_size=0.2, random_state=42)

    dynamic_basal_train_arrays, dynamic_basal_val_arrays, dynamic_basal_train_labels, dynamic_basal_val_labels = train_test_split(
        dynamic_training_arrays_basal, dynamic_basal_labels, test_size=0.2, random_state=42)

    dynamic_radial_train_arrays, dynamic_radial_val_arrays, dynamic_radial_train_labels, dynamic_radial_val_labels = train_test_split(
        dynamic_training_arrays_radial, dynamic_radial_labels, test_size=0.2, random_state=42)

    train_dynamic_arrays = np.concatenate((dynamic_basal_train_arrays, dynamic_radial_train_arrays, dynamic_goblet_train_arrays ))
    train_dynamic_labels = np.concatenate((dynamic_basal_train_labels, dynamic_radial_train_labels, dynamic_goblet_train_labels ))
    val_dynamic_arrays = np.concatenate((dynamic_basal_val_arrays, dynamic_radial_val_arrays, dynamic_goblet_val_arrays ))
    val_dynamic_labels = np.concatenate((dynamic_basal_val_labels, dynamic_radial_val_labels, dynamic_goblet_val_labels ))
   
    dynamic_h5_training_data = {
        'train_arrays': train_dynamic_arrays,
        'train_labels': train_dynamic_labels,
        'val_arrays': val_dynamic_arrays,
        'val_labels': val_dynamic_labels
    }


    with h5py.File(os.path.join(train_save_dir, f'dynamic_training_data_gbr_{tracklet_length}.h5'), 'w') as hf:
        for key, value in dynamic_h5_training_data.items():
            hf.create_dataset(key, data=value)



home_folder = '/lustre/fsstor/projects/rech/jsy/uzj81mi/'
dataset_name = ['Second', 'Fifth']
tracklet_lengths = [10,25,50,75,100]
strides = [10,5,4,4,4]
for index, tracklet_length in enumerate(tracklet_lengths):
  stride = strides[index]
  process_datasets(home_folder, dataset_name, tracklet_length=tracklet_length, stride=stride)