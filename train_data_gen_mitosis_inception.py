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

def process_datasets(home_folder, dataset_names, channel='nuclei_', tracking_directory_name = 'nuclei_membrane_tracking/', tracklet_length=50, stride=4):
    shape_training_arrays_dividing = []
    dynamic_training_arrays_dividing = []
    shape_training_arrays_non_dividing = []
    dynamic_training_arrays_non_dividing = []

    for dataset_name in dataset_names:
        tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/{tracking_directory_name}'
        data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
        normalized_dataframe_file = os.path.join(data_frames_dir, f'results_dataframe_normalized_{channel}.csv')
        dataset_dataframe = pd.read_csv(normalized_dataframe_file)
        analysis_vectors, vanilla_dataframe = create_analysis_tracklets(dataset_dataframe)
        dividing_tracklet_ids = dataset_dataframe[dataset_dataframe['Dividing'] == 1]['Track ID']
        non_dividing_tracklet_ids = dataset_dataframe[dataset_dataframe['Dividing'] == 0]['Track ID']

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

        for track_id in dividing_tracklet_ids:
            try:
                (shape_dynamic_dataframe_list, shape_dataframe_list, dynamic_dataframe_list, full_dataframe_list) = analysis_vectors[track_id]
                shape_track_array = np.array([[item for item in record.values()] for record in shape_dataframe_list])
                dynamic_track_array = np.array([[item for item in record.values()] for record in dynamic_dataframe_list])
                shape_training_arrays_dividing.extend(create_training_arrays(shape_track_array, tracklet_length, stride))
                dynamic_training_arrays_dividing.extend(create_training_arrays(dynamic_track_array, tracklet_length, stride))
            except KeyError:
                print(f'key {track_id} not found, skipping')

        for track_id in non_dividing_tracklet_ids:
            try:
                (shape_dynamic_dataframe_list, shape_dataframe_list, dynamic_dataframe_list, full_dataframe_list) = analysis_vectors[track_id]
                shape_track_array = np.array([[item for item in record.values()] for record in shape_dataframe_list])
                dynamic_track_array = np.array([[item for item in record.values()] for record in dynamic_dataframe_list])
                shape_training_arrays_non_dividing.extend(create_training_arrays(shape_track_array, tracklet_length, stride))
                dynamic_training_arrays_non_dividing.extend(create_training_arrays(dynamic_track_array, tracklet_length, stride))
            except KeyError:
                print(f'key {track_id} not found, skipping')

    shape_training_arrays_dividing = np.asarray(shape_training_arrays_dividing)
    dynamic_training_arrays_dividing = np.asarray(dynamic_training_arrays_dividing)
    shape_training_arrays_non_dividing = np.asarray(shape_training_arrays_non_dividing)
    dynamic_training_arrays_non_dividing = np.asarray(dynamic_training_arrays_non_dividing)

    shape_dividing_labels = np.ones(len(shape_training_arrays_dividing), dtype=int)
    dynamic_dividing_labels = np.ones(len(dynamic_training_arrays_dividing), dtype=int)
    shape_non_dividing_labels = np.zeros(len(shape_training_arrays_non_dividing), dtype=int)
    dynamic_non_dividing_labels = np.zeros(len(dynamic_training_arrays_non_dividing), dtype=int)

    train_save_dir = f'{home_folder}Mari_Data_Training/track_training_data/'

    shape_dividing_train_arrays, shape_dividing_val_arrays, shape_dividing_train_labels, shape_dividing_val_labels = train_test_split(
        shape_training_arrays_dividing, shape_dividing_labels, test_size=0.2, random_state=42)
    shape_non_dividing_train_arrays, shape_non_dividing_val_arrays, shape_non_dividing_train_labels, shape_non_dividing_val_labels = train_test_split(
        shape_training_arrays_non_dividing, shape_non_dividing_labels, test_size=0.2, random_state=42)
    
    train_shape_arrays = np.concatenate( (shape_dividing_train_arrays, shape_non_dividing_train_arrays))
    train_shape_labels = np.concatenate((shape_dividing_train_labels, shape_non_dividing_train_labels))
    val_shape_arrays = np.concatenate( (shape_dividing_val_arrays, shape_non_dividing_val_arrays))
    val_shape_labels = np.concatenate((shape_dividing_val_labels, shape_non_dividing_val_labels))


    shape_h5_training_data = {
        'train_arrays': train_shape_arrays,
        'train_labels': train_shape_labels,
        'val_arrays': val_shape_arrays,
        'val_labels': val_shape_labels
    }

    with h5py.File(os.path.join(train_save_dir, f'shape_training_data_mitosis_{tracklet_length}.h5'), 'w') as hf:
        for key, value in shape_h5_training_data.items():
            hf.create_dataset(key, data=value)

    dynamic_dividing_train_arrays, dynamic_dividing_val_arrays, dynamic_dividing_train_labels, dynamic_dividing_val_labels = train_test_split(
        dynamic_training_arrays_dividing, dynamic_dividing_labels, test_size=0.2, random_state=42)
    dynamic_non_dividing_train_arrays, dynamic_non_dividing_val_arrays, dynamic_non_dividing_train_labels, dynamic_non_dividing_val_labels = train_test_split(
        dynamic_training_arrays_non_dividing, dynamic_non_dividing_labels, test_size=0.2, random_state=42)
    
    train_dynamic_arrays = np.concatenate( (dynamic_dividing_train_arrays, dynamic_non_dividing_train_arrays))
    train_dynamic_labels = np.concatenate((dynamic_dividing_train_labels, dynamic_non_dividing_train_labels))
    val_dynamic_arrays = np.concatenate( (dynamic_dividing_val_arrays, dynamic_non_dividing_val_arrays))
    val_dynamic_labels = np.concatenate((dynamic_dividing_val_labels, dynamic_non_dividing_val_labels))



    dynamic_h5_training_data = {
        'train_arrays': train_dynamic_arrays,
        'train_labels': train_dynamic_labels,
        'val_arrays': val_dynamic_arrays,
        'val_labels': val_dynamic_labels
    }

    with h5py.File(os.path.join(train_save_dir, f'dynamic_training_data_mitosis_{tracklet_length}.h5'), 'w') as hf:
        for key, value in dynamic_h5_training_data.items():
            hf.create_dataset(key, data=value)

home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
dataset_names = ['Third', 'Second', 'Fourth']
tracklet_lengths = [10,25,50,75,100]
strides = [10,5,4,4,4]
for index, tracklet_length in enumerate(tracklet_lengths):
  stride = strides[index]
  process_datasets(home_folder, dataset_names, tracklet_length=tracklet_length, stride=stride)

