#!/usr/bin/env python
# coding: utf-8



import os
from napatrackmater.Trackvector import (
    train_gbr_neural_net
)




home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
#/gpfsstore/rech/jsy/uzj81mi/
channel = 'nuclei_'
base_dir =  f'{home_folder}Mari_Data_Training/track_training_data/'
model_dir = f'{home_folder}Mari_Models/TrackModels/'
shape_model_dir = os.path.join(model_dir, f'shape_feature_lightning_densenet_gbr_25_{channel}lite/')
os.makedirs(shape_model_dir, exist_ok = True)
shape_gbr_h5_file = f'shape_training_data_gbr_25_{channel}.h5'
num_classes = 3
batch_size = 10240
epochs = 100
block_config = (6,12, 24)
growth_rate = 4
train_gbr_neural_net(
    save_path = shape_model_dir,
    h5_file = os.path.join(base_dir, shape_gbr_h5_file),
    num_classes = num_classes,
    batch_size = batch_size,
    epochs = epochs,
    model_type='densenet',
    experiment_name='shape_densenet',
    num_workers = 10,
    block_config = block_config,
    growth_rate = growth_rate
    
)




