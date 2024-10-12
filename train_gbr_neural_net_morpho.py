#!/usr/bin/env python
# coding: utf-8


import os
from napatrackmater.Trackvector import (
    train_gbr_neural_net
)




home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
#/home/debian/jz/
#/gpfsstore/rech/jsy/uzj81mi/
channel = 'nuclei_'
base_dir =  f'{home_folder}Mari_Data_Training/track_training_data/'
model_dir = f'{home_folder}Mari_Models/TrackModels/'
morpho_model_dir = os.path.join(model_dir, f'morpho_feature_lightning_attention_gbr_25_{channel}shallowest_litest/')
os.makedirs(morpho_model_dir, exist_ok = True)
morpho_gbr_h5_file = f'morpho_training_data_gbr_25_{channel}.h5'
num_classes = 3
batch_size = 9800
epochs = 100
block_config = (6)
attention_dim=64
n_pos=(8,)
growth_rate = 4
train_gbr_neural_net(
    save_path = morpho_model_dir,
    h5_file = os.path.join(base_dir, morpho_gbr_h5_file),
    num_classes = num_classes,
    batch_size = batch_size,
    epochs = epochs,
    model_type = 'attention',
    experiment_name='morpho_attention',
    num_workers = 10,
    block_config=block_config,
    attention_dim = attention_dim,
    n_pos = n_pos,
    growth_rate = growth_rate
)




