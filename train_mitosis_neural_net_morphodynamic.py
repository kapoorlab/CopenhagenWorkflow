#!/usr/bin/env python
# coding: utf-8


import os
from napatrackmater.Trackvector import (
    train_mitosis_neural_net
)

home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
base_dir =  f'{home_folder}Mari_Data_Training/track_training_data/'
model_dir = f'{home_folder}Mari_Models/TrackModels/'

dynamic_model_dir = os.path.join(model_dir, 'morphodynamic_features_mitosis')
os.makedirs(dynamic_model_dir, exist_ok = True)
morphodynamic_mitosis_h5_file = 'morphodynamic_training_data_mitosis_nuclei_25.h5'

num_classes = 2
batch_size = 98000
epochs = 100
block_config = (6)
growth_rate = 4

train_mitosis_neural_net(
    h5_file = os.path.join(base_dir, morphodynamic_mitosis_h5_file),
    save_path = dynamic_model_dir,
    num_classes = num_classes,
    batch_size = batch_size,
    epochs = epochs,
    model_type = 'densenet',
    experiment_name='morphodynamic_mitosis_densenet',
    num_workers = 10,
    block_config=block_config,
    growth_rate = growth_rate,
    attention_dim=64,
    n_pos=(8,)
)




