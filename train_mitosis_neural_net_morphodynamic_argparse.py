#!/usr/bin/env python
# coding: utf-8


import os
import argparse
from napatrackmater.Trackvector import (
    train_mitosis_neural_net
)


parser = argparse.ArgumentParser(description='Train Morphodynamic Neural Network with Attention for Mitosis.')
parser.add_argument('--morpho_model_dir', type=str, required=True, help='Path to save the trained model.')
parser.add_argument('--block_config', type=str, required=True, help='Configuration of blocks (e.g., "(6, 12, 24)").')
parser.add_argument('--growth_rate', type=int, default=32, help='Growth rate for the model.')
parser.add_argument('--morphodynamic_mitosis_h5_file', type=str, required=True, help='H5 file containing training data.')

args = parser.parse_args()


home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
base_dir =  f'{home_folder}Mari_Data_Training/track_training_data/'

os.makedirs(args.morpho_model_dir, exist_ok = True)
block_config_str = args.block_config.strip("()")
if "," in block_config_str:
    block_config = tuple(map(int, block_config_str.split(',')))
else:
    block_config = (int(block_config_str),)

num_classes = 2
batch_size = 98000
epochs = 100
growth_rate = args.growth_rate

train_mitosis_neural_net(
    h5_file = os.path.join(base_dir, args.morphodynamic_mitosis_h5_file),
    save_path = args.morpho_model_dir,
    num_classes = num_classes,
    batch_size = batch_size,
    epochs = epochs,
    model_type = 'attention',
    experiment_name='morphodynamic_mitosis_densenet',
    num_workers = 10,
    block_config=block_config,
    growth_rate = growth_rate,
    attention_dim=64,
    n_pos=(8,)
)




