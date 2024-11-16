import os
import argparse
from napatrackmater.Trackvector import train_gbr_neural_net



block_config = (6)
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
base_dir = f'{home_folder}Mari_Data_Training/track_training_data/'
morpho_model_dir = '/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/morpho_nuclei_membrane/'


morpho_gbr_h5_file = 'morphodynamic_training_data_gbr_25_nuclei_membrane_.h5'
growth_rate = 4
os.makedirs(morpho_model_dir, exist_ok=True)

# Train the model
train_gbr_neural_net(
    save_path=morpho_model_dir,
    h5_file=os.path.join(base_dir, morpho_gbr_h5_file),
    num_classes=3,
    batch_size=98000,
    epochs=100,
    model_type='attention',
    experiment_name='morpho_attention',
    num_workers=10,
    block_config=block_config,
    attention_dim=64,
    n_pos=(8,),
    growth_rate=growth_rate,
)
