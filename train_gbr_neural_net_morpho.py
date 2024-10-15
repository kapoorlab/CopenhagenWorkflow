import os
import argparse
from napatrackmater.Trackvector import train_gbr_neural_net

# Argument parsing to configure parameters for multiple runs
parser = argparse.ArgumentParser(description='Train Morphodynamic Neural Network with Attention.')
parser.add_argument('--morpho_model_dir', type=str, required=True, help='Path to save the trained model.')
parser.add_argument('--block_config', type=str, required=True, help='Configuration of blocks (e.g., "(6, 12, 24)").')
parser.add_argument('--growth_rate', type=int, default=32, help='Growth rate for the model.')
parser.add_argument('--channel', type=str, default='nuclei_', help='Channel type (e.g., "membrane_", "nuclei_").')
parser.add_argument('--morpho_gbr_h5_file', type=str, required=True, help='H5 file containing training data.')

args = parser.parse_args()

# Convert block_config from string to tuple of integers

block_config_str = args.block_config.strip("()")
if "," in block_config_str:
    block_config = tuple(map(int, block_config_str.split(',')))
else:
    block_config = (int(block_config_str),)
# Static configurations
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
base_dir = f'{home_folder}Mari_Data_Training/track_training_data/'

# Create model save directory
os.makedirs(args.morpho_model_dir, exist_ok=True)

# Train the model
train_gbr_neural_net(
    save_path=args.morpho_model_dir,
    h5_file=os.path.join(base_dir, args.morpho_gbr_h5_file),
    num_classes=3,
    batch_size=98000 // 2,
    epochs=100,
    model_type='attention',
    experiment_name='morpho_attention',
    num_workers=10,
    block_config=block_config,
    attention_dim=64,
    n_pos=(8,),
    growth_rate=args.growth_rate,
)
