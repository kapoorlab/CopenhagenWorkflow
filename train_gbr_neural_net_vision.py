import os
import argparse
from napatrackmater.Trackvector import train_gbr_vision_neural_net

# Argument parsing to configure parameters for multiple runs
parser = argparse.ArgumentParser(description='Train Cell Fate Vision Model.')
parser.add_argument('--vision_model_dir', type=str, required=True, help='Path to save the trained model.')
parser.add_argument('--input_shape', type=str, required=True, help='Input TZYX shape tuple.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size.')
parser.add_argument('--vision_gbr_h5_file', type=str, required=True, help='H5 file containing training data.')


args = parser.parse_args()
depth = {'depth_0': 6,'depth_1': 12,'depth_2': 24,'depth_3': 16 }
stage_number = 4
block_config_str = args.block_config.strip("()")
if "," in block_config_str:
    block_config = tuple(map(int, block_config_str.split(',')))
else:
    block_config = (int(block_config_str),)
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
base_dir = f'{home_folder}Mari_Data_Training/vision_track_training_data/'

os.makedirs(args.vision_model_dir, exist_ok=True)

train_gbr_vision_neural_net(
    save_path=args.vision_model_dir,
    h5_file=os.path.join(base_dir, args.vision_gbr_h5_file),
    input_shape = args.input_shape,
    num_classes=3,
    batch_size=64,
    epochs=100,
    stage_number=stage_number,
    depth=depth,
    experiment_name='vision_cellfate',
    num_workers=10
)
