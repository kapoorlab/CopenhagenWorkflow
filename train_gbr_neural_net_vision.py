import os
import argparse
from napatrackmater.Trackvector import train_gbr_vision_neural_net

# Argument parsing to configure parameters for multiple runs
parser = argparse.ArgumentParser(description='Train Cell Fate Vision Model.')
parser.add_argument('--vision_model_dir', type=str, required=True, help='Path to save the trained model.')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size.')
parser.add_argument('--vision_gbr_h5_file', type=str, required=True, help='H5 file containing training data.')


args = parser.parse_args()
depth = {'depth_0': 6,'depth_1': 12,'depth_2': 24,'depth_3': 16 }
stage_number = 4
input_shape = [10,128,128,8]
batch_size = 32

home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
base_dir = f'{home_folder}Mari_Data_Training/vision_track_training_data/'



train_gbr_vision_neural_net(
    save_path=args.vision_model_dir,
    h5_file=os.path.join(base_dir, args.vision_gbr_h5_file),
    input_shape = input_shape,
    num_classes=3,
    batch_size=batch_size,
    epochs=100,
    stage_number=stage_number,
    depth=depth,
    experiment_name='vision_cellfate',
    num_workers=10
)
