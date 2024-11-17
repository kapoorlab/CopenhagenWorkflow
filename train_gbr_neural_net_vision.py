import os
from pathlib import Path
from napatrackmater.Trackvector import train_gbr_vision_neural_net


vision_model_dir = '/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/vision_inception/'

Path(vision_model_dir).mkdir(exist_ok=True)
depth = {'depth_0': 12,'depth_1': 24,'depth_2': 16 }
crop_size = [75, 8, 128, 128]
input_shape = crop_size
batch_size = 16
growth_rate = 4
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
base_dir = f'{home_folder}Mari_Data_Training/vision_track_training_data_large_window_75/'
vision_gbr_h5_file = 'cellfate_vision_training_data_gbr_large.h5'


train_gbr_vision_neural_net(
    save_path=vision_model_dir,
    h5_file=os.path.join(base_dir, vision_gbr_h5_file),
    input_shape = input_shape,
    num_classes=3,
    batch_size=batch_size,
    epochs=100,
    depth=depth,
    experiment_name='vision_cellfate',
    num_workers=10,
    crop_size = crop_size,
    growth_rate = growth_rate
    
)
