# %%
from pathlib import Path 
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_models import DenseNet
from kapoorlabs_lightning.lightning_trainer import LightningModel
from napatrackmater.Trackvector import (
    inception_model_prediction,
    save_cell_type_predictions,
    SHAPE_FEATURES,
    DYNAMIC_FEATURES
)

dataset_name = 'Fifth'
home_folder = '/home/debian/jz/'
#'/lustre/fsstor/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
model_dir = f'{home_folder}Mari_Models/TrackModels/'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
print(f'reading data from {normalized_dataframe}')
tracks_dataframe = pd.read_csv(normalized_dataframe)


t_initials = [0,50,100,120]
t_finals = [100,150,200,250]
tracklet_length = 75
num_samples = 20
gbr_shape_model_json = f'{model_dir}shape_feature_lightning_densenet_gbr_{tracklet_length}/shape_densenet.json'
gbr_dynamic_model_json = f'{model_dir}dynamic_feature_lightning_densenet_gbr_{tracklet_length}/dynamic_densenet.json'

class_map_gbr = {
    0: "Basal",
    1: "Radial",
    2: "Goblet"
}

loss_func =  CrossEntropyLoss()

gbr_shape_lightning_model, gbr_shape_torch_model = LightningModel.extract_mitosis_model(
    DenseNet,
    gbr_shape_model_json,
    loss_func,
    Adam,
    map_location=torch.device(device),
    local_model_path=os.path.join(home_folder, f'Mari_Models/TrackModels/shape_feature_lightning_densenet_gbr_{tracklet_length}/')
    
)
gbr_dynamic_lightning_model, gbr_dynamic_torch_model = LightningModel.extract_mitosis_model(
    DenseNet,
    gbr_dynamic_model_json,
    loss_func,
    Adam,
    map_location=torch.device(device),
    local_model_path=os.path.join(home_folder, f'Mari_Models/TrackModels/dynamic_feature_lightning_densenet_gbr_{tracklet_length}/')
)



gbr_shape_torch_model.eval()
gbr_dynamic_torch_model.eval()
for index, t_initial in enumerate(t_initials):
   
        t_final = t_finals[index]
        tracks_dataframe_short = tracks_dataframe[(tracks_dataframe['t'] > t_initial) & (tracks_dataframe['t'] <= t_final)]
        annotations_prediction_dir = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_tracklet_length_{tracklet_length}_t_initial_{t_initial}_t_final_{t_final}/'
        Path(annotations_prediction_dir).mkdir(exist_ok=True)
        tracks_dataframe_short = tracks_dataframe_short[tracks_dataframe_short['Track Duration'] >= tracklet_length]
        gbr_prediction = {}
        for track_id in tqdm(tracks_dataframe_short['Track ID'].unique()):
            gbr_prediction[track_id] = inception_model_prediction(tracks_dataframe_short, track_id, tracklet_length, class_map_gbr, dynamic_model= gbr_dynamic_torch_model, shape_model=gbr_shape_torch_model, num_samples=num_samples,device=device )

        filtered_gbr_prediction = {k: v for k, v in gbr_prediction.items() if v is not None and v != "UnClassified"}
        save_cell_type_predictions(tracks_dataframe_short, class_map_gbr, filtered_gbr_prediction, annotations_prediction_dir, channel)




