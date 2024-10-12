# %%
from pathlib import Path 
import os
import torch
import pandas as pd
from tqdm import tqdm
from torch.nn.modules.loss import CrossEntropyLoss
from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_models import HybridAttentionDenseNet
from kapoorlabs_lightning.lightning_trainer import LightningModel
from napatrackmater.Trackvector import (
    inception_model_prediction,
    save_cell_type_predictions,
)

dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/' #'/lustre/fsstor/projects/rech/jsy/uzj81mi/'
#'/home/debian/jz/'
#'/lustre/fsstor/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

model_dir = f'{home_folder}Mari_Models/TrackModels/'
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
print(f'reading data from {normalized_dataframe}')
tracks_dataframe = pd.read_csv(normalized_dataframe)


t_initials = [50]
t_finals = [400]
tracklet_length = 25
gbr_morpho_model_json = f'{model_dir}morpho_feature_lightning_attention_gbr_{tracklet_length}_{channel}shallowest_litest/shape_attention.json'

class_map_gbr = {
    0: "Basal",
    1: "Radial",
    2: "Goblet"
}

loss_func =  CrossEntropyLoss()

gbr_morpho_lightning_model, gbr_morpho_torch_model = LightningModel.extract_mitosis_model(
    HybridAttentionDenseNet,
    gbr_morpho_model_json,
    loss_func,
    Adam,
    map_location=torch.device(device),
    local_model_path=os.path.join(home_folder, f'Mari_Models/TrackModels/morpho_feature_lightning_attention_gbr_{tracklet_length}_{channel}shallowest_litest/')
    
)




gbr_morpho_torch_model.eval()
for index, t_initial in enumerate(t_initials):
   
        t_final = t_finals[index]
        tracks_dataframe_short = tracks_dataframe[(tracks_dataframe['t'] > t_initial) & (tracks_dataframe['t'] <= t_final)]
        annotations_prediction_dir = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_attention_tracklet_length_{tracklet_length}_t_initial_{t_initial}_t_final_{t_final}_{channel}shallowest_litest_morpho_dynamic/'
        Path(annotations_prediction_dir).mkdir(exist_ok=True)
        tracks_dataframe_short = tracks_dataframe_short[tracks_dataframe_short['Track Duration'] >= tracklet_length]
        gbr_prediction = {}
        for trackmate_id in tqdm(tracks_dataframe_short['TrackMate Track ID'].unique()):
            gbr_prediction[trackmate_id] = inception_model_prediction(tracks_dataframe_short, trackmate_id, tracklet_length, class_map_gbr,morphodynamic_model=gbr_morpho_torch_model,device=device )

        filtered_gbr_prediction = {k: v for k, v in gbr_prediction.items() if v is not None and v != "UnClassified"}
        save_cell_type_predictions(tracks_dataframe_short, class_map_gbr, filtered_gbr_prediction, annotations_prediction_dir, channel)




