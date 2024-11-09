import argparse
from pathlib import Path 
import os
import torch
import pandas as pd
from tqdm import tqdm
from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_models import DenseVollNet
from kapoorlabs_lightning.pytorch_losses import VolumeYoloLoss
from kapoorlabs_lightning.lightning_trainer import LightningModel
from napatrackmater.Trackvector import (
    vision_inception_model_prediction,
    save_cell_type_predictions,
)

def main(args):
    dataset_name = args.dataset_name
    home_folder = args.home_folder
    t_initials = args.t_initials
    t_finals = args.t_finals
    tracklet_length = args.tracklet_length
    model_dir = args.model_dir
    model_name = args.model_name

    tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
    data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    box_vector = 7
    categories = 3
    channel = 'nuclei_'
    input_shape = [25,8,128,128]
    normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
    print(f'reading data from {normalized_dataframe}')
    tracks_dataframe = pd.read_csv(normalized_dataframe)

    vision_inception_model_json = f'{model_dir}{model_name}/vision_cellfate.json'

    class_map_gbr = {
        0: "Basal",
        1: "Radial",
        2: "Goblet"
    }

    loss_func =  VolumeYoloLoss(categories, box_vector, device)

    vision_inception_lightning_model, vision_inception_torch_model = LightningModel.extract_vision_inception_model(
        DenseVollNet,
        vision_inception_model_json,
        input_shape,
        box_vector,
        loss_func,
        Adam,
        map_location=torch.device(device),
        local_model_path=os.path.join(model_dir, model_name + f'_{channel}')
    )

    vision_inception_torch_model.eval()

    for index, t_initial in enumerate(t_initials):
        t_final = t_finals[index]
        tracks_dataframe_short = tracks_dataframe[(tracks_dataframe['t'] > t_initial) & (tracks_dataframe['t'] <= t_final)]
        annotations_prediction_dir = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted_attention_{model_name}_{channel}vision/'
        Path(annotations_prediction_dir).mkdir(exist_ok=True)
        tracks_dataframe_short = tracks_dataframe_short[tracks_dataframe_short['Track Duration'] >= tracklet_length]
        gbr_prediction = {}
        for trackmate_id in tqdm(tracks_dataframe_short['TrackMate Track ID'].unique()):
            gbr_prediction[trackmate_id] = vision_inception_model_prediction(tracks_dataframe_short, trackmate_id, tracklet_length, class_map_gbr, morphodynamic_model=vision_inception_torch_model, device=device)

        filtered_gbr_prediction = {k: v for k, v in gbr_prediction.items() if v is not None and v != "UnClassified"}
        save_cell_type_predictions(tracks_dataframe_short, class_map_gbr, filtered_gbr_prediction, annotations_prediction_dir, channel + 'vision')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for morpho prediction script")
    parser.add_argument('--dataset_name', type=str, default='Sixth', help='Name of the dataset')
    parser.add_argument('--home_folder', type=str, default='/lustre/fsn1/projects/rech/jsy/uzj81mi/', help='Home folder path')
    parser.add_argument('--t_initials', type=int, nargs='+', default=[50], help='List of initial timepoints')
    parser.add_argument('--t_finals', type=int, nargs='+', default=[400], help='List of final timepoints')
    parser.add_argument('--tracklet_length', type=int, default=25, help='Tracklet length value')
    parser.add_argument('--model_dir', type=str, default='/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/', help='Model directory path')
    parser.add_argument('--model_name', type=str, default='vision_inception', help='Model name including full path')

    args = parser.parse_args()
    main(args)
