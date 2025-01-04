import argparse
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

def main(args):
    dataset_name = args.dataset_name
    home_folder = args.home_folder
    channel = args.channel
    t_initials = args.t_initials
    t_finals = args.t_finals
    tracklet_length = args.tracklet_length
    model_dir = args.model_dir
    model_name = args.model_name

    tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
    data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
    print(f'reading data from {normalized_dataframe}')
    tracks_dataframe = pd.read_csv(normalized_dataframe)

    gbr_morpho_model_json = f'{model_dir}{model_name}/morpho_attention.json'

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
        local_model_path=os.path.join(model_dir, model_name)
    )

    gbr_morpho_torch_model.eval()

    for index, t_initial in enumerate(t_initials):
        t_final = t_finals[index]
        tracks_dataframe_short = tracks_dataframe[(tracks_dataframe['t'] > t_initial) & (tracks_dataframe['t'] <= t_final)]
        annotations_prediction_dir = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted/'
        Path(annotations_prediction_dir).mkdir(exist_ok=True)
        tracks_dataframe_short = tracks_dataframe_short[tracks_dataframe_short['Track Duration'] >= tracklet_length]
        gbr_prediction = {}
        for trackmate_id in tqdm(tracks_dataframe_short['TrackMate Track ID'].unique()):
            gbr_prediction[trackmate_id] = inception_model_prediction(tracks_dataframe_short, trackmate_id, tracklet_length, class_map_gbr, morphodynamic_model=gbr_morpho_torch_model, device=device)

        filtered_gbr_prediction = {k: v for k, v in gbr_prediction.items() if v is not None and v != "UnClassified"}
        save_cell_type_predictions(tracks_dataframe_short, class_map_gbr, filtered_gbr_prediction, annotations_prediction_dir, channel)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for morpho prediction script")
    parser.add_argument('--dataset_name', type=str, default='Second', help='Name of the dataset')
    parser.add_argument('--home_folder', type=str, default='/lustre/fsn1/projects/rech/jsy/uzj81mi/', help='Home folder path')
    parser.add_argument('--channel', type=str, default='membrane_', help='Channel name, e.g., nuclei_ or membrane_')
    parser.add_argument('--t_initials', type=int, nargs='+', default=[0], help='List of initial timepoints')
    parser.add_argument('--t_finals', type=int, nargs='+', default=[400], help='List of final timepoints')
    parser.add_argument('--tracklet_length', type=int, default=25, help='Tracklet length value')
    parser.add_argument('--model_dir', type=str, default='/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/', help='Model directory path')
    parser.add_argument('--model_name', type=str, default='membrane_inception_cell_type', help='Model name including full path')

    args = parser.parse_args()
    main(args)
