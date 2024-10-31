import argparse
import os
import torch
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from torch.nn.modules.loss import CrossEntropyLoss
from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_models import HybridAttentionDenseNet, get_attention_importance
from kapoorlabs_lightning.lightning_trainer import LightningModel
from napatrackmater.Trackvector import (SHAPE_FEATURES, 
                                        DYNAMIC_FEATURES, 
                                        SHAPE_DYNAMIC_FEATURES,
                                        
                                        )
def plot_feature_importance_heatmap(model, inputs, save_dir, save_name):
    """
    Saves a heatmap of feature importance across multiple tracks.

    Parameters:
        model (nn.Module): The trained model with attention layers.
        inputs (list of torch.Tensor): Input tensors, each with shape (N, F, T) for each track.
        save_dir (str): Directory to save the plot.
        save_name (str): Filename to save the plot as.
    """
    
    # Ensure the save directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect feature importance for each track
    all_importances = []
    avg_importance = get_attention_importance(model, inputs)
    all_importances.append(avg_importance)

    # Convert to a 2D array where each row is a feature and each column is a track
    importance_matrix = np.array(all_importances).T  # Transpose for correct orientation
    
    # Plot the heatmap
    plt.figure(figsize=(20, 10))  # Larger figure for visibility
    sns.heatmap(importance_matrix[:,:,0], annot=False, cmap="coolwarm", yticklabels=SHAPE_DYNAMIC_FEATURES)
    plt.xlabel("Track IDs")
    plt.ylabel("Features")
    plt.title("Feature Importance Across Tracks")
    
    # Save the plot
    save_path = os.path.join(save_dir, save_name)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    print(f"Saved heatmap to {save_path}")

def main(args):
    dataset_name = args.dataset_name
    home_folder = args.home_folder
    channel = args.channel
    tracklet_length = args.tracklet_length
    model_dir = args.model_dir
    model_name = args.model_name
  
    tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
    data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
    goblet_basal_radial_dataframe = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}predicted_{model_name}.csv')
    tracks_goblet_basal_radial_dataframe = pd.read_csv(goblet_basal_radial_dataframe)
    unique_cell_types = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]['Cell_Type'].unique()
    print("Number of unique cell types:", len(unique_cell_types))
    print("Unique cell types:", unique_cell_types)
    cell_type_label_mapping = {
        "Basal": 1,
        "Radial": 2, 
        "Goblet": 3
    }
    print("Cell type counts, unique TrackMate Track IDs, and count of mitotic tracks:")
    for cell_type in unique_cell_types:
        cell_type_df = tracks_goblet_basal_radial_dataframe[tracks_goblet_basal_radial_dataframe['Cell_Type'] == cell_type]
        unique_track_ids = cell_type_df['TrackMate Track ID'].unique()
        
        dividing_count = 0
        for track_id in unique_track_ids:
            track_df = cell_type_df[cell_type_df['TrackMate Track ID'] == track_id]
            if track_df['Dividing'].iloc[0] == 1:
                dividing_count += 1
        
        count = len(cell_type_df)
        print(f"{cell_type}: {count} rows, unique TrackMate Track IDs: {len(unique_track_ids)}, mitotic tracks: {dividing_count}")

    cell_type_dataframe = tracks_goblet_basal_radial_dataframe[~tracks_goblet_basal_radial_dataframe['Cell_Type'].isna()]
    cell_type_dataframe['Cell_Type_Label'] = cell_type_dataframe['Cell_Type'].map(cell_type_label_mapping)
    correlation_dataframe = cell_type_dataframe.copy()
    correlation_dataframe.head()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

   
    gbr_morpho_model_json = f'{model_dir}{model_name}_{channel}/morpho_attention.json'
    loss_func =  CrossEntropyLoss()
    gbr_morpho_lightning_model, gbr_morpho_torch_model = LightningModel.extract_mitosis_model(
        HybridAttentionDenseNet,
        gbr_morpho_model_json,
        loss_func,
        Adam,
        map_location=torch.device(device),
        local_model_path=os.path.join(model_dir, model_name + f'_{channel}')
    )

    gbr_morpho_torch_model.eval()
    save_dir = os.path.join(tracking_directory, "feature_importance_plots")
    os.makedirs(save_dir, exist_ok=True)

    for cell_type in ["Basal", "Goblet", "Radial"]:
        cell_type_df = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == cell_type]
        
        track_lengths = cell_type_df.groupby("TrackMate Track ID").size()
        longest_track_ids = track_lengths.nlargest(args.N).index  
        batch_tracklets = []
        for track_id in longest_track_ids:
            # Get the track data
            track_df = cell_type_df[cell_type_df["TrackMate Track ID"] == track_id]

            if len(track_df) < tracklet_length:
                continue

            tracklet_features = track_df[SHAPE_DYNAMIC_FEATURES].iloc[:tracklet_length].values 
            tracklet_tensor = torch.tensor(tracklet_features, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)  # Shape (1, T, F)
            batch_tracklets.append(tracklet_tensor)
        if batch_tracklets:    
            batch_tensor = torch.cat(batch_tracklets, dim=0).to(device)
            print(batch_tensor.shape)
            save_name = f"{cell_type}_feature_importance.png"
            plot_feature_importance_heatmap(
                gbr_morpho_torch_model,
                batch_tensor,
                save_dir=save_dir,
                save_name=save_name,
            )
            print(f"Saved feature importance plot for {cell_type}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for morpho prediction script")
    parser.add_argument('--dataset_name', type=str, default='Sixth', help='Name of the dataset')
    parser.add_argument('--home_folder', type=str, default='/lustre/fsn1/projects/rech/jsy/uzj81mi/', help='Home folder path')
    parser.add_argument('--channel', type=str, default='nuclei_', help='Channel name, e.g., nuclei_ or membrane_')
    parser.add_argument('--tracklet_length', type=int, default=25, help='Tracklet length value')
    parser.add_argument('--model_dir', type=str, default='/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/', help='Model directory path')
    parser.add_argument('--model_name', type=str, default='morpho_feature_attention_shallowest_litest', help='Model name including full path')
    parser.add_argument('--N', type=int, default=200, help='Number of longest tracks per cell type to analyze')

    args = parser.parse_args()
    main(args)
