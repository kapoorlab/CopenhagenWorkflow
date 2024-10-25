import argparse
import os
import torch
from torch.nn.modules.loss import CrossEntropyLoss
from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_models import HybridAttentionDenseNet,plot_feature_importance_heatmap
from kapoorlabs_lightning.lightning_trainer import LightningModel


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
    tracks_goblet_basal_radial_dataframe = os.path.join(data_frames_dir , f'goblet_basal_dataframe_normalized_{channel}predicted_{model_name}.csv')
    
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

    # Directory to save feature importance plots
    save_dir = os.path.join(tracking_directory, "feature_importance_plots")
    os.makedirs(save_dir, exist_ok=True)

    # Loop through each cell type and plot feature importance for the top tracks
    for cell_type in ["Basal", "Goblet", "Radial"]:
        # Filter dataframe by cell type and get longest tracks
        cell_type_df = cell_type_dataframe[cell_type_dataframe['Cell_Type'] == cell_type]
        
        # Sort TrackMate Track IDs by track length and select the longest tracks
        track_lengths = cell_type_df.groupby("TrackMate Track ID").size()
        longest_track_ids = track_lengths.nlargest(args.N).index  # Replace N with the number of tracks to select

        for track_id in longest_track_ids:
            # Get the track data
            track_df = cell_type_df[cell_type_df["TrackMate Track ID"] == track_id]

            # Ensure tracklet meets minimum length requirement
            if len(track_df) < tracklet_length:
                continue

            # Create a tracklet tensor with shape (1, T, F)
            tracklet_features = track_df.iloc[:tracklet_length, 4:].values  # Assuming features start from 5th column
            tracklet_tensor = torch.tensor(tracklet_features, dtype=torch.float32).unsqueeze(0).to(device)  # Shape (1, T, F)
            
            # Calculate and save the feature importance plot
            save_name = f"{cell_type}_Track_{track_id}_feature_importance.png"
            plot_feature_importance_heatmap(
                gbr_morpho_torch_model,
                [tracklet_tensor],
                save_dir=save_dir,
                save_name=save_name,
                feature_names=track_df.columns[4:],  # Assuming features start from 5th column
                track_labels=[f"Track {track_id}"]
            )
            print(f"Saved feature importance plot for {cell_type} Track ID {track_id}.")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for morpho prediction script")
    parser.add_argument('--dataset_name', type=str, default='Sixth', help='Name of the dataset')
    parser.add_argument('--home_folder', type=str, default='/lustre/fsn1/projects/rech/jsy/uzj81mi/', help='Home folder path')
    parser.add_argument('--channel', type=str, default='membrane_', help='Channel name, e.g., nuclei_ or membrane_')
    parser.add_argument('--t_initials', type=int, nargs='+', default=[50], help='List of initial timepoints')
    parser.add_argument('--t_finals', type=int, nargs='+', default=[400], help='List of final timepoints')
    parser.add_argument('--tracklet_length', type=int, default=25, help='Tracklet length value')
    parser.add_argument('--model_dir', type=str, default='/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Models/TrackModels/', help='Model directory path')
    parser.add_argument('--model_name', type=str, default='morpho_feature_lightning_attention_gbr_25_membrane_', help='Model name including full path')
    parser.add_argument('--N', type=int, default=30, help='Number of longest tracks per cell type to analyze')

    args = parser.parse_args()
    main(args)
