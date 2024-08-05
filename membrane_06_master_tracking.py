import hydra
from hydra.core.config_store import ConfigStore
from scenario_track import NapaTrackMater
from pathlib import Path
import os 
from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.lightning_trainer import AutoLightningModel
from napatrackmater.Trackmate import TrackMate
from napatrackmater import load_json
from cellshape_cloud import CloudAutoEncoder
from tifffile import imread
from kapoorlabs_lightning.pytorch_losses import ChamferLoss
configstore = ConfigStore.instance()
configstore.store(name = 'NapaTrackMater' , node = NapaTrackMater )


@hydra.main(version_base ="1.3", config_path = 'conf', config_name = 'scenario_track')
def main(config:NapaTrackMater):

    
    channel = 'membrane_'
    timelapse_to_track =  config.experiment_data_paths.timelapse_membrane_to_track
    base_dir = config.experiment_data_paths.tracking_directory
    
    xml_name = channel + timelapse_to_track + ".xml"
    edges_csv = channel + timelapse_to_track + "_edges.csv"
    spot_csv = channel + timelapse_to_track + "_spots.csv"
    track_csv = channel + timelapse_to_track + "_tracks.csv"
    seg_image_name = timelapse_to_track + ".tif"
    tissue_mask_name = timelapse_to_track + ".tif"
    xml_path = Path(os.path.join(base_dir, xml_name))
    track_csv = Path(os.path.join(base_dir, track_csv))
    spot_csv = Path(os.path.join(base_dir, spot_csv))
    edges_csv = Path(os.path.join(base_dir, edges_csv))
    modelconfig = load_json(os.path.join(config.model_paths.cloud_nuclei_model_dir, config.model_paths.cloud_nuclei_model_json))
    learning_rate = config.parameters.learning_rate
    accelerator = config.parameters.accelerator
    devices = config.parameters.devices
    model_class_cloud_auto_encoder = CloudAutoEncoder
    loss = ChamferLoss()
    optimizer = Adam(lr=learning_rate)
    scale_z = modelconfig["scale_z"]
    scale_xy = modelconfig["scale_xy"]
    compute_with_autoencoder = config.parameters.compute_with_autoencoder
    cloud_autoencoder = model_class_cloud_auto_encoder(
        num_features=modelconfig["num_features"],
        k=modelconfig["k_nearest_neighbours"],
        encoder_type=modelconfig["encoder_type"],
        decoder_type=modelconfig["decoder_type"],
    )
    
    autoencoder_model = AutoLightningModel.load_from_checkpoint(os.path.join(config.model_paths.cloud_nuclei_model_dir, 
                        config.model_paths.cloud_nuclei_model_name),map_location='cpu', network = cloud_autoencoder, 
                        loss_func = loss, optim_func = optimizer, scale_z = scale_z, scale_xy = scale_xy)

    
    axes = config.parameters.axes
   
    num_points = modelconfig["num_points"] 
    seg_image = imread(os.path.join(config.experiment_data_paths.timelapse_seg_nuclei_directory,seg_image_name))
    if config.experiment_data_paths.timelapse_region_of_interest_directory is None:
        mask_image = None
    else:
       mask_image = imread(os.path.join(config.experiment_data_paths.timelapse_region_of_interest_directory, tissue_mask_name))
    batch_size = config.parameters.batch_size
    AttributeBoxname = "AttributeIDBox"
    TrackAttributeBoxname = "TrackAttributeIDBox"
    TrackidBox = "All"
    TrackMate(
            xml_path,
            spot_csv,
            track_csv,
            edges_csv,
            AttributeBoxname, 
            TrackAttributeBoxname, 
            TrackidBox,
            axes,
            seg_image=seg_image,
            mask=mask_image,
            autoencoder_model=autoencoder_model,
            num_points=num_points,
            batch_size=batch_size,
            accelerator=accelerator,
            devices=devices, 
            scale_z=scale_z,
            scale_xy=scale_xy,
            master_extra_name = "marching_cubes_",
            compute_with_autoencoder=compute_with_autoencoder
        )

if __name__=='__main__':
        main()
