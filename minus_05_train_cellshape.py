from kapoorlabs_lightning.optimizers import Adam
from kapoorlabs_lightning.pytorch_losses import ChamferLoss
from kapoorlabs_lightning.lightning_trainer import AutoLightningTrain
from kapoorlabs_lightning.pytorch_loggers import CustomNPZLogger
from kapoorlabs_lightning import  PointCloudDataset
import hydra
from scenario_train_cellshape import TrainCellShape
from hydra.core.config_store import ConfigStore
from cellshape_cloud import CloudAutoEncoder

import os
import json
from pathlib import Path
configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellShape', node = TrainCellShape)

@hydra.main(config_path = 'conf', config_name = 'scenario_train_cellshape')
def main( config : TrainCellShape):


        base_dir = config.train_data_paths.base_nuclei_dir
        cloud_dataset_dir = os.path.join(base_dir, config.train_data_paths.cloud_mask_nuclei_dir)
        point_cloud_file = os.path.join(cloud_dataset_dir, config.train_data_paths.point_cloud_filename)
        cloud_model_dir = config.model_paths.cloud_nuclei_model_dir
        
        cloud_model_name = config.model_paths.cloud_nuclei_model_name
        cloud_model_name_json = config.model_paths.cloud_nuclei_model_json
        batch_size = config.parameters.batch_size
        num_features = config.parameters.num_features
        encoder_type = config.parameters.encoder_type
        decoder_type = config.parameters.decoder_type
        k_nearest_neighbours = config.parameters.k_nearest_neighbours
        learning_rate = config.parameters.learning_rate
        num_epochs = config.parameters.epochs
        output_dir = cloud_model_dir
        ckpt_file = os.path.join(cloud_model_dir, f"{cloud_model_name}")
        npz_logger = CustomNPZLogger(save_dir=output_dir,experiment_name='nuclei_autoencoder')
        scale_z = config.parameters.scale_z
        scale_xy = config.parameters.scale_xy
        model = CloudAutoEncoder(
        num_features=num_features,
        k=k_nearest_neighbours,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        )
        
        model_type = config.parameters.model_type
        num_points = config.parameters.num_cloud_points
        dataset = PointCloudDataset(point_cloud_file, scale_z = scale_z, scale_xy = scale_xy)
        loss = ChamferLoss()
        optimizer = Adam(lr=learning_rate)
         
        model_params = {
        "num_features": num_features,
        "k_nearest_neighbours": k_nearest_neighbours,
        "model_type": model_type,
        "encoder_type": encoder_type,
        "decoder_type": decoder_type,
        "num_points": num_points,
        "scale_z": scale_z,
        "scale_xy": scale_xy,
    }
        model_params_file = os.path.join(cloud_model_dir, f"{cloud_model_name_json}")
        if not os.path.exists(ckpt_file):
               ckpt_file = None
        with open(model_params_file, "w") as json_file:
                json.dump(model_params, json_file) 
        # Now we have everything (except the logger) to start the training
        lightning_special_train = AutoLightningTrain(dataset, loss, model, optimizer,output_dir,  ckpt_file = ckpt_file, batch_size= batch_size, epochs = num_epochs,
                                                 accelerator = 'gpu', logger = npz_logger, devices = -1)

        lightning_special_train._train_model()

if __name__=='__main__':
      main()        
