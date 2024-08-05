import hydra
from scenario_train_vollseg_cellpose_sam import TrainCellPose
from hydra.core.config_store import ConfigStore
from vollseg import CellPose

configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellPose', node = TrainCellPose)

@hydra.main(config_path = 'conf', config_name = 'scenario_train_cellpose')
def main( config : TrainCellPose):


        base_membrane_dir =  config.train_data_paths.base_membrane_dir
        raw_membrane_dir = config.train_data_paths.raw_membrane_patch_dir 
        real_mask_membrane_dir = config.train_data_paths.real_mask_membrane_patch_dir
        test_raw_membrane_patch_dir = config.train_data_paths.test_raw_membrane_patch_dir
        test_real_mask_membrane_patch_dir = config.train_data_paths.test_real_mask_membrane_patch_dir 

        cellpose_model_dir = config.model_paths.cellpose2D_model_dir
        cellpose_model_name = config.model_paths.cellpose2D_model_name

        epochs = config.parameters.epochs
        learning_rate = config.parameters.learning_rate
        gpu = config.parameters.gpu
        in_channels = config.parameters.in_channels


        CellPose(base_membrane_dir,
            cellpose_model_name,
            cellpose_model_dir,
            raw_membrane_dir,
            real_mask_membrane_dir,
            test_raw_membrane_patch_dir,
            test_real_mask_membrane_patch_dir,
            n_epochs=epochs,
            learning_rate=learning_rate,
            channels=in_channels,
            gpu=gpu)
        
if __name__=='__main__':
    main()        