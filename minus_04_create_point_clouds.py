import hydra
from scenario_train_cellshape import TrainCellShape
from hydra.core.config_store import ConfigStore
from cellshape_helper import conversions
import os

configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellShape', node = TrainCellShape)

@hydra.main(config_path = 'conf', config_name = 'scenario_train_cellshape')
def main( config : TrainCellShape):

        base_dir = config.train_data_paths.base_membrane_dir
        cloud_dataset_dir = os.path.join(base_dir, config.train_data_paths.cloud_mask_membrane_dir)
        real_mask_dir = os.path.join(base_dir, config.train_data_paths.real_mask_membrane_patch_dir)
        num_points = config.parameters.num_cloud_points
        min_size = (5,16,16)
        conversions.label_tif_to_pc_directory(real_mask_dir, cloud_dataset_dir,num_points, min_size = min_size)


if __name__=='__main__':
        main()         
