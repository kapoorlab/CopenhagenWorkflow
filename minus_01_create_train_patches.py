import hydra
from scenario_train_cellshape import TrainCellShape
from hydra.core.config_store import ConfigStore
from vollseg import SmartPatches

configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellShape', node = TrainCellShape)

@hydra.main(config_path = 'conf', config_name = 'scenario_train_cellshape')
def main( config : TrainCellShape):


        base_membrane_dir =  config.train_data_paths.base_membrane_dir 
        raw_membrane_dir = config.train_data_paths.raw_membrane_dir
        base_nuclei_dir =  config.train_data_paths.base_nuclei_dir 
        raw_nuclei_dir = config.train_data_paths.raw_nuclei_dir
        

        membrane_channel_results_directory = config.train_data_paths.real_mask_membrane_dir
        membrane_raw_save_dir = config.train_data_paths.raw_membrane_patch_dir
        membrane_real_mask_patch_dir = config.train_data_paths.real_mask_membrane_patch_dir
        membrane_binary_mask_patch_dir = config.train_data_paths.binary_mask_membrane_patch_dir
        nuclei_channel_results_directory = config.train_data_paths.real_mask_nuclei_dir
        nuclei_raw_save_dir = config.train_data_paths.raw_nuclei_patch_dir
        nuclei_real_mask_patch_dir = config.train_data_paths.real_mask_nuclei_patch_dir
        nuclei_binary_mask_patch_dir = config.train_data_paths.binary_mask_nuclei_patch_dir
        lower_ratio_fore_to_back=config.parameters.lower_ratio_fore_to_back
        upper_ratio_fore_to_back=config.parameters.upper_ratio_fore_to_back
        patch_size = tuple(config.parameters.patch_size)
        erosion_iterations = config.parameters.erosion_iterations
        create_for_channel = config.parameters.create_for_channel

        
        SmartPatches(base_membrane_dir,
        raw_membrane_dir,
        base_nuclei_dir,
        raw_nuclei_dir,
        nuclei_channel_results_directory,
        membrane_channel_results_directory,
        nuclei_raw_save_dir,
        membrane_raw_save_dir,
        nuclei_real_mask_patch_dir,
        membrane_real_mask_patch_dir,
        nuclei_binary_mask_patch_dir,
        membrane_binary_mask_patch_dir, patch_size, erosion_iterations,
        create_for_channel = create_for_channel, 
        lower_ratio_fore_to_back=lower_ratio_fore_to_back,
        upper_ratio_fore_to_back=upper_ratio_fore_to_back)

if __name__=='__main__':
        main()        
