import hydra
from scenario_train_vollseg_cellpose_sam import TrainCellPose
from hydra.core.config_store import ConfigStore
from vollseg import SmartSeeds3D

configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellPose', node = TrainCellPose)

@hydra.main(config_path = 'conf', config_name = 'scenario_train_vollseg_cellpose_sam')
def main( config : TrainCellPose):

        base_membrane_dir =  config.train_data_paths.base_membrane_dir 
        raw_membrane_dir = config.train_data_paths.raw_membrane_patch_dir
        real_mask_membrane_dir = config.train_data_paths.real_mask_membrane_dir
        binary_mask_membrane_dir = config.train_data_paths.binary_mask_membrane_patch_dir
        nuclei_npz_filename =  config.model_paths.nuclei_npz_filename 
        membrane_npz_filename = config.model_paths.membrane_npz_filename

        base_nuclei_dir =  config.train_data_paths.base_nuclei_dir 
        raw_nuclei_dir = config.train_data_paths.raw_nuclei_patch_dir
        real_mask_nuclei_dir = config.train_data_paths.real_mask_nuclei_patch_dir
        binary_mask_nuclei_dir = config.train_data_paths.binary_mask_nuclei_patch_dir


        unet_model_dir = config.model_paths.unet_model_dir
        star_model_dir = config.model_paths.star_model_dir
        unet_nuclei_model_name = config.model_paths.unet_nuclei_model_name
        unet_membrane_model_name = config.model_paths.unet_membrane_model_name
        star_nuclei_model_name = config.model_paths.star_nuclei_model_name
        star_membrane_model_name = config.model_paths.star_membrane_model_name



        in_channels = config.parameters.in_channels
        grid_x = config.parameters.grid_x
        grid_y = config.parameters.grid_y
        backbone = config.parameters.network_type
        batch_size = config.parameters.batch_size
        learning_rate = config.parameters.learning_rate
        patch_size = tuple(config.parameters.patch_size)
        generate_npz = config.parameters.generate_npz
        load_data_sequence = config.parameters.load_data_sequence
        validation_split = config.parameters.validation_split
        n_patches_per_image = config.parameters.n_patches_per_image
        train_loss = config.parameters.train_loss
        train_unet = config.parameters.train_unet
        train_star = config.parameters.train_star
        erosion_iterations = config.parameters.erosion_iterations
        use_gpu_opencl = config.parameters.use_gpu_opencl
        depth = config.parameters.depth
        kern_size = config.parameters.kern_size
        startfilter = config.parameters.startfilter
        n_rays = config.parameters.n_rays
        epochs = config.parameters.epochs

        train_nuclei = True 
        train_membrane = True 
        if train_nuclei:
                SmartSeeds3D(base_dir = base_nuclei_dir, 
                            unet_model_name = unet_nuclei_model_name,
                            star_model_name = star_nuclei_model_name,  
                            unet_model_dir = unet_model_dir,
                            star_model_dir = star_model_dir,
                            npz_filename = nuclei_npz_filename, 
                            raw_dir = raw_nuclei_dir,
                            real_mask_dir = real_mask_nuclei_dir,
                            binary_mask_dir = binary_mask_nuclei_dir,
                            n_channel_in = in_channels,
                            backbone = backbone,
                            load_data_sequence = load_data_sequence, 
                            validation_split = validation_split, 
                            n_patches_per_image = n_patches_per_image, 
                            generate_npz = generate_npz,
                            patch_size = patch_size,
                            grid_x = grid_x,
                            grid_y = grid_y,
                            erosion_iterations = erosion_iterations,  
                            train_loss = train_loss,
                            train_star = train_star,
                            train_unet = train_unet,
                            use_gpu = use_gpu_opencl,  
                            batch_size = batch_size, 
                            depth = depth, 
                            kern_size = kern_size, 
                            startfilter = startfilter, 
                            n_rays = n_rays, 
                            epochs = epochs, 
                            learning_rate = learning_rate)
                
        if train_membrane:
               
                SmartSeeds3D(base_dir = base_membrane_dir, 
                            unet_model_name = unet_membrane_model_name,
                            star_model_name = star_membrane_model_name,  
                            unet_model_dir = unet_model_dir,
                            star_model_dir = star_model_dir,
                            npz_filename = membrane_npz_filename, 
                            raw_dir = raw_membrane_dir,
                            real_mask_dir = real_mask_membrane_dir,
                            binary_mask_dir = binary_mask_membrane_dir,
                            n_channel_in = in_channels,
                            backbone = backbone,
                            load_data_sequence = load_data_sequence, 
                            validation_split = validation_split, 
                            n_patches_per_image = n_patches_per_image, 
                            generate_npz = generate_npz,
                            patch_size = patch_size,
                            grid_x = grid_x,
                            grid_y = grid_y,
                            erosion_iterations = erosion_iterations,  
                            train_loss = train_loss,
                            train_star = train_star,
                            train_unet = train_unet,
                            use_gpu = use_gpu_opencl,  
                            batch_size = batch_size, 
                            depth = depth, 
                            kern_size = kern_size, 
                            startfilter = startfilter, 
                            n_rays = n_rays, 
                            epochs = epochs, 
                            learning_rate = learning_rate) 
                       

if __name__ == "__main__":
    main()        