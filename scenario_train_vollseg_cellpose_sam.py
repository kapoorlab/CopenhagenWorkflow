from dataclasses import dataclass

@dataclass
class Params:

          patch_size : list
          overlap: list
          crop: list
          in_channels: int
          out_activation: str
          norm_method: str
          background_weight: int 
          flow_weight: int 
          out_channels: int 
          feat_channels: list 
          num_levels: int 
          min_size_mask: int 
          min_size: int 
          max_size:  int 
          n_tiles: list
          axes: str 
          channel_membrane: int 
          channel_nuclei: int 
          file_type: str 
          batch_size: int 
          learning_rate: float 
          num_gpu: int 
          epochs: int 
          network_type: str 
          generate_npz: bool 
          grid_x: int 
          grid_y: int 
          grid_z: int
          diameter_cellpose: float 
          stitch_threshold: float 
          flow_threshold: float 
          cellprob_threshold: float 
          anisotropy: float   
          cellpose_model_path: str  
          gpu: bool 
          prob_thresh_nuclei: float 
          nms_thresh_nuclei: float 
          prob_thresh_membrane: float 
          nms_thresh_membrane: float 
          UseProbability: bool 
          ExpandLabels: bool 
          donormalize: bool 
          lower_perc: float 
          upper_perc: float 
          dounet: bool 
          seedpool: bool 
          save_dir: str 
          Name: str 
          slice_merge: bool 
          do_3D: bool 
          z_thresh: int 
          load_data_sequence: bool
          validation_split: float 
          n_patches_per_image: int 
          train_loss: str 
          train_unet: bool 
          train_star: bool
          erosion_iterations: int 
          use_gpu_opencl: bool 
          depth: int 
          kern_size: int 
          n_rays: int 
          startfilter: int 
          patch_x: int
          patch_y: int
 


@dataclass 
class Model_Paths:

     model_dir:  str
     star_model_dir: str
     unet_model_dir:  str
     roi_model_dir:  str
     den_model_dir:  str
     cellpose3D_model_dir: str
     cellpose2D_model_dir: str
    
     unet_nuclei_model_name: str 
     star_nuclei_model_name: str
     unet_membrane_model_name: str
     star_membrane_model_name: str
     roi_nuclei_model_name: str
     den_nuclei_model_name: str
    
     cellpose3D_model_name: str
     cellpose2D_model_name: str
     ckpt_path: str

     sam_ckpt_directory: str
     sam_ckpt_model_name: str
     sam_model_type: str
     nuclei_npz_filename: str
     membrane_npz_filename: str
     roi_npz_filename: str
     ckpt_path: str
     

@dataclass 
class Train_Data_Paths:

          base_membrane_dir: str 

          raw_membrane_dir: str
          real_mask_membrane_dir: str
          raw_membrane_patch_dir: str
          real_mask_membrane_patch_dir: str
          raw_membrane_patch_h5_dir: str
          real_mask_membrane_patch_h5_dir: str
          binary_mask_membrane_patch_dir: str
          binary_erode_mask_membrane_patch_dir: str
          test_raw_membrane_patch_dir: str
          test_real_mask_membrane_patch_dir: str

          base_nuclei_dir: str 

          raw_nuclei_dir: str
          real_mask_nuclei_dir: str
          raw_nuclei_patch_dir: str
          real_mask_nuclei_patch_dir: str
          raw_nuclei_patch_h5_dir: str
          real_mask_nuclei_patch_h5_dir: str
          binary_mask_nuclei_patch_dir: str
          binary_erode_mask_nuclei_patch_dir: str
          test_raw_nuclei_patch_dir: str
          test_real_mask_nuclei_patch_dir: str

          base_roi_dir: str
          raw_roi_dir : str
          binary_mask_roi_dir : str 
 
          identifier: str
          save_train: str
          save_test: str 
          save_val: str 

@dataclass
class  TrainCellPose:
    
      train_data_paths: Train_Data_Paths
      model_paths: Model_Paths 
      parameters: Params 
           
