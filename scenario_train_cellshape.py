from dataclasses import dataclass

@dataclass
class Params:

    depth: int 
    epochs: int 
    learning_rate: float 
    batch_size: int 
    num_features: int 
    pretrain: bool 
    num_clusters: int 
    model_type: str 
    encoder_type: str 
    decoder_type: str  
    k_nearest_neighbours: int 
    gamma: int 
    alpha: float 
    num_cloud_points: int 
    divergence_tolerance: float 
    update_interval: int 
    patch_size : list
    erosion_iterations: int
    lower_ratio_fore_to_back: float 
    upper_ratio_fore_to_back: float 
    create_for_channel: str
    scale_z: int
    scale_xy: int

@dataclass 
class Model_Paths:

     cloud_nuclei_model_dir:  str
     cloud_membrane_model_dir: str
     cloud_nuclei_model_name: str
     cloud_nuclei_model_json: str
     cluster_nuclei_model_name: str
     cloud_membrane_model_name: str
     cloud_membrane_model_json: str
     cluster_membrane_model_name: str
        

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
    oneat_nuclei_train_data: str
    cloud_mask_membrane_dir: str 

    cloud_mask_nuclei_dir: str

    point_cloud_filename: str

    identifier: str
     

@dataclass
class  TrainCellShape:
    
      train_data_paths: Train_Data_Paths
      model_paths: Model_Paths 
      parameters: Params 
           
