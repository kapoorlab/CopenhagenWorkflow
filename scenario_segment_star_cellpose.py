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
        feat_channels: int
        min_size_mask: int
        min_size: int
        max_size:  int
        n_tiles: list
        axes: str
        channel_membrane: int
        channel_nuclei: int
        file_type: str
        batch_size: int
        network_type: str
        diameter_cellpose: float
        stitch_threshold: float 
        flow_threshold: float 
        cellprob_threshold: float 
        anisotropy: float   
        custom_cellpose_model: bool 
        pretrained_cellpose_model_path: str  
        cellpose_model_name: str 
        gpu: bool 
        prob_thresh_nuclei: float  
        nms_thresh_nuclei: float 
        prob_thresh_membrane: float 
        nms_thresh_membrane: float
        min_size_mask: int 
        min_size: int 
        max_size: int 
        n_tiles: list 
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
        grid_x: int
        grid_y: int
        load_data_sequence: bool
        validation_split: float
        n_patches_per_image: int
 

@dataclass
class Model_Paths:

    model_dir:  str
    star_model_dir: str
    unet_model_dir:  str
    roi_model_dir:  str
    den_model_dir:  str
    cellpose3D_model_dir: str
    cellpose2D_model_dir: str
    cloud_nuclei_model_dir: str
    cellpose_membrane_model_dir: str
    npz_filename: str
    unet_nuclei_model_name: str
    star_nuclei_model_name: str
    unet_membrane_model_name: str
    star_membrane_model_name: str
    roi_nuclei_model_name: str
    den_nuclei_model_name: str
    cloud_nuclei_model_name: str
    cloud_nuclei_model_json: str
    cloud_membrane_model_name: str
    cloud_membrane_model_json: str
    cluster_nuclei_model_name: str
    cellpose3D_model_name: str
    cellpose2D_model_name: str
    cellpose2D_nuclei_model_name: str
    cellpose_model_type: str


@dataclass
class Experiment_Data_Path:
        timelapse_nuclei_to_track: str
        dual_channel_split_directory: str
        timelapse_membrane_to_track: str
        timelapse_membrane_to_track: str
        timelapse_nuclei_directory: str
        timelapse_seg_nuclei_directory : str
        timelapse_seg_nuclei_vollseg_directory : str
        timelapse_membrane_directory: str
        timelapse_seg_membrane_directory : str
        timelapse_oneat_directory: str
        timelapse_oneat_vollseg_directory: str
        variable_t_calibration: dict
        voxel_size_xyz: list
        timelapse_region_of_interest_directory: str
        tracking_directory: str
        tracking_vollseg_directory: str
        oneat_nuclei_csv_file: str
        oneat_nuclei_vollseg_csv_file: str 
        oneat_membrane_csv_file: str
   

@dataclass
class  VollCellSegPose:
    
      experiment_data_paths: Experiment_Data_Path
      model_paths: Model_Paths 
      parameters: Params
