from dataclasses import dataclass 


@dataclass
class Params:

        startfilter: int 
        start_kernel: int 
        mid_kernel: int 
        learning_rate: float 
        batch_size: int 
        epochs: int 
        show: bool 
        stage_number: int 
        size_tminus: int 
        size_tplus: int 
        imagex: int 
        imagey: int 
        imagez: int 
        depth: dict 
        reduction: float 
        n_tiles: list
        event_threshold: float 
        event_confidence: float 
        file_type: str 
        nms_space: int 
        nms_time: int 
        categories_json: str 
        cord_json: str 
        training_class: str
 

@dataclass
class Model_Paths:

    model_dir:  str
    star_nucle_model_dir: str
    unet_nuclei_model_dir:  str
    roi_nuclei_model_dir:  str
    den_nuclei_model_dir:  str
    cellpose_model_dir: str
    cloud_nuclei_model_dir: str

    oneat_nuclei_model_dir: str 
    unet_nuclei_model_name: str
    star_nuclei_model_name: str
    unet_membrane_model_name: str
    
    roi_nuclei_model_name: str
    den_nuclei_model_name: str
    cloud_nuclei_model_name: str
    cloud_nuclei_model_json: str
    cluster_nuclei_model_name: str
    cellpose_model_name: str


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
        base_directory: str
   

@dataclass
class  VollOneat:
    
      experiment_data_paths: Experiment_Data_Path
      model_paths: Model_Paths 
      parameters: Params
