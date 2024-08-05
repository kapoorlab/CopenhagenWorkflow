from dataclasses import dataclass 
from typing import List

@dataclass
class Params:

       batch_size: int
       axes: str
       learning_rate: float
       num_clusters: int
       accelerator: str
       devices: List[int] | str | int
       scale_z: float
       scale_xy: float
       compute_with_autoencoder: bool
       calibration_z: float
       calibration_x: float
       calibration_y: float
       oneat_threshold_cutoff: float
       enhance_trackmate_xml: bool
 

@dataclass
class Model_Paths:

    model_dir:  str
    star_nucle_model_dir: str
    unet_nuclei_model_dir:  str
    roi_nuclei_model_dir:  str
    den_nuclei_model_dir:  str
    cellpose_membrane_model_dir: str
    cloud_nuclei_model_dir: str
    cloud_membrane_model_dir: str
    oneat_nuclei_model_dir: str
    track_model_dir: str

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
    cellpose_model_name: str
    track_shape_model_name: str
    track_dynamic_model_name: str
    track_shape_dynamic_model_name: str
    track_model_training_data_dir: str
    track_shape_model_no_latent_name: str
    track_dynamic_model_no_latent_name: str
    track_shape_dynamic_model_no_latent_name: str



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
        timelapse_region_of_interest_directory: str
        tracking_directory: str
        tracking_vollseg_directory: str
        ground_truth_csv: str
        variable_t_calibration: dict
        voxel_size_xyz: list
        oneat_nuclei_csv_file: str
        oneat_nuclei_vollseg_csv_file: str 
        oneat_membrane_csv_file: str
        goblet_cells_nuclei_predicted: str
        basal_cells_nuclei_predicted: str
        radial_cells_nuclei_predicted: str

        goblet_cells_nuclei_gt: str
        basal_cells_nuclei_gt: str
        radial_cells_nuclei_gt: str

        goblet_cells_membrane_predicted: str
        basal_cells_membrane_predicted: str
        radial_cells_membrane_predicted: str


   

@dataclass
class  NapaTrackMater:
    
      experiment_data_paths: Experiment_Data_Path
      model_paths: Model_Paths 
      parameters: Params