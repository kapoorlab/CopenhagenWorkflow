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
        tshift: int 
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
        normalizeimage: bool 
        event_name: list 
        event_label: list 
        event_position_name : list 
        event_position_label : list 
        categories_json: str 
        cord_json: str
        training_class: type
        training_config: type

@dataclass 
class Model_Paths:

     oneat_nuclei_model_dir:  str
     oneat_nuclei_model_name: str

@dataclass 
class Train_Data_Paths:

    base_nuclei_dir: str 
    oneat_timelapse_nuclei_raw : str
    oneat_timelapse_nuclei_csv : str
    oneat_timelapse_nuclei_seg : str
    oneat_nuclei_patch_dir: str
    oneat_h5_file: str
    identifier: str
     

@dataclass
class  TrainOneat:
    
      train_data_paths: Train_Data_Paths
      model_paths: Model_Paths 
      parameters: Params 
           
