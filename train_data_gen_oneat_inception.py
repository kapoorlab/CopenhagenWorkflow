import hydra
from scenario_train_oneat import TrainOneat
from hydra.core.config_store import ConfigStore
from kapoorlabs_lightning import save_json, VolumeLabelDataSet, OneatConfig
import os
configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellShape', node = TrainOneat)

@hydra.main(version_base="1.3",config_path = 'conf', config_name = 'scenario_train_oneat')
def main( config : TrainOneat):

    base_dir = config.train_data_paths.base_nuclei_dir
    image_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_raw)
    csv_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_csv)
    seg_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_seg)
    model_dir = config.model_paths.oneat_nuclei_model_dir
    class_name = config.parameters.event_name
    class_label = config.parameters.event_label
    size_tminus = config.parameters.size_tminus
    size_tplus = config.parameters.size_tplus
    tshift = config.parameters.tshift
    imagex  = config.parameters.imagex
    imagey = config.parameters.imagey
    imagez = config.parameters.imagez
    normalizeimage = config.parameters.normalizeimage
    oneat_h5_file = config.train_data_paths.oneat_h5_file
    crop_size = [imagex,imagey,imagez, size_tminus,size_tplus]
    event_position_name = config.parameters.event_position_name
    event_position_label = config.parameters.event_position_label
    dynamic_config = OneatConfig(class_name, class_label, event_position_name, event_position_label)

    dynamic_json, dynamic_cord_json = dynamic_config.to_json()

    save_json(dynamic_json, model_dir +config.parameters.categories_json)

    save_json(dynamic_cord_json, model_dir + config.parameters.cord_json)
    
    VolumeLabelDataSet(image_dir, 
                                seg_dir, 
                                csv_dir, 
                                oneat_h5_file,
                                class_name, 
                                class_label, 
                                crop_size,
                                normalizeimage = normalizeimage,
                                tshift = tshift, 
                               )

if __name__=='__main__':
    main()  
