import hydra
from scenario_train_oneat import TrainOneat
from hydra.core.config_store import ConfigStore
from oneat.NEATUtils.utils import save_json
from oneat.NEATUtils import MovieCreator
from oneat.NEATModels.TrainConfig import TrainConfig
import os
configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellShape', node = TrainOneat)

@hydra.main(version_base="1.3",config_path = 'conf', config_name = 'scenario_train_oneat')
def main( config : TrainOneat):

    base_dir = config.train_data_paths.base_nuclei_dir
    
    image_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_raw)
    csv_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_csv)
    seg_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_seg)
    save_patch_dir = os.path.join(base_dir, config.train_data_paths.oneat_nuclei_patch_dir)
    model_dir = config.model_paths.oneat_nuclei_model_dir
    
    #Name of the  events
    event_type_name = config.parameters.event_name
    event_type_label = config.parameters.event_label

    size_tminus = config.parameters.size_tminus
    size_tplus = config.parameters.size_tplus
    tshift = config.parameters.tshift
    imagex  = config.parameters.imagex
    imagey = config.parameters.imagey
    imagez = config.parameters.imagez
    normalizeimage = config.parameters.normalizeimage
    npz_name = config.train_data_paths.oneat_h5_file + '.npz'
    npz_val_name = config.train_data_paths.oneat_h5_file + '_val.npz'
    crop_size = [imagex,imagey,imagez, size_tminus,size_tplus]
    event_position_name = config.parameters.event_position_name
    event_position_label = config.parameters.event_position_label
    dynamic_config = TrainConfig(event_type_name, event_type_label, event_position_name, event_position_label)

    dynamic_json, dynamic_cord_json = dynamic_config.to_json()

    save_json(dynamic_json, model_dir +config.parameters.categories_json)

    save_json(dynamic_cord_json, model_dir + config.parameters.cord_json)
    
    MovieCreator.VolumeLabelDataSet(image_dir, 
                                seg_dir, 
                                csv_dir, 
                                save_patch_dir, 
                                event_type_name, 
                                event_type_label, 
                                '',
                                crop_size,
                                normalizeimage = normalizeimage,
                                tshift = tshift, 
                                )
    MovieCreator.createNPZ(save_patch_dir, axes = 'STZYXC', save_name = npz_name, save_name_val = npz_val_name)

if __name__=='__main__':
    main()  
