import os
from scenario_predict_oneat import VollOneat
import hydra
from hydra.core.config_store import ConfigStore
from oneat.NEATUtils.utils import generate_membrane_locations
from pathlib import Path


configstore = ConfigStore.instance()
configstore.store(name = 'VollOneat', node = VollOneat)

@hydra.main(version_base="1.3",config_path = 'conf', config_name = 'scenario_predict_oneat')
def main( config : VollOneat):
    timelapse_oneat_directory = config.experiment_data_paths.timelapse_oneat_directory
    timelapse_nuclei_to_track = config.experiment_data_paths.timelapse_nuclei_to_track
    timelapse_seg_membrane_directory = config.experiment_data_paths.timelapse_seg_membrane_directory
    membranesegimage = os.path.join(timelapse_seg_membrane_directory, timelapse_nuclei_to_track + '.tif')

    csv_files = [
    file_name
    for label in ["mitosis"]
    for file_name in [
        f"oneat_{label}_locations_nuclei_{timelapse_nuclei_to_track}.csv",
        f"non_maximal_oneat_{label}_locations_nuclei_{timelapse_nuclei_to_track}.csv"
    ]
    ]

    
    save_files = [
    file_name
    for label in ["mitosis"]
    for file_name in [
        f"oneat_{label}_locations_membrane_{timelapse_nuclei_to_track}.csv",
        f"non_maximal_oneat_{label}_locations_membrane_{timelapse_nuclei_to_track}.csv"
    ]
    ]
    for index, csv_file in enumerate(csv_files):
                    csvfile = os.path.join(timelapse_oneat_directory, csv_file)
                    savefile = os.path.join(timelapse_oneat_directory, save_files[index] )
                    print(membranesegimage,csvfile,savefile)
                    generate_membrane_locations(membranesegimage, csvfile, savefile)


if __name__=='__main__':
    main()    
     
     
     
