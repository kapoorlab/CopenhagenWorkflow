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
    timelapse_seg_nuclei_vollseg_directory = config.experiment_data_paths.timelapse_seg_nuclei_vollseg_directory
   
    timelapse_oneat_vollseg_directory = config.experiment_data_paths.timelapse_oneat_vollseg_directory
    Path(timelapse_oneat_vollseg_directory).mkdir(exist_ok=True)
    vollsegsegimage = os.path.join(timelapse_seg_nuclei_vollseg_directory, timelapse_nuclei_to_track + '.tif')
    csv_files = [
                    f"oneat_{label}_locations_nuclei_{timelapse_nuclei_to_track}.csv"
                    for label in ["mitosis"]
                ]
    save_files = [
                    f"oneat_{label}_locations_nuclei_vollseg_{timelapse_nuclei_to_track}.csv"
                    for label in ["mitosis"]
                ]

    for index, csv_file in enumerate(csv_files):
                    csvfile = os.path.join(timelapse_oneat_directory, csv_file)
                    savefile = os.path.join(timelapse_oneat_vollseg_directory, save_files[index] )
                    print(vollsegsegimage,csvfile,savefile)
                    generate_membrane_locations(vollsegsegimage, csvfile, savefile)


if __name__=='__main__':
    main()    
     
     
     
