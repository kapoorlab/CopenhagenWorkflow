import os
import hydra
from hydra.core.config_store import ConfigStore
from napatrackmater.Trackmate import transfer_fate_location
from pathlib import Path
from scenario_track import NapaTrackMater
configstore = ConfigStore.instance()
configstore.store(name = 'NapaTrackMater' , node = NapaTrackMater )


@hydra.main(version_base ="1.3", config_path = 'conf', config_name = 'scenario_track')
def main(config:NapaTrackMater):
    timelapse_nuclei_to_track = config.experiment_data_paths.timelapse_nuclei_to_track
    timelapse_seg_membrane_directory = config.experiment_data_paths.timelapse_seg_membrane_directory
    membranesegimage = os.path.join(timelapse_seg_membrane_directory, timelapse_nuclei_to_track + '.tif')
    csv_files = [
                    config.experiment_data_paths.basal_cells_nuclei_predicted, config.experiment_data_paths.radial_cells_nuclei_predicted,
                    config.experiment_data_paths.goblet_cells_nuclei_predicted
                ]
    save_files = [
                   config.experiment_data_paths.basal_cells_membrane_transferred, config.experiment_data_paths.radial_cells_membrane_transferred,
                    config.experiment_data_paths.goblet_cells_membrane_transferred
                ]

    for index, csv_file in enumerate(csv_files):
                    save_file = save_files[index]
                    Path(save_file).mkdir(exist_ok=True)
                    print(membranesegimage,csv_file,save_file)
                    transfer_fate_location(membranesegimage, csv_file, save_file)


if __name__=='__main__':
    main()    
     
     
     
