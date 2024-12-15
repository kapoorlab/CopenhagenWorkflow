import os
from tifffile import imread
import hydra
import numpy as np
from scenario_segment_star_cellpose import VollCellSegPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
import concurrent.futures
from natsort import natsorted
from tifffile import imread, imwrite
configstore = ConfigStore.instance()
configstore.store(name='VollCellSegPose', node=VollCellSegPose)


def stitch_files(input_files):
    images = [imread(file) for file in input_files]
    return images

@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_segment_star_cellpose')
def main(config: VollCellSegPose):
    
    input_parent_folder = config.experiment_data_paths.dual_channel_split_directory
    parent_directory = Path(input_parent_folder).parent
    timelapse_name = config.experiment_data_paths.timelapse_nuclei_to_track

    input_folders = [
        #os.path.join(input_parent_folder, 'VollSeg/Roi'),
        #os.path.join(input_parent_folder, 'VollSeg/StarDist'),
        #os.path.join(input_parent_folder, 'VollSeg/VollSeg'),
        os.path.join(input_parent_folder, 'VollCellPoseSeg/VollCellPose'),
        #os.path.join(input_parent_folder, 'VollCellPoseSeg/VollCellPose')

    ]

    output_folders = [
        #os.path.join(parent_directory, 'region_of_interest'),
        #os.path.join(parent_directory, 'seg_nuclei_timelapses'),
        #os.path.join(parent_directory, 'seg_nuclei_vollseg_timelapses'),
        os.path.join(parent_directory, 'seg_membrane_timelapses'),
        #os.path.join(parent_directory, 'watershed_seg_membrane_timelapses'),
    ]
    for input_folder, output_folder in zip(input_folders, output_folders):
        input_files = [os.path.join(input_folder, filename) for filename in natsorted(os.listdir(input_folder))]

        with concurrent.futures.ThreadPoolExecutor(os.cpu_count()) as executor:
                results = list(executor.map(stitch_files, [input_files]))

        timelapse = results[0]
        Path(output_folder).mkdir(exist_ok=True)
        print(f'Saving data in {output_folder}')
        output_path = os.path.join(output_folder, timelapse_name + '.tif')
        
        imwrite(output_path, timelapse, dtype=np.uint16, bigtiff=True)
        
    
if __name__ == '__main__':
    main()    