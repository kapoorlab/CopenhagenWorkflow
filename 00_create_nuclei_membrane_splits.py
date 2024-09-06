import os
from tifffile import imread
import hydra
import numpy as np
from scenario_segment_star_cellpose import VollCellSegPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
from concurrent.futures import ProcessPoolExecutor
from tifffile import imread, imwrite
configstore = ConfigStore.instance()
configstore.store(name='VollCellSegPose', node=VollCellSegPose)


def save_image(image_data, image_name, output_dir, time_point):
    os.makedirs(output_dir, exist_ok=True)
    
    image_path = os.path.join(output_dir, f"{image_name}-{time_point}.tif")
    
    imwrite(image_path, image_data)

@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_segment_star_cellpose')
def main(config: VollCellSegPose):
    
    nuclei_image_dir = config.experiment_data_paths.timelapse_nuclei_directory
    membrane_image_dir = config.experiment_data_paths.timelapse_membrane_directory
    Path(nuclei_image_dir).mkdir(exist_ok=True)
    Path(membrane_image_dir).mkdir(exist_ok=True)
    parent_directory = Path(nuclei_image_dir).parent
    channel_nuclei = config.parameters.channel_nuclei
    channel_membrane = config.parameters.channel_membrane
    merged_path = os.path.join(parent_directory, 'Merged.tif')
    merged_data = imread(merged_path)
    membrane_data = merged_data[:, :, channel_membrane, :, :]
    nuclei_data = merged_data[:, :, channel_nuclei, :, :]
    data_name =  os.path.splitext(os.path.basename(merged_path))[0]
    voxel_size_xyz = config.experiment_data_paths.voxel_size_xyz

    imwrite(os.path.join(nuclei_image_dir, 'timelapse_sixth_dataset.tif'), nuclei_data, imagej=True,
            photometric='minisblack',
            resolution=(1 / voxel_size_xyz[0], 1 / voxel_size_xyz[1]),
            metadata={'spacing': voxel_size_xyz[2], 'unit': 'um', 
                        'axes': 'TZYX'})
    
    imwrite(os.path.join(membrane_image_dir, 'timelapse_sixth_dataset.tif'), membrane_data, imagej=True,
            photometric='minisblack',
            resolution=(1 / voxel_size_xyz[0], 1 / voxel_size_xyz[1]),
            metadata={'spacing': voxel_size_xyz[2], 'unit': 'um', 
                        'axes': 'TZYX'})

    save_dir = config.experiment_data_paths.dual_channel_split_directory
    Path(save_dir).mkdir(exist_ok=True)
    
    merged_path = os.path.join(parent_directory, 'Merged.tif')
    data = np.asarray([membrane_data, nuclei_data])
    data = np.transpose(data, (1, 2, 0, 3, 4))

    imwrite(merged_path, data, imagej=True,
            photometric='minisblack',
            resolution=(1 / voxel_size_xyz[0], 1 / voxel_size_xyz[1]),
            metadata={'spacing': voxel_size_xyz[2], 'unit': 'um', 
                      'axes': 'TZCYX'})
    

    time_points = data.shape[0]
    print(data.shape)
    with ProcessPoolExecutor() as executor:
        futures = []

        for time_point in range(time_points):
            image_data = data[time_point]
            futures.append(executor.submit(save_image, image_data, data_name, save_dir, time_point))

        for future in futures:
            future.result()
    
    
if __name__ == '__main__':
    main()    