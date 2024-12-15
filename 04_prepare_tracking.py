import os
from pathlib import Path
from tifffile import imread, imwrite
import hydra
from scenario_track import NapaTrackMater
from hydra.core.config_store import ConfigStore
import numpy as np
import shutil
import gc
import tqdm
configstore = ConfigStore.instance()
configstore.store(name='VollOneat', node=NapaTrackMater)

@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_track')
def main(config: NapaTrackMater):
    
        do_vollseg = False
        do_membrane = True
        do_nuclei = False
        
            
        timelapse_directory = config.experiment_data_paths.timelapse_nuclei_directory
        timelapse_seg_directory = config.experiment_data_paths.timelapse_seg_nuclei_directory
        timelapse_oneat_directory = config.experiment_data_paths.timelapse_oneat_directory
        tracking_directory = config.experiment_data_paths.tracking_directory
        Path(tracking_directory).mkdir(exist_ok=True)
        
        if do_nuclei:
                files = os.listdir(timelapse_directory)
                for fname in files:
                        image = imread(os.path.join(timelapse_directory, fname), dtype=np.uint8)
                        print(f'Image shape: {image.shape}')
                        segimage = imread(os.path.join(timelapse_seg_directory, fname),dtype=np.uint16)
                        print(f'Segimage shape: {segimage.shape}')
                        name = fname.replace('.tif', '')
                        
                        
                        csv_files = [
                            f"non_maximal_oneat_{label}_locations_nuclei_{name}.csv"
                            for label in ["mitosis"]
                        ]

                        for csv_file in csv_files:
                            csv_path_src = os.path.join(timelapse_oneat_directory, csv_file)
                            csv_path_dst = os.path.join(tracking_directory, csv_file)
                            shutil.copy(csv_path_src, csv_path_dst)

                        hyperstack = np.asarray([segimage, image], dtype=np.uint16)
                        del image 
                        del segimage
                        gc.collect()
                        
                        hyperstack = np.transpose(hyperstack, (1, 2, 0, 3, 4))
                        voxel_size_xyz = config.experiment_data_paths.voxel_size_xyz  
                        
                        hyperstack_path = os.path.join(tracking_directory, f"nuclei_{name}.tif")
                        imwrite(hyperstack_path, hyperstack, imagej=True, bigtiff=True,
                        photometric='minisblack',
                        resolution=(1 / voxel_size_xyz[0], 1 / voxel_size_xyz[1]),
                        metadata={'spacing': voxel_size_xyz[2], 'unit': 'um', 
                                    'axes': 'TZCYX'})
                        
                        del hyperstack
                        gc.collect() 
        if do_vollseg:
            timelapse_seg_directory = config.experiment_data_paths.timelapse_seg_nuclei_vollseg_directory
            timelapse_oneat_vollseg_directory = config.experiment_data_paths.timelapse_oneat_vollseg_directory
            tracking_vollseg_directory = config.experiment_data_paths.tracking_vollseg_directory
            Path(tracking_vollseg_directory).mkdir(exist_ok=True)
            files = os.listdir(timelapse_directory)
            for fname in files:
                    image = imread(os.path.join(timelapse_directory, fname), dtype=np.uint8)
                    print(f'Image shape: {image.shape}')
                    segimage = imread(os.path.join(timelapse_seg_directory, fname),dtype=np.uint16)
                    print(f'VollSeg Segimage shape: {segimage.shape}')
                    name = fname.replace('.tif', '')
                    
                    
                    csv_files = [
                        f"non_maximal_oneat_{label}_locations_nuclei_vollseg_{name}.csv"
                        for label in ["mitosis"]
                    ]

                    for csv_file in csv_files:
                        csv_path_src = os.path.join(timelapse_oneat_vollseg_directory, csv_file)
                        csv_path_dst = os.path.join(tracking_vollseg_directory, csv_file)
                        shutil.copy(csv_path_src, csv_path_dst)

                    hyperstack = np.asarray([segimage, image], dtype=np.uint16)
                    del image 
                    del segimage
                    gc.collect()
                    
                    hyperstack = np.transpose(hyperstack, (1, 2, 0, 3, 4))
                    voxel_size_xyz = config.experiment_data_paths.voxel_size_xyz  
                    
                    hyperstack_path = os.path.join(tracking_vollseg_directory, f"nuclei_vollseg_{name}.tif")
                    imwrite(hyperstack_path, hyperstack, imagej=True, bigtiff=True,
                    photometric='minisblack',
                    resolution=(1 / voxel_size_xyz[0], 1 / voxel_size_xyz[1]),
                    metadata={'spacing': voxel_size_xyz[2], 'unit': 'um', 
                                'axes': 'TZCYX'})
                    
                    del hyperstack
                    gc.collect()         
        
        if do_membrane:
            timelapse_directory = config.experiment_data_paths.timelapse_membrane_directory
            timelapse_seg_directory = config.experiment_data_paths.timelapse_seg_membrane_directory
            tracking_directory = config.experiment_data_paths.tracking_directory
            
            files = os.listdir(timelapse_directory)
            for fname in files:
                file_path = os.path.join(timelapse_directory, fname)
                if os.path.isfile(file_path):
                    image = imread(os.path.join(timelapse_directory, fname), dtype=np.uint8)
                    print(f'Image shape: {image.shape}')
                    segimage = imread(os.path.join(timelapse_seg_directory, fname),dtype=np.uint16)
                    print(f'Segimage shape: {segimage.shape}')
                    name = fname.replace('.tif', '')
                    csv_files = [
                        f"non_maximal_oneat_{label}_locations_membrane_{name}.csv"
                        for label in ["mitosis"]
                    ]

                    for csv_file in csv_files:
                        csv_path_src = os.path.join(timelapse_oneat_directory, csv_file)
                        csv_path_dst = os.path.join(tracking_directory, csv_file)
                        shutil.copy(csv_path_src, csv_path_dst)
                    hyperstack = np.asarray([segimage, image], dtype=np.uint16)
                    hyperstack = np.transpose(hyperstack, (1, 2, 0, 3, 4))
                    voxel_size_xyz = config.experiment_data_paths.voxel_size_xyz    
                    del image 
                    del segimage
                    gc.collect()
                    hyperstack_path = os.path.join(tracking_directory, f"membrane_{name}.tif")
                    imwrite(hyperstack_path, hyperstack, imagej=True, bigtiff=True,
                    photometric='minisblack',
                    resolution=(1 / voxel_size_xyz[0], 1 / voxel_size_xyz[1]),
                    metadata={'spacing': voxel_size_xyz[2], 'unit': 'um', 
                                'axes': 'TZCYX'})   

                    del hyperstack
                    gc.collect()                 

if __name__ == "__main__":
    main()