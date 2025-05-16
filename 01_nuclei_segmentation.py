import os
import glob
from tifffile import imread, imwrite
from vollseg import StarDist3D, MASKUNET
from vollseg.utils import VollSeg
import hydra
import numpy as np
from tqdm import tqdm 
from scenario_segment_star_cellpose import VollCellSegPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
from natsort import natsorted
configstore = ConfigStore.instance()
configstore.store(name='VollCellSegPose', node=VollCellSegPose)

@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_segment_star_cellpose')
def main(config: VollCellSegPose):
    
    base_directory = config.experiment_data_paths.base_directory
    timelapse_nuclei_directory = os.path.join(base_directory, config.experiment_data_paths.timelapse_nuclei_directory)
    timelapse_seg_directory = os.path.join(base_directory, config.experiment_data_paths.timelapse_seg_nuclei_directory)
    split_timelapse_nuclei_directory =  os.path.join(base_directory, config.experiment_data_paths.dual_channel_split_directory)
    voxel_size_xyz = config.experiment_data_paths.voxel_size_xyz  
    Path(split_timelapse_nuclei_directory).mkdir(exist_ok=True)
    Path(timelapse_seg_directory).mkdir(exist_ok=True)
    channel_nuclei = config.parameters.channel_nuclei
    stardist_nucli_model_name = config.model_paths.star_nuclei_model_name
    stardist_nuclei_model_dir = config.model_paths.star_model_dir
    

    axes = config.parameters.axes
    min_size = config.parameters.min_size
    n_tiles = config.parameters.n_tiles
    star_model = StarDist3D(config = None, name = stardist_nucli_model_name, basedir = stardist_nuclei_model_dir)
    roi_model = MASKUNET(config=None,
                         name=config.model_paths.roi_nuclei_model_name,
                         basedir=config.model_paths.roi_model_dir)
    Raw_path = os.path.join(timelapse_nuclei_directory, config.parameters.file_type)
    filesRaw = glob.glob(Raw_path)
    filesRaw = natsorted(filesRaw)
    raw_save_dir = os.path.join(split_timelapse_nuclei_directory, 'Raw')
    Path(raw_save_dir).mkdir(exist_ok=True)
    seg_save_dir = os.path.join(split_timelapse_nuclei_directory, 'StarDist')
    min_size = config.parameters.min_size
    min_size_mask = config.parameters.min_size_mask
    max_size = config.parameters.max_size
    donormalize = config.parameters.donormalize
    ExpandLabels = config.parameters.ExpandLabels
    slice_merge = config.parameters.slice_merge
    UseProbability = config.parameters.UseProbability
    for fname in tqdm(filesRaw):
        image = imread(fname)
        Name = os.path.basename(os.path.splitext(fname)[0])
        if len(image.shape) == 5:

            image_gfp_time = image[ :, :, channel_nuclei, :, :]
        elif len(image.shape) == 4:
            image_gfp_time =  image
        else:
            print(f'Image should be 4 or 5 D but found {len(image.shape)}')    
        for time in range(image_gfp_time.shape[0]):
            image_nuclei = image_gfp_time[time]

           
            result_file = os.path.join(seg_save_dir, f'{Name}_{time}.tif')
            if os.path.exists(result_file):
                print(f"Skipping {fname} as {result_file} already exists.")
                continue 
            imwrite(os.path.join(raw_save_dir, f'{Name}_{time}.tif'), image_nuclei)
            VollSeg( image_nuclei, 
                            star_model = star_model, 
                            roi_model= roi_model,
                            axes = axes, 
                            min_size = min_size,  
                            min_size_mask = min_size_mask,
                            max_size = max_size,
                            donormalize=donormalize,
                            n_tiles = n_tiles,
                            ExpandLabels = ExpandLabels,
                            slice_merge = slice_merge, 
                            UseProbability = UseProbability, 
                            save_dir = raw_save_dir, 
                            Name = Name) 

        seg_files = glob.glob(os.path.join(seg_save_dir, f"{Name}_*.tif"))
        seg_files = natsorted(seg_files)  

        seg_timelapse = []

        for seg_file in seg_files:
            img = imread(seg_file)
            seg_timelapse.append(img)

        seg_timelapse = np.array(seg_timelapse)

        
        timelapse_tif_path = os.path.join(timelapse_seg_directory, f'{Name}.tif')

        imwrite(timelapse_tif_path, seg_timelapse, imagej=True, bigtiff=True,
                photometric='minisblack',
                resolution=(1 / voxel_size_xyz[0], 1 / voxel_size_xyz[1]),
                metadata={'spacing': voxel_size_xyz[2], 'unit': 'um', 'axes': 'TZYX'})

        print(f"Timelapse saved as: {timelapse_tif_path}") 
    
    
if __name__ == '__main__':
    main()    
