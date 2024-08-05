import os
import glob
from tifffile import imread
from vollseg import StarDist3D, UNET, MASKUNET
from vollseg.utils import VollSeg
import hydra
from scenario_segment_star_cellpose import VollCellSegPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
from natsort import natsorted
configstore = ConfigStore.instance()
configstore.store(name='VollCellSegPose', node=VollCellSegPose)

@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_segment_star_cellpose')
def main(config: VollCellSegPose):
    
    image_dir = config.experiment_data_paths.dual_channel_split_directory
   
    save_dir = os.path.join(image_dir, 'VollSeg')
    Path(save_dir).mkdir(exist_ok=True)
    unet_model_dir = config.model_paths.unet_model_dir
    star_model_dir = config.model_paths.star_model_dir
    roi_model_dir = config.model_paths.roi_model_dir

    unet_model_name = config.model_paths.unet_nuclei_model_name
    star_model_name = config.model_paths.star_nuclei_model_name
    roi_model_name = config.model_paths.roi_nuclei_model_name


    unet_model = UNET(config = None, name = unet_model_name, basedir = unet_model_dir)
    star_model = StarDist3D(config = None, name = star_model_name, basedir = star_model_dir)
    roi_model =  MASKUNET(config = None, name = roi_model_name, basedir = roi_model_dir)



    Raw_path = os.path.join(image_dir, config.parameters.file_type)
    filesRaw = glob.glob(Raw_path)
    filesRaw =natsorted(filesRaw)
    
    min_size = config.parameters.min_size
    min_size_mask = config.parameters.min_size_mask
    max_size = config.parameters.max_size
    n_tiles = config.parameters.n_tiles
    channel_nuclei = config.parameters.channel_nuclei
    dounet = config.parameters.dounet
    seedpool = config.parameters.seedpool
    slice_merge = config.parameters.slice_merge
    UseProbability = config.parameters.UseProbability
    donormalize = config.parameters.donormalize
    axes = config.parameters.axes
    ExpandLabels = config.parameters.ExpandLabels
    for fname in filesRaw:
                    # Check if the result file already exists
                    result_file = os.path.join(save_dir, 'StarDist', f'{os.path.splitext(os.path.basename(fname))[0]}.tif')
                    if os.path.exists(result_file):
                        print(f"Skipping {fname} as {result_file} already exists.")
                        continue 
                    image = imread(fname)
                    image_nuclei = image[:, channel_nuclei, :, :]
                    Name = os.path.basename(os.path.splitext(fname)[0])
                    VollSeg( image_nuclei, 
                            unet_model = unet_model, 
                            star_model = star_model, 
                            roi_model= roi_model,
                            seedpool = seedpool, 
                            axes = axes, 
                            min_size = min_size,  
                            min_size_mask = min_size_mask,
                            max_size = max_size,
                            donormalize=donormalize,
                            n_tiles = n_tiles,
                            ExpandLabels = ExpandLabels,
                            slice_merge = slice_merge, 
                            UseProbability = UseProbability, 
                            save_dir = save_dir, 
                            Name = Name,
                            dounet = dounet) 

    
    
if __name__ == '__main__':
    main()    
