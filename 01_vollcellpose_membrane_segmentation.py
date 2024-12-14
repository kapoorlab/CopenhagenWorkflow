import os
import glob
from tifffile import imread
from vollseg.utils import VollCellSeg
import hydra
from tifffile import imread
from scenario_segment_star_cellpose import VollCellSegPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
configstore = ConfigStore.instance()
configstore.store(name='VollCellSegPose', node=VollCellSegPose)
from natsort import natsorted
from tqdm import tqdm 


@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_segment_star_cellpose')
def main(config: VollCellSegPose):
    dual_channel_image_dir = config.experiment_data_paths.dual_channel_split_directory
    save_dir = os.path.join(dual_channel_image_dir, 'VollCellPoseSeg')
    Path(save_dir).mkdir(exist_ok=True)
    nuclei_save_dir = os.path.join(dual_channel_image_dir, 'VollSeg')
    channel_membrane = config.parameters.channel_membrane
    Raw_path = os.path.join(dual_channel_image_dir, config.parameters.file_type)
    filesRaw = glob.glob(Raw_path)
    filesRaw =natsorted(filesRaw)
    n_tiles = tuple(config.parameters.n_tiles)
    axes = config.parameters.axes

    for fname in tqdm(filesRaw):
        image = imread(fname)
        Name = os.path.basename(os.path.splitext(fname)[0])
        extension = os.path.splitext(fname)[1]
        mask_folder = os.path.join(nuclei_save_dir, 'Roi')  
        nuclei_segmentation_folder = os.path.join(nuclei_save_dir, 'Markers') 
        edge_enhanced_folder_path = os.path.join(dual_channel_image_dir, 'Membrane_Enhanced')
        cellpose_folder_path =  os.path.join(save_dir, 'VollCellPose')
        Path(edge_enhanced_folder_path).mkdir(exist_ok=True)
        result_file = os.path.join(cellpose_folder_path, f'{Name}.tif')
        if os.path.exists(result_file):
            print(f"Skipping {fname} as {result_file} already exists.")
            continue
                
        nuclei_seg_image = imread(os.path.join(nuclei_segmentation_folder, Name + extension))
        denoised_image_membrane = imread(os.path.join(edge_enhanced_folder_path, Name + extension))
        mask = imread(os.path.join(mask_folder, Name + extension))
        image[ :, channel_membrane, :, :] = denoised_image_membrane
        
        VollCellSeg(
                    image,
                    nuclei_seg_image = nuclei_seg_image,
                    mask = mask,
                    channel_membrane = channel_membrane,
                    axes = axes,
                    n_tiles = n_tiles,
                    save_dir=save_dir,
                    Name = Name,
                    
                )



if __name__ == '__main__':
    main()
