import os
import glob
from tifffile import imread
from vollseg import StarDist3D, UNET, MASKUNET
import hydra
from vollseg import VollSeg3D, CARE
from tifffile import imread, imwrite
from scenario_segment_star_cellpose import VollCellSegPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
configstore = ConfigStore.instance()
configstore.store(name='VollCellSegPose', node=VollCellSegPose)
from natsort import natsorted



@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_segment_star_cellpose')
def main(config: VollCellSegPose):
    dual_channel_image_dir = config.experiment_data_paths.dual_channel_split_directory
    save_dir = os.path.join(dual_channel_image_dir, 'VollCellPoseSeg')
    channel_membrane = config.parameters.channel_membrane

    den_model_dir = config.model_paths.den_model_dir
    edge_enhancement_model_name = config.model_paths.edge_enhancement_model_name

    Raw_path = os.path.join(dual_channel_image_dir, config.parameters.file_type)
    filesRaw = glob.glob(Raw_path)
    filesRaw =natsorted(filesRaw)
    
    n_tiles = tuple(config.parameters.n_tiles)
    
    edge_enhancement_model = CARE(config = None, name = edge_enhancement_model_name, basedir = den_model_dir)

    for fname in filesRaw:
        image = imread(fname)
        Name = os.path.basename(os.path.splitext(fname)[0])
        edge_enhanced_folder_path = os.path.join(save_dir, 'Membrane_Enhanced')
        Path(edge_enhanced_folder_path).mkdir(exist_ok=True)
        image_membrane = image[ :, channel_membrane, :, :]
        denoised_image_membrane = VollSeg3D(image_membrane,unet_model = None, star_model = None,  noise_model=edge_enhancement_model,n_tiles= n_tiles, dounet=False,  axes='ZYX')
        imwrite(edge_enhanced_folder_path + '/' + Name + '.tif', denoised_image_membrane)                                        
        
              



if __name__ == '__main__':
    main()
