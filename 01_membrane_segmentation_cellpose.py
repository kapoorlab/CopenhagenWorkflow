import os
import glob
from tifffile import imread
from vollseg.utils import CellPoseSeg
import hydra
from vollseg import VollSeg3D, CARE
from scenario_segment_star_cellpose import VollCellSegPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
from natsort import natsorted
from tifffile import imread, imwrite
configstore = ConfigStore.instance()
configstore.store(name='VollCellSegPose', node=VollCellSegPose)

@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_segment_star_cellpose')
def main(config: VollCellSegPose):
    
    image_dir = config.experiment_data_paths.dual_channel_split_directory
    den_model_dir = config.model_paths.den_model_dir
    edge_enhancement_model_name = config.model_paths.edge_enhancement_model_name
    save_dir = os.path.join(image_dir, 'MembraneSeg')
    Path(save_dir).mkdir(exist_ok=True)
    
    cellpose_model_name = config.model_paths.cellpose2D_model_name
    
    diameter_cellpose = config.parameters.diameter_cellpose
    stitch_threshold = config.parameters.stitch_threshold
    channel_membrane = config.parameters.channel_membrane
    cellpose_model_dir = config.model_paths.cellpose2D_model_dir
    flow_threshold = config.parameters.flow_threshold
    cellprob_threshold = config.parameters.cellprob_threshold
    gpu = config.parameters.gpu
    n_tiles = config.parameters.n_tiles

    Raw_path = os.path.join(image_dir, config.parameters.file_type)
    filesRaw = glob.glob(Raw_path)
    filesRaw =natsorted(filesRaw)
    channel_membrane = config.parameters.channel_membrane
    axes = config.parameters.axes
    do_3D = config.parameters.do_3D
    edge_enhancement_model = CARE(config = None, name = edge_enhancement_model_name, basedir = den_model_dir)
    for fname in filesRaw:
                    
                    result_file = os.path.join(save_dir, 'CellPose', f'{os.path.splitext(os.path.basename(fname))[0]}.tif')
                    if os.path.exists(result_file):
                        print(f"Skipping {fname} as {result_file} already exists.")
                        continue 

                    image = imread(fname)
                    image_membrane = image[:, channel_membrane, :, :]
                    Name = os.path.basename(os.path.splitext(fname)[0])
                    extension = os.path.splitext(fname)[1]
                    inner_folder_path = os.path.join(save_dir, 'CellPose')  
                    edge_enhanced_folder_path = os.path.join(save_dir, 'Membrane_Enhanced')
                    if not os.path.exists(os.path.join(inner_folder_path, Name + extension)):
                            

                            denoised_image_membrane = VollSeg3D(image_membrane,unet_model = None, star_model = None,  noise_model=edge_enhancement_model,n_tiles= n_tiles, dounet=False,  axes='ZYX')
                            imwrite(edge_enhanced_folder_path + '/' + Name + '.tif', denoised_image_membrane)  
                            CellPoseSeg( denoised_image_membrane, 
                            diameter_cellpose= diameter_cellpose,
                            stitch_threshold = stitch_threshold,
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold,
                            cellpose_model_path= os.path.join(cellpose_model_dir, cellpose_model_name),
                            gpu = gpu,
                            axes = axes,
                            save_dir=save_dir,
                            Name = Name,
                            do_3D=do_3D
                            ) 
                            

    
    
if __name__ == '__main__':
    main()    