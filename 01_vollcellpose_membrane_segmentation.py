import os
import glob
from tifffile import imread
from vollseg import StarDist3D, UNET, MASKUNET
from vollseg.utils import VollCellSeg
import hydra
from vollseg import VollSeg3D, CARE
from tifffile import imread, imwrite
from scenario_segment_star_cellpose import VollCellSegPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
configstore = ConfigStore.instance()
configstore.store(name='VollCellSegPose', node=VollCellSegPose)
from pynvml.smi import nvidia_smi
import tensorflow as tf
from natsort import natsorted



@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_segment_star_cellpose')
def main(config: VollCellSegPose):
    dual_channel_image_dir = config.experiment_data_paths.dual_channel_split_directory
    save_dir = os.path.join(dual_channel_image_dir, 'VollCellPoseSeg')
    Path(save_dir).mkdir(exist_ok=True)
    nuclei_save_dir = os.path.join(dual_channel_image_dir, 'VollSeg')
    

    cellpose_model_name = config.model_paths.cellpose2D_model_name

    
    cellpose_model_dir = config.model_paths.cellpose2D_model_dir

    diameter_cellpose = config.parameters.diameter_cellpose
    stitch_threshold = config.parameters.stitch_threshold
    channel_membrane = config.parameters.channel_membrane
    channel_nuclei = config.parameters.channel_nuclei
    flow_threshold = config.parameters.flow_threshold
    cellprob_threshold = config.parameters.cellprob_threshold
    gpu = config.parameters.gpu

   

    den_model_dir = config.model_paths.den_model_dir
    edge_enhancement_model_name = config.model_paths.edge_enhancement_model_name

    Raw_path = os.path.join(dual_channel_image_dir, config.parameters.file_type)
    filesRaw = glob.glob(Raw_path)
    filesRaw =natsorted(filesRaw)
    min_size = config.parameters.min_size
    min_size_mask = config.parameters.min_size_mask
    max_size = config.parameters.max_size
    do_3D = config.parameters.do_3D
    n_tiles = tuple(config.parameters.n_tiles)
    dounet = config.parameters.dounet
    seedpool = config.parameters.seedpool
    slice_merge = config.parameters.slice_merge
    UseProbability = config.parameters.UseProbability
    donormalize = config.parameters.donormalize
    axes = config.parameters.axes
    ExpandLabels = config.parameters.ExpandLabels
    z_thresh = config.parameters.z_thresh
    edge_enhancement_model = CARE(config = None, name = edge_enhancement_model_name, basedir = den_model_dir)

    for fname in filesRaw:
        image = imread(fname)
        Name = os.path.basename(os.path.splitext(fname)[0])
        extension = os.path.splitext(fname)[1]
        inner_folder_path = os.path.join(save_dir, 'CellPose')  
        nuclei_segmentation_folder = os.path.join(nuclei_save_dir, 'StarDist') 
        edge_enhanced_folder_path = os.path.join(save_dir, 'Membrane_Enhanced')
        Path(edge_enhanced_folder_path).mkdir(exist_ok=True)
        if not os.path.exists(os.path.join(inner_folder_path, Name + extension)):
                
                nuclei_seg_image = imread(os.path.join(nuclei_segmentation_folder, Name + extension))
                
                denoised_image_membrane = imread(os.path.join(edge_enhanced_folder_path, Name + extension))
 

                image_membrane = image[ :, channel_membrane, :, :]
                
                denoised_image_membrane = VollSeg3D(image_membrane,unet_model = None, star_model = None,  noise_model=edge_enhancement_model,n_tiles= n_tiles, dounet=False,  axes='ZYX')
                imwrite(edge_enhanced_folder_path + '/' + Name + '.tif', denoised_image_membrane)                                        
                
                image[ :, channel_membrane, :, :] = denoised_image_membrane
                
                VollCellSeg(
                            image,
                            nuclei_seg_image = nuclei_seg_image,
                            diameter_cellpose = diameter_cellpose,
                            stitch_threshold = stitch_threshold,
                            channel_membrane = channel_membrane,
                            channel_nuclei = channel_nuclei,
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold,
                            
                            cellpose_model_path= os.path.join(cellpose_model_dir, cellpose_model_name),
                            gpu = gpu,
                            axes = axes,
                            min_size_mask = min_size_mask,
                            min_size = min_size,
                            max_size = max_size,
                            n_tiles = n_tiles,
                            UseProbability= UseProbability,
                            ExpandLabels = ExpandLabels,
                            donormalize = donormalize,
                            dounet = dounet,
                            seedpool=seedpool,
                            save_dir=save_dir,
                            Name = Name,
                            slice_merge=slice_merge,
                            do_3D=do_3D,
                            z_thresh = z_thresh
                        )



if __name__ == '__main__':
    main()
