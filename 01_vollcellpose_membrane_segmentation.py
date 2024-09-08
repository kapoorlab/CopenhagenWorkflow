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
    

    cellpose_model_name = config.model_paths.cellpose2D_model_name

    
    cellpose_model_dir = config.model_paths.cellpose2D_model_dir

    diameter_cellpose = config.parameters.diameter_cellpose
    stitch_threshold = config.parameters.stitch_threshold
    channel_membrane = config.parameters.channel_membrane
    channel_nuclei = config.parameters.channel_nuclei
    flow_threshold = config.parameters.flow_threshold
    cellprob_threshold = config.parameters.cellprob_threshold
    gpu = config.parameters.gpu
    max_size = config.parameters.max_size

    Raw_path = os.path.join(dual_channel_image_dir, config.parameters.file_type)
    filesRaw = glob.glob(Raw_path)
    filesRaw =natsorted(filesRaw)
    do_3D = config.parameters.do_3D
    n_tiles = tuple(config.parameters.n_tiles)
    axes = config.parameters.axes

    for fname in tqdm(filesRaw):
        image = imread(fname)
        Name = os.path.basename(os.path.splitext(fname)[0])
        extension = os.path.splitext(fname)[1]
        inner_folder_path = os.path.join(save_dir, 'ollCellPose')  
        stitch_threshold = config.parameters.stitch_threshold
        nuclei_segmentation_folder = os.path.join(nuclei_save_dir, 'StarDist') 
        roi_segmentation_folder = os.path.join(nuclei_save_dir, 'Roi')
        edge_enhanced_folder_path = os.path.join(save_dir, 'Membrane_Enhanced')
        Path(edge_enhanced_folder_path).mkdir(exist_ok=True)
        if True: #not os.path.exists(os.path.join(inner_folder_path, Name + extension)):
                
                nuclei_seg_image = imread(os.path.join(nuclei_segmentation_folder, Name + extension))
                roi_image = imread(os.path.join(roi_segmentation_folder, Name + extension))
                denoised_image_membrane = imread(os.path.join(edge_enhanced_folder_path, Name + extension))
               
                image[ :, channel_membrane, :, :] = denoised_image_membrane
                
                VollCellSeg(
                            image,
                            nuclei_seg_image = nuclei_seg_image,
                            roi_image = roi_image, 
                            diameter_cellpose = diameter_cellpose,
                            stitch_threshold = stitch_threshold,
                            channel_membrane = channel_membrane,
                            channel_nuclei = channel_nuclei,
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold,
                            
                            cellpose_model_path= None, #os.path.join(cellpose_model_dir, cellpose_model_name),
                            gpu = gpu,
                            axes = axes,
                            
                            n_tiles = n_tiles,
                            
                            
                            save_dir=save_dir,
                            Name = Name,
                            max_size = max_size,
                            do_3D=do_3D,
                            
                        )



if __name__ == '__main__':
    main()
