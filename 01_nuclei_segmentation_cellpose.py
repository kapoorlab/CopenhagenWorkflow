import os
import glob
from tifffile import imread
from vollseg.utils import CellPoseSeg
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
   
    save_dir = os.path.join(image_dir, 'Pure_CellPose_Nuclei')
    Path(save_dir).mkdir(exist_ok=True)
    
    cellpose_model_name = config.model_paths.cellpose2D_nuclei_model_name
    diameter_cellpose = config.parameters.diameter_cellpose
    stitch_threshold = config.parameters.stitch_threshold
    channel_nuclei = config.parameters.channel_nuclei
    cellpose_model_dir = config.model_paths.cellpose2D_model_dir
    flow_threshold = config.parameters.flow_threshold
    cellprob_threshold = config.parameters.cellprob_threshold
    gpu = config.parameters.gpu
    
    cellpose_model_type = config.model_paths.cellpose_model_type

    Raw_path = os.path.join(image_dir, config.parameters.file_type)
    filesRaw = glob.glob(Raw_path)
    filesRaw =natsorted(filesRaw)

    axes = config.parameters.axes
    do_3D = config.parameters.do_3D
    for fname in filesRaw:
    
                    image = imread(fname)
                    image_nuclei = image[:, channel_nuclei, :, :]
                    Name = os.path.basename(os.path.splitext(fname)[0])
                    CellPoseSeg( image_nuclei, 
                            diameter_cellpose= diameter_cellpose,
                            stitch_threshold = stitch_threshold,
                            flow_threshold = flow_threshold,
                            cellprob_threshold = cellprob_threshold,
                            cellpose_model_type = cellpose_model_type,
                            #cellpose_model_path= os.path.join(cellpose_model_dir, cellpose_model_name),
                            gpu = gpu,
                            axes = axes,
                            save_dir=save_dir,
                            Name = Name,
                            do_3D=do_3D
                            ) 

    
    
if __name__ == '__main__':
    main()    