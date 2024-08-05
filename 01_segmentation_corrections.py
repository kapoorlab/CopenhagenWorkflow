import os
import glob
from tifffile import imread
from vollseg.utils import SegCorrect
import hydra
from scenario_train_vollseg_cellpose_sam import TrainCellPose
from hydra.core.config_store import ConfigStore
from pathlib import Path 
configstore = ConfigStore.instance()
configstore.store(name='TrainCellPose', node=TrainCellPose)

@hydra.main(version_base="1.3", config_path = 'conf', config_name = 'scenario_train_vollseg_cellpose_sam')
def main( config : TrainCellPose):

    imagedir = os.path.join(config.train_data_paths.base_roi_dir, config.train_data_paths.raw_roi_dir)
    segmentationdir = os.path.join(config.train_data_paths.base_roi_dir,config.train_data_paths.binary_mask_roi_dir)

    segcorrect = SegCorrect(imagedir, segmentationdir)

    segcorrect.showNapari()

if __name__ == '__main__':

    main() 