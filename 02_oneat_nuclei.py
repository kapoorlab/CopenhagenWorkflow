#!/usr/bin/env python
# coding: utf-8
import os
from oneat.NEATModels import NEATDenseVollNet
from oneat.NEATUtils.utils import load_json
from pathlib import Path
from tifffile import imread
import hydra
from scenario_predict_oneat import VollOneat
from hydra.core.config_store import ConfigStore
import numpy as np
configstore = ConfigStore.instance()
configstore.store(name = 'VollOneat', node = VollOneat)

@hydra.main(version_base="1.3", config_path = 'conf', config_name = 'scenario_predict_oneat')
def main( config : VollOneat):
          n_tiles = config.parameters.n_tiles
          event_threshold = list(config.parameters.event_threshold)
          event_confidence = list(config.parameters.event_confidence)
          timelapse_directory = config.experiment_data_paths.timelapse_nuclei_directory
          timelapse_seg_directory = config.experiment_data_paths.timelapse_seg_nuclei_directory
          model_dir = config.model_paths.oneat_nuclei_model_dir
          savedir =  Path(timelapse_directory).parent / 'oneat_detections'
          Path(savedir).mkdir(exist_ok=True)
          
          division_categories_json = model_dir + config.parameters.categories_json
          catconfig = load_json(division_categories_json)
          division_cord_json = model_dir + config.parameters.cord_json
          cordconfig = load_json(division_cord_json)
          model = NEATDenseVollNet(None, model_dir, catconfig, cordconfig)
          
          files = os.listdir(timelapse_directory)
          for fname in files:
                    image = imread(os.path.join(timelapse_directory, fname))
                    segimage = imread(os.path.join(timelapse_seg_directory, fname))
                    marker_tree =  model.get_markers(segimage)
                    name = fname.replace('.tif', '')
                    savename = 'nuclei_' + name
                    model.predict(image,
                                   savedir = savedir,
                                   savename = savename, 
                                   n_tiles = n_tiles, 
                                   dtype = np.float32,
                                   event_threshold = event_threshold, 
                                   event_confidence = event_confidence,
                                   
                                   marker_tree = marker_tree)
          
               
if __name__=="__main__":
     main()
