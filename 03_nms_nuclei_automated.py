#!/usr/bin/env python

# coding: utf-8

from oneat.NEATUtils import NEATViz
import hydra
from scenario_predict_oneat import VollOneat
from hydra.core.config_store import ConfigStore
import os
from pathlib import Path
configstore = ConfigStore.instance()
configstore.store(name = 'VollOneat', node = VollOneat)

@hydra.main(version_base="1.3",config_path = 'conf', config_name = 'scenario_predict_oneat')
def main( config : VollOneat):
            tracking_nuclei_directory = config.experiment_data_paths.timelapse_nuclei_directory
            csvdir = config.experiment_data_paths.timelapse_oneat_directory
            model_dir = config.model_paths.oneat_nuclei_model_dir
           
            categories_json = os.path.join(model_dir, config.parameters.categories_json)
            fileextension = config.parameters.file_type
            event_threshold = config.parameters.event_threshold
            nms_space = config.parameters.nms_space
            nms_time = config.parameters.nms_time
            NEATViz(tracking_nuclei_directory,
                    csvdir,
                  
                    categories_json,
                    fileextension=fileextension,
                    event_threshold=event_threshold,
                    headless=True,
                    volume=True,
                    nms_space=nms_space,
                    nms_time=nms_time)


if __name__=='__main__':
     main() 



