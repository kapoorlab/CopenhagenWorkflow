import hydra
from hydra.core.config_store import ConfigStore
from scenario_track import NapaTrackMater
from pathlib import Path
import os 
from napatrackmater.Trackmate import TrackMate
from tifffile import imread 

configstore = ConfigStore.instance()
configstore.store(name = 'NapaTrackMater' , node = NapaTrackMater )


@hydra.main(version_base ="1.3", config_path = 'conf', config_name = 'scenario_track')
def main(config:NapaTrackMater):

    AttributeBoxname = "AttributeIDBox"
    TrackAttributeBoxname = "TrackAttributeIDBox"
    TrackidBox = "All"
    initial_channel = 'nuclei_'
    axes = config.parameters.axes
    timelapse_to_track =  config.experiment_data_paths.timelapse_nuclei_to_track
    base_dir = config.experiment_data_paths.tracking_directory
    xml_name = initial_channel + timelapse_to_track + ".xml"
   
    seg_image_name = timelapse_to_track + ".tif"
    xml_path = Path(os.path.join(base_dir, xml_name))
   

    channel_seg_image =  imread(os.path.join(config.experiment_data_paths.timelapse_seg_membrane_directory,seg_image_name))

    TrackMate(
    xml_path,    
    AttributeBoxname=AttributeBoxname,
    TrackAttributeBoxname=TrackAttributeBoxname,
    TrackidBox=TrackidBox,
    axes=axes,
    second_channel_name='membrane',
    channel_seg_image = channel_seg_image,
    channel_xml_path=base_dir   
    )

if __name__=='__main__':
        main()    