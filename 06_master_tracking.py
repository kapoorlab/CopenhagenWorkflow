import hydra
from hydra.core.config_store import ConfigStore

from pathlib import Path
import os 
from napatrackmater.Trackmate import TrackMate
from tifffile import imread
from scenario_track import NapaTrackMater
configstore = ConfigStore.instance()
configstore.store(name = 'NapaTrackMater' , node = NapaTrackMater )


@hydra.main(version_base ="1.3", config_path = 'conf', config_name = 'scenario_track')
def main(config:NapaTrackMater):
    
    do_nuclei = True 
    do_nuclei_vollseg = True
    do_membrane = True
    compute_with_autoencoder = config.parameters.compute_with_autoencoder
    variable_t_calibration = config.experiment_data_paths.variable_t_calibration
    oneat_nuclei_csv_file = config.experiment_data_paths.oneat_nuclei_csv_file
    oneat_nuclei_vollseg_csv_file = config.experiment_data_paths.oneat_nuclei_vollseg_csv_file
    oneat_membrane_csv_file = config.experiment_data_paths.oneat_membrane_csv_file
    oneat_threshold_cutoff = config.parameters.oneat_threshold_cutoff
    enhance_trackmate_xml = config.parameters.enhance_trackmate_xml
    if compute_with_autoencoder:
                master_extra_name = "autoencoder_"
    else:
                master_extra_name = "marching_cubes_filled_"
    timelapse_to_track =  config.experiment_data_paths.timelapse_membrane_to_track            
    tissue_mask_name = timelapse_to_track + ".tif"            
    if config.experiment_data_paths.timelapse_region_of_interest_directory is None:
                mask_image = None
    else:
                mask_image = imread(os.path.join(config.experiment_data_paths.timelapse_region_of_interest_directory, tissue_mask_name))            
    if do_nuclei:
            channel = 'nuclei_'
            timelapse_to_track =  config.experiment_data_paths.timelapse_nuclei_to_track
            base_dir = config.experiment_data_paths.tracking_directory
            xml_name = channel + timelapse_to_track + ".xml"
            seg_image_name = timelapse_to_track + ".tif"
            xml_path = Path(os.path.join(base_dir, xml_name))
            seg_image = imread(os.path.join(config.experiment_data_paths.timelapse_seg_nuclei_directory,seg_image_name))
            
            TrackMate(
                    xml_path,
                    seg_image=seg_image,
                    mask=mask_image,
                    enhance_trackmate_xml=enhance_trackmate_xml,
                    master_extra_name = master_extra_name,
                    compute_with_autoencoder=compute_with_autoencoder,
                    variable_t_calibration=variable_t_calibration,
                    oneat_csv_file=oneat_nuclei_csv_file,
                    oneat_threshold_cutoff=oneat_threshold_cutoff
                )
    if do_nuclei_vollseg:
            
            channel = 'nuclei_vollseg_'
            timelapse_to_track =  config.experiment_data_paths.timelapse_nuclei_to_track
            base_dir = config.experiment_data_paths.tracking_vollseg_directory
            xml_name = channel + timelapse_to_track + ".xml"
            seg_image_name = timelapse_to_track + ".tif"
            xml_path = Path(os.path.join(base_dir, xml_name))
            seg_image = imread(os.path.join(config.experiment_data_paths.timelapse_seg_nuclei_vollseg_directory,seg_image_name))
            
            TrackMate(
                    xml_path,
                    seg_image=seg_image,
                    mask=mask_image,
                    enhance_trackmate_xml=enhance_trackmate_xml,
                    master_extra_name = master_extra_name,
                    compute_with_autoencoder=compute_with_autoencoder,
                    variable_t_calibration=variable_t_calibration,
                    oneat_csv_file=oneat_nuclei_vollseg_csv_file,
                    oneat_threshold_cutoff=oneat_threshold_cutoff
                )


    if do_membrane:        
            channel = 'membrane_'
            timelapse_to_track =  config.experiment_data_paths.timelapse_membrane_to_track
            base_dir = config.experiment_data_paths.tracking_directory
            xml_name = channel + timelapse_to_track + ".xml"
            seg_image_name = timelapse_to_track + ".tif"
            tissue_mask_name = timelapse_to_track + ".tif"
            xml_path = Path(os.path.join(base_dir, xml_name))
        
            seg_image = imread(os.path.join(config.experiment_data_paths.timelapse_seg_membrane_directory,seg_image_name))
           
            TrackMate(
                    xml_path,
                    seg_image=seg_image,
                    mask=mask_image,
                    enhance_trackmate_xml=enhance_trackmate_xml,
                    master_extra_name = master_extra_name,
                    compute_with_autoencoder=compute_with_autoencoder,
                    variable_t_calibration=variable_t_calibration,
                    oneat_csv_file=oneat_membrane_csv_file,
                    oneat_threshold_cutoff=oneat_threshold_cutoff
                )

if __name__=='__main__':
        main()
