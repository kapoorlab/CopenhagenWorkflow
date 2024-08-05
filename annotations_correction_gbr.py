# %%

from pathlib import Path 
import os
import numpy as np
import pandas as pd
import napari
from tifffile import imread 
from qtpy.QtWidgets import QPushButton
from napatrackmater.Trackvector import TrackVector

# %%
dataset_name = 'Fifth'
#/lustre/fsstor/projects/rech/jsy/uzj81mi/
home_folder = '/lustre/fsstor/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
timelapse_image = imread(f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/membrane_timelapses/{timelapse_to_track}.tif', dtype=np.uint8)
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')
master_xml_name = 'master_' + 'marching_cubes_filled_' + channel + timelapse_to_track + ".xml"
xml_path = Path(os.path.join(tracking_directory, master_xml_name))
   
goblet_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted/goblet_cells_{channel}annotations_inception.csv'
basal_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted/basal_cells_{channel}annotations_inception.csv'
radial_cells_file = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_predicted/radially_intercalating_cells_{channel}annotations_inception.csv'


goblet_cells_dataframe = pd.read_csv(goblet_cells_file)
basal_cells_dataframe = pd.read_csv(basal_cells_file)
radial_cells_dataframe = pd.read_csv(radial_cells_file)
normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')



save_dir = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/annotations_corrected/'
Path(save_dir).mkdir(exist_ok=True)


# %%
track_vectors = TrackVector(master_xml_path=xml_path)

track_vectors.t_minus = 0
track_vectors.t_plus = track_vectors.tend
track_vectors.y_start = 0
track_vectors.y_end = track_vectors.ymax
track_vectors.x_start = 0
track_vectors.x_end = track_vectors.xmax

print(f'reading data from {normalized_dataframe}')
correlation_dataframe = pd.read_csv(normalized_dataframe)

# %%

def get_last_time_point_cell_type(track_ids, dataframe):
        results = []
        global_t_min = np.inf
        for track_id in track_ids:
        
            selected_dataframe = dataframe[dataframe['Track ID'] == track_id]
            t_max = selected_dataframe['t'].max()
            if t_max < global_t_min:
                  global_t_min = t_max
            row_selected_dataframe = selected_dataframe[selected_dataframe['t']==t_max]

            z_max = row_selected_dataframe.iloc[0]['z']
            y_max = row_selected_dataframe.iloc[0]['y']
            x_max = row_selected_dataframe.iloc[0]['x']
            results.append((t_max, z_max, y_max, x_max))
            
        return np.asarray(results), global_t_min 





# %%


# %%
goblet_ids = goblet_cells_dataframe['Track ID'].unique()
basal_ids = basal_cells_dataframe['Track ID'].unique()
radial_ids = radial_cells_dataframe['Track ID'].unique()

goblet_locations, goblet_t_min = get_last_time_point_cell_type(goblet_ids, correlation_dataframe)
basal_locations, basal_t_min = get_last_time_point_cell_type(basal_ids, correlation_dataframe)
radial_locations, radial_t_min = get_last_time_point_cell_type(radial_ids, correlation_dataframe)

t_min = min(goblet_t_min, basal_t_min, radial_t_min)


# %%



# %%
last_timepoint_goblet = pd.DataFrame(
                        goblet_locations, index=None, columns=["T", "Z", "Y", "X"]
                    )
last_timepoint_basal = pd.DataFrame(
                        basal_locations, index=None, columns=["T", "Z", "Y", "X"]
                    )
last_timepoint_radial = pd.DataFrame(
                        radial_locations, index=None, columns=["T", "Z", "Y", "X"]
                    )

# %%
viewer = napari.Viewer()
typesavebutton = QPushButton("Save Clicks")

def save_layers():
    for layer in viewer.layers:
        if isinstance(layer, napari.layers.Points):
            data = layer.data
            filename = os.path.join(save_dir, f"{layer.name}.csv")
            df = pd.DataFrame(data, columns=['T', 'Z', 'Y', 'X'])
            df.to_csv(filename, index=False)
            print(f"Saved {layer.name} to {filename}")



            
typesavebutton.clicked.connect(save_layers)
viewer.window.add_dock_widget(
            typesavebutton, name="Save Clicks", area="bottom"
        )
viewer.add_image(timelapse_image, name='Image')
viewer.add_points(
                        data=last_timepoint_goblet,
                        name='Goblet',
                        face_color='Red',
                        ndim = 4,
                        size= 15

                    )
viewer.add_points(
                        data=last_timepoint_basal,
                        name='Basal',
                        face_color='Green',
                        ndim = 4,
                        size= 15
                    )
viewer.add_points(
                        data=last_timepoint_radial,
                        name='Radial',
                        face_color='Blue',
                        ndim = 4, 
                        size= 15
                    )
napari.run()

# %%



