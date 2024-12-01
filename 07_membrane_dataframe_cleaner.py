
import os
import pandas as pd


dataset_name = 'Sixth'
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
timelapse_to_track = f'timelapse_{dataset_name.lower()}_dataset'
tracking_directory = f'{home_folder}Mari_Data_Oneat/Mari_{dataset_name}_Dataset_Analysis/nuclei_membrane_tracking/'
channel = 'nuclei_'
second_channel = 'membrane_'
data_frames_dir = os.path.join(tracking_directory, f'dataframes/')

normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{channel}.csv')
second_normalized_dataframe = os.path.join(data_frames_dir , f'results_dataframe_normalized_{second_channel}.csv')


tracks_dataframe = pd.read_csv(normalized_dataframe)
second_tracks_dataframe = pd.read_csv(second_normalized_dataframe)


trackmate_ids = tracks_dataframe["TrackMate Track ID"].unique()
track_ids_to_remove = []
for tracklet_id in tracks_dataframe["Track ID"].unique():
        tracklet_sub_dataframe = tracks_dataframe[
            tracks_dataframe["Track ID"] == tracklet_id
        ]
        second_tracklet_sub_dataframe = second_tracks_dataframe[
            second_tracks_dataframe["Track ID"] == tracklet_id
        ]

        if len(tracklet_sub_dataframe) != len(second_tracklet_sub_dataframe):
       
           track_ids_to_remove.append(tracklet_id)


second_tracks_dataframe = second_tracks_dataframe[~second_tracks_dataframe["Track ID"].isin(track_ids_to_remove)]

second_tracks_dataframe.to_csv(second_normalized_dataframe, index=False)
