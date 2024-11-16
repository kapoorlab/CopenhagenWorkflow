import os
import numpy as np
import pandas as pd
from napatrackmater import create_analysis_tracklets
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm 
def process_datasets(home_folder, dataset_names, channel='nuclei_', tracking_directory_name='nuclei_membrane_tracking/', tracklet_length=25, stride=4):
    dividing_arrays, non_dividing_arrays = [], []  # Combined morphodynamic arrays

    def create_training_arrays(array, tracklet_length, stride):
        processed_arrays = []
        N, F = array.shape
        if N > tracklet_length:
            num_subarrays = (N - tracklet_length) // stride + 1
            for i in range(num_subarrays):
                start_index = i * stride
                end_index = start_index + tracklet_length
                sub_array = array[start_index:end_index, :]
                processed_arrays.append(sub_array)
        return np.asarray(processed_arrays)

    for dataset_name in tqdm(dataset_names):
        tracking_directory = os.path.join(home_folder, f'Mari_Data_Oneat/Mari_{dataset_name}/{tracking_directory_name}')
        data_frames_dir = os.path.join(tracking_directory, 'dataframes/')
        normalized_dataframe_file = os.path.join(data_frames_dir, f'results_dataframe_normalized_{channel}.csv')
        dataset_dataframe = pd.read_csv(normalized_dataframe_file)
        analysis_vectors, _ = create_analysis_tracklets(dataset_dataframe)

        # Process dividing and non-dividing tracklets
        for trackmate_track_id, group_df in tqdm(dataset_dataframe.groupby('TrackMate Track ID')):
            dividing_tracklet_ids = group_df[group_df['Dividing'] == 1]['Track ID']
            non_dividing_tracklet_ids = group_df[group_df['Dividing'] == 0]['Track ID']

            for tracklet_ids, target_array in [
                (dividing_tracklet_ids, dividing_arrays),
                (non_dividing_tracklet_ids, non_dividing_arrays)
            ]:
                for track_id in tracklet_ids:
                    try:
                        shape_dynamic_df, shape_df, dynamic_df, _ = analysis_vectors[track_id]
                        shape_track_array = np.array([[item for item in record.values()] for record in shape_df])
                        dynamic_track_array = np.array([[item for item in record.values()] for record in dynamic_df])

                        # Combine shape and dynamic features into morphodynamic array
                        combined_track_array = np.concatenate((shape_track_array, dynamic_track_array), axis=-1)
                        morphodynamic_subarrays = create_training_arrays(combined_track_array, tracklet_length, stride)
                        target_array.extend(morphodynamic_subarrays)
                    except KeyError:
                        print(f'Key {track_id} not found, skipping')

    # Convert lists to numpy arrays
    dividing_arrays = np.asarray(dividing_arrays)
    non_dividing_arrays = np.asarray(non_dividing_arrays)

    # Create labels for dividing and non-dividing tracklets
    dividing_labels = np.ones(len(dividing_arrays), dtype=int)
    non_dividing_labels = np.zeros(len(non_dividing_arrays), dtype=int)

    # Split data into training and validation sets
    train_arrays, val_arrays, train_labels, val_labels = train_test_split(
        np.concatenate((dividing_arrays, non_dividing_arrays)),
        np.concatenate((dividing_labels, non_dividing_labels)),
        test_size=0.2, random_state=42
    )

    # Save combined morphodynamic features to a single H5 file
    train_save_dir = f'{home_folder}Mari_Data_Training/track_training_data/'
    morphodynamic_h5_data = {
        'train_arrays': train_arrays,
        'train_labels': train_labels,
        'val_arrays': val_arrays,
        'val_labels': val_labels
    }

    with h5py.File(os.path.join(train_save_dir, f'morphodynamic_training_data_mitosis_{channel}{tracklet_length}.h5'), 'w') as hf:
        for key, value in morphodynamic_h5_data.items():
            hf.create_dataset(key, data=value)

# Parameters
home_folder = '/lustre/fsn1/projects/rech/jsy/uzj81mi/'
dataset_names = ['Third_Dataset_Analysis', 'Second_Dataset_Analysis', 'Fifth_Dataset_Analysis', 'Sixth_Dataset_Analysis']
tracklet_lengths = [25]
strides = [4]

for index, tracklet_length in enumerate(tracklet_lengths):
    stride = strides[index]
    process_datasets(home_folder, dataset_names, channel='nuclei_', tracklet_length=tracklet_length, stride=stride)
