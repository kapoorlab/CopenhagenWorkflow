import os
from pathlib import Path
from tifffile import imread, imwrite
from tqdm import tqdm

def split_channels(source_dir, membrane_dir, nuclei_dir):
    # Create the output directories if they don't exist
    Path(membrane_dir).mkdir(parents=True, exist_ok=True)
    Path(nuclei_dir).mkdir(parents=True, exist_ok=True)

    # List all files in the source directory
    files = [f for f in os.listdir(source_dir) if f.endswith('.tif')]

    for fname in tqdm(files, desc="Processing images"):
        # Read the image
        image = imread(os.path.join(source_dir, fname))

        membrane_image = image[:,0,:,:]  # Channel 0
        nuclei_image = image[:,1,:,:]    # Channel 1

        membrane_save_path = os.path.join(membrane_dir, fname)
        nuclei_save_path = os.path.join(nuclei_dir, fname)

        imwrite(membrane_save_path, membrane_image)
        imwrite(nuclei_save_path, nuclei_image)

        print(f"Saved membrane to {membrane_save_path} and nuclei to {nuclei_save_path}")

if __name__ == "__main__":
    source_dir = "/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_Second_Dataset_Analysis/split_nuclei_membrane_raw/"
    membrane_dir = "/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_Second_Dataset_Analysis/split_nuclei_membrane_raw/membrane_split/"  
    nuclei_dir = "/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_Second_Dataset_Analysis/split_nuclei_membrane_raw/nuclei_split/"  

    split_channels(source_dir, membrane_dir, nuclei_dir)
