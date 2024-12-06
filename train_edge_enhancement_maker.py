from tifffile import imread, imwrite
import os 
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.segmentation import find_boundaries
from skimage.morphology import binary_opening, binary_closing, disk

def main():
  source_dir = '/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_Second_Dataset_Analysis/split_nuclei_membrane_raw/VollCellPoseSeg/CellPose/'
  destination_dir = '/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_Second_Dataset_Analysis/split_nuclei_membrane_raw/VollCellPoseSeg/CellPoseEdges/'
  Path(destination_dir).mkdir(exist_ok=True)
  acceptable_formats = [".tif"] 
  nthreads = os.cpu_count()
  



  def denoisemaker(path, save_path, dtype, append_name = 'dec_second'):
              
              files = os.listdir(path)
              for fname in tqdm(files):
                if any(fname.endswith(f) for f in acceptable_formats):
                    image = imread(os.path.join(path,fname))
                    
            
                    # Apply simple_dist only to the filtered image
                    if np.max(image) > 0:
                            image = simple_dist(image.astype('uint16'))
                            imwrite(save_path + '/' + os.path.splitext(fname)[0] + append_name + '.tif' , image.astype(dtype))
                    else:
                       print('image is empty: ' + fname)   
  futures = []                   
  with ThreadPoolExecutor(max_workers = nthreads) as executor:
                    futures.append(executor.submit(denoisemaker, source_dir, destination_dir, 'float32')) 

  [r.result() for r in futures]  


def simple_dist(label_image, opening_size=2, closing_size=2):
    """
    Processes 3D labeled image to extract, smooth boundaries and remove small objects.
    
    Parameters:
        label_image (numpy array): 3D labeled image.
        opening_size (int): Size of the structuring element for opening.
        closing_size (int): Size of the structuring element for closing.
        
    Returns:
        numpy array: Processed 3D image with refined boundaries.
    """
    binary_image = np.zeros_like(label_image, dtype=np.float32)
    struct_elem_open = disk(opening_size)
    struct_elem_close = disk(closing_size)

    for i in range(binary_image.shape[0]):
        boundary = find_boundaries(label_image[i], mode="outer") * 255

        smoothed = gaussian_filter(boundary, sigma=1)

        binary = smoothed > 0

        opened = binary_opening(binary, struct_elem_open)

        closed = binary_closing(opened, struct_elem_close)

        binary_image[i] = closed.astype(np.float32)

    max_value = np.max(binary_image)
    if max_value > 0:
        return binary_image / max_value
    return binary_image


if __name__=='__main__':

  main()  
