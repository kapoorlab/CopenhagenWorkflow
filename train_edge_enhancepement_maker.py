from tifffile import imread, imwrite
import os 
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tqdm import tqdm
import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt, binary_dilation
from skimage.segmentation import find_boundaries
from skimage.measure import regionprops

def main():
  source_dir = '/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_Second_Dataset_Analysis/split_nuclei_membrane_raw/MembraneSeg/CellPose/'
  destination_dir = '/lustre/fsn1/projects/rech/jsy/uzj81mi/Mari_Data_Oneat/Mari_Second_Dataset_Analysis/split_nuclei_membrane_raw/MembraneSeg/CellPoseEdges/'
  Path(destination_dir).mkdir(exist_ok=True)
  acceptable_formats = [".tif"] 
  nthreads = os.cpu_count()
  
  def filter_labels_by_size(image, min_label_size):
        """Filter out small labels based on their size."""
        membrane_prop = regionprops(image.astype(np.uint16))
        filtered_image = np.zeros_like(image, dtype=np.uint16)

        def process_region(region):
            if region.area >= min_label_size:
                filtered_image[image == region.label] = region.label

        with ThreadPoolExecutor(max_workers=nthreads) as executor:
            list(tqdm(executor.map(process_region, membrane_prop), total=len(membrane_prop), desc='Taking only Clean Labels'))

        return filtered_image


  def denoisemaker(path, save_path, dtype, min_label_size = 100):
              
              files = os.listdir(path)
              for fname in tqdm(files):
                if any(fname.endswith(f) for f in acceptable_formats):
                    image = imread(os.path.join(path,fname))
                    filtered_image = filter_labels_by_size(image, min_label_size)
            
                    # Apply simple_dist only to the filtered image
                    if np.max(filtered_image) > 0:
                            image = simple_dist(image.astype('uint16'))
                            imwrite(save_path + '/' + os.path.splitext(fname)[0]  + '.tif' , image.astype(dtype))
                    else:
                       print('image is empty: ' + fname)   
  futures = []                   
  with ThreadPoolExecutor(max_workers = nthreads) as executor:
                    futures.append(executor.submit(denoisemaker, source_dir, destination_dir, 'float32')) 

  [r.result() for r in futures]  

def simple_dist(label_image):


    

    # Create an empty output image
    binary_image = np.zeros_like(label_image, dtype=np.float32)
    for i in range(output_image.shape[0]):
       binary_image[i] = find_boundaries(label_image[i], mode="outer") * 255
       output_image[i] = gaussian_filter(binary_image[i], sigma = 1)
    output_image = output_image / np.max(output_image)
    return output_image    




if __name__=='__main__':

  main()  
