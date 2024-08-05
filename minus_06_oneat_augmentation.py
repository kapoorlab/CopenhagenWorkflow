import hydra
from scenario_train_oneat import TrainOneat
from hydra.core.config_store import ConfigStore
import os
from tifffile import imread, imwrite
from pathlib import Path 
from caped_ai_augmentations import AugmentTZYXCsv
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import concurrent
configstore = ConfigStore.instance()
configstore.store(name='TrainCellShape', node=TrainOneat)

def process_augmentation(fname, image_dir, seg_dir, csv_dir, event_type_name,mean, sigma, distribution, aug_image_dir, aug_seg_dir, aug_csv_dir, csv_name_diff):
    name = os.path.basename(os.path.splitext(fname)[0])   
    acceptable_formats = [".tif", ".TIFF", ".TIF", ".png"]
    if any(fname.endswith(f) for f in acceptable_formats):
            image = imread(os.path.join(image_dir,fname))
            segimage = imread(os.path.join(seg_dir,fname)).astype('uint16')
            i = 0
        #for i in range(1, len(event_type_name)):
            event_name = event_type_name[i]
            Csvname = csv_name_diff + event_name + name
            csvfname = os.path.join(csv_dir, Csvname + '.csv')
            if os.path.exists(csvfname):
                flip_pixels = AugmentTZYXCsv(flip=True)
                aug_flip_pixels, aug_flip_pixels_label, aug_flip_pixels_csv = flip_pixels.build(image=np.copy(image), labelimage=segimage, labelcsv=csvfname)

                save_name_raw = str(aug_image_dir) + '/' + 'flip_'  + name + '.tif'
                save_name_seg = str(aug_seg_dir) + '/' + 'flip_'  + name + '.tif'
                save_name_csv = str(aug_csv_dir) + '/' + csv_name_diff + event_name + 'flip_' + name + '.csv'
                if not os.path.exists(save_name_raw):
                        imwrite(save_name_raw, aug_flip_pixels.astype('float32'))
                if not os.path.exists(save_name_seg):
                        imwrite(save_name_seg, aug_flip_pixels_label.astype('uint16'))
                if not os.path.exists(save_name_csv):
                        aug_flip_pixels_csv.to_csv(save_name_csv, index=False, mode='w')
                addnoise_pixels = AugmentTZYXCsv(mean=mean, sigma=sigma, distribution=distribution)
                aug_addnoise_pixels, _, aug_addnoise_pixels_csv = addnoise_pixels.build(image=np.copy(image), labelimage=segimage, labelcsv=csvfname)

                save_name_raw = str(aug_image_dir) + '/' + 'noise_' + str(sigma) + name + '.tif'
                save_name_seg = str(aug_seg_dir) + '/' + 'noise_' + str(sigma) + name + '.tif'
                save_name_csv = str(aug_csv_dir) + '/' + csv_name_diff + event_name + 'noise_' + str(sigma) + name + '.csv'
                if not os.path.exists(save_name_raw):
                    imwrite(save_name_raw, aug_addnoise_pixels.astype('float32'))
                if not os.path.exists(save_name_seg):
                    imwrite(save_name_seg, segimage)
                if not os.path.exists(save_name_csv):
                    aug_addnoise_pixels_csv.to_csv(save_name_csv, index=False, mode='w')        

        

@hydra.main(version_base="1.3", config_path='conf', config_name='scenario_train_oneat')
def main(config: TrainOneat):
    base_dir = config.train_data_paths.base_nuclei_dir
    
    image_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_raw)
    csv_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_csv)
    seg_dir = os.path.join(base_dir, config.train_data_paths.oneat_timelapse_nuclei_seg)

    aug_image_dir = Path(image_dir) / "background"
    aug_csv_dir = Path(csv_dir) / "background"
    aug_seg_dir = Path(seg_dir) / "background"

    Path(aug_image_dir).mkdir(exist_ok = True)
    Path(aug_seg_dir).mkdir(exist_ok = True)
    Path(aug_csv_dir).mkdir(exist_ok = True)

    files_raw = os.listdir(image_dir)
    event_type_name = ["Normal", "Division"]
    csv_name_diff = 'ONEAT'
    mean = 0.0
    sigma = 10.0
    distribution = 'Both'
    future_labels = []
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:  
        
                        future_labels = [executor.submit(
                            process_augmentation, fname, image_dir, seg_dir, csv_dir, event_type_name,mean, sigma, distribution, aug_image_dir, aug_seg_dir, aug_csv_dir, csv_name_diff
                        ) for fname in files_raw]

                        [r.result()
                            for r in concurrent.futures.as_completed(
                                future_labels
                            )] 




if __name__ == '__main__':
    main()


