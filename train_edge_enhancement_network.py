from tifffile import imread
from csbdeep.utils import download_and_extract_zip_file, plot_some
from csbdeep.data import RawData, create_patches
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
from csbdeep.utils import axes_dict
import hydra
from scenario_train_vollseg_cellpose_sam import TrainCellPose
from hydra.core.config_store import ConfigStore
import os
configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellPose', node = TrainCellPose)

@hydra.main(config_path = 'conf', config_name = 'scenario_train_vollseg_cellpose_sam')
def main( config : TrainCellPose):
        basepath = config.train_data_paths.base_membrane_dir
        npz_file = os.path.join(basepath, config.train_data_paths.membrane_enhancement_npzfile)
        low = config.train_data_paths.edge_enhancement_low_dir 
        GT = config.train_data_paths.edge_enhancement_gt_dir

        #raw_data = RawData.from_folder (
        #    basepath    = basepath,
        #    source_dirs = [low],
        #    target_dir  = GT,
        #    axes        = 'ZYX',
        #)

        #X, Y, XY_axes = create_patches (
        #    raw_data            = raw_data,
        #    patch_size          = tuple(config.parameters.patch_size),
        #    n_patches_per_image = 20,
        #    save_file           = npz_file,
        #)

        (X,Y), (X_val,Y_val), axes = load_training_data(npz_file, validation_split=0.01, verbose=True)



        configtrain = Config('ZYX', 1, 1,train_batch_size = config.parameters.batch_size, unet_n_depth=config.parameters.depth, unet_kern_size=config.parameters.kern_size, unet_n_first=config.parameters.startfilter,  train_loss='mae', train_epochs=config.parameters.epochs)
        print(configtrain)
        vars(configtrain)
        den_membrane_model_name = config.model_paths.edge_enhancement_model_name
        # the base directory in which our model will live
        den_model_dir = config.model_paths.den_model_dir
        model = CARE(configtrain, den_membrane_model_name, basedir=den_model_dir)
        if os.path.exists(
                os.path.join(
                    den_model_dir,
                    os.path.join(den_membrane_model_name, "weights_now.h5"),
                )
            ):
                print("Loading checkpoint model")
                model.load_weights(
                    os.path.join(
                        den_model_dir,
                        os.path.join(den_membrane_model_name, "weights_now.h5"),
                    )
                )

        if os.path.exists(
            os.path.join(
                den_model_dir,
                os.path.join(den_membrane_model_name, "weights_last.h5"),
            )
        ):
            print("Loading checkpoint model")
            model.load_weights(
                os.path.join(
                    den_model_dir,
                    os.path.join(den_membrane_model_name, "weights_last.h5"),
                )
            )

        if os.path.exists(
            os.path.join(
                den_model_dir,
                os.path.join(den_membrane_model_name, "weights_best.h5"),
            )
        ):
            print("Loading checkpoint model")
            model.load_weights(
                os.path.join(
                    den_model_dir,
                    os.path.join(den_membrane_model_name, "weights_best.h5"),
                )
            )
        history = model.train(X,Y, validation_data=(X_val,Y_val))

if __name__ == "__main__":
    main() 
