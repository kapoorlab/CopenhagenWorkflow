import hydra
from scenario_train_oneat import TrainOneat
from hydra.core.config_store import ConfigStore
import os
from oneat.NEATModels import NEATDenseVollNet
from oneat.NEATModels.config import volume_config
from oneat.NEATUtils.utils import save_json, load_json
configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellShape', node = TrainOneat)

@hydra.main(version_base="1.3",config_path = 'conf', config_name = 'scenario_train_oneat')
def main( config : TrainOneat):

    npz_directory = config.train_data_paths.base_nuclei_dir
    model_dir = config.model_paths.oneat_nuclei_model_dir
    npz_name = config.train_data_paths.oneat_npzfile
    npz_val_name = config.train_data_paths.oneat_npzvalfile
    #Neural network parameters
    division_categories_json = os.path.join(model_dir, config.parameters.categories_json)
    key_categories = load_json(division_categories_json)
    
    division_cord_json = os.path.join(model_dir, config.parameters.cord_json)
    key_cord = load_json(division_cord_json)

    #Number of starting convolutional filters, is doubled down with increasing depth
    startfilter = config.parameters.startfilter
    #CNN network start layer, mid layers and lstm layer kernel size
    start_kernel = config.parameters.start_kernel
    mid_kernel = config.parameters.mid_kernel
    #Size of the gradient descent length vector, start small and use callbacks to get smaller when reaching the minima
    learning_rate = config.parameters.learning_rate
    #For stochastic gradient decent, the batch size used for computing the gradients
    batch_size = config.parameters.batch_size
    #Training epochs, longer the better with proper chosen learning rate
    epochs = config.parameters.epochs
    
    #The inbuilt model stride which is equal to the nulber of times image was downsampled by the network
    show = config.parameters.show
    stage_number = config.parameters.stage_number
    size_tminus = config.parameters.size_tminus
    size_tplus = config.parameters.size_tplus
    imagex = config.parameters.imagex
    imagey = config.parameters.imagey
    imagez = config.parameters.imagez
    trainclass = eval(config.parameters.training_class)
    trainconfig = eval(config.parameters.training_config)
    depth = dict(config.parameters.depth)
    reduction = config.parameters.reduction
    config= trainconfig(npz_directory = npz_directory, npz_name = npz_name, npz_val_name = npz_val_name,  
                            key_categories = key_categories, key_cord = key_cord, imagex = imagex,
                            reduction = reduction,
                            imagey = imagey, imagez = imagez, size_tminus = size_tminus, size_tplus = size_tplus, epochs = epochs,learning_rate = learning_rate,
                            depth = depth, start_kernel = start_kernel, mid_kernel = mid_kernel, stage_number = stage_number,
                            show = show,startfilter = startfilter, batch_size = batch_size)

    config_json = config.to_json()
    print(config)
    save_json(config_json, model_dir + '/' + 'parameters.json')
    Train: NEATDenseVollNet = trainclass(config, model_dir, key_categories, key_cord)
    Train.loadData()
    Train.TrainModel()

if __name__ == '__main__':
    main()  
