import hydra
from scenario_train_oneat import TrainOneat
from hydra.core.config_store import ConfigStore
import os
from oneat.NEATUtils import TrainDataMaker
configstore = ConfigStore.instance()
configstore.store(name = 'TrainCellShape', node = TrainOneat)

@hydra.main(version_base="1.3",config_path = 'conf', config_name = 'scenario_train_oneat')
def main( config : TrainOneat):
    base_dir = config.train_data_paths.base_nuclei_dir
    oneat_nuclei_train_data = os.path.join(base_dir,config.train_data_paths.oneat_timelapse_nuclei_raw)
    TrainDataMaker(oneat_nuclei_train_data)

if __name__=='__main__':
    main()  
