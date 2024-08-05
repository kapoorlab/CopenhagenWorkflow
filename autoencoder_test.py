import numpy as np
from kapoorlabs_lightning.lightning_trainer import ClusterLightningModel, AutoLightningModel

from pyntcloud import PyntCloud
from cellshape_cloud import CloudAutoEncoder
from napatrackmater import load_json
from kapoorlabs_lightning.pytorch_losses import ChamferLoss

from kapoorlabs_lightning.optimizers import Adam
import torch
import pandas as pd

def main():
	point_cloud = PyntCloud.from_file("/gpfsscratch/rech/jsy/uzj81mi/Mari_Data_Training/xenopus_segmentation_cellshape/cloud_mask_nuclei/point_cloud/Merged-237_Second_Dataset12051555.ply")
	
	autoencoder_model_path = '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/NucleiCloud/xenopus_nuclei_autoencoder.ckpt'
	model_path_json = '/gpfsstore/rech/jsy/uzj81mi/Mari_Models/NucleiCloud/xenopus_nuclei_autoencoder.json'

	print(point_cloud)
	point_cloud.plot(mesh=True)
	loss = ChamferLoss()
	
	optimizer = Adam(lr=0.001)
	modelconfig = load_json(model_path_json)
	cloud_autoencoder = CloudAutoEncoder(
		num_features=modelconfig["num_features"],
		k=modelconfig["k_nearest_neighbours"],
		encoder_type=modelconfig["encoder_type"],
		decoder_type=modelconfig["decoder_type"],
	    )
	autoencoder = AutoLightningModel.load_from_checkpoint(autoencoder_model_path, network = cloud_autoencoder, loss_func = loss, optim_func = optimizer)
	
	point_cloud = torch.tensor(point_cloud.points.values)
	mean = torch.mean(point_cloud, 0)
	scale = torch.tensor([[8, 16, 16]])
	point_cloud = (point_cloud - mean) / scale
	
	outputs, features = autoencoder(point_cloud.unsqueeze(0).to('cuda'))
	print(outputs.shape, scale.shape, mean.shape)
	outputs = outputs.detach().cpu().numpy()
	outputs = outputs * scale.detach().cpu().numpy() + mean.detach().cpu().numpy()
	print(outputs.shape)
	outputs = outputs[0,:]
	
	points = pd.DataFrame(outputs)
	points = pd.DataFrame(points.values, columns=["x", "y", "z"])
	cloud = PyntCloud(points)
	cloud.plot(mesh=True)
	

if __name__=='__main__':

      main()

