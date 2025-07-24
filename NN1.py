import sys 
import os   
import jax
import numpy as np
import flax.linen as nn   
from flax.training import train_state, checkpoints 
from pathlib import Path   


def Scaling(en,th): 
	s=10**9/np.cos(th/180*np.pi/2)/en 
	return s * (1.0/137.0)**2 *np.cos(th/180*np.pi/2)**2 /(4*en**2 *np.sin(th/180*np.pi/2)**4) 


class MyNeuralNetwork(nn.Module): 
	dim_hidden: list
	act_hidden: list 
	dim_output: int 
	dropout_rate: float

	def setup(self):  
		self.dense_hidden = [nn.Dense(features=k) for k in self.dim_hidden] 
		self.dense_output = nn.Dense(features=self.dim_output)
  
	@nn.compact
	def __call__(self, x, training: bool, isdropout: bool):
		for k in range(len(self.dim_hidden)):
			x = self.dense_hidden[k](x)
			x = nn.BatchNorm(use_running_average=not training)(x)
			x = self.act_hidden[k](x)
			x = nn.Dropout(rate=self.dropout_rate, deterministic=not isdropout)(x)
		x = self.dense_output(x)
		x = self.act_hidden[-1](x)
		return x
	
class TrainState(train_state.TrainState):
	key: any
	batch_stats: any






def GenerateDropout( dim_layers, number_of_versions, dropout_key_array, ckpt_dir, energy, theta, energy_transfer, number_of_points): 

	files=os.listdir(ckpt_dir)
	restored_state_dict=checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir+"/"+files[-1], target=None) 
	restored_state=restored_state_dict['state']
	dim=restored_state_dict['config']['dimensions']
	OutputSize= dim[1]
	drop=restored_state_dict['dropout']  
	parent_dir =  Path(__file__).parent 
	
	model = MyNeuralNetwork(
		dim_hidden=dim_layers,  
		act_hidden=[nn.relu,nn.relu,nn.relu,nn.relu,nn.relu,  
			nn.relu,nn.relu,nn.relu,nn.relu,nn.relu,
			nn.sigmoid], 
		dim_output=OutputSize, 
		dropout_rate=drop)  
	
	dir_name="Results_Dropout"
	resulsdir = os.path.join(parent_dir, dir_name)
	if not os.path.exists(resulsdir):
		os.makedirs(resulsdir)	 

	mean_array=[]
	std_array=[]  
	minw=energy_transfer[0]
	maxw=energy_transfer[1] 
 

	energy_transfer_array=np.arange(minw,maxw+(maxw-minw)/(number_of_points-1),(maxw-minw)/(number_of_points-1))[0:number_of_points] 
	scaling_factor=Scaling(energy,theta) 
	xarray= np.array([ [energy, j, theta,  np.cos(theta/180.0*np.pi),  2*energy*(energy-j)*(1 - np.cos(theta/180.0*np.pi)) ] for j in energy_transfer_array])
	predarray=[[  model.apply(
		{'params': restored_state['params'], 'batch_stats': restored_state['batch_stats']}, 
		xarray[s], 
		training=False, 
		isdropout=True, 
		rngs={'dropout': dropout_key_array[t]}) for t in range(number_of_versions)] for s in range(number_of_points)] 
	predarray=np.multiply(predarray, scaling_factor)

	for s in range(number_of_points):  
		mean= np.mean( np.array([predarray[s][t][0] for t in range(number_of_versions)])) 
		stddev= np.sqrt(np.var(np.array([predarray[s][t][0] for t in range(number_of_versions)]))) 
		mean_array.append(mean)
		std_array.append(stddev)

	output_file_name="DropoutModel_energy="+str(energy)+"theta="+str(theta)+"dropout="+str(drop)+"nov"+str(number_of_versions)
	output_file_name= output_file_name.replace(".", "_").lower() 
	full_array = np.stack([energy_transfer_array, mean_array, std_array], axis=1)
	np.savetxt(os.path.join(resulsdir,output_file_name+".txt"), full_array, delimiter="\t", header="energy transfer\mean prediction\std dev", comments='')
 




def GenerateBootstrap( dim_layers, number_of_versions, ckpt_dir,  energy, theta, energy_transfer, number_of_points):  

	files=os.listdir(ckpt_dir+"/"+"clones_jax_train_drop=0.0_ver_0")
	restored_state_dict=checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir+"/"+"clones_jax_train_drop=0.0_ver_0/"+files[-1], target=None)  
	dim=restored_state_dict['config']['dimensions']
	OutputSize= dim[1]

	model = MyNeuralNetwork(
		dim_hidden=dim_layers,  
		act_hidden=[nn.relu,nn.relu,nn.relu,nn.relu,nn.relu,  
			nn.relu,nn.relu,nn.relu,nn.relu,nn.relu,
			nn.sigmoid], 
		dim_output=OutputSize, 
		dropout_rate=0.0)  
	
	dir_name="Results_Bootstrap"  
	resulsdir = os.path.join(parent_dir, dir_name)
	if not os.path.exists(resulsdir):
		os.makedirs(resulsdir)	

	mean_array=[]
	std_array=[]  
	minw=energy_transfer[0]
	maxw=energy_transfer[1]  
	
	restored_state_array=[]
	for h in range(number_of_versions):
		files=os.listdir(ckpt_dir+"/"+"clones_jax_train_drop=0.0_ver_"+str(h))
		cp=checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir+"/"+"clones_jax_train_drop=0.0_ver_"+str(h)+"/"+files[-1], target=None) 
		restored_state_array.append(cp['state'])

	energy_transfer_array=np.arange(minw,maxw+(maxw-minw)/(number_of_points-1),(maxw-minw)/(number_of_points-1))[0:number_of_points] 
	xarray= np.array([ [energy, j, theta,  np.cos(theta/180.0*np.pi),  2*energy*(energy-j)*(1 - np.cos(theta/180.0*np.pi)) ] for j in energy_transfer_array])
	scaling_factor=Scaling(energy,theta)	 

	predarray=[[  model.apply(
		{'params': restored_state_array[t]['params'], 'batch_stats': restored_state_array[t]['batch_stats']}, 
		xarray[s], 
		training=False, 
		isdropout=False) for t in range(number_of_versions)] for s in range(number_of_points)] 
	predarray=np.multiply(predarray, scaling_factor)

	for s in range(number_of_points):  
		mean= np.mean( np.array([predarray[s][t][0] for t in range(number_of_versions)])) 
		stddev= np.sqrt(np.var(np.array([predarray[s][t][0] for t in range(number_of_versions)]))) 
		mean_array.append(mean)
		std_array.append(stddev)

	output_file_name="BootstrapModel_energy="+str(energy)+"theta="+str(theta)+"nov"+str(number_of_versions)
	output_file_name= output_file_name.replace(".", "_").lower()
	full_array = np.stack([energy_transfer_array, mean_array, std_array], axis=1)
	np.savetxt(os.path.join(resulsdir,output_file_name+".txt"), full_array, delimiter="\t", header="energy transfer\mean prediction\std dev", comments='')
