import sys 
import os   
import keras
import numpy as np
from pathlib import Path   


def Scaling(en,th): 
	s=10**9/np.cos(th/180*np.pi/2)/en 
	return s * (1.0/137.0)**2 *np.cos(th/180*np.pi/2)**2 /(4*en**2 *np.sin(th/180*np.pi/2)**4) 





def GenerateBootstrap(number_of_versions, ckpt_dir,  energy, theta, energy_transfer, number_of_points):  


	state_list=[]  
	parent_dir =  Path(__file__).parent 
	dir_name="Results_Bootstrap"  
	resulsdir = os.path.join(parent_dir, dir_name)
	if not os.path.exists(resulsdir):
		os.makedirs(resulsdir)	

	mean_array=[]
	std_array=[]  
	minw=energy_transfer[0]
	maxw=energy_transfer[1]  

	for h in range(number_of_versions): 
		loaded_model = keras.saving.load_model(ckpt_dir+"/"+"model_keras_electron_A_12_ver_"+str(h)+".keras")
		state_list.append(loaded_model)

	energy_transfer_array=np.arange(minw,maxw+(maxw-minw)/(number_of_points-1),(maxw-minw)/(number_of_points-1))[0:number_of_points] 
	xarray= np.array([ [energy/20.0, j/20.0, theta/180.0,  np.cos(theta/180.0*np.pi),  2*energy*(energy-j)*(1 - np.cos(theta/180.0*np.pi))/100.0 ] for j in energy_transfer_array])
	scaling_factor=Scaling(energy,theta) 
	predarray=np.array([ state_list[t].predict(xarray) for t in range(number_of_versions)]) 
	predarray=np.multiply(predarray, scaling_factor)

	for s in range(number_of_points):  
		mean= np.mean( np.array([predarray[t][s] for t in range(number_of_versions)])) 
		stddev= np.sqrt(np.var(np.array([predarray[t][s] for t in range(number_of_versions)]))) 
		mean_array.append(mean)
		std_array.append(stddev)
	output_file_name="BootstrapModel_NewFit_energy="+str(energy)+"theta="+str(theta)+"nov"+str(number_of_versions)
	output_file_name= output_file_name.replace(".", "_").lower()
	full_array = np.stack([energy_transfer_array, mean_array, std_array], axis=1)
	print(full_array)
	np.savetxt(os.path.join(resulsdir, output_file_name+".txt"), full_array, delimiter="\t", header="energy transfer\mean prediction\std dev", comments='')

