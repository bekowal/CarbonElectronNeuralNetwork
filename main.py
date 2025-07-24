import sys 
import os   
import jax
import numpy as np
from pathlib import Path   
import NN1
import NN2


if __name__ == '__main__':

	number_of_versions=50
	energy=0.68
	theta=60.0
	energy_transfer_min=0.0
	energy_transfer_max=energy
	number_of_points=100

	for arg in enumerate(sys.argv):		
		if arg[1][:4]=="nov=":
			number_of_versions=max(1,int(arg[1][4:]))	
		if arg[1][:7]=="energy=":
			energy=float(arg[1][7:])	 
		if arg[1][:6]=="theta=":
			theta=float(arg[1][6:])	
		if arg[1][:4]=="min=":
			energy_transfer_min=float(arg[1][4:])	 
		if arg[1][:4]=="max=":
			energy_transfer_max=float(arg[1][4:])   
		if arg[1][:4]=="nop=":
			number_of_points=max(2,int(arg[1][4:]))


	layers_dims=[300,300,300,300,300, 300,300,300,300,300]  
	parent_dir =  Path(__file__).parent 


	print("Electron energy: {} GeV".format(energy))
	print("Theta angle: {} degree".format(theta))
	print("Energy transfer range: [{},{}] GeV".format(energy_transfer_min,energy_transfer_max))
	print("Number of points: {}".format(number_of_points))

	for arg in enumerate(sys.argv):
		if arg[1][:7]=="dropout":
			print("Number of versions: {}".format(number_of_versions))
			rng=jax.random.PRNGKey(0) 
			main_key, *dropout_key_array = jax.random.split(key=rng, num=(number_of_versions+1))  
			dir_name='dropout_model' 
			path = os.path.join(parent_dir, dir_name)
			NN1.GenerateDropout(dim_layers=layers_dims, number_of_versions=number_of_versions, dropout_key_array=dropout_key_array, ckpt_dir=path, energy=energy, theta=theta, energy_transfer=[energy_transfer_min,energy_transfer_max], number_of_points= number_of_points)

		if arg[1][:9]=="bootstrap": 
			dir_name='bootstrap_model' 
			number_of_versions=min(50,number_of_versions)
			print("Number of versions: {}".format(number_of_versions))
			path = os.path.join(parent_dir, dir_name)
			NN1.GenerateBootstrap(dim_layers=layers_dims, number_of_versions=number_of_versions,  ckpt_dir=path, energy=energy, theta=theta, energy_transfer=[energy_transfer_min,energy_transfer_max], number_of_points= number_of_points)

		if arg[1][:6]=="newfit": 
			dir_name='bootstrap_model_newfit' 
			number_of_versions=min(50, number_of_versions)
			print("Number of versions: {}".format(number_of_versions))
			path = os.path.join(parent_dir, dir_name)
			NN2.GenerateBootstrap(number_of_versions=number_of_versions,  ckpt_dir=path, energy=energy, theta=theta, energy_transfer=[energy_transfer_min,energy_transfer_max], number_of_points= number_of_points)
