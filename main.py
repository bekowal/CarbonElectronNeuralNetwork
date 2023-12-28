import sys 
import os   
import jax
import numpy as np
import flax.linen as nn   
from flax.training import train_state, checkpoints 
import matplotlib.pyplot as plt 
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
	key: jax.random.KeyArray
	batch_stats: any



def GenerateDropout( dim_layers, number_of_versions, dropout_key_array, ckpt_dir, energy, theta, energy_transfer, number_of_points): 

    files=os.listdir(ckpt_dir)
    restored_state_dict=checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir+"/"+files[-1], target=None) 
    restored_state=restored_state_dict['state']
    dim=restored_state_dict['config']['dimensions']
    OutputSize= dim[1]
    drop=restored_state_dict['dropout']  
    sizew=8
    sizeh=6

    model = MyNeuralNetwork(
        dim_hidden=dim_layers,  
        act_hidden=[nn.relu,nn.relu,nn.relu,nn.relu,nn.relu,  
            nn.relu,nn.relu,nn.relu,nn.relu,nn.relu,
            nn.sigmoid], 
        dim_output=OutputSize, 
        dropout_rate=drop)  
    
    dir_name="Results_Dropout_"+str(drop)  
    resulsdir = os.path.join(parent_dir, dir_name)
    if not os.path.exists(resulsdir):
        os.makedirs(resulsdir)     

    mean_array=[]
    std_array=[]  
    minw=energy_transfer[0]
    maxw=energy_transfer[1] 

    f, ax = plt.subplots(figsize=(sizew, sizeh)) 

    energy_transfer_array=np.arange(minw,maxw+(maxw-minw)/(number_of_points-1),(maxw-minw)/(number_of_points-1))[0:number_of_points] 
    scaling_factor=Scaling(energy,theta)
    print("Prediction starts")
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
    mean_array_top=np.add(np.array(mean_array),np.array(std_array))
    mean_array_bottom=np.add(np.array(mean_array),-np.array(std_array))

    ax.plot(energy_transfer_array, mean_array_top.flatten(), color='wheat',  linewidth=0.0)
    ax.plot(energy_transfer_array, mean_array_bottom.flatten(), color='wheat', linewidth=0.0) 
    ax.fill_between(energy_transfer_array, mean_array_bottom.flatten(),  mean_array_top.flatten(), color='wheat')
    ax.plot(energy_transfer_array,np.array(mean_array).flatten(), color='blue', linewidth=0.5)


    ax.set_title(str(energy)+' GeV, '+str(theta)+r'$^{\circ}$', fontsize=6) 

    ax.yaxis.get_offset_text().set_fontsize(6) 
    ax.ticklabel_format(axis='y', style='sci',scilimits=(-2,2)) 
    ax.tick_params(axis='both', labelsize=6)  
    ax.set_xlim([minw, maxw]) 
    ax.set_ylabel(r'$d^2\sigma/(d\omega d\Omega) \quad [nb/GeV/sr]$', fontsize=6)
    ax.set_xlabel(r'$\omega [GeV]$', fontsize=6)
    ax.set_ylim(bottom=0)    
    f.tight_layout()
    output_file_name="DropoutModel_energy="+str(energy)+"theta="+str(theta)+"dropout="+str(drop)
    output_file_name= output_file_name.replace(".", "_").lower() 
    plt.savefig(os.path.join(resulsdir,output_file_name+".eps"), format='eps')
    full_array = np.stack([energy_transfer_array, mean_array, std_array], axis=1)
    np.savetxt(os.path.join(resulsdir,output_file_name+".txt"), full_array, delimiter="\t", header="energy transfer\mean preduction\std dev", comments='')
 

def GenerateClones( dim_layers, number_of_versions, ckpt_dir,  energy, theta, energy_transfer, number_of_points):  

    files=os.listdir(ckpt_dir+"/"+"clones_jax_train_drop=0.0_ver_0")
    restored_state_dict=checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir+"/"+"clones_jax_train_drop=0.0_ver_0/"+files[-1], target=None)  
    dim=restored_state_dict['config']['dimensions']
    OutputSize= dim[1]
  
    sizew=8
    sizeh=6

    model = MyNeuralNetwork(
        dim_hidden=dim_layers,  
        act_hidden=[nn.relu,nn.relu,nn.relu,nn.relu,nn.relu,  
            nn.relu,nn.relu,nn.relu,nn.relu,nn.relu,
            nn.sigmoid], 
        dim_output=OutputSize, 
        dropout_rate=0.0)  
    
    dir_name="Results_Clones"  
    resulsdir = os.path.join(parent_dir, dir_name)
    if not os.path.exists(resulsdir):
        os.makedirs(resulsdir)    

    mean_array=[]
    std_array=[]  
    minw=energy_transfer[0]
    maxw=energy_transfer[1] 

    f, ax = plt.subplots(figsize=(sizew, sizeh)) 

    restored_state_array=[]
    for h in range(number_of_versions):
        files=os.listdir(ckpt_dir+"/"+"clones_jax_train_drop=0.0_ver_"+str(h))
        cp=checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir+"/"+"clones_jax_train_drop=0.0_ver_"+str(h)+"/"+files[-1], target=None) 
        restored_state_array.append(cp['state'])

    energy_transfer_array=np.arange(minw,maxw+(maxw-minw)/(number_of_points-1),(maxw-minw)/(number_of_points-1))[0:number_of_points] 
    xarray= np.array([ [energy, j, theta,  np.cos(theta/180.0*np.pi),  2*energy*(energy-j)*(1 - np.cos(theta/180.0*np.pi)) ] for j in energy_transfer_array])
    scaling_factor=Scaling(energy,theta)    
    print("Prediction starts")

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
    mean_array_top=np.add(np.array(mean_array),np.array(std_array))
    mean_array_bottom=np.add(np.array(mean_array),-np.array(std_array))

    ax.plot(energy_transfer_array, mean_array_top.flatten(), color='lightgreen',  linewidth=0.0)
    ax.plot(energy_transfer_array, mean_array_bottom.flatten(), color='lightgreen', linewidth=0.0) 
    ax.fill_between(energy_transfer_array, mean_array_bottom.flatten(),  mean_array_top.flatten(), color='lightgreen')
    ax.plot(energy_transfer_array,np.array(mean_array).flatten(), color='blue', linewidth=0.5)


    ax.set_title(str(energy)+' GeV, '+str(theta)+r'$^{\circ}$', fontsize=6)

    ax.yaxis.get_offset_text().set_fontsize(6) 
    ax.ticklabel_format(axis='y', style='sci',scilimits=(-2,2)) 
    ax.tick_params(axis='both', labelsize=6)  
    ax.set_xlim([minw, maxw]) 

    ax.set_ylabel(r'$d^2\sigma/(d\omega d\Omega) \quad [nb/GeV/sr]$', fontsize=6)
    ax.set_xlabel(r'$\omega [GeV]$', fontsize=6)
    ax.set_ylim(bottom=0) 
    f.tight_layout()
    output_file_name="ClonesModel_energy="+str(energy)+"theta="+str(theta)+"dropout="+str(drop)
    output_file_name= output_file_name.replace(".", "_").lower()
    plt.savefig(os.path.join(resulsdir,output_file_name+".eps"), format='eps')
    full_array = np.stack([energy_transfer_array, mean_array, std_array], axis=1)
    np.savetxt(os.path.join(resulsdir,output_file_name+".txt"), full_array, delimiter="\t", header="energy transfer\mean preduction\std dev", comments='')

if __name__ == '__main__':

    number_of_versions=50
    energy=0.68
    theta=60.0
    energy_transfer_min=0.0
    energy_transfer_max=energy
    number_of_points=100

    for arg in enumerate(sys.argv):        
        if arg[1][:4]=="nov=":
            number_of_versions=int(arg[1][4:])        
        if arg[1][:7]=="energy=":
            energy=float(arg[1][7:])     
        if arg[1][:6]=="theta=":
            theta=float(arg[1][6:])    
        if arg[1][:4]=="min=":
            energy_transfer_min=float(arg[1][4:])     
        if arg[1][:4]=="max=":
            energy_transfer_max=float(arg[1][4:])   
        if arg[1][:4]=="nop=":
            number_of_points=float(arg[1][4:])   
 


    parent_dir = str(Path(__file__).parent)+"/"  
    layers_dims=[300,300,300,300,300, 300,300,300,300,300]  
    parent_dir =  Path(__file__).parent 

    for arg in enumerate(sys.argv):
        if arg[1][:7]=="dropout":
            rng=jax.random.PRNGKey(0) 
            main_key, *dropout_key_array = jax.random.split(key=rng, num=(number_of_versions+1))  
            dir_name='dropout_model' 
            path = os.path.join(parent_dir, dir_name)
            GenerateDropout(dim_layers=layers_dims, number_of_versions=number_of_versions, dropout_key_array=dropout_key_array, ckpt_dir=path, energy=energy, theta=theta, energy_transfer=[energy_transfer_min,energy_transfer_max], number_of_points= number_of_points)

        if arg[1][:6]=="clones": 
            dir_name='clones_model' 
            path = os.path.join(parent_dir, dir_name)
            GenerateClones(dim_layers=layers_dims, number_of_versions=number_of_versions,  ckpt_dir=path, energy=energy, theta=theta, energy_transfer=[energy_transfer_min,energy_transfer_max], number_of_points= number_of_points)


