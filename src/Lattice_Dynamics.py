#%%
%matplotlib qt
import numpy as np
from numpy.linalg import norm
import netCDF4 as nc
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import scipy as sp

#######################   USER INPUT   ######################
Path                 = "./lammps"
Dyn_Mat_Filename     = "Dyn_Mat.gz"
Structure_Filename   = "Structure.nc"
k_vec_filename       = 'k-vectorsGKMG.dat'
N_a                  = 20
ticks                = [0, 6.67, 10, 20]
tick_labels          = [r"$ \Gamma $", r"$K$", r"$M$", r"$ \Gamma $"]
save_eigs            = False
save_figures         = False
#######################   USER INPUT   ######################


#%%
## 1. Load Input Data

# Load k-vector list
k_vec_list=np.loadtxt(k_vec_filename, dtype=np.float64, comments='#')
no_kpoints=k_vec_list.shape[0]

# Load Structure Data
pos_variable_name  = 'coordinates'  
ds=nc.Dataset(Path+"/"+Structure_Filename, mode='a')

types_map=np.ma.getdata(ds['type'][0, :]) 
no_types=np.unique(types_map).shape[0]
cell_dimensions=np.ma.getdata(ds['cell_lengths'][0, :])
cell_origin=np.ma.getdata(ds['cell_origin'][0, :])
Positions=np.ma.getdata(ds[pos_variable_name][0, :, :]) 
no_atoms=len(types_map)

# Load Force Constant Matrix (+ make it complex)
dyn_mat=np.loadtxt(Path+"/"+Dyn_Mat_Filename)
dyn_mat=np.reshape(dyn_mat, (3*no_atoms, 3*no_atoms))
dyn_mat= dyn_mat+1j*np.zeros(dyn_mat.shape)

# Conversion Factor
mole=6.02214076e+23 
eV=1.60217646e-19
gram=1e-3
Angstrom=1e-10
Conversion=eV/(gram/mole*Angstrom**2)                      # [Dyn_Mat] = [energy] / ([mass]*[distance]^2)
#w_Conversion=np.sqrt(Conversion)/(2*np.pi)*1e-12          # In THz
w_Conversion=np.sqrt(Conversion)/(2*np.pi)*1e-12*33.356    # In cm-1

#%%

## 2. Dispersion Curves:

R_l=Positions[(types_map==1)]              # Take the position of atom type #1 to be the position of the corresponidng unit cellDisp_Curves
lattice_constant=cell_dimensions[0]/N_a
print('Lattice Constant: '+"{:.3f}".format(lattice_constant)+' Angstroms')
Disp_Curves  = np.zeros((3*no_types, no_kpoints), dtype=np.float64)
Eigen_Vecs   = np.zeros((3*no_types, 3*no_types, no_kpoints), dtype=np.complex128) # Cartesian for each atom, each mode, each k-point

# Reshape the dynamical matrix into a tensor to take advantage of numpy's tensor product. 
# Order from faster to slower changing is  b > k' > l' > a > k > l
# CAUTION: This requires that LAMMPS outputs the atom types at each unit cell with the same order !! 
# (It is the case for pristine systems !!)

# Spatial Fourier of the Force Constant Matrix
dyn_tensor = dyn_mat.reshape(N_a**2, no_types, 3, N_a**2, no_types, 3)
for k, k_vec in enumerate(k_vec_list):
    exp_ixk    = np.exp(-2*np.pi*1.0j*np.dot(R_l, k_vec)/lattice_constant)              
    exp_tensor = np.tensordot(np.conj(exp_ixk),exp_ixk, axes=0)
    D_k= np.tensordot(dyn_tensor, exp_tensor, axes=([3,0], [1,0])).reshape(3*no_types, 3*no_types)
    
    eigenval,eigenvec =np.linalg.eig(D_k)
    eigenval=abs(eigenval.real)
    w =np.sqrt(eigenval)*w_Conversion/N_a

    # Sort ascending and zero threshold
    arg_sort= np.argsort(w)
    w=np.sort(w)
    eigenvec=eigenvec[:,arg_sort]
    
    # Store results
    Disp_Curves[:, k]=w
    Eigen_Vecs[:,:,k]=eigenvec

#%%
## 3. Visualise the dispersion Curves
############################################
no_branches=2
k_points=np.arange(1, len(k_vec_list)-1)
#############################################

y=Disp_Curves[:no_branches, :]
fig, ax= plt.subplots(figsize=(6,3))
for k in k_points:
    k_= k*np.ones(y.shape[0])
    ax.scatter(k_,y[:,k], s=10, color='black')

# Gamma Point especially
y=Disp_Curves[:6, 0]
k_= 0*np.ones(len(y))
ax.scatter(k_,y, s=10, color='black')
k_= (len(k_vec_list)-1)*np.ones(len(y))
ax.scatter(k_,y, s=10, color='black')

fig.suptitle('Phonon Dispersion Curves', fontweight ="bold", fontsize = 15,
                     transform=ax.transAxes, y=1.15)
ax.set_ylim(0, 550)
#ax.set_title('Lennard-Jones', style='italic', verticalalignment='center', fontsize=13,  pad=12)
ax.set_ylabel(r'Energy ($cm^{-1}$)', fontsize = 12)
ax.set_xticks(ticks, tick_labels, fontsize=12)

## Save the Results
if save_figures is True:
    #np.savetxt(Path+"/LJ_Dispersion_Curves.dat", Disp_Curves, fmt='%.3f', delimiter='\t')
    fig.savefig(Path+"/KC_Dispersion_Curves.svg",bbox_inches='tight',transparent=True)

#%%
# 4. Plot eigenvectors on atoms (Unit Cell)
#####################
k_point_choice=6  # Choice of k-point index to plot
mode_choice=2     # Choice of mode index to plot for this specific k-point
#####################

ex=Eigen_Vecs[0::3,mode_choice-1,k_point_choice-1].real
ey=Eigen_Vecs[1::3,mode_choice-1,k_point_choice-1].real
ez=Eigen_Vecs[2::3,mode_choice-1,k_point_choice-1].real

fig, ax= plt.subplots(figsize=(5,5), subplot_kw={"projection": "3d"})
for i in range(no_types):
    ax.scatter(Positions[i,0],Positions[i,1],Positions[i,2], s=60)

ax.quiver(Positions[:no_types,0],Positions[:no_types,1], Positions[:no_types,2], 
          ex, ey, ez, label=r"$\omega = \ %.1f \ cm^{-1}$"%(w[mode_choice-1]))# 3D

ax.set_xlabel(r'x ($\AA$)'); ax.set_ylabel(r'y ($\AA$)'); ax.set_zlabel(r'z ($\AA$)')
ax.set_aspect('auto', 'box')
ax.legend()

if save_figures:
    fig.savefig(Path+"Eig_%s.svg"%mode_choice,bbox_inches='tight',transparent=True)

if save_eigs:
    np.savez(Path+"/Eigs.npz", w= w, eigenvec=eigenvec)

ds.close()

# %%
