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
## Visualise the dispersion Curves
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
if save_eigs is True:
    #np.savetxt(Path+"/LJ_Dispersion_Curves.dat", Disp_Curves, fmt='%.3f', delimiter='\t')
    fig.savefig(Path+"/KC_Dispersion_Curves.svg",bbox_inches='tight',transparent=True)

#%%
# Plot eigenvectors on atoms (Unit Cell)
#####################
k_point_choice=1  # Choice of k-point index to plot
mode_choice=5     # Choice of mode index to plot for this specific k-point
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



#%%
## 3. Full Diagonalisation of the Force Constant Matrix

# Diagonalise
eigenval,eigenvec =np.linalg.eig(dyn_mat)  # Eigenvectors output column-wise!

# Check that we don't have negative and/or imaginary eigenvalues
tolerance=5e-8
if not (abs(eigenval.imag)<tolerance).all(): print("Imaginary Eigenvalues Detected")
if not (eigenval.real>=-tolerance).all(): print("Negative Eigenvalues Detected")       
eigenval=abs(eigenval.real)

w=np.sqrt(eigenval)*w_Conversion

# Sort ascending and zero threshold
arg_sort= np.argsort(w)
w=np.sort(w)
eigenvec=eigenvec[:,arg_sort]
eigenvec[eigenvec < 1e-5] = 0

del(eigenval)
del(dyn_mat)

#%%
# Participation Ratio

## UPDATE AND CORRECT FORMULA
ex_2=(eigenvec[0::3,:]*np.conj(eigenvec[0::3,:])).real
ey_2=(eigenvec[1::3,:]*np.conj(eigenvec[1::3,:])).real
ez_2=(eigenvec[2::3,:]*np.conj(eigenvec[2::3,:])).real
enumerator=(norm(eigenvec, axis=0)**4) # Possibly always unity
denominator=np.sum((ex_2+ey_2+ez_2)**2, axis=0)

Part_Ratio=1/no_atoms*enumerator/denominator
Part_Map=np.sum(1/np.sqrt(no_atoms*denominator)*(ex_2+ey_2+ez_2)**2, axis=1)

del(enumerator,denominator,ex_2, ey_2, ez_2)

#%%
# Plot eigenvectors on atoms
mode_choice=5

ex=(eigenvec[0::3,mode_choice-1]).real
ey=(eigenvec[1::3,mode_choice-1]).real

fig, ax= plt.subplots(figsize=(5,5))
for i in range(no_types):
    mask=(types_map==(i+1))
    ax.scatter(Positions[mask,0],Positions[mask,1], s=20)

ax.quiver(Positions[:,0],Positions[:,1], ex, ey,
            units='xy', label=r"$\omega = \ %.1f \ cm^{-1}$"%(w[mode_choice-1]))# 3D

ax.set_xlabel(r'x ($\AA$)')
ax.set_ylabel(r'y ($\AA$)')
ax.set_xlim(cell_origin[0], 1.5*(cell_origin[0]+cell_dimensions[0]))
ax.set_ylim(cell_origin[1], 1.1*(cell_origin[1]+cell_dimensions[1]))
ax.set_aspect('equal', 'box')
ax.legend()

#%%
# Save Eigenvectors in the NetCDF file of the positions
save_eigs=True
if save_eigs:
    ds=nc.Dataset(Path+"/"+Structure_Filename, mode='a')
    # fig.savefig(Path+"Eig_%s.svg"%mode_choice,bbox_inches='tight',transparent=True)
    # np.savez(Path+"/Eigs.npz", w= w, Part_Ratio=Part_Ratio,eigenvec=eigenvec)
    eig=ds.createVariable('Eig_w_%.0f'%w[mode_choice-1], np.float32, ('frame', 'atom', 'spatial'))
    pr=ds.createVariable('PR_w_%.0f'%w[mode_choice-1], np.float32, ('frame', 'atom'))
    # pr=ds['PR_w_%.0f'%w[mode_choice-1]]
    # eig=ds['Eig_w_%.0f'%w[mode_choice-1]]
    
    pr[:,:]=Part_Map
    eig[:,:,:]=eigenvec[:, mode_choice-1].real.reshape(1, no_atoms, 3)

ds.close()

# %%
