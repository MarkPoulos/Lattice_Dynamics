#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 16:12:35 2019

@author: pdesmarche
"""


import numpy as np
from Functions import ReadNbAt,matRead,extractPosDump,plotEigExp
from mpl_toolkits.mplot3d import Axes3D  
from scipy.sparse import csc_matrix,load_npz,save_npz,eye,rand,isspmatrix_coo,isspmatrix_csc,isspmatrix_csr,isspmatrix_lil
from scipy.sparse.linalg import eigsh
from scipy.fftpack import dst
from  scipy.signal import decimate

import matplotlib.pyplot as plt
import os
import tarfile
import matplotlib.pylab as pylab
import scipy as sp
import sys
params = {'legend.fontsize': 'x-large',
         'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'x-large',
           'xtick.direction': 'in',
         'ytick.direction': 'in'}

pylab.rcParams.update(params)


n_replica=12
fig, ax = plt.subplots(figsize=(8,5),sharex=True,gridspec_kw={'hspace':0})
origin_step=10054502  #10060386 #10071111#10101838  
#10060386 
mypath="/home/paul/Documents/LYS/AQSLong/Patch58/"
steps=np.loadtxt(mypath+"step.txt")
# mypath="/home/paul/Documents/MERLINBlock/Patch9ART/"#sys.argv[2]
# steps=np.arange(4,59,2)


   
fig1, ax1 = plt.subplots(figsize=(5,5)) #
for j, step in enumerate(steps[:]):
    fold="/NEB%d-%d/"%(step[0],origin_step)
    # fold="/NEB%d/"%(step,)

    Path=mypath+fold
    if os.path.isfile(Path+'sparse-DynMatMin.npz'):
                print("from binary")
                log= [x for x in os.listdir(Path) if "log" in x]
                n= ReadNbAt(Path+"dynMat.log")
                mat_or=load_npz(Path+'sparse-DynMatMin.npz')
    else:
                print("from matFile")
                log= [x for x in os.listdir(Path) if "log" in x]
                n= ReadNbAt(Path+"dynMat.log")
           
                mat_or=matRead(Path+"dynmat-Min.dat",n,2)
                save_npz(Path+'sparse-DynMatMin.npz',  mat_or)
    
    mat_or_full=mat_or.toarray()# stupid but I needed to process the full matrix rather than the sparse
    
  
    PosOr,Type =extractPosDump(Path+"dump.neb.1",n)# get the postition of the atom of which we have the dynamical matrix
    

    evalue_or,evec_or =np.linalg.eig(mat_or_full)
    arg_evalue_or= np.argsort(evalue_or)
    evalue_or_ord=np.sort(evalue_or)
    #%%
    # Plot the different modes
    even= [ x for x in range(len(evec_or[0])) if x % 2 != 0] # to get x component of the eigen vector
    odd= [ x for x in range(len(evec_or[0])) if x % 2 == 0] # to get x component of the eigen vector
 
    
    for i in range(len(arg_evalue_or)):
       fig, ax= plt.subplots(figsize=(5,5))
      
       ax.quiver(PosOr[:,0],PosOr[:,1], evec_or[odd,arg_evalue_or[i]], evec_or[even,arg_evalue_or[i]],
                    width=.01,label="Eig. Vec. %d"%(i+1))# 3D
       ax.set_xlabel(r'x ($\sigma_{LJ}$)')
       ax.set_ylabel('y ($\sigma_{LJ}$)')
       ax.set_aspect('equal', 'box')
       ax.legend()
       # #   #   plt.tight_layout()
       fig.savefig(Path+"Eig%s.svg"%i,bbox_inches='tight',transparent=True)
       plt.close()
    

# %%
