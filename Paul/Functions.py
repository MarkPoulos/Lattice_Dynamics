#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 15:41:04 2020

@author: pdesmarche
"""

import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
from scipy.sparse import csc_matrix,load_npz,save_npz,eye,rand,isspmatrix_coo,isspmatrix_csc,isspmatrix_csr,isspmatrix_lil
from scipy.sparse.linalg import eigsh
from scipy.fftpack import dst
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.ticker as ticker

import tarfile




def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False
def iter_spmatrix(matrix):
    """ Iterator for iterating the elements in a ``scipy.sparse.*_matrix`` 

    This will always return:
    >>> (row, column, matrix-element)

    Currently this can iterate `coo`, `csc`, `lil` and `csr`, others may easily be added.

    Parameters
    ----------
    matrix : ``scipy.sparse.sp_matrix``
      the sparse matrix to iterate non-zero elements
    """
    if isspmatrix_coo(matrix):
        for r, c, m in zip(matrix.row, matrix.col, matrix.data):
            yield r, c, m

    elif isspmatrix_csc(matrix):
        for c in range(matrix.shape[1]):
            for ind in range(matrix.indptr[c], matrix.indptr[c+1]):
                yield matrix.indices[ind], c, matrix.data[ind]

    elif isspmatrix_csr(matrix):
        for r in range(matrix.shape[0]):
            for ind in range(matrix.indptr[r], matrix.indptr[r+1]):
                yield r, matrix.indices[ind], matrix.data[ind]

    elif isspmatrix_lil(matrix):
        for r in range(matrix.shape[0]):
            for c, d in zip(matrix.rows[r], matrix.data[r]):
                yield r, c, d

    else:
        raise NotImplementedError("The iterator for this sparse matrix has not been implemented")

def ReadNbAt(File, second=False):
    """
    Read nb atom from dump
    """

    file_object= open(File, "r")
    for line in file_object:
          
          if "atoms in group group_1" in line and not second:
              parts=line.split()
              n=int(parts[0])
              break
          if "atoms in group group_2\n" in line and second:
              parts=line.split()
              n=int(parts[0])
              break
    return n
def extractPosDump(File,Natoms): # Extractio
    print("Pos Extract")
    # Pos versus time in 2 dim array
    
   
    PosOr=np.zeros((int(Natoms),2))
    Type=np.zeros((int(Natoms)+1))
    PosBeg=False
    atom=0
    with open(File, 'r') as f:
        for line in f:
          
          if "ITEM: TIMESTEP" in line:
              line=next(f)
              parts=line.split()
              final_ts=parts[0]

    with open(File, 'r') as f:
         for line in f:
             if "ITEM: TIMESTEP" in line:
                 line=next(f)
                 parts=line.split()
                 if parts[0]==final_ts:
                     for i in range(8):# ad hoc line count
                         line=next(f)
                         parts =line.split()
                     while True:
                        for j in range(2,4):
                                    k=j-2
                                    PosOr[atom,k]=float(parts[j]) # Origin configuration
                        Type[atom]=int(parts[1])
                        atom+=1
                        if Natoms<=atom:
                            break
                        line=next(f)
                        parts =line.split()
                     break
        
                     
    return PosOr,Type
def matRead(File,nbat,d):
    # read the matrix from file, the file is as the command dynamical_matrix leaves it
    ###he output for the dynamical matrix is printed three elements at a time.
    ####The three elements are the three elements for a respective i/alpha/j combination.
    ####Each line is printed in order of j increasing first, alpha second, and i last.
    #### reading ok
    if "tar" in File:
         tar= tarfile.open(File, 'r') # open archive
         Lfile=tar.getmembers() # list of file
         file_object=tar.extractfile(Lfile[0])# read the first file   
    else: 
        file_object= open(File, "r")
    lcols=[]
    lrows=[]
    ldat=[]
 
    step=0
    i=0
    if "_num" not in File:
      print("LAMMPS output reading")
      for k,line in enumerate(file_object):
     	  if int(k/nbat)%3!=2:
             parts =line.split()
             #read nbat*3 then switch line

             for l in range(2):
                 i= int(step/(nbat*d))
                 j=step%(nbat*d)
                 if float(parts[l]) !=0:
                     lcols.append(j)
                     lrows.append(i)
                     ldat.append(float(parts[l]))
                 step+=1


    cols=np.array(lcols)#from fortran index to python index
    rows=np.array(lrows)
    data=np.array(ldat)
    file_object.close()
  
    return csc_matrix((data,(rows,cols)),shape=(nbat*d,nbat*d))

       
def plotEigMod2D(Mat,n,Path,PosOr,Load=False,WishedFreq=np.array([0.2,1,13,14.3,17,21,22,23,24])):
    Avo=6.02214076e+23 
    eV=1.60217646e-19
    Conversion=Avo*eV/(1e-23)#10-23 for gram to kg * Angst**2 to m**2
    
    if Load: 
       data=np.load(Path+"Eigs.npz")
       eigenval=data['eigenval']
       eigenvec=data['eigenvec']
       data.close
    else:  
        eigenval, eigenvec =eigsh(Mat, k=3*n-2)
        np.savez(Path+"Eigs.npz", eigenval=eigenval, eigenvec= eigenvec)
    if "InMat" in Path:
        In="In"
    else:
        In=""
    w=np.sqrt(eigenval*Conversion)/(2*np.pi)*1e-12
    indWOrder=np.argsort(w)
    
  
    IndexOfWishedFreq=np.array([np.nanargmax(w>Freq) for Freq in WishedFreq ])# select the last frequency bew the wished frequency
    SelectedFreq=w[IndexOfWishedFreq]
    
    for i in IndexOfWishedFreq: # arrow from the top
        ##################################################
         #Quiver mode representation
        ###################################################"
        # Mapping of atoms on the right position OK
        # Mapping of the modes on the atoms...
        ### Diagonalisation does not mess with vector
        ### Lammps : group map = atom id ?
        ### Does the indexing of atom vector correspond to atom ID
        fig, ax= plt.subplots()
        # atom numbering for debug
        for j in range(len(PosOr[:,1])):
            ax.scatter(PosOr[j,0],PosOr[j,1],PosOr[j,2], marker="$%d$"%j,s=150)
        
            ax.quiver(PosOr[j,0],PosOr[j,1], eigenvec[j,i], eigenvec[j+1,i],width=.001)# 3D
        ax.set_xlabel(r'x ($\AA$)')
        ax.set_ylabel(r'y ($\AA$)')
     #   plt.tight_layout()
        fig.savefig(Path+"%sModes%.2f.png"%(In,w[i]),transparent=True,dpi=300)
        

        

