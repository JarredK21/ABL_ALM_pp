import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy.stats import pearsonr
import glob 
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
from multiprocessing import Pool
import time


def offset_data(p_rotor,no_cells_offset,it,i,velocity_comp):

    if velocity_comp == "coordinates":
        u = np.array(p_rotor.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_rotor.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice


def Ux_it_offset(it):

    velocityx = offset_data(p_rotor, no_cells_offset,it,i=2,velocity_comp="velocityx")
    velocityy = offset_data(p_rotor, no_cells_offset,it,i=2,velocity_comp="velocityy")
    hvelmag = np.add( np.multiply(velocityx,np.cos(np.radians(29))) , np.multiply( velocityy,np.sin(np.radians(29))) )


    hvelmag = hvelmag.reshape((y,z))

    Ux_rotor = []
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)):
            r = np.sqrt(ys[j]**2 + zs[k]**2)
            if r <= 63 and r >= 1.5:
                Ux_rotor.append(hvelmag[j,k])

    return np.average(Ux_rotor)


#sampling data
sampling = glob.glob("../../../jarred/ALM_sensitivity_analysis/test10/post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]
offsets = p_rotor.offsets

print(p_rotor)

dq = dict()

time_sample = np.array(a.variables["time"])
time_sample = time_sample - time_sample[0]

tstart = 50
tend = 150

tstart_sample_idx = np.searchsorted(time_sample,tstart)
tend_sample_idx = np.searchsorted(time_sample,tend)

no_cells = len(p_rotor.variables["coordinates"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

y = p_rotor.ijk_dims[0] #no. data points
z = p_rotor.ijk_dims[1] #no. data points

coordinates = offset_data(p_rotor,no_cells_offset,it=0,i=2,velocity_comp="coordinates")

xo = coordinates[0:y,0]
yo = coordinates[0:y,1]
zo = np.linspace(p_rotor.origin[2],p_rotor.axis2[2],z)

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-29)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
zs = zo - rotor_coordiates[2]

print(len(ys),len(zs))
print(zs)


dy = ys[1]-ys[0]
dz = zs[1] - zs[0]
dA = dy * dz


i = 2
Ux_it = []
with Pool() as pool:
    for Ux_i in pool.imap(Ux_it_offset, np.arange(0,1)):
        print(Ux_i)
        Ux_it.append(Ux_i)
#Ux_it = df["RtVAvgxh_[m/s]"][tstart_OF_idx:tend_OF_idx]
dq["Ux_{}".format(offsets[i])] = Ux_it