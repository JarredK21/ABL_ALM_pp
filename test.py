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

start_time = time.time()

#openfast data
df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

#sampling data
sampling = glob.glob("../post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]

offsets = p_rotor.offsets

Variables = ["Time_OF","Time_sample","Ux_{}".format(offsets[2]),"RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[s]","[s]", "[m/s]","[N]","[N-m]","[N-m]","[rads]"]


dq = dict()

time_OF = np.array(df["Time_[s]"])
time_sample = np.array(a.variables["time"])
time_sample = time_sample - time_sample[0]

tstart = 50
tend = 52
tstart_OF_idx = np.searchsorted(time_OF,tstart)
tend_OF_idx = np.searchsorted(time_OF,tend)
tstart_sample_idx = np.searchsorted(time_sample,tstart)
tend_sample_idx = np.searchsorted(time_sample,tend)


dq["Time_OF"] = time_OF[tstart_OF_idx:tend_OF_idx]
dq["Time_sample"] = time_sample[tstart_sample_idx:tend_sample_idx]
print(dq["Time_OF"])
print(dq["Time_sample"])

no_cells = len(p_rotor.variables["coordinates"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

y = p_rotor.ijk_dims[0] #no. data points
z = p_rotor.ijk_dims[1] #no. data points

rotor_coordinates = np.array([2560,2560,90])
ly = 400
Oy = 2560 - ly/2

Oz = p_rotor.origin[2]
lz = p_rotor.axis2[2]

ys = np.linspace(Oy,Oy+ly,y) - rotor_coordinates[1]
zs = np.linspace(Oz,Oz+lz,z) - rotor_coordinates[2]

dy = ys[1]-ys[0]
dz = zs[1] - zs[0]
dA = dy * dz


print(time.time() - start_time)

def offset_data(p_rotor,no_cells_offset,it,i,velocity_comp):

    u = np.array(p_rotor.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice


def it_offset(it):

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

    print("line 99",time.time()-start_time)
    return np.average(Ux_rotor)



for iv in np.arange(2,len(Variables)):
    Variable = Variables[iv]
    if Variable[0:2] == "Ux":
        i = 2
        Ux_it = []
        with Pool() as pool:
            for Ux_i in pool.imap(it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
                Ux_it.append(Ux_i)
        dq["Ux_{}".format(offsets[i])] = Ux_it
        print(dq["Ux_{}".format(offsets[i])])

    elif Variable == "MR" or Variable == "Theta":
        signaly = df["RtAeroMyh_[N-m]"][tstart_OF_idx:tend_OF_idx]
        signalz = df["RtAeroMzh_[N-m]"][tstart_OF_idx:tend_OF_idx]
        
        if Variable == "MR":
            signal = np.sqrt( np.square(signaly) + np.square(signalz) ) 
        elif Variable == "Theta": 
            signal = np.arctan(np.true_divide(signalz,signaly)) 

        dq[Variable] = signal  

    else:
        txt = "{0}_{1}".format(Variable,units[iv])
        signal = df[txt][tstart_OF_idx:tend_OF_idx]
        dq[Variable] = signal


dw = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in dq.items()]))
print(dw)

dw.to_csv("../post_processing/out.csv")