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


#openfast data
df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

#sampling data
sampling = glob.glob("../post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]

offsets = p_rotor.offsets

Variables = ["Time_OF","Time_sample","Ux_{}".format(offsets[2]),"IA_{}".format(offsets[2]),"RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[s]","[s]", "[m/s]", "[m/s]", "[m/s]","[$m^4/s$]","[$m^4/s$]","[$m^4/s$]","[N]","[N-m]","[N-m]","[rads]"]


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

no_cells = len(p_rotor.variables["velocityx"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

print(no_offsets,no_cells_offset)

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

#create R,theta space over rotor
R = np.linspace(1.5,63,100)
Theta = np.arange(0,2*np.pi,(2*np.pi)/300)


def offset_data(p_rotor,no_cells_offset,i,it,velocity_comp):

    u = np.array(p_rotor.variables[velocity_comp][it]) #only time step
    print(np.shape(u))

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]
    print(np.shape(u_slice))

    return u_slice


def it_offset(i,it):
    
    start_time2 = time.time()

    velocityx = offset_data(p_rotor, no_cells_offset,i,it,velocity_comp="velocityx")
    velocityy = offset_data(p_rotor, no_cells_offset,i,it,velocity_comp="velocityy")
    print(np.shape(velocityx))
    hvelmag = np.add( np.multiply(velocityx[i],np.cos(np.radians(29))) , np.multiply( velocityy[i],np.sin(np.radians(29))) )

    hvelmag = hvelmag.reshape((z,y))
    f = interpolate.interp2d(ys,zs,hvelmag,kind="linear")

    hvelmag = hvelmag.reshape((y,z))

    print(time.time()-start_time2)

    IA = 0
    Ux_rotor = 0
    ic = 0
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)):
            r = np.sqrt(ys[j]**2 + zs[k]**2)
            if r <= 63 and r >= 1.5:
                Ux_rotor += hvelmag[j,k]

                delta_Ux_i = delta_Ux(r,ys[j],zs[k],f)
                IA += r * delta_Ux_i * dA

                ic+=1

    return Ux_rotor/ic, IA


def delta_Ux(r,y0,z0,f):

    Y_0 = y0
    Z_0 = z0

    theta = np.arccos(Y_0/r)

    if theta + ((2*np.pi)/3) > (2*np.pi):
        theta_1 = theta +(2*np.pi)/3 - (2*np.pi)
    else:
        theta_1 = theta + (2*np.pi)/3

    Y_1 = r*np.cos(theta_1)
    Z_1 = r*np.sin(theta_1)


    if theta - ((2*np.pi)/3) < 0:
        theta_2 = theta - ((2*np.pi)/3) + (2*np.pi)
    else:
        theta_2 = theta - ((2*np.pi)/3)

    Y_2 = r*np.cos(theta_2)
    Z_2 = r*np.sin(theta_2)

    Ux_0 =  f(Y_0,Z_0)
    Ux_1 =  f(Y_1,Z_1)
    Ux_2 =  f(Y_2,Z_2)

    delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

    return delta_Ux


for iv in np.arange(3,len(Variables)):
    Variable = Variables[iv]
    if Variable[0:2] == "IA":
        i = 2
        Ux_it = []
        IA_it = []
        start_time = time.time()
        for it in np.arange(tstart_sample_idx,tend_sample_idx):
            Ux_i,IA_i = it_offset(i,it)
            Ux_it.append(Ux_i)
            IA_it.append(IA_i)
            print(len(Ux_it),time.time()-start_time)
        dq["Ux_{}".format(offsets[i])] = Ux_it
        dq["IA_{}".format(offsets[i])] = IA_it

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
dw.to_csv("../post_processing/out.csv")