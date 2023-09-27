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

#review codes for load correlations

start_time = time.time()

#loop over coordinates with counter
#calc range of y and z coordinates rotor falls into
#if true sum velocity
#divide by len
def Ux_it_offset(it):

    Ux_rotor = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            Ux_rotor.append(velocityx[it,ijk]*np.cos(np.radians(29))+velocityy[it,ijk]*np.sin(np.radians(29)))
        ijk+=1
    return np.average(Ux_rotor)


def IA_it_offset(it):

    velx = np.reshape(velocityx[it],(y,x)); vely = np.reshape(velocityy[it],(y,x))

    fx = interpolate.interp2d(Y,Z,velx,kind="linear"); fy = interpolate.interp2d(Y,Z,vely,kind="linear")

    IA = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            delta_Ux_i = delta_Ux(j,k,r,fx,fy)
            IA += r * delta_Ux_i * dA
    return IA


def delta_Ux(j,k,r,fx,fy):

    theta = np.arccos(j/r)

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

    vx = fx(j,k); vy = fy(j,k)
    vx_1 = fx(Y_1,Z_1); vy_1 = fy(Y_1,Z_1)
    vx_2 = fx(Y_2,Z_2); vy_2 = fy(Y_2,Z_2)

    Ux_0 = vx*np.cos(np.radians(29))+vy*np.sin(np.radians(29))
    Ux_1 = vx_1*np.cos(np.radians(29))+vy_1*np.sin(np.radians(29))
    Ux_2 = vx_2*np.cos(np.radians(29))+vy_2*np.sin(np.radians(29))

    delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

    return delta_Ux



#create netcdf file
ncfile = Dataset("./Dataset.nc",mode="w",format='NETCDF4')
ncfile.title = "AMR-Wind data sampling output combined"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)
sampling_dim = ncfile.createDimension("sampling",None)

#create variables
time_OF = ncfile.createVariable("time_OF", np.float64, ('OF',),zlib=True)
time_sampling = ncfile.createVariable("time_sampling", np.float64, ('sampling',),zlib=True)

RtAeroFxh = ncfile.createVariable("RtAeroFxh", np.float64, ('OF',),zlib=True)
RtAeroMxh = ncfile.createVariable("RtAeroMxh", np.float64, ('OF',),zlib=True)
RtAeroMyh = ncfile.createVariable("RtAeroMyh", np.float64, ('OF',),zlib=True)
RtAeroMzh = ncfile.createVariable("RtAeroMzh", np.float64, ('OF',),zlib=True)
LSSGagMys = ncfile.createVariable("LSSGagMys", np.float64, ('OF',),zlib=True)
LSSGagMzs = ncfile.createVariable("LSSGagMzs", np.float64, ('OF',),zlib=True)
LSSTipMys = ncfile.createVariable("LSSTipMys", np.float64, ('OF',),zlib=True)
LSSTipMzs = ncfile.createVariable("LSSTipMzs", np.float64, ('OF',),zlib=True)
LSShftFys = ncfile.createVariable("LSShftFys", np.float64, ('OF',),zlib=True)
LSShftFzs = ncfile.createVariable("LSShftFzs", np.float64, ('OF',),zlib=True)
Theta_Aero = ncfile.createVariable("Theta_Aero", np.float64, ('OF',),zlib=True)
Theta_Tip = ncfile.createVariable("Theta_Tip", np.float64, ('OF',),zlib=True)
Theta_LSS = ncfile.createVariable("Theta_LSS", np.float64, ('OF',),zlib=True)

print("line 148",time.time()-start_time)


#openfast data
df = io.fast_output_file.FASTOutputFile("./NREL_5MW_Main.out").toDataFrame()

time_OF[:] = np.array(df["Time_[s]"])

print("line 156",time.time()-start_time)

Variables = ["RtAeroFxh","RtAeroMxh","RtAeroMyh","RtAeroMzh","Theta","LSSGagMys","LSSGagMzs",
             "LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs","Theta_Aero","Theta_Tip", "Theta_LSS"]
units = ["[N]","[N-m]","[N-m]","[N-m]","[rads]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[rads]","[rads]","[rads]"]
for iv in np.arange(0,len(Variables)):
    Variable = Variables[iv]

    if Variable[0:4] == "Theta":
        if Variable[6:] == "Aero":
            signaly = np.array(df["RtAeroMyh_[N-m]"])
            signalz = np.array(df["RtAeroMzh_[N-m]"])
            
            signal = np.arctan2(signalz,signaly)
            Theta_Aero[:] = signal; del signal

        elif Variable[6:] == "Tip":
            signaly = np.array(df["LSSTipMys_[kN-m]"])
            signalz = np.array(df["LSSTipMzs_[kN-m]"])
            
            signal = np.arctan2(signalz,signaly)
            Theta_Tip[:] = signal; del signal
        
        elif Variable[6:] == "LSS":
            signaly = np.array(df["LSSGagMys_[kN-m]"])
            signalz = np.array(df["LSSGagMzs_[kN-m]"])
            
            signal = np.arctan2(signalz,signaly)
            Theta_LSS[:] = signal; del signal

    else:
        txt = "{0}_{1}".format(Variable,units[iv])
        signal = np.array(df[txt])
        if Variable == "RtAeroFxh":
            RtAeroFxh[:] = signal; del signal
        elif Variable == "RtAeroMxh":
            RtAeroMxh[:] = signal; del signal
        elif Variable == "RtAeroMyh":
            RtAeroMyh[:] = signal; del signal
        elif Variable == "RtAeroMzh":
            RtAeroMzh[:] = signal; del signal
        elif Variable == "LSSGagMys":
            LSSGagMys[:] = signal; del signal
        elif Variable == "LSSGagMzs":
            LSSGagMzs[:] = signal; del signal
        elif Variable == "LSSTipMys":
            LSSTipMys[:] = signal; del signal
        elif Variable == "LSSTipMzs":
            LSSTipMzs[:] = signal; del signal
        elif Variable == "LSShftFys":
            LSShftFys[:] = signal[:,0]; del signal
        elif Variable == "LSShftFzs":
            LSShftFzs[:] = signal; del signal


del df

print("line 191",time.time()-start_time)

#sampling data
a = Dataset("./sampling_r_0.0.nc")

#sampling time
Time_sample = np.array(a.variables["time"])
time_idx = len(Time_sample)
time_sampling[:] = Time_sample; del Time_sample

print("line 201", time_idx, time.time()-start_time)

offsets = [0.0,-63.0]
group_label = [0.0,63.0]

ic = 0
for offset in offsets:

    a = Dataset("./sampling_r_{}.nc".format(offset))

    group = ncfile.createGroup("{}".format(group_label[ic]))

    Ux = group.createVariable("Ux", np.float64, ('sampling'),zlib=True)
    IA = group.createVariable("IA", np.float64, ('sampling'),zlib=True)

    p_rotor = a.groups["p_r"]; del a

    velocityx = np.array(p_rotor.variables["velocityx"]); velocityy = np.array(p_rotor.variables["velocityy"])

    Variables = ["Ux_{0}".format(offset), "IA_{0}".format(offset)]
    units = ["[m/s]", "[$m^4/s$]"]

    x = p_rotor.ijk_dims[0] #no. data points
    y = p_rotor.ijk_dims[1] #no. data points

    coordinates = np.array(p_rotor.variables["coordinates"])

    xo = coordinates[:,0]
    yo = coordinates[:,1]
    zo = coordinates[:,2]

    rotor_coordiates = [2560,2560,90]

    x_trans = xo - rotor_coordiates[0]
    y_trans = yo - rotor_coordiates[1]

    phi = np.radians(-29)
    xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
    ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zs = zo - rotor_coordiates[2]

    Y = np.linspace(round(np.min(ys),0), round(np.max(ys),0),x )
    Z = np.linspace(round(np.min(zs),0), round(np.max(zs),0),y )

    del coordinates

    dy = (max(Y) - min(Y))/x
    dz = (max(Z) - min(Z))/y
    dA = dy * dz

    del p_rotor

    print("line 249",time.time()-start_time)

    for iv in np.arange(0,len(Variables)):
        Variable = Variables[iv]
        print(Variable[0:2],"_",Variable[3:])

        if Variable[0:2] == "Ux":
            Ux_it = []
            print("Ux calcs")
            with Pool() as pool:
                i_Ux = 1
                for Ux_i in pool.imap(Ux_it_offset, np.arange(0,time_idx)):
                    Ux_it.append(Ux_i)
                    print(i_Ux,time.time()-start_time)
                    i_Ux+=1
                Ux_it = np.array(Ux_it)
                Ux[:] = Ux_it; del Ux_it

        elif Variable[0:2] == "IA":
            IA_it = []
            print("IA calcs")
            with Pool() as pool:
                i_IA = 1
                for IA_i in pool.imap(IA_it_offset, np.arange(0,time_idx)):
                    IA_it.append(IA_i)
                    print(i_IA,time.time()-start_time)
                    i_IA+=1
                IA_it = np.array(IA_it)
                IA[:] = IA_it; del IA_it


    print(ncfile.groups)
    ic+=1
print(ncfile)
ncfile.close()

print("line 308",time.time() - start_time)