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


def Uy_it_offset(it):
    Uy_rotor = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            Uy_rotor.append(-velocityx[it,ijk]*np.sin(np.radians(29))+velocityy[it,ijk]*np.cos(np.radians(29)))
        ijk+=1
    return np.average(Uy_rotor)


def Uz_it_offset(it):
    Uz_rotor = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            Uz_rotor.append(velocityz[it,ijk])
        ijk+=1
    return np.average(Uz_rotor)


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


def delta_y_Ux(it):

    velx = np.reshape(velocityx[it],(y,x)); vely = np.reshape(velocityy[it],(y,x))

    fx = interpolate.interp2d(Y,Z,velx,kind="linear"); fy = interpolate.interp2d(Y,Z,vely,kind="linear")
    vx1 = fx(63,0); vy1 = fy(63,0); vx2 = fx(-63,0); vy2 = fy(-63,0)

    Ux_1 = vx1*np.cos(np.radians(29))+vy1*np.sin(np.radians(29))
    Ux_2 = vx2*np.cos(np.radians(29))+vy2*np.sin(np.radians(29))

    delta_y_Ux = Ux_1 - Ux_2

    return delta_y_Ux


def delta_z_Ux(it):

    velx = np.reshape(velocityx[it],(y,x)); vely = np.reshape(velocityy[it],(y,x))

    fx = interpolate.interp2d(Y,Z,velx,kind="linear"); fy = interpolate.interp2d(Y,Z,vely,kind="linear")
    vx1 = fx(0,63); vy1 = fy(0,63); vx2 = fx(0,-63); vy2 = fy(0,-63)

    Ux_1 = vx1*np.cos(np.radians(29))+vy1*np.sin(np.radians(29))
    Ux_2 = vx2*np.cos(np.radians(29))+vy2*np.sin(np.radians(29))

    delta_z_Ux = Ux_1 - Ux_2

    return delta_z_Ux



#directories
in_dir = "./"
#in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"
out_dir = in_dir

#create netcdf file
ncfile = Dataset(out_dir+"Dataset.nc",mode="w",format='NETCDF4')
ncfile.title = "OpenFAST data sampling output"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)
sampling_dim = ncfile.createDimension("sampling",None)

#create variables
time_OF = ncfile.createVariable("time_OF", np.float64, ('OF',),zlib=True)
time_sampling = ncfile.createVariable("time_sampling", np.float64, ('sampling',),zlib=True)

# Alpha_75 = ncfile.createVariable("Alpha_75", np.float64, ('OF',),zlib=True)
# Vrel_75 = ncfile.createVariable("Vrel_75", np.float64, ('OF',),zlib=True)
# Cl_75 = ncfile.createVariable("Cl_75", np.float64, ('OF',),zlib=True)
# Cd_75 = ncfile.createVariable("Cd_75", np.float64, ('OF',),zlib=True)
# Fn_75 = ncfile.createVariable("Fn_75", np.float64, ('OF',),zlib=True)
# Ft_75 = ncfile.createVariable("Ft_75", np.float64, ('OF',),zlib=True)
# Fx_75 = ncfile.createVariable("Fx_75", np.float64, ('OF',),zlib=True)
# Fy_75 = ncfile.createVariable("Fy_75", np.float64, ('OF',),zlib=True)
# Vx_75 = ncfile.createVariable("Vx_75", np.float64, ('OF',),zlib=True)
# Vy_75 = ncfile.createVariable("Vy_75", np.float64, ('OF',),zlib=True)

Azimuth = ncfile.createVariable("Azimuth", np.float64, ('OF',),zlib=True)
RtAeroFxh = ncfile.createVariable("RtAeroFxh", np.float64, ('OF',),zlib=True)
RtAeroFyh = ncfile.createVariable("RtAeroFyh", np.float64, ('OF',),zlib=True)
RtAeroFzh = ncfile.createVariable("RtAeroFzh", np.float64, ('OF',),zlib=True)
RtAeroMxh = ncfile.createVariable("RtAeroMxh", np.float64, ('OF',),zlib=True)
RtAeroMyh = ncfile.createVariable("RtAeroMyh", np.float64, ('OF',),zlib=True)
RtAeroMzh = ncfile.createVariable("RtAeroMzh", np.float64, ('OF',),zlib=True)

LSSGagMys = ncfile.createVariable("LSSGagMys", np.float64, ('OF',),zlib=True)
LSSGagMzs = ncfile.createVariable("LSSGagMzs", np.float64, ('OF',),zlib=True)
LSShftMxa = ncfile.createVariable("LSShftMxa", np.float64, ('OF',),zlib=True)
LSSTipMys = ncfile.createVariable("LSSTipMys", np.float64, ('OF',),zlib=True)
LSSTipMzs = ncfile.createVariable("LSSTipMzs", np.float64, ('OF',),zlib=True)
LSShftFxa = ncfile.createVariable("LSShftFxa", np.float64, ('OF',),zlib=True)
LSShftFys = ncfile.createVariable("LSShftFys", np.float64, ('OF',),zlib=True)
LSShftFzs = ncfile.createVariable("LSShftFzs", np.float64, ('OF',),zlib=True)

print("line 148",time.time()-start_time)

#openfast data
df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

time_OF[:] = np.array(df["Time_[s]"])

print("line 156",time.time()-start_time)

# Variables = ["AB1N225Alpha","AB1N225Vrel","AB1N225Cl","AB1N225Cd","AB1N225Fn","AB1N225Ft","AB1N225Fx","AB1N225Fy","AB1N225Vx","AB1N225Vy",
#              "RtAeroFxh","RtAeroMxh","RtAeroMyh","RtAeroMzh","LSSGagMys","LSSGagMzs",
#              "LSSTipMys","LSSTipMzs","LSShftFys","LSShftFzs","Theta_Aero","Theta_Tip", "Theta_LSS", "LSShftFxa", "LSShftMxa"]
# units = ["[deg]","[m/s]","[-]","[-]","[N/m]","[N/m]","[N/m]","[N/m]","[m/s]","[m/s]",
#          "[N]","[N-m]","[N-m]","[N-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[rads]","[rads]","[rads]","[kN]","[kN-m]"]

Variables = ["Azimuth","RtAeroFxh","RtAeroFyh","RtAeroFzh","RtAeroMxh","RtAeroMyh","RtAeroMzh",
             "LSSGagMys","LSSGagMzs", "LSShftMxa","LSSTipMys","LSSTipMzs",
             "LSShftFxa","LSShftFys","LSShftFzs"]
units = ["[deg]","[N]","[N]","[N]","[N-m]","[N-m]","[N-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]",
         "[kN]","[kN]","[kN]"]
for i in np.arange(0,len(Variables)):
    Variable = Variables[i]

    txt = "{0}_{1}".format(Variable,units[i])
    signal = np.array(df[txt])
    if Variable == "RtAeroFxh":
        RtAeroFxh[:] = signal; del signal
    elif Variable == "RtAeroFyh":
        RtAeroFyh[:] = signal; del signal
    elif Variable == "RtAeroFzh":
        RtAeroFzh[:] = signal; del signal
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
    elif Variable == "LSShftMxa":
        LSShftMxa[:] = signal; del signal
    elif Variable == "LSSTipMys":
        LSSTipMys[:] = signal; del signal
    elif Variable == "LSSTipMzs":
        LSSTipMzs[:] = signal; del signal
    elif Variable == "LSShftFxa":
        LSShftFxa[:] = signal[:,0]; del signal
    elif Variable == "LSShftFys":
        LSShftFys[:] = signal[:,0]; del signal
    elif Variable == "LSShftFzs":
        LSShftFzs[:] = signal; del signal
    elif Variable == "Azimuth":
        Azimuth[:] = signal; del signal
    # elif Variable == "AB1N225Alpha":
    #     Alpha_75[:] = signal; del signal
    # elif Variable == "AB1N225Vrel":
    #     Vrel_75 = signal; del signal
    # elif Variable == "AB1N225Cl":
    #     Cl_75 = signal; del signal
    # elif Variable == "AB1N225Cd":
    #     Cd_75 = signal; del signal
    # elif Variable == "AB1N225Fn":
    #     Fn_75 = signal; del signal
    # elif Variable == "AB1N225Ft":
    #     Ft_75 = signal; del signal
    # elif Variable == "AB1N225Fx":
    #     Fx_75 = signal; del signal
    # elif Variable == "AB1N225Fy":
    #     Fy_75 = signal; del signal
    # elif Variable == "AB1N225Vx":
    #     Vx_75 = signal; del signal
    # elif Variable == "AB1N225Vy":
    #     Vy_75 = signal; del signal

del df

print("line 191",time.time()-start_time)

#sampling data
a = Dataset(in_dir+"sampling_r_0.0.nc")

#sampling time
Time_sample = np.array(a.variables["time"])
time_idx = len(Time_sample)
time_sampling[:] = Time_sample; del Time_sample

print("line 201", time_idx, time.time()-start_time)

offsets = [0.0,-63.0]
group_label = [0.0,63.0]

ic = 0
for offset in offsets:

    a = Dataset(in_dir+"sampling_r_{}.nc".format(offset))

    group = ncfile.createGroup("{}".format(group_label[ic]))

    Ux = group.createVariable("Ux", np.float64, ('sampling'),zlib=True)
    Uy = group.createVariable("Uy", np.float64, ('sampling'),zlib=True)
    Uz = group.createVariable("Uz", np.float64, ('sampling'),zlib=True)
    IA = group.createVariable("IA", np.float64, ('sampling'),zlib=True)
    Iy = group.createVariable("Iy", np.float64, ('sampling'),zlib=True)
    Iz = group.createVariable("Iz", np.float64, ('sampling'),zlib=True)

    p_rotor = a.groups["p_r"]; del a

    velocityx = np.array(p_rotor.variables["velocityx"]); velocityy = np.array(p_rotor.variables["velocityy"])
    velocityz = np.array(p_rotor.variables["velocityz"])

    Variables = ["Ux_{0}".format(offset), "IA_{0}".format(offset), "Uy_{0}".format(offset), "Uz_{0}".format(offset),
                 "Iy_{0}".format(offset), "Iz_{0}".format(offset)]

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


        elif Variable[0:2] == "Uy":
            Uy_it = []
            print("Uy calcs")
            with Pool() as pool:
                i_Uy = 1
                for Uy_i in pool.imap(Uy_it_offset, np.arange(0,time_idx)):
                    Uy_it.append(Uy_i)
                    print(i_Uy,time.time()-start_time)
                    i_Uy+=1
                Uy_it = np.array(Uy_it)
                Uy[:] = Uy_it; del Uy_it


        elif Variable[0:2] == "Uz":
            Uz_it = []
            print("Uz calcs")
            with Pool() as pool:
                i_Uz = 1
                for Uz_i in pool.imap(Uz_it_offset, np.arange(0,time_idx)):
                    Uz_it.append(Uz_i)
                    print(i_Uz,time.time()-start_time)
                    i_Uz+=1
                Uz_it = np.array(Uz_it)
                Uz[:] = Uz_it; del Uz_it


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


        elif Variable[0:2] == "Iy":
            Iy_it = []
            print("Iy calcs")
            with Pool() as pool:
                i_Iy = 1
                for Iy_i in pool.imap(delta_y_Ux, np.arange(0,time_idx)):
                    Iy_it.append(Iy_i)
                    print(i_Iy,time.time()-start_time)
                    i_Iy+=1
                Iy_it = np.array(Iy_it)
                Iy[:] = Iy_it; del Iy_it


        elif Variable[0:2] == "Iz":
            Iz_it = []
            print("Iz calcs")
            with Pool() as pool:
                i_Iz = 1
                for Iz_i in pool.imap(delta_z_Ux, np.arange(0,time_idx)):
                    Iz_it.append(Iz_i)
                    print(i_Iz,time.time()-start_time)
                    i_Iz+=1
                Iz_it = np.array(Iz_it)
                Iz[:] = Iz_it; del Iz_it


    del velocityx; velocityy; velocityz

    print(ncfile.groups)
    ic+=1

print(ncfile)
ncfile.close()

print("line 308",time.time() - start_time)