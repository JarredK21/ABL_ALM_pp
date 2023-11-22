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


def Ux_it_offset(it):

    Ux_rotor = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            Ux_rotor.append(velocityx[it,ijk]*np.cos(np.radians(29))+velocityy[it,ijk]*np.sin(np.radians(29)))
        ijk+=1
    return np.average(Ux_rotor)


def IA_it(it):

    velx = np.reshape(velocityx[it],(y,x)); vely = np.reshape(velocityy[it],(y,x))

    fx = interpolate.interp2d(Y,Z,velx,kind="linear"); fy = interpolate.interp2d(Y,Z,vely,kind="linear")

    IA = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            delta_Ux_i = delta_Ux(j,k,r,fx,fy)
            IA += r * delta_Ux_i * dA
    return IA


def IB_it(it):

    velx = np.reshape(velocityx[it],(y,x)); vely = np.reshape(velocityy[it],(y,x))

    fx = interpolate.interp2d(Y,Z,velx,kind="linear"); fy = interpolate.interp2d(Y,Z,vely,kind="linear")

    IB = 0
    dr = (63-1.5)/200
    for r in np.arange(1.5,63,200):
            delta_Ux_i = delta_Ux_r(r,fx,fy,it)
            IB += r * delta_Ux_i * dr
    return IB


def Iy_it(it):

    M_velZ = 0
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            N = 0
            M_velZ += (velocityx[it,ijk]*np.cos(np.radians(29))+velocityy[it,ijk]*np.sin(np.radians(29)))*k*dA
            N+=1
        ijk+=1
    Iy = M_velZ/N

    return Iy


def Iz_it(it):

    M_velY = 0
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            N = 0
            M_velY += (velocityx[it,ijk]*np.cos(np.radians(29))+velocityy[it,ijk]*np.sin(np.radians(29)))*j*dA
            N+=1
        ijk+=1
    Iz = M_velY/N

    return Iz


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


def delta_Ux_r(r,fx,fy,it):
    theta = Azimuth[it]

    Y = r*np.cos(theta)
    Z = r*np.sin(theta)

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

    vx = fx(Y,Z); vy = fy(Y,Z)
    vx_1 = fx(Y_1,Z_1); vy_1 = fy(Y_1,Z_1)
    vx_2 = fx(Y_2,Z_2); vy_2 = fy(Y_2,Z_2)

    Ux_0 = vx*np.cos(np.radians(29))+vy*np.sin(np.radians(29))
    Ux_1 = vx_1*np.cos(np.radians(29))+vy_1*np.sin(np.radians(29))
    Ux_2 = vx_2*np.cos(np.radians(29))+vy_2*np.sin(np.radians(29))

    delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 ), abs(Ux_1 - Ux_2)] )

    return delta_Ux





#directories
in_dir = "./"
#in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"
out_dir = in_dir

#create netcdf file
ncfile = Dataset(out_dir+"Dataset_test.nc",mode="w",format='NETCDF4')
ncfile.title = "OpenFAST data sampling output"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)
sampling_dim = ncfile.createDimension("sampling",None)

#openfast data
df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

time_OF = np.array(df["Time_[s]"])

Azimuth = np.radians(np.array(df["Azimuth_[deg]"]))

del df

print("line 191",time.time()-start_time)

#sampling data
a = Dataset(in_dir+"sampling_r_0.0.nc")

#sampling time
Time_sample = np.array(a.variables["time"])
time_idx = len(Time_sample)

print("line 201", time_idx, time.time()-start_time)

offsets = [0.0]
group_label = [0.0]


ic = 0
for offset in offsets:

    a = Dataset(in_dir+"sampling_r_{}.nc".format(offset))

    group = ncfile.createGroup("{}".format(group_label[ic]))

    IB = group.createVariable("IB",np.float64, ('sampling'),zlib=True)
    Iy = group.createVariable("Iy", np.float64, ('sampling'),zlib=True)
    Iz = group.createVariable("Iz", np.float64, ('sampling'),zlib=True)

    p_rotor = a.groups["p_r"]; del a

    velocityx = np.array(p_rotor.variables["velocityx"]); velocityy = np.array(p_rotor.variables["velocityy"])
    f = interpolate.interp1d(Azimuth,time_OF)
    Azimuth = f(Time_sample)

    Variables = ["IB_{0}".format(offset), "Iy_{0}".format(offset),"Iz_{0}".format(offset)]

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


        if Variable[0:2] == "IB":
            IB_it_Arr = []
            print("IB calcs")
            with Pool() as pool:
                i_IB = 1
                for IB_i in pool.imap(IB_it, np.arange(0,time_idx)):
                    IB_it_Arr.append(IB_i)
                    print(i_IB,time.time()-start_time)
                    i_IB+=1
                IB_it_Arr = np.array(IB_it_Arr)
                IB[:] = IB_it_Arr; del IB_it_Arr


        elif Variable[0:2] == "Iy":
            Iy_arr = []
            print("Iy calcs")
            with Pool() as pool:
                i_I = 1
                for Iy_i in pool.imap(Iy_it, np.arange(0,time_idx)):
                    Iy_arr.append(Iy_i)
                    print(i_I,time.time()-start_time)
                    i_I+=1
                Iy_arr = np.array(Iy_arr)
                Iy[:] = Iy_arr; del Iy_arr


        elif Variable[0:2] == "Iz":
            Iz_arr = []
            print("Iz calcs")
            with Pool() as pool:
                i_I = 1
                for Iz_i in pool.imap(Iz_it, np.arange(0,time_idx)):
                    Iz_arr.append(Iz_i)
                    print(i_I,time.time()-start_time)
                    i_I+=1
                Iz_arr = np.array(Iz_arr)
                Iz[:] = Iz_arr; del Iz_arr


    del velocityx; velocityy

    print(ncfile.groups)
    ic+=1

print(ncfile)
ncfile.close()

print("line 308",time.time() - start_time)