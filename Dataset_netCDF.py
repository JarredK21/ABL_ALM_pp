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


# def Uy_it_offset(it):
#     Uy_rotor = []
#     ijk = 0
#     for j,k in zip(ys,zs):
#         r = np.sqrt(j**2 + k**2)
#         if r <= 63 and r > 1.5:
#             Uy_rotor.append(-velocityx[it,ijk]*np.sin(np.radians(29))+velocityy[it,ijk]*np.cos(np.radians(29)))
#         ijk+=1
#     return np.average(Uy_rotor)


# def Uz_it_offset(it):
#     Uz_rotor = []
#     ijk = 0
#     for j,k in zip(ys,zs):
#         r = np.sqrt(j**2 + k**2)
#         if r <= 63 and r > 1.5:
#             Uz_rotor.append(velocityz[it,ijk])
#         ijk+=1
#     return np.average(Uz_rotor)


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


def Iy_it(it):

    velx = np.reshape(velocityx[it],(y,x)); vely = np.reshape(velocityy[it],(y,x))

    fx = interpolate.interp2d(Y,Z,velx,kind="linear"); fy = interpolate.interp2d(Y,Z,vely,kind="linear")

    delta_z_Ux = []
    for L_top_i in L_top:
        for L_bottom_i in L_bottom:
            r = L_top_i[1] - L_bottom_i[1]
            ux_top = fx(L_top_i[0],L_top_i[1]); uy_top = fy(L_top_i[0],L_top_i[1])
            Ux_1 = ux_top*np.cos(np.radians(29))+uy_top*np.sin(np.radians(29))
            ux_bottom = fx(L_bottom_i[0],L_bottom_i[1]); uy_bottom = fy(L_bottom_i[0],L_bottom_i[1])
            Ux_2 = ux_bottom*np.cos(np.radians(29))+uy_bottom*np.sin(np.radians(29))
            
            delta_z_Ux.append(r*(Ux_1-Ux_2))

    return max(delta_z_Ux)


def Iz_it(it):

    velx = np.reshape(velocityx[it],(y,x)); vely = np.reshape(velocityy[it],(y,x))

    fx = interpolate.interp2d(Y,Z,velx,kind="linear"); fy = interpolate.interp2d(Y,Z,vely,kind="linear")

    delta_y_Ux = []
    for L_right_i in L_right:
        for L_left_i in L_left:
            r = L_right_i[0] - L_left_i[0]
            ux_right = fx(L_right_i[0],L_right_i[1]); uy_right = fy(L_right_i[0],L_right_i[1])
            Ux_1 = ux_right*np.cos(np.radians(29))+uy_right*np.sin(np.radians(29))
            ux_left = fx(L_left_i[0],L_left_i[1]); uy_left = fy(L_left_i[0],L_left_i[1])
            Ux_2 = ux_left*np.cos(np.radians(29))+uy_left*np.sin(np.radians(29))
            
            delta_y_Ux.append(r*(Ux_1-Ux_2))

    return max(delta_y_Ux)


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


#defining asymmetry domains
L_bottom_int = np.linspace(-63,-1.5,200)
L_top_int = np.linspace(63,1.5,200)
l = np.linspace(-1.5,1.5,10)

L_bottom = []
L_top = []
L_right = []
L_left = []
for L_bottom_i,L_top_i in zip(L_bottom_int,L_top_int):
    for l_i in l:
        L_bottom.append([l_i,L_bottom_i])
        L_top.append([l_i,L_top_i])
        L_right.append([L_top_i,l_i])
        L_left.append([L_bottom_i,l_i])


ic = 0
for offset in offsets:

    a = Dataset(in_dir+"sampling_r_{}.nc".format(offset))

    group = ncfile.createGroup("{}".format(group_label[ic]))

    Ux = group.createVariable("Ux", np.float64, ('sampling'),zlib=True)
    #Uy = group.createVariable("Uy", np.float64, ('sampling'),zlib=True)
    #Uz = group.createVariable("Uz", np.float64, ('sampling'),zlib=True)
    IA = group.createVariable("IA", np.float64, ('sampling'),zlib=True)
    Iy = group.createVariable("Iy", np.float64, ('sampling'),zlib=True)
    Iz = group.createVariable("Iz", np.float64, ('sampling'),zlib=True)

    p_rotor = a.groups["p_r"]; del a

    velocityx = np.array(p_rotor.variables["velocityx"]); velocityy = np.array(p_rotor.variables["velocityy"])
    #velocityz = np.array(p_rotor.variables["velocityz"])

    # Variables = ["Ux_{0}".format(offset), "IA_{0}".format(offset), "Uy_{0}".format(offset), "Uz_{0}".format(offset),
    #              "Iy_{0}".format(offset), "Iz_{0}".format(offset)]
    Variables = ["Ux_{0}".format(offset), "IA_{0}".format(offset), "Iy_{0}".format(offset), "Iz_{0}".format(offset)]

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


        # elif Variable[0:2] == "Uy":
        #     Uy_it = []
        #     print("Uy calcs")
        #     with Pool() as pool:
        #         i_Uy = 1
        #         for Uy_i in pool.imap(Uy_it_offset, np.arange(0,time_idx)):
        #             Uy_it.append(Uy_i)
        #             print(i_Uy,time.time()-start_time)
        #             i_Uy+=1
        #         Uy_it = np.array(Uy_it)
        #         Uy[:] = Uy_it; del Uy_it


        # elif Variable[0:2] == "Uz":
        #     Uz_it = []
        #     print("Uz calcs")
        #     with Pool() as pool:
        #         i_Uz = 1
        #         for Uz_i in pool.imap(Uz_it_offset, np.arange(0,time_idx)):
        #             Uz_it.append(Uz_i)
        #             print(i_Uz,time.time()-start_time)
        #             i_Uz+=1
        #         Uz_it = np.array(Uz_it)
        #         Uz[:] = Uz_it; del Uz_it


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
            Iy = []
            print("Iy calcs")
            with Pool() as pool:
                i_Iy = 1
                for Iy_i in pool.imap(Iy_it, np.arange(0,time_idx)):
                    Iy.append(Iy_i)
                    print(i_Iy,time.time()-start_time)
                    i_Iy+=1
                Iy = np.array(Iy)
                Iy[:] = Iy; del Iy


        elif Variable[0:2] == "Iz":
            Iz = []
            print("Iz calcs")
            with Pool() as pool:
                i_Iz = 1
                for Iz_i in pool.imap(Iz_it, np.arange(0,time_idx)):
                    Iz.append(Iz_i)
                    print(i_Iz,time.time()-start_time)
                    i_Iz+=1
                Iz = np.array(Iz)
                Iz[:] = Iz; del Iz


    del velocityx; velocityy; #velocityz

    print(ncfile.groups)
    ic+=1

print(ncfile)
ncfile.close()

print("line 308",time.time() - start_time)