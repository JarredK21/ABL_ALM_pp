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


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    mag_horz_vel = []
    for i in np.arange(0,len(zs)):
        u_i = u[it,i*x:(i+1)*x]; v_i = v[it,i*x:(i+1)*x]
        if zs[i] < h[0]:
            twist_h = f(h[0])

        elif zs[i] > h[-1]:
            twist_h = f(h[-1])
        else:
            twist_h = f(zs[i])

        mag_horz_vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
        mag_horz_vel.extend(mag_horz_vel_i)
    mag_horz_vel = np.array(mag_horz_vel)
    return mag_horz_vel



#def dyUx_calc(it):
def dyUx_calc():
    #U = u[it]
    U = u
    u_plane = U.reshape(y,x)

    



#directories
#in_dir = "./"
in_dir = "../../ABL_precursor_2_restart/"
#out_dir = in_dir

#defining twist angles with height from precursor
#precursor = Dataset("./abl_statistics76000.nc")
precursor = Dataset(in_dir+"abl_statistics70000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38200)
t_end = np.searchsorted(precursor.variables["time"],39200)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 67", time.time()-start_time)

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

# #create netcdf file
# ncfile = Dataset(out_dir+"Dataset.nc",mode="w",format='NETCDF4')
# ncfile.title = "OpenFAST data sampling gradients output"

# #create global dimensions
# sampling_dim = ncfile.createDimension("sampling",None)

# #create variables
# time_sampling = ncfile.createVariable("time_sampling", np.float64, ('sampling',),zlib=True)

#sampling data
# a = Dataset(in_dir+"sampling_r_-5.5.nc")

# #sampling time
# Time_sample = np.array(a.variables["time"])
# Time_steps = np.arange(0,len(Time_sample))
# Time_sample = Time_sample - Time_sample[0]
# time_idx = len(Time_sample)
#time_sampling[:] = Time_sample

#print("line 201", time_idx, time.time()-start_time)

# offsets = [-5.5,-63.0]
# group_label = [5.5,63.0]
offsets = [-63.0]
group_label = [63.0]


ic = 0
for offset in offsets:

    # a = Dataset(in_dir+"sampling_r_{}.nc".format(offset))
    a = Dataset(in_dir+"sampling_r_{}_0.nc".format(offset))

    # group = ncfile.createGroup("{}".format(group_label[ic]))

    # dyUx = group.createVariable("dyUx", np.float64, ('sampling'),zlib=True)
    # dzUx = group.createVariable("dzUx", np.float64, ('sampling'),zlib=True)
    # drUx = group.createVariable("drUx", np.float64, ('sampling'),zlib=True)

    p_rotor = a.groups["p_r"]; del a
    

    Variables = ["dyUx_{}".format(offset),"dzUx_{}".format(offset),"drUx_{}".format(offset)]

    x = p_rotor.ijk_dims[0] #no. data points
    y = p_rotor.ijk_dims[1] #no. data points


    normal = 29

    #define plotting axes
    coordinates = np.array(p_rotor.variables["coordinates"])


    xo = coordinates[0:x,0]
    yo = coordinates[0:x,1]

    rotor_coordiates = [2560,2560,90]

    x_trans = xo - rotor_coordiates[0]
    y_trans = yo - rotor_coordiates[1]

    phi = np.radians(-normal)
    xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
    ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    xs = xs + rotor_coordiates[0]
    ys = ys + rotor_coordiates[1]
    zs = np.linspace(p_rotor.origin[2],p_rotor.origin[2]+p_rotor.axis2[2],y)

    dy = ys[1]-ys[0]
    dz = zs[1]-zs[0]

    #velocity field
    u = np.array(p_rotor.variables["velocityx"])
    v = np.array(p_rotor.variables["velocityy"])
    del p_rotor

    u[u<0]=0; v[v<0]=0 #remove negative velocities

    # with Pool() as pool:
    #     u_hvel = []
    #     for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
            
    #         u_hvel.append(u_hvel_it)
    #         print(len(u_hvel),time.time()-start_time)
    # u = np.array(u_hvel); del u_hvel; del v

    f = interpolate.interp1d(h,twist)
    mag_horz_vel = []
    for i in np.arange(0,len(zs)):
        u_i = u[i*x:(i+1)*x]; v_i = v[i*x:(i+1)*x]
        if zs[i] < h[0]:
            twist_h = f(h[0])

        elif zs[i] > h[-1]:
            twist_h = f(h[-1])
        else:
            twist_h = f(zs[i])

        mag_horz_vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
        mag_horz_vel.extend(mag_horz_vel_i)
    u = np.array(mag_horz_vel); del v; del mag_horz_vel; mag_horz_vel_i


    print("line 139",time.time()-start_time)

    for iv in np.arange(0,len(Variables)):
        Variable = Variables[iv]
        print(Variable[0:2],"_",Variable[3:])

        if Variable[0:2] == "dy":
            dyUx = dyUx_calc()
            # dyUx_array = []
            # print("dyUx calcs")
            # with Pool() as pool:
            #     cc = 1
            #     for dyUx_it in pool.imap(dyUx_calc, np.arange(0,time_idx)):
            #         dyUx_array.append(dyUx_it)
            #         print(cc,time.time()-start_time)
            #         cc+=1
            #     dyUx[:] = np.array(dyUx_array); del dyUx_array


        #elif Variable[0:2] == "dz":
            # dzUx_array = []
            # print("dzUx calcs")
            # with Pool() as pool:
            #     cc = 1
            #     for dzUx_it in pool.imap(dzUx_calc, np.arange(0,time_idx)):
            #         dzUx_array.append(dzUx_it)
            #         print(cc,time.time()-start_time)
            #         cc+=1
            #     dzUx[:] = np.array(dzUx_array); del dzUx_array


        #elif Variable[0:2] == "dr":
            # drUx_array = []
            # print("drUx calcs")
            # with Pool() as pool:
            #     cc = 1
            #     for drUx_it in pool.imap(drUx_calc, np.arange(0,time_idx)):
            #         drUx_array.append(drUx_it)
            #         print(cc,time.time()-start_time)
            #         cc+=1
            #     drUx[:] = np.array(drUx_array); del drUx_array


    del u

#     print(ncfile.groups)
#     ic+=1

# print(ncfile)
# ncfile.close()

print("line 308",time.time() - start_time)