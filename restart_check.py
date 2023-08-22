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

# def coriolis_twist(u,v):
#     twist = np.arctan(np.true_divide(v,u))

#     return twist


# def magnitude_horizontal_velocity(u,v,twist,x,zs,h):

#     mag_horz_vel = []
#     for i in np.arange(0,len(zs)):
#         u_i = u[i*x:(i+1)*x]; v_i = v[i*x:(i+1)*x]
#         height = zs[i]
#         h_idx = np.searchsorted(h,height,side="left")
#         if h_idx > 127:
#             h_idx = 127
#         mag_horz_vel_i = np.add( np.multiply(u_i,np.cos(twist[h_idx])) , np.multiply( v_i,np.sin(twist[h_idx])) )
#         mag_horz_vel.extend(mag_horz_vel_i)
#     mag_horz_vel = np.array(mag_horz_vel)

#     return mag_horz_vel

# #loop over coordinates with counter
# #calc range of y and z coordinates rotor falls into
# #if true sum velocity
# #divide by len
# def Ux_it_offset(it):

#     Hvelmag = hvelmag[it]
#     Hvelmag = np.reshape(Hvelmag,(y,x))

#     Ux_rotor = []
#     for j in np.arange(0,len(ys)):
#         for k in np.arange(0,len(zs)):
#             r = np.sqrt(ys[j]**2 + zs[k]**2)
#             if r <= 63:
#                 Ux_rotor.append(Hvelmag[j,k])

#     return np.average(Ux_rotor)


# def Uz_it_offset(it):

#     Velocityz = velocityz[it]
#     Velocityz = np.reshape(Velocityz,(y,x))

#     Uz_rotor = []
#     for j in np.arange(0,len(ys)):
#         for k in np.arange(0,len(zs)):
#             r = np.sqrt(ys[j]**2 + zs[k]**2)
#             if r <= 63 and r >= 1.5:
#                 Uz_rotor.append(Velocityz[j,k])

#     return np.average(Uz_rotor)


# def IA_it_offset(it):


#     hvelmag_interp = hvelmag[it]
#     hvelmag_interp = np.reshape(hvelmag_interp,(y,x))
#     f = interpolate.interp2d(ys,zs,hvelmag_interp)

#     Hvelmag = hvelmag[it]
#     Hvelmag = np.reshape(Hvelmag,(y,x))

#     IA = 0
#     for j in np.arange(0,len(ys)):
#         for k in np.arange(0,len(zs)):
#             r = np.sqrt(ys[j]**2 + zs[k]**2)
#             if r <= 63 and r >= 1.5:

#                 delta_Ux_i = delta_Ux(r,j,k,f,Hvelmag)
#                 IA += r * delta_Ux_i * dA
#     return IA


# def delta_Ux(r,j,k,f,Hvelmag):

#     Y_0 = ys[j]
#     Z_0 = zs[k]

#     theta = np.arccos(Y_0/r)

#     if theta + ((2*np.pi)/3) > (2*np.pi):
#         theta_1 = theta +(2*np.pi)/3 - (2*np.pi)
#     else:
#         theta_1 = theta + (2*np.pi)/3

#     Y_1 = r*np.cos(theta_1)
#     Z_1 = r*np.sin(theta_1)


#     if theta - ((2*np.pi)/3) < 0:
#         theta_2 = theta - ((2*np.pi)/3) + (2*np.pi)
#     else:
#         theta_2 = theta - ((2*np.pi)/3)

#     Y_2 = r*np.cos(theta_2)
#     Z_2 = r*np.sin(theta_2)

#     Ux_0 =  Hvelmag[j][k]
#     Ux_1 =  f(Y_1,Z_1)
#     Ux_2 =  f(Y_2,Z_2)

#     delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

#     return delta_Ux


# def HV_it_offset(it):

#     hvelmag_interp = hvelmag[it]
#     hvelmag_interp = np.reshape(hvelmag_interp,(y,x))
#     f = interpolate.interp2d(ys,zs,hvelmag_interp)

#     HV = f(rotor_coordiates[1],rotor_coordiates[2])

#     return HV



# #defining twist angles with height from precursor
# precursor = Dataset("./abl_statistics60000.nc")
# mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
# t_start = np.searchsorted(precursor.variables["time"],32300)
# t_end = np.searchsorted(precursor.variables["time"],33500)
# u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
# v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
# h = mean_profiles["h"][:]
# twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
# del precursor

OF_files = ["../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out",
             "../../NREL_5MW_MCBL_R_CRPM_100320/NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out"]

#AMR_files = ["./sampling65000.nc","./sampling100320.nc"]

Variables = ["Wind1VelX","RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[m/s]","[N]","[N-m]","[N-m]","[rads]"]


for iv in np.arange(0,len(Variables)):
    Variable = Variables[iv]
    unit = units[iv]

    fig = plt.figure(figsize=(14,8))

    ic = 0
    for file in OF_files:


        #openfast data
        da = io.fast_output_file.FASTOutputFile(file).toDataFrame()

        restart_time = 137.748
        Time_OF = np.array(da["Time_[s]"])
        if ic == 1:
            Time_OF = Time_OF+restart_time


        if Variable == "MR" or Variable == "Theta":
            signaly = np.array(da["RtAeroMyh_[N-m]"])
            signalz = np.array(da["RtAeroMzh_[N-m]"])
            
            if Variable == "MR":
                signal = np.sqrt( np.square(signaly) + np.square(signalz) ) 
            elif Variable == "Theta": 
                signal = np.arctan2(signalz,signaly)

        else:
            txt = "{0}_{1}".format(Variable,units[iv])
            signal = np.array(da[txt])

        plt.plot(Time_OF,signal)

        ic+=1
        print("line 195",time.time()-start_time)

    plt.xlabel("Time [s]")
    plt.ylabel("{0} {1}".format(Variable,unit))
    plt.xlim([120,160])
    plt.savefig("./{0}".format(Variable))
    plt.close(fig)

        


# print("line 235",time.time()-start_time)

# #sampling data
# a = Dataset("./sampling_r_0.0.nc")

# #sampling time
# Time_sample = np.array(a.variables["time"])
# Time_sample = Time_sample - Time_sample[0]

# plot_all_times = True
# if plot_all_times == False:
#     tstart = 50
#     tend = 150
#     tstart_sample_idx = np.searchsorted(Time_sample,tstart)
#     tend_sample_idx = np.searchsorted(Time_sample,tend)
# else:
#     tstart_sample_idx = 0
#     tend_sample_idx = np.searchsorted(Time_sample,Time_sample[-1])

# print(tend_sample_idx)

# time_sampling[:] = Time_sample[tstart_sample_idx:tend_sample_idx]

# print("line 254", time.time()-start_time)

# offsets = [0.0, -63.0, -126]
# group_label = [0.0, 63.0, 126.0]

# ic = 0
# for offset in offsets:

#     a = Dataset("./sampling_r_{}.nc".format(offset))

#     group = ncfile.createGroup("{}".format(group_label[ic]))

#     Ux = group.createVariable("Ux", np.float64, ('sampling'),zlib=True)
#     Uz = group.createVariable("Uz", np.float64, ('sampling'),zlib=True)
#     IA = group.createVariable("IA", np.float64, ('sampling'),zlib=True)
#     HV = group.createVariable("HV", np.float64, ('sampling'),zlib=True)

#     p_rotor = a.groups["p_r"]

#     Variables = ["Ux_{0}".format(offset), "Uz_{0}".format(offset), "IA_{0}".format(offset), "HV_{0}".format(offset)]
#     units = ["[m/s]", "[m/s]", "[$m^4/s$]"]


#     x = p_rotor.ijk_dims[0] #no. data points
#     y = p_rotor.ijk_dims[1] #no. data points

#     coordinates = p_rotor.variables["coordinates"]

#     xo = coordinates[0:x,0]
#     yo = coordinates[0:x,1]
#     zo = np.linspace(p_rotor.origin[2],p_rotor.axis2[2],y)

#     rotor_coordiates = [2560,2560,90]

#     x_trans = xo - rotor_coordiates[0]
#     y_trans = yo - rotor_coordiates[1]

#     phi = np.radians(-29)
#     xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
#     ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
#     zs = zo - rotor_coordiates[2]


#     dy = ys[1]-ys[0]
#     dz = zs[1] - zs[0]
#     dA = dy * dz

#     print("line 295",time.time()-start_time)


#     def velocity_field(it):

#         hvelmag_it = magnitude_horizontal_velocity(velocityx[it],velocityy[it],twist,x,zs,h)

#         return hvelmag_it
        


#     #velocity field
#     velocityx = p_rotor.variables["velocityx"]
#     velocityy = p_rotor.variables["velocityy"]
#     velocityz = p_rotor.variables["velocityz"]
#     hvelmag = []
#     with Pool() as pool:
#         for hvelmag_it in pool.imap(velocity_field,np.arange(tstart_sample_idx,tend_sample_idx)):
            
#             hvelmag.append(hvelmag_it)
#             print(len(hvelmag),time.time()-start_time)

#     np.array(hvelmag)

#     print("line 323",np.shape(hvelmag))

#     for iv in np.arange(0,len(Variables)):
#         Variable = Variables[iv]
#         print(Variable[0:2],"_",Variable[3:])

#         if Variable[0:2] == "Ux":
#             Ux_it = []
#             print("Ux calcs",len(np.arange(tstart_sample_idx,tend_sample_idx)))
#             with Pool() as pool:
#                 for Ux_i in pool.imap(Ux_it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
#                     Ux_it.append(Ux_i)
#                     print(len(Ux_it),time.time()-start_time)
#                 Ux_it = np.array(Ux_it)
#                 Ux[:] = Ux_it; del Ux_it

#         elif Variable[0:2] == "Uz":
#             Uz_it = []
#             print("Uz calcs",len(np.arange(tstart_sample_idx,tend_sample_idx)))
#             with Pool() as pool:
#                 for Uz_i in pool.imap(Uz_it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
#                     Uz_it.append(Uz_i)
#                     print(len(Uz_it),time.time()-start_time)
#                 Uz_it = np.array(Uz_it)
#                 Uz[:] = Uz_it; del Uz_it

#         elif Variable[0:2] == "IA":
#             IA_it = []
#             print("IA calcs",len(np.arange(tstart_sample_idx,tend_sample_idx)))
#             with Pool() as pool:
#                 for IA_i in pool.imap(IA_it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
#                     IA_it.append(IA_i)
#                     print(len(IA_it),time.time()-start_time)
#                 IA_it = np.array(IA_it)
#                 IA[:] = IA_it; del IA_it

#         elif Variable[0:2] == "HV":
#             HV_it = []
#             print("HV calcs",len(np.arange(tstart_sample_idx,tend_sample_idx)))
#             with Pool() as pool:
#                 for HV_i in pool.imap(HV_it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
#                     HV_it.append(HV_i)
#                     print(len(HV_it),time.time()-start_time)
#                 HV_it = np.array(HV_it)
#                 HV[:] = HV_it; del HV_it

#     print(ncfile.groups)
#     ic+=1
# print(ncfile)
# ncfile.close()

# print("line 352",time.time() - start_time)