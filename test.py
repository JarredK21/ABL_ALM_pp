from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob 
import os
from matplotlib import cm
import operator
import math
import sys
import time
from multiprocessing import Pool
import pyFAST.input_output as io
from scipy import interpolate


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_u = interpolate.interp1d(h,u_mean_profile)
    vel = []
    fluct_vel = []
    for i in np.arange(0,len(zs)):

        u_i = u[it,i*x:(i+1)*x]; v_i = v[it,i*x:(i+1)*x]


        if zs[i] < h[0]:
            twist_h = f(h[0])
            u_mean = f_u(h[0])

        elif zs[i] > h[-1]:
            twist_h = f(h[-1])
            u_mean = f_u(h[-1])
        else:
            twist_h = f(zs[i])
            u_mean = f_u(zs[i])


        vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)


        fluc_vel_i = np.subtract(vel_i,u_mean)
        vel.extend(vel_i)
        fluct_vel.extend(fluc_vel_i)
    vel = np.array(vel)
    fluct_vel = np.array(fluct_vel)
    return vel,fluct_vel


def Horz_vel_2(it):
    f = interpolate.interp1d(h,twist)
    f_u = interpolate.interp1d(h,u_mean_profile)

    u_i = u_H[it]; v_i = v_H[it]
    twist_h = f(90); u_mean = f_u(90)

    vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)

    fluc_vel_i = np.subtract(vel_i,u_mean)

    return fluc_vel_i


def blade_positions(it):

    R = 63
    Az = -Azimuth[it]
    Y = [2560]; Y2 = [2560]; Y3 = [2560]
    Z = [90]; Z2 = [90]; Z3 = [90]

    Y.append(Y[0]+R*np.sin(Az))
    Z.append(Z[0]+R*np.cos(Az))

    Az2 = Az-(2*np.pi)/3
    if Az2 < -2*np.pi:
        Az2 += (2*np.pi)
    
    Az3 = Az-(4*np.pi)/3
    if Az2 < -2*np.pi:
        Az2 += (2*np.pi)

    Y2.append(Y2[0]+R*np.sin(Az2))
    Z2.append(Z2[0]+R*np.cos(Az2))

    Y3.append(Y3[0]+R*np.sin(Az3))
    Z3.append(Z3[0]+R*np.cos(Az3))

    return Y, Z, Y2, Z2, Y3, Z3


start_time = time.time()


precursor_df = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor_df.variables["time"])
mean_profiles = precursor_df.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor_df.variables["time"],38200)
u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation

u_mean_profile = (u * np.cos(twist)) + (v * np.sin(twist))

del precursor_df; del Time_pre; del mean_profiles; del t_start; del u; del v



df = Dataset("Dataset.nc")
Time_OF = np.array(df.variables["Time_OF"])
OF_vars = df.groups["OpenFAST_Variables"]

Azimuth = np.radians(np.array(OF_vars.variables["Azimuth"]))

Azimuth = Azimuth+np.radians(334)

del df; del OF_vars

a = Dataset("./sampling_r_-63.0.nc")

p = a.groups["p_r"]

#time options
Time = np.array(a.variables["time"])
Time = Time - Time[0]
tstart = 700
tstart_idx = np.searchsorted(Time,tstart)
tend = 1000
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]

f = interpolate.interp1d(Time_OF,Azimuth)
Azimuth = f(Time)
print(len(Azimuth))
print(len(Time))


x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points


normal = 29.29

#define plotting axes
coordinates = np.array(p.variables["coordinates"])


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
zs = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y)


u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

u[u<0]=0; v[v<0]=0 #remove negative velocities

u_hvel = [];u_pri = []
ix=0
with Pool() as pool:
    for u_hvel_it, u_hvel_pri_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        u_pri.append(u_hvel_pri_it)
        print(ix)
        ix+=1

u = np.array(u_pri); del u_hvel; del u_pri; del v

                         

cmin = -7
cmax = 7


nlevs = int((cmax-cmin)/2)
if nlevs>abs(cmin) or nlevs>cmax:
    nlevs = min([abs(cmin),cmax])+1

levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
levels = np.concatenate((levs_min,levs_max[1:]))


print("line 370",levels)


a = Dataset("./sampling_l_85.nc")

p = a.groups["p_l"]

x_H = p.ijk_dims[0] #no. data points
y_H = p.ijk_dims[1] #no. data points

xs_H = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x_H)
ys_H = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y_H)


u_H = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v_H = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

u_H[u_H<0]=0; v_H[v_H<0]=0 #remove negative velocities

u_pri = []
ix=0
with Pool() as pool:
    for u_hvel_pri_it in pool.imap(Horz_vel_2,Time_steps):
        
        u_pri.append(u_hvel_pri_it)
        print(ix)
        ix+=1

u_H = np.array(u_pri); del u_pri; del v_H


cmin = -7
cmax = 7


nlevs = int((cmax-cmin)/2)
if nlevs>abs(cmin) or nlevs>cmax:
    nlevs = min([abs(cmin),cmax])+1

levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
levels_H = np.concatenate((levs_min,levs_max[1:]))



plt.rcParams['font.size'] = 40

def Update(it):

    U = u[it] #velocity time step it
    U_H = u_H[it]
    
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    u_plane = U.reshape(y,x)
    X,Y = np.meshgrid(ys,zs)


    u_plane_H = U_H.reshape(x_H,y_H)
    X_H,Y_H = np.meshgrid(xs_H,ys_H)

    T = Time[it]


    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(100,30))
    plt.rcParams['font.size'] = 40

    cs = ax1.contourf(X,Y,u_plane,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    ax1.set_xlabel("y' axis (rotor frame of reference) [m]")
    ax1.set_ylabel("z' axis (rotor frame of reference) [m]")

    cb = plt.colorbar(cs)

    ax1.set_title("Rotor plane: -63.0m offset\nFluctuating streamwise velocity $u'_{x'}$")

    cz = ax2.contourf(X_H,Y_H,u_plane_H,levels=levels_H, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    ax2.set_xlabel("x axis [m]")
    ax2.set_ylabel("y axis [m]")

    ax2.set_title("Horizontal plane: hub height 90m\nFluctuating streamwise velocity $u'_{x'}$")

    x_lims = [2524.5,2585.5]; y_lims = [2615.1,2504.9]
    ax2.plot(x_lims,y_lims,linewidth=1.0,color="k")



    YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

    ax1.plot(YB1,ZB1,color="k",linewidth = 1)
    ax1.plot(YB2,ZB2,color="r",linewidth = 1)
    ax1.plot(YB3,ZB3,color="b",linewidth = 1)  


    fig.suptitle("Time: {}s".format(T))
    plt.savefig("ISOplots/"+Time_idx+".png")
    plt.cla()
    cb.remove()
    plt.close(fig)

    return T



with Pool() as pool:
    for T in pool.imap(Update,Time_steps):

        print(T,time.time()-start_time)