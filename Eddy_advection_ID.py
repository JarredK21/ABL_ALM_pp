from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
import time
from multiprocessing import Pool
from scipy import interpolate
from matplotlib.patches import Circle
from math import ceil



def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)

    height = offset+7.5
    twist_h = f(height)
    ux_mean = f_ux(height)
    
    mag_horz_vel = u[it]*np.cos(twist_h) + v[it]*np.sin(twist_h)
    mag_fluc_horz_vel = np.array(np.subtract(mag_horz_vel,ux_mean))

    return mag_fluc_horz_vel


def Update(it):

    U = u[it] #velocity time step it
    
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)



    Z = U.reshape(x,y)
    X,Y = np.meshgrid(xs,ys)

    fu = interpolate.interp2d(X[xsminidx:xsmaxidx,ysminidx:ysmaxidx],Y[xsminidx:xsmaxidx,ysminidx:ysmaxidx],Z[xsminidx:xsmaxidx,ysminidx:ysmaxidx])
    Zrotor = []
    for ix,iy in zip(xrotor,yrotor):
        Zrotor.append(fu(ix,iy)[0])
    print(Zrotor)
    CS = plt.contour(X, Y, Z, levels=levels_pos)
    CZ = plt.contour(X,Y,Z, levels=levels_neg)


    HSR = []
    LSS = []
    for i in np.arange(0,len(Zrotor)):
        if Zrotor[i] >= 0.7:
            HSR.append([xrotor[i],yrotor[i]])
        elif Zrotor[i] <= -0.7:
            LSS.append([xrotor[i],yrotor[i]]) 
    print(HSR)
    print(LSS)
    lines = CS.allsegs[0] #plot only threshold velocity
    for line in lines:
        Xline, Yline = line[:,0], line[:,1]

        Xlinemax = np.max(Xline); Xlinemin = np.min(Xline)
        Ylinemax = np.max(Yline); Ylinemin = np.min(Yline)
        print(Xlinemin,Xlinemax)
        print(Ylinemin,Ylinemax)
        for HSR_i in HSR:
            if Xlinemin <= HSR_i[0] <= Xlinemax and Ylinemin <= HSR_i[1] <= Ylinemax:
                plt.plot(Xline,Yline,"-r",linewidth=3)
                break

    
    lines = CZ.allsegs[-1] #plot only threshold velocity
    for line in lines:
        Xline, Yline = line[:,0], line[:,1]

        Xlinemax = np.max(Xline); Xlinemin = np.min(Xline)
        Ylinemax = np.max(Yline); Ylinemin = np.min(Yline)
        print(Xlinemin,Xlinemax)
        print(Ylinemin,Ylinemax)
        for LSS_i in LSS:
            if Xlinemin <= LSS_i[0] <= Xlinemax and Ylinemin <= LSS_i[1] <= Ylinemax:
                plt.plot(Xline,Yline,"--b",linewidth=3)
                break            


    T = Time[it]

    fig = plt.figure(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = plt.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    plt.xlim([2000,3000]); plt.ylim([2000,3000])
    plt.plot(xrotor,yrotor,linewidth=1.0,color="k")
    plt.xlabel("x axis [m]")
    plt.ylabel("y axis [m]")

    cb = plt.colorbar(cs)

    Title = "Horizontal Plane. \nFluctuating velocity Streamwise velocity [m/s]: Offset = {}, Time = {}[s]".format(offset,round(T,4))
    filename = "Horz_Fluc_vel_Horz_vel_{}_{}.png".format(offset,Time_idx)

    plt.title(Title)
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return T



start_time = time.time()

#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38200)
u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 67", time.time()-start_time)

#directories
in_dir = "./"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)

# csv_out_dir = in_dir + "csv_files_2/"
# isExist = os.path.exists(csv_out_dir)
# if isExist == False:
#     os.makedirs(csv_out_dir)

offsets = [22.5,85,142.5]
for offset in offsets:

    a = Dataset("./sampling_l_{}.nc".format(offset))

    #time options
    Time = np.array(a.variables["time"])
    dt = Time[1] - Time[0]
    tstart = 38200
    tstart_idx = np.searchsorted(Time,tstart)
    tend = 39201
    tend_idx = np.searchsorted(Time,tend)
    Time_steps = np.arange(0, tend_idx-tstart_idx)
    Time = Time[tstart_idx:tend_idx]


    #rotor data
    p = a.groups["p_l"]; del a

    x = p.ijk_dims[0] #no. data points
    y = p.ijk_dims[1] #no. data points

    coordinates = np.array(p.variables["coordinates"])


    xs = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x)
    ys = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y)
    
    xsminidx = np.searchsorted(xs,2455); xsmaxidx = np.searchsorted(xs,2555)
    ysminidx = np.searchsorted(ys,2455); ysmaxidx = np.searchsorted(ys,2605)


    #velocity field
    u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
    v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
    del p

    u[u<0]=0; v[v<0]=0 #remove negative velocities

    with Pool() as pool:
        u_pri = []
        for u_fluc_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
            
            u_pri.append(u_fluc_hvel_it)
            print(len(u_pri),time.time()-start_time)
    u = np.array(u_pri); del u_pri; del v

    print("line 139",time.time()-start_time)

    #find vmin and vmax for isocontour plots            
    #min and max over data
    cmin = math.floor(np.min(u))
    cmax = math.ceil(np.max(u))


    nlevs = int((cmax-cmin)/2)
    if nlevs>abs(cmin) or nlevs>cmax:
        nlevs = min([abs(cmin),cmax])+1

    levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
    levels = np.concatenate((levs_min,levs_max[1:]))
    print("line 153", levels)

    #define thresholds with number of increments
    levels_pos = np.linspace(0.7,cmax,4)
    print("line 157", levels_pos)
    levels_neg = np.linspace(cmin,-0.7,4)
    print("line 159", levels_neg)

    if offset == 85:
        xrotor = np.linspace(2474.36,2535.44,201)
        yrotor = np.linspace(2584.56,2474.36,201)
    else:
        xrotor = np.linspace(2486.27,2523.53,101)
        yrotor = np.linspace(2563.06,2495.86,101)

    folder = out_dir+"Horizontal_Plane_Fluctutating_horz_{}_advection/".format(offset)
    isExist = os.path.exists(folder)
    if isExist == False:
        os.makedirs(folder)

    for it in np.arange(0,1):
        T = Update(it)
        print(T)
    # with Pool() as pool:
    #     for T in pool.imap(Update,Time_steps):
    #         print(offset)
    #         print(T,time.time()-start_time)




