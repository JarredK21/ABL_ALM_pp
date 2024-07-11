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
import math


start_time = time.time()


def Rotor_Avg_calc(it):

    U = u[it]

    Ux_rotor = []
    Iy = 0
    Iz = 0
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            Ux_rotor.append(U[ijk])
            Iy+=U[ijk]*k*dA
            Iz+=U[ijk]*j*dA

        ijk+=1
    return np.average(Ux_rotor),Iy,Iz


def IA_calc(it):

    U = np.reshape(u[it], (y,x))

    f = interpolate.interp2d(Y,Z,U,kind="linear")

    IA = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            delta_Ux_i = delta_Ux(j,k,r,f)
            IA += r * delta_Ux_i * dA
    return IA


def delta_Ux(j,k,r,f):

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

    Ux_0 = f(j,k)
    Ux_1 = f(Y_1,Z_1)
    Ux_2 = f(Y_2,Z_2)

    delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

    return delta_Ux


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    mag_horz_vel = []
    mag_fluc_horz_vel = []
    for i in np.arange(0,len(zs)):
        u_i = u[it,i*x:(i+1)*x]; v_i = v[it,i*x:(i+1)*x]
        if zs[i] < h[0]:
            twist_h = f(h[0])
            ux_mean = f_ux(h[0])

        elif zs[i] > h[-1]:
            twist_h = f(h[-1])
            ux_mean = f_ux(h[-1])
        else:
            twist_h = f(zs[i])
            ux_mean = f_ux(zs[i])

        mag_horz_vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
        mag_fluc_horz_vel_i = np.subtract(mag_horz_vel_i,ux_mean)
        mag_horz_vel.extend(mag_horz_vel_i)
        mag_fluc_horz_vel.extend(mag_fluc_horz_vel_i)
    mag_horz_vel = np.array(mag_horz_vel)
    mag_fluc_horz_vel = np.array(mag_fluc_horz_vel)
    return mag_horz_vel,mag_fluc_horz_vel


def IHL_calc(it):

    U = u[it]
    U_pri = u_pri[it]


    AH = 0; AL = 0; AI = 0
    IyH = 0; IyL = 0; IyI = 0; Iy = 0
    IzH = 0; IzL = 0; IzI = 0; Iz = 0
    UxH = []; UxL = []; UxI = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)

        if r <= 63 and r > 1.5:

            Uijk = U[ijk]
            U_pri_ijk = U_pri[ijk]
            if cmin<=Uijk<=cmax:
                Iy+=(Uijk*k*dA)
                Iz+=(Uijk*j*dA)

                if U_pri_ijk >= 0.7:
                    AH+=dA
                    IyH+=(Uijk*k*dA)
                    IzH+=(Uijk*j*dA)
                    UxH.append(Uijk)
                elif U_pri_ijk <= -0.7:
                    AL+=dA
                    IyL+=(Uijk*k*dA)
                    IzL+=(Uijk*j*dA)
                    UxL.append(Uijk)
                else:
                    AI+=dA
                    IyI+=(Uijk*k*dA)
                    IzI+=(Uijk*j*dA)
                    UxI.append(Uijk)
        ijk+=1

    if len(UxH) > 0:
        UxH_avg = np.average(UxH)
    else:
        UxH_avg = 0

    if len(UxL) > 0:
        UxL_avg = np.average(UxL)
    else:
        UxL_avg = 0

    if len(UxI) > 0:
        UxI_avg = np.average(UxI)
    else:
        UxI_avg = 0


    return AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy,Iz,UxH_avg,UxL_avg,UxI_avg


def dUx_calc(it):
    U = u[it]
    u_plane = U.reshape(y,x)

    du_dy = []
    for k in np.arange(0,len(u_plane)):
        du_dy_k = np.subtract(u_plane[k][1:],u_plane[k][:-1])/dy
        du_dy_k = np.insert(du_dy_k,0,du_dy_k[0])
        du_dy.append(du_dy_k)
    du_dy = np.array(du_dy).flatten()

    du_dz = []
    for j in np.arange(0,x):
        du_dz_j = np.subtract(u_plane[1:,j],u_plane[:-1,j])/dz
        du_dz_j = np.insert(du_dz_j,0,du_dz_j[0])
        du_dz.append(du_dz_j)
    du_dz = np.array(du_dz).reshape(y,x).flatten()

    du_dr = np.sqrt(np.add(np.square(du_dy),np.square(du_dz)))

    ijk = 0
    du_dy_avg = []
    du_dz_avg = []
    du_dr_avg = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            du_dy_avg.append(du_dy[ijk])
            du_dz_avg.append(du_dz[ijk])
            du_dr_avg.append(du_dr[ijk])

    return np.average(du_dy_avg),np.average(du_dz_avg),np.average(du_dr_avg)


def Ejection_heights_calc(it):

    #algorithm for ejections

    U_pri = u_pri[it] #velocity time step it

    u_plane = U_pri.reshape(y,x)

    H = np.zeros(len(YS))
    for j in np.arange(0,len(YS)):
        for k in np.arange(0,len(ZS)-1):

            if u_plane[k+1,j] > threshold:
                H[j] = ZS[k]
                break

    return H


def isInside(x, y):
     
    if ((x - 2560) * (x - 2560) +
        (y - 90) * (y - 90) < 63 * 63):
        return True
    else:
        return False
    

def Ejections_Asymmetry_calc(it):

    H = H_array[it]
    y = []; z = []
    for j in np.arange(0,len(YS)):
        #is coordinate inside rotor disk
        cc = isInside(YS[j],H[j])

        if cc == True:

            z_i = np.min( np.roots([1,-180,(90**2-63**2+(YS[j]-2560)**2)]) )
            y.append(YS[j]); y.append(YS[j])
            z.append(z_i); z.append(H[j])


        #is coordinate above rotor disk so it is still covering it
        elif YS[j] > 2497 and YS[j] < 2623 and H[j] > 153:
            z_i = np.roots([1,-180,(90**2-63**2+(YS[j]-2560)**2)])
            y.append(YS[j]); y.append(YS[j])
            z.append(np.min(z_i)); z.append(np.max(z_i))

    if len(y) > 0:
        Iy,Iz = I_it_calc(it,y,z)
    else:
        Iy = 0; Iz = 0

    return Iy, Iz


def I_it_calc(it,y,z):

    Iy = 0
    Iz = 0
    ijk = 0
    for j,k in zip(ys,zs):
    
        if (ys[ijk]+rotor_coordiates[1]) in y:
            idx = y.index(ys[ijk]+rotor_coordiates[1])

            if z[idx] <= (zs[ijk]+rotor_coordiates[2]) <= z[idx+1]:
                Iy += u[it,ijk]*k*dA
                Iz += u[it,ijk]*j*dA
        ijk+=1

    return Iy,Iz


def Rotor_split_calc(it):

    U = u[it]

    IyL = 0
    IzL = 0
    IyM = 0
    IzM = 0
    IyH = 0
    IzH = 0
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            if k+rotor_coordiates[2] < 69:
                IyL+=U[ijk]*k*dA
                IzL+=U[ijk]*j*dA
            elif 69 <= k+rotor_coordiates[2] <= 111:
                IyM+=U[ijk]*k*dA
                IzM+=U[ijk]*j*dA
            elif 111 < k+rotor_coordiates[2] <= 153:
                IyH+=U[ijk]*k*dA
                IzH+=U[ijk]*j*dA

        ijk+=1
    return IyL,IzL,IyM,IzM,IyH,IzH


def Split_dUx_calc(it):
    U = u[it]
    u_plane = U.reshape(y,x)

    du_dy = []
    for k in np.arange(0,len(u_plane)):
        du_dy_k = np.subtract(u_plane[k][1:],u_plane[k][:-1])/dy
        du_dy_k = np.insert(du_dy_k,0,du_dy_k[0])
        du_dy.append(du_dy_k)
    du_dy = np.array(du_dy).flatten()

    du_dz = []
    for j in np.arange(0,x):
        du_dz_j = np.subtract(u_plane[1:,j],u_plane[:-1,j])/dz
        du_dz_j = np.insert(du_dz_j,0,du_dz_j[0])
        du_dz.append(du_dz_j)
    du_dz = np.array(du_dz).reshape(y,x).flatten()

    du_dr = np.sqrt(np.add(np.square(du_dy),np.square(du_dz)))

    ijk = 0
    du_dyH_avg = []; du_dyM_avg = []; du_dyL_avg = []
    du_dzH_avg = []; du_dzM_avg = []; du_dzL_avg = []
    du_drH_avg = []; du_drM_avg = []; du_drL_avg = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            if k+rotor_coordiates[2] < 69:
                du_dyL_avg.append(du_dy[ijk])
                du_dzL_avg.append(du_dz[ijk])
                du_drL_avg.append(du_dr[ijk])
            elif 69 <= k+rotor_coordiates[2] <= 111:
                du_dyM_avg.append(du_dy[ijk])
                du_dzM_avg.append(du_dz[ijk])
                du_drM_avg.append(du_dr[ijk])
            elif 111 < k+rotor_coordiates[2] <= 153:
                du_dyH_avg.append(du_dy[ijk])
                du_dzH_avg.append(du_dz[ijk])
                du_drH_avg.append(du_dr[ijk])


    return np.average(du_dyL_avg),np.average(du_dzL_avg),np.average(du_drL_avg), np.average(du_dyM_avg),np.average(du_dzM_avg),np.average(du_drM_avg),np.average(du_dyH_avg),np.average(du_dzH_avg),np.average(du_drH_avg)


def IHL_dUx_calc(it):
    U = u[it]
    U_pri = u_pri[it]
    u_plane = U.reshape(y,x)

    du_dy = []
    for k in np.arange(0,len(u_plane)):
        du_dy_k = np.subtract(u_plane[k][1:],u_plane[k][:-1])/dy
        du_dy_k = np.insert(du_dy_k,0,du_dy_k[0])
        du_dy.append(du_dy_k)
    du_dy = np.array(du_dy).flatten()

    du_dz = []
    for j in np.arange(0,x):
        du_dz_j = np.subtract(u_plane[1:,j],u_plane[:-1,j])/dz
        du_dz_j = np.insert(du_dz_j,0,du_dz_j[0])
        du_dz.append(du_dz_j)
    du_dz = np.array(du_dz).reshape(y,x).flatten()

    du_dr = np.sqrt(np.add(np.square(du_dy),np.square(du_dz)))

    ijk = 0
    du_dyH_avg = []; du_dyL_avg = []; du_dyI_avg = []
    du_dzH_avg = []; du_dzL_avg = []; du_dzI_avg = []
    du_drH_avg = []; du_drL_avg = []; du_drI_avg = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:

            Uijk = U[ijk]
            U_pri_ijk = U_pri[ijk]
            if cmin<=Uijk<=cmax:
                if U_pri_ijk >= 0.7:
                    du_dyH_avg.append(du_dy[ijk])
                    du_dzH_avg.append(du_dz[ijk])
                    du_drH_avg.append(du_dr[ijk])
                elif U_pri_ijk <= -0.7:
                    du_dyL_avg.append(du_dy[ijk])
                    du_dzL_avg.append(du_dz[ijk])
                    du_drL_avg.append(du_dr[ijk])
                else:
                    du_dyI_avg.append(du_dy[ijk])
                    du_dzI_avg.append(du_dz[ijk])
                    du_drI_avg.append(du_dr[ijk])
        ijk+=1

    if len(du_dyH_avg) > 0:
        du_dyH_avg = np.average(du_dyH_avg); du_dzH_avg = np.average(du_dzH_avg); du_drH_avg = np.average(du_drH_avg)
    else:
        du_dyH_avg = 0; du_dzH_avg = 0; du_drH_avg = 0

    if len(du_dyL_avg) > 0:
        du_dyL_avg = np.average(du_dyL_avg); du_dzL_avg = np.average(du_dzL_avg); du_drL_avg = np.average(du_drL_avg)
    else:
        du_dyL_avg = 0; du_dzL_avg = 0; du_drL_avg = 0

    if len(du_dyI_avg) > 0:
        du_dyI_avg = np.average(du_dyI_avg); du_dzI_avg = np.average(du_dzI_avg); du_drI_avg = np.average(du_drI_avg)
    else:
        du_dyI_avg = 0; du_dzI_avg = 0; du_drI_avg = 0

    return du_dyH_avg,du_dzH_avg, du_drH_avg, du_dyL_avg, du_dzL_avg, du_drL_avg, du_dyI_avg, du_dzI_avg, du_drI_avg


#directories
in_dir = "./"
out_dir = in_dir


#create netcdf file
ncfile = Dataset(out_dir+"Dataset_2.nc",mode="w",format='NETCDF4')
ncfile.title = "OpenFAST data sampling output"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)
sampling_dim = ncfile.createDimension("sampling",None)

#create variables
time_OF = ncfile.createVariable("Time_OF", np.float64, ('OF',),zlib=True)
time_sampling = ncfile.createVariable("Time_sampling", np.float64, ('sampling',),zlib=True)

print("Outputting openfast variables",time.time()-start_time)
group = ncfile.createGroup("OpenFAST_Variables")

Azimuth = group.createVariable("Azimuth", np.float64, ('OF',),zlib=True)
RtAeroFxh = group.createVariable("RtAeroFxh", np.float64, ('OF',),zlib=True)
RtAeroFyh = group.createVariable("RtAeroFyh", np.float64, ('OF',),zlib=True)
RtAeroFzh = group.createVariable("RtAeroFzh", np.float64, ('OF',),zlib=True)
RtAeroMxh = group.createVariable("RtAeroMxh", np.float64, ('OF',),zlib=True)
RtAeroMyh = group.createVariable("RtAeroMyh", np.float64, ('OF',),zlib=True)
RtAeroMzh = group.createVariable("RtAeroMzh", np.float64, ('OF',),zlib=True)

LSSGagMys = group.createVariable("LSSGagMys", np.float64, ('OF',),zlib=True)
LSSGagMzs = group.createVariable("LSSGagMzs", np.float64, ('OF',),zlib=True)
LSShftMxa = group.createVariable("LSShftMxa", np.float64, ('OF',),zlib=True)
LSSTipMys = group.createVariable("LSSTipMys", np.float64, ('OF',),zlib=True)
LSSTipMzs = group.createVariable("LSSTipMzs", np.float64, ('OF',),zlib=True)
LSShftFxa = group.createVariable("LSShftFxa", np.float64, ('OF',),zlib=True)
LSShftFys = group.createVariable("LSShftFys", np.float64, ('OF',),zlib=True)
LSShftFzs = group.createVariable("LSShftFzs", np.float64, ('OF',),zlib=True)


#openfast data
df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

time_OF[:] = np.array(df["Time_[s]"])

print("line 466",time.time()-start_time)

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

print(ncfile.groups)


#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38000)
t_end = np.searchsorted(precursor.variables["time"],39200)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v



print("line 531", time.time()-start_time)

#sampling data
a = Dataset(in_dir+"sampling_r_-63.0.nc")

#sampling time
Time_sample = np.array(a.variables["time"])
Time_steps = np.arange(0,len(Time_sample))
Time_sample = Time_sample - Time_sample[0]
time_sampling[:] = Time_sample

print("line 542", time.time()-start_time)


p_rotor = a.groups["p_r"]; del a

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

print("line 572",time.time()-start_time)

#velocity field
u = np.array(p_rotor.variables["velocityx"])
v = np.array(p_rotor.variables["velocityy"])
del p_rotor

u[u<0]=0; v[v<0]=0 #remove negative velocities

#fluctuating streamwise velocity
with Pool() as pool:
    u_hvel = []; u_pri = []
    for u_hvel_it,u_fluc_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_pri.append(u_fluc_hvel_it)
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u_pri = np.array(u_pri)
u = np.array(u_hvel); del u_hvel; del v

print("line 592",time.time()-start_time)

YS = ys
ZS = zs


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

print("line 623",time.time()-start_time)


print("Rotor avg calcs",time.time()-start_time)
group = ncfile.createGroup("Rotor_Avg_Variables")

Ux = group.createVariable("Ux", np.float64, ('sampling'),zlib=True)
IA = group.createVariable("IA", np.float64, ('sampling'),zlib=True)
Iy = group.createVariable("Iy", np.float64, ('sampling'),zlib=True)
Iz = group.createVariable("Iz", np.float64, ('sampling'),zlib=True)

Ux_array = []; Iy_array = []; Iz_array = []
with Pool() as pool:
    ix = 1
    for Ux_it,Iy_it,Iz_it in pool.imap(Rotor_Avg_calc, Time_steps):
        Ux_array.append(Ux_it)
        Iy_array.append(Iy_it)
        Iz_array.append(Iz_it)
        print(ix,time.time()-start_time)
        ix+=1
Ux[:] = np.array(Ux_array); del Ux_array
Iy[:] = np.array(Iy_array); del Iy_array
Iz[:] = np.array(Iz_array); del Iz_array



IA_array = []
print("IA calcs")
with Pool() as pool:
    ix = 1
    for IA_it in pool.imap(IA_calc, Time_steps):
        IA_array.append(IA_it)
        print(ix,time.time()-start_time)
        ix+=1
IA[:] = np.array(IA_array); del IA_array

print(ncfile.groups)


#IHL calc
print("IHL variables output",time.time()-start_time)
group = ncfile.createGroup("IHL_Variables")

cmin = math.floor(np.min(u_pri))
cmax = math.ceil(np.max(u_pri))
print("line 244",cmin,cmax)


cmin = math.floor(np.min(u))
cmax = math.ceil(np.max(u))
print("line 249",cmin,cmax)


A_High_arr = []; A_Low_arr = []; A_Int_arr = []
Iy_High_arr = []; Iy_Low_arr = []; Iy_Int_arr = []
Iz_High_arr = []; Iz_Low_arr = []; Iz_Int_arr = []
Ux_High_arr = []; Ux_Low_arr = []; Ux_Int_arr = []
Iy_arr = []; Iz_arr = []
with Pool() as pool:
    ix = 1
    for AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy_it,Iz_it,UxH_it,UxL_it,UxI_it in pool.imap(IHL_calc,Time_steps):        
    
        A_High_arr.append(AH); A_Low_arr.append(AL); A_Int_arr.append(AI)
        Iy_High_arr.append(IyH); Iy_Low_arr.append(IyL); Iy_Int_arr.append(IyI)
        Iz_High_arr.append(IzH); Iz_Low_arr.append(IzL); Iz_Int_arr.append(IzI)
        Ux_High_arr.append(UxH_it); Ux_Low_arr.append(UxL_it); Ux_Int_arr.append(UxI_it)
        Iy_arr.append(Iy_it); Iz_arr.append(Iz_it)

        print(ix,time.time()-start_time)

        ix+=1


Area_high = group.createVariable("Area_high", np.float64, ('sampling',),zlib=True)
Area_low = group.createVariable("Area_low", np.float64, ('sampling',),zlib=True)
Area_int = group.createVariable("Area_int", np.float64, ('sampling',),zlib=True)

Iy_high = group.createVariable("Iy_high", np.float64, ('sampling',),zlib=True)
Iy_low = group.createVariable("Iy_low", np.float64, ('sampling',),zlib=True)
Iy_int = group.createVariable("Iy_int", np.float64, ('sampling',),zlib=True)

Iz_high = group.createVariable("Iz_high", np.float64, ('sampling',),zlib=True)
Iz_low = group.createVariable("Iz_low", np.float64, ('sampling',),zlib=True)
Iz_int = group.createVariable("Iz_int", np.float64, ('sampling',),zlib=True)

Ux_high = group.createVariable("Ux_high", np.float64, ('sampling',),zlib=True)
Ux_low = group.createVariable("Ux_low", np.float64, ('sampling',),zlib=True)
Ux_int = group.createVariable("Ux_int", np.float64, ('sampling',),zlib=True)

Iy = group.createVariable("Iy", np.float64, ('sampling',),zlib=True)
Iz = group.createVariable("Iz", np.float64, ('sampling',),zlib=True)

Area_high[:] = np.array(A_High_arr); del A_High_arr
Area_low[:] = np.array(A_Low_arr); del A_Low_arr
Area_int[:] = np.array(A_Int_arr); del A_Int_arr

Iy_high[:] = np.array(Iy_High_arr); del Iy_High_arr
Iy_low[:] = np.array(Iy_Low_arr); del Iy_Low_arr
Iy_int[:] = np.array(Iy_Int_arr); del Iy_Int_arr

Iz_high[:] = np.array(Iz_High_arr); del Iz_High_arr
Iz_low[:] = np.array(Iz_Low_arr); del Iz_Low_arr
Iz_int[:] = np.array(Iz_Int_arr); del Iz_Int_arr

Ux_high[:] = np.array(Ux_High_arr); del Ux_High_arr
Ux_low[:] = np.array(Ux_Low_arr); del Ux_Low_arr
Ux_int[:] = np.array(Ux_Int_arr); del Ux_Int_arr

Iy[:] = np.array(Iy_arr); del Iy_arr
Iz[:] = np.array(Iz_arr); del Iz_arr


dyUx_high = group.createVariable("dyUx_high", np.float64, ('sampling'),zlib=True)
dzUx_high = group.createVariable("dzUx_high", np.float64, ('sampling'),zlib=True)
drUx_high = group.createVariable("drUx_high", np.float64, ('sampling'),zlib=True)
dyUx_low = group.createVariable("dyUx_low", np.float64, ('sampling'),zlib=True)
dzUx_low = group.createVariable("dzUx_low", np.float64, ('sampling'),zlib=True)
drUx_low = group.createVariable("drUx_low", np.float64, ('sampling'),zlib=True)
dyUx_int = group.createVariable("dyUx_int", np.float64, ('sampling'),zlib=True)
dzUx_int = group.createVariable("dzUx_int", np.float64, ('sampling'),zlib=True)
drUx_int = group.createVariable("drUx_int", np.float64, ('sampling'),zlib=True)

print(ncfile.groups)


#IHL gradients
print("IHL gradient output",time.time()-start_time)
dyUx_high_array = []; dyUx_low_array = []; dyUx_int_array = []
dzUx_high_array = []; dzUx_low_array = []; dzUx_int_array = []
drUx_high_array = []; drUx_low_array = []; drUx_int_array = []
print("dUx calcs")
with Pool() as pool:
    ix = 1
    for dyUx_high_it,dzUx_high_it,drUx_high_it,dyUx_low_it,dzUx_low_it,drUx_low_it,dyUx_int_it,dzUx_int_it,drUx_int_it in pool.imap(IHL_dUx_calc, Time_steps):
        dyUx_high_array.append(dyUx_high_it)
        dzUx_high_array.append(dzUx_high_it)
        drUx_high_array.append(drUx_high_it)
        dyUx_low_array.append(dyUx_low_it)
        dzUx_low_array.append(dzUx_low_it)
        drUx_low_array.append(drUx_low_it)
        dyUx_int_array.append(dyUx_int_it)
        dzUx_int_array.append(dzUx_int_it)
        drUx_int_array.append(drUx_int_it)
        print(ix,time.time()-start_time)
        ix+=1

dyUx_high[:] = np.array(dyUx_high_array); del dyUx_high_array
dzUx_high[:] = np.array(dzUx_high_array); del dzUx_high_array
drUx_high[:] = np.array(drUx_high_array); del drUx_high_array
dyUx_low[:] = np.array(dyUx_low_array); del dyUx_low_array
dzUx_low[:] = np.array(dzUx_low_array); del dzUx_low_array
drUx_low[:] = np.array(drUx_low_array); del drUx_low_array
dyUx_int[:] = np.array(dyUx_int_array); del dyUx_int_array
dzUx_int[:] = np.array(dzUx_int_array); del dzUx_int_array
drUx_int[:] = np.array(drUx_int_array); del drUx_int_array

print(ncfile.groups)


#rotor gradients calc
print("Rotor gradients output",time.time()-start_time)
group = ncfile.createGroup("Rotor_Gradients")

dyUx = group.createVariable("dyUx", np.float64, ('sampling'),zlib=True)
dzUx = group.createVariable("dzUx", np.float64, ('sampling'),zlib=True)
drUx = group.createVariable("drUx", np.float64, ('sampling'),zlib=True)


dyUx_array = []
dzUx_array = []
drUx_array = []
print("dUx calcs")
with Pool() as pool:
    ix = 1
    for dyUx_it,dzUx_it,drUx_it in pool.imap(dUx_calc, Time_steps):
        dyUx_array.append(dyUx_it)
        dzUx_array.append(dzUx_it)
        drUx_array.append(drUx_it)
        print(ix,time.time()-start_time)
        ix+=1
dyUx[:] = np.array(dyUx_array); del dyUx_array
dzUx[:] = np.array(dzUx_array); del dzUx_array
drUx[:] = np.array(drUx_array); del drUx_array


print(ncfile.groups)



#Ejections calc
print("Ejection outputs",time.time()-start_time)
group = ncfile.createGroup("Ejections_Variables")

#create global dimensions
y_dim = ncfile.createDimension("num_points",None)

y_locs = group.createVariable("ys", np.float64, ('num_points',),zlib=True)
y_locs[:] = YS

#thresholds to output data
thresholds = [-5.0,-4.5,-4.0,-3.5,-3.0,-2.5,-2.0,-1.4]


for threshold in thresholds:

    print("line 293",threshold)

    group_inner = group.createGroup("{}".format(abs(threshold)))

    H_ejection = group_inner.createVariable("Height_ejection", np.float64, ('sampling','num_points'),zlib=True)

    H_array = []
    ix = 1
    with Pool() as pool:
        for H_it in pool.imap(Ejection_heights_calc,Time_steps):

            H_array.append(H_it)

            print(ix,time.time()-start_time)

            ix+=1

    H_ejection[:] = np.array(H_array)

    print(group.groups)


    Iy_ejection = group_inner.createVariable("Iy", np.float64, ('sampling'),zlib=True)
    Iz_ejection = group_inner.createVariable("Iz", np.float64, ('sampling'),zlib=True)

    print("line 853",threshold)

    Iy_array = []
    Iz_array = []
    ix = 1
    with Pool() as pool:
        for Iy_it,Iz_it in pool.imap(Ejections_Asymmetry_calc,Time_steps):

            Iy_array.append(Iy_it); Iz_array.append(Iz_it)

            print(ix,time.time()-start_time)

            ix+=1
    
    Iy_ejection[:] = np.array(Iy_array); del Iy_array
    Iz_ejection[:] = np.array(Iz_array); del Iz_array
    del H_array
    
    print(group.groups)

print(ncfile.groups)


#Separating rotor into thirds
print("Separating rotor outputs",time.time()-start_time)
group = ncfile.createGroup("Split_rotor_Variables")

IyL = group.createVariable("IyL", np.float64, ('sampling'),zlib=True)
IzL = group.createVariable("IzL", np.float64, ('sampling'),zlib=True)
IyM = group.createVariable("IyM", np.float64, ('sampling'),zlib=True)
IzM = group.createVariable("IzM", np.float64, ('sampling'),zlib=True)
IyH = group.createVariable("IyH", np.float64, ('sampling'),zlib=True)
IzH = group.createVariable("IzH", np.float64, ('sampling'),zlib=True)

IyL_array = []; IyM_array = []; IyH_array = []
IzL_array = []; IzM_array = []; IzH_array = []
ix = 1
with Pool() as pool:
    for IyL_it,IzL_it,IyM_it,IzM_it,IyH_it,IzH_it in pool.imap(Rotor_split_calc,Time_steps):

        IyL_array.append(IyL_it); IzL_array.append(IzL_it)
        IyM_array.append(IyM_it); IzM_array.append(IzM_it)
        IyH_array.append(IyH_it); IzH_array.append(IzH_it)

        print(ix,time.time()-start_time)

        ix+=1

IyL[:] = np.array(IyL_array); del IyL_array
IzL[:] = np.array(IzL_array); del IzL_array
IyM[:] = np.array(IyM_array); del IyM_array
IzM[:] = np.array(IzM_array); del IzM_array
IyH[:] = np.array(IyH_array); del IyH_array
IzH[:] = np.array(IzH_array); del IzH_array


#separate rotor gradients calc
print("Separate rotor gradient outputs",time.time()-start_time)

dyUxL = group.createVariable("dyUxL", np.float64, ('sampling'),zlib=True)
dzUxL = group.createVariable("dzUxL", np.float64, ('sampling'),zlib=True)
drUxL = group.createVariable("drUxL", np.float64, ('sampling'),zlib=True)
dyUxM = group.createVariable("dyUxM", np.float64, ('sampling'),zlib=True)
dzUxM = group.createVariable("dzUxM", np.float64, ('sampling'),zlib=True)
drUxM = group.createVariable("drUxM", np.float64, ('sampling'),zlib=True)
dyUxH = group.createVariable("dyUxH", np.float64, ('sampling'),zlib=True)
dzUxH = group.createVariable("dzUxH", np.float64, ('sampling'),zlib=True)
drUxH = group.createVariable("drUxH", np.float64, ('sampling'),zlib=True)


dyUxL_array = []; dyUxM_array = []; dyUxH_array = []
dzUxL_array = []; dzUxM_array = []; dzUxH_array = []
drUxL_array = []; drUxM_array = []; drUxH_array = []
print("dUx calcs")
with Pool() as pool:
    ix = 1
    for dyUxL_it,dzUxL_it,drUxL_it,dyUxM_it,dzUxM_it,drUxM_it,dyUxH_it,dzUxH_it,drUxH_it in pool.imap(Split_dUx_calc, Time_steps):
        dyUxL_array.append(dyUxL_it)
        dzUxL_array.append(dzUxL_it)
        drUxL_array.append(drUxL_it)
        dyUxM_array.append(dyUxM_it)
        dzUxM_array.append(dzUxM_it)
        drUxM_array.append(drUxM_it)
        dyUxH_array.append(dyUxH_it)
        dzUxH_array.append(dzUxH_it)
        drUxH_array.append(drUxH_it)
        print(ix,time.time()-start_time)
        ix+=1
dyUxL[:] = np.array(dyUxL_array); del dyUxL_array
dzUxL[:] = np.array(dzUxL_array); del dzUxL_array
drUxL[:] = np.array(drUxL_array); del drUxL_array
dyUxM[:] = np.array(dyUxM_array); del dyUxM_array
dzUxM[:] = np.array(dzUxM_array); del dzUxM_array
drUxM[:] = np.array(drUxM_array); del drUxM_array
dyUxH[:] = np.array(dyUxH_array); del dyUxH_array
dzUxH[:] = np.array(dzUxH_array); del dzUxH_array
drUxH[:] = np.array(drUxH_array); del drUxH_array


print(ncfile.groups)



print(ncfile)
ncfile.close()

print("line 959",time.time()-start_time)