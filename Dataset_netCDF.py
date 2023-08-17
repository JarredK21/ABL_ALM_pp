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

def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def magnitude_horizontal_velocity(u,v,twist,x,zs,h):

    mag_horz_vel = []
    for i in np.arange(0,len(zs)):
        u_i = u[i*x:(i+1)*x]; v_i = v[i*x:(i+1)*x]
        height = zs[i]
        h_idx = np.searchsorted(h,height,side="left")
        if h_idx > 127:
            h_idx = 127
        mag_horz_vel_i = np.add( np.multiply(u_i,np.cos(twist[h_idx])) , np.multiply( v_i,np.sin(twist[h_idx])) )
        mag_horz_vel.extend(mag_horz_vel_i)
    mag_horz_vel = np.array(mag_horz_vel)

    return mag_horz_vel


def Ux_it_offset(it):

    Velocityx = velocityx[it]
    Velocityy = velocityy[it]
    hvelmag = magnitude_horizontal_velocity(Velocityx,Velocityy,twist,x,zs,h)


    hvelmag = hvelmag.reshape((y,x))

    Ux_rotor = []
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)):
            r = np.sqrt(ys[j]**2 + zs[k]**2)
            if r <= 63 and r >= 1.5:
                Ux_rotor.append(hvelmag[j,k])

    return np.average(Ux_rotor)


def Uz_it_offset(it):

    Velocityz = velocityz[it]

    Velocityz = Velocityz.reshape((y,x))

    Uz_rotor = []
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)):
            r = np.sqrt(ys[j]**2 + zs[k]**2)
            if r <= 63 and r >= 1.5:
                Uz_rotor.append(Velocityz[j,k])

    return np.average(Uz_rotor)


def IA_it_offset(it):

    Velocityx = velocityx[it]
    Velocityy = velocityy[it]
    hvelmag = magnitude_horizontal_velocity(Velocityx,Velocityy,twist,x,zs,h)

    hvelmag_interp = hvelmag.reshape((y,x))
    f = interpolate.interp2d(ys,zs,hvelmag_interp)

    hvelmag = np.reshape(hvelmag,(y,x))

    IA = 0
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)):
            r = np.sqrt(ys[j]**2 + zs[k]**2)
            if r <= 63 and r >= 1.5:

                delta_Ux_i = delta_Ux(r,j,k,f,hvelmag)
                IA += r * delta_Ux_i * dA
    return IA


def delta_Ux(r,j,k,f,Hvelmag):

    Y_0 = ys[j]
    Z_0 = zs[k]

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

    Ux_0 =  Hvelmag[j][k]
    Ux_1 =  f(Y_1,Z_1)
    Ux_2 =  f(Y_2,Z_2)

    delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

    return delta_Ux


#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics60000.nc")
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],32300)
t_end = np.searchsorted(precursor.variables["time"],33500)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
del precursor

print("line 143",time.time()-start_time)


#create netcdf file
ncfile = Dataset("./Dataset.nc",mode="w",format='NETCDF4') #change name
ncfile.title = "AMR-Wind data sampling output combined"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)
sampling_dim = ncfile.createDimension("sampling",None)

#create variables
time_OF = ncfile.createVariable("time_OF", np.float64, ('OF',),zlib=True)
time_sampling = ncfile.createVariable("time_sampling", np.float64, ('sampling',),zlib=True)

Ux_1 = ncfile.createVariable("Ux_0.0", np.float64, ('sampling',),zlib=True)
# Ux_2 = ncfile.createVariable("Ux_-63.0", np.float64, ('sampling',),zlib=True)
# Ux_3 = ncfile.createVariable("Ux_-126.0", np.float64, ('sampling',),zlib=True)

Uz_1 = ncfile.createVariable("Uz_0.0", np.float64, ('sampling',),zlib=True)
# Uz_2 = ncfile.createVariable("Uz_-63.0", np.float64, ('sampling',),zlib=True)
# Uz_3 = ncfile.createVariable("Uz_-126.0", np.float64, ('sampling',),zlib=True)

IA_1 = ncfile.createVariable("IA_0.0", np.float64, ('sampling',),zlib=True)
# IA_2 = ncfile.createVariable("IA_-63.0", np.float64, ('sampling',),zlib=True)
# IA_3 = ncfile.createVariable("IA_-126.0", np.float64, ('sampling',),zlib=True)

RtAeroVxh = ncfile.createVariable("RtAeroVxh", np.float64, ('OF',),zlib=True)
RtAeroFxh = ncfile.createVariable("RtAeroFxh", np.float64, ('OF',),zlib=True)
RtAeroMxh = ncfile.createVariable("RtAeroMxh", np.float64, ('OF',),zlib=True)
RtAeroMrh = ncfile.createVariable("RtAeroMrh", np.float64, ('OF',),zlib=True)
Theta = ncfile.createVariable("Theta", np.float64, ('OF',),zlib=True)

print("line 175",time.time()-start_time)


#openfast data
da = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()
db = io.fast_output_file.FASTOutputFile("../../NREL_5MW_MCBL_R_CRPM_100320/NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

#combine time
restart_time = 137.748
Time_a_OF = np.array(da["Time_[s]"]); Time_b_OF = np.array(db["Time_[s]"]); Time_b_OF = Time_b_OF+restart_time
restart_idx = np.searchsorted(Time_a_OF,restart_time); restart_idx-=1
Time_OF = np.concatenate((Time_a_OF[0:restart_idx],Time_b_OF))


plot_all_times = True
if plot_all_times == False:
    tstart = 50
    tend = 150
    tstart_OF_idx = np.searchsorted(Time_OF,tstart)
    tend_OF_idx = np.searchsorted(Time_OF,tend)
else:
    tstart_OF_idx = 0
    tend_OF_idx = np.searchsorted(Time_OF,Time_OF[-1])

time_OF[:] = Time_OF[tstart_OF_idx:tend_OF_idx]; del Time_OF


#combine openFAST outputs
df = pd.concat((da[:][0:restart_idx],db[:])); del da; del db

print("line 205",time.time()-start_time)

Variables = ["Wind1VelX","RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[m/s]","[N]","[N-m]","[N-m]","[rads]"]


for iv in np.arange(0,len(Variables)):
    Variable = Variables[iv]

    if Variable == "MR" or Variable == "Theta":
        signaly = df["RtAeroMyh_[N-m]"][tstart_OF_idx:tend_OF_idx]
        signalz = df["RtAeroMzh_[N-m]"][tstart_OF_idx:tend_OF_idx]
        
        if Variable == "MR":
            signal = np.sqrt( np.square(signaly) + np.square(signalz) ) 
            RtAeroMrh[:] = signal; del signal
        elif Variable == "Theta": 
            signal = np.arctan2(signalz,signaly)
            Theta[:] = signal; del signal

    else:
        txt = "{0}_{1}".format(Variable,units[iv])
        signal = df[txt][tstart_OF_idx:tend_OF_idx]
        if Variable == "RtAeroFxh":
            RtAeroFxh[:] = signal; del signal
        elif Variable == "RtAeroMxh":
            RtAeroMxh[:] = signal; del signal
        elif Variable == "Wind1VelX":
            RtAeroVxh[:] = signal; del signal

del df

print("line 235",time.time()-start_time)

#sampling data
a = Dataset("./sampling_r_0.0.nc")

#sampling time
Time_sample = np.array(a.variables["time"])
Time_sample = Time_sample - Time_sample[0]

plot_all_times = True
if plot_all_times == False:
    tstart = 50
    tend = 150
    tstart_sample_idx = np.searchsorted(Time_sample,tstart)
    tend_sample_idx = np.searchsorted(Time_sample,tend)
else:
    tstart_sample_idx = 0
    tend_sample_idx = np.searchsorted(Time_sample,Time_sample[-1])

print(tend_sample_idx)

time_sampling[:] = Time_sample[tstart_sample_idx:tend_sample_idx]

p_rotor = a.groups["p_r"]

offsets = [0.0] #only rotor plane for now

Variables = []
units = []

for offset in offsets:
    txt = ["Ux_{0}".format(offset), "Uz_{0}".format(offset), "IA_{0}".format(offset)]
    unit = ["[m/s]", "[m/s]", "[$m^4/s$]"]
    for x,y in zip(txt,unit):
        Variables.append(x)
        units.append(y)


x = p_rotor.ijk_dims[0] #no. data points
y = p_rotor.ijk_dims[1] #no. data points

coordinates = p_rotor.variables["coordinates"]

xo = coordinates[0:x,0]
yo = coordinates[0:x,1]
zo = np.linspace(p_rotor.origin[2],p_rotor.axis2[2],y)

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-29)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
zs = zo - rotor_coordiates[2]


dy = ys[1]-ys[0]
dz = zs[1] - zs[0]
dA = dy * dz

#velocity field
velocityx = p_rotor.variables["velocityx"]
velocityy = p_rotor.variables["velocityy"]
velocityz = p_rotor.variables["velocityz"]

print("line 323")

for iv in np.arange(0,len(Variables)):
    Variable = Variables[iv]
    print(Variable[0:2])
    print(Variable[3:])
    if Variable[3:] == "0.0":
        i = 0
    elif Variable[3:] == "-63.0":
        i = 1
    elif Variable[3:] == "-126.0":
        i = 2
    if Variable[0:2] == "Ux":
        Ux_it = []
        print("Ux calcs",len(np.arange(tstart_sample_idx,tend_sample_idx)))
        with Pool() as pool:
            for Ux_i in pool.imap(Ux_it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
                Ux_it.append(Ux_i)
                print(len(Ux_it),time.time()-start_time)
            Ux_it = np.array(Ux_it)
        if Variable[3:] == "0.0":
            Ux_1[:] = Ux_it; del Ux_it
        # elif Variable[3:] == "-63.0":
        #     Ux_2[:] = Ux_it; del Ux_it
        # elif Variable[3:] == "-126.0":
        #     Ux_3[:] = Ux_it; del Ux_it

    elif Variable[0:2] == "Uz":
        Uz_it = []
        print("Uz calcs",len(np.arange(tstart_sample_idx,tend_sample_idx)))
        with Pool() as pool:
            for Uz_i in pool.imap(Uz_it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
                Uz_it.append(Uz_i)
                print(len(Uz_it),time.time()-start_time)
            Uz_it = np.array(Uz_it)
        if Variable[3:] == "0.0":
            Uz_1[:] = Uz_it; del Uz_it
        # elif Variable[3:] == "-63.0":
        #     Uz_2[:] = Uz_it; del Uz_it
        # elif Variable[3:] == "-126.0":
        #     Uz_3[:] = Uz_it; del Uz_it
    elif Variable[0:2] == "IA":
        IA_it = []
        print("IA calcs",len(np.arange(tstart_sample_idx,tend_sample_idx)))
        with Pool() as pool:
            for IA_i in pool.imap(IA_it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
                IA_it.append(IA_i)
                print(len(IA_it),time.time()-start_time)
            IA_it = np.array(IA_it)
        if Variable[3:] == "0.0":
            IA_1[:] = IA_it; del IA_it
        # elif Variable[3:] == "-63.0":
        #     IA_2[:] = IA_it; del IA_it
        # elif Variable[3:] == "-126.0":
        #     IA_3[:] = IA_it; del IA_it

print(ncfile)
ncfile.close()

print("line 352",time.time() - start_time)