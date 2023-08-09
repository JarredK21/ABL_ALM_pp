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

def offset_data(p_rotor,no_cells_offset,it,i,velocity_comp):

    if velocity_comp == "coordinates":
        u = np.array(p_rotor.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_rotor.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice


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

    velocityx = offset_data(p_rotor, no_cells_offset,it,i=2,velocity_comp="velocityx")
    velocityy = offset_data(p_rotor, no_cells_offset,it,i=2,velocity_comp="velocityy")
    hvelmag = magnitude_horizontal_velocity(velocityx,velocityy,twist,x,zs,h)


    hvelmag = hvelmag.reshape((y,x))

    Ux_rotor = []
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)):
            r = np.sqrt(ys[j]**2 + zs[k]**2)
            if r <= 63 and r >= 1.5:
                Ux_rotor.append(hvelmag[j,k])

    return np.average(Ux_rotor)


def IA_it_offset(it):

    velocityx = offset_data(p_rotor, no_cells_offset,it,i=0,velocity_comp="velocityx")
    velocityy = offset_data(p_rotor, no_cells_offset,it,i=0,velocity_comp="velocityy")
    hvelmag = magnitude_horizontal_velocity(velocityx,velocityy,twist,x,zs,h)

    hvelmag_interp = hvelmag.reshape((y,x))
    f = interpolate.interp2d(ys,zs,hvelmag_interp)

    hvelmag = hvelmag.reshape((y,x))

    IA = 0
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)):
            r = np.sqrt(ys[j]**2 + zs[k]**2)
            if r <= 63 and r >= 1.5:

                delta_Ux_i = delta_Ux(r,j,k,f,hvelmag)
                IA += r * delta_Ux_i * dA
    return IA


def delta_Ux(r,j,k,f,hvelmag):

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

    Ux_0 =  hvelmag[j][k]
    Ux_1 =  f(Y_1,Z_1)
    Ux_2 =  f(Y_2,Z_2)

    delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

    return delta_Ux


#defining twist angles with height from precursor
a = Dataset("./abl_statistics60000.nc")

mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

t_start = np.searchsorted(a.variables["time"],32300)
t_end = np.searchsorted(a.variables["time"],33500)

u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)

h = mean_profiles["h"][:]

twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation


#openfast data
df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

#sampling data
sampling = glob.glob("./sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_r"]

offsets = p_rotor.offsets

Variables = ["Time_OF","Time_sample","Ux","IA","RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[s]","[s]", "[m/s]","[m^4/s]","[N]","[N-m]","[N-m]","[rads]"]


dq = dict()

time_OF = np.array(df["Time_[s]"])
time_sample = np.array(a.variables["time"])
time_sample = time_sample - time_sample[0]

plot_all_times = True
if plot_all_times == False:
    tstart = 50
    tend = 350
    tstart_OF_idx = np.searchsorted(time_OF,tstart)
    tend_OF_idx = np.searchsorted(time_OF,tend)
    tstart_sample_idx = np.searchsorted(time_sample,tstart)
    tend_sample_idx = np.searchsorted(time_sample,tend)
else:
    tstart_OF_idx = 0
    tend_OF_idx = np.searchsorted(time_OF,time_OF[-1])
    tstart_sample_idx = 0
    tend_sample_idx = np.searchsorted(time_sample,time_sample[-1])


dq["Time_OF"] = time_OF[tstart_OF_idx:tend_OF_idx]
dq["Time_sample"] = time_sample[tstart_sample_idx:tend_sample_idx]

no_cells = len(p_rotor.variables["coordinates"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

x = p_rotor.ijk_dims[0] #no. data points
y = p_rotor.ijk_dims[1] #no. data points

coordinates = offset_data(p_rotor,no_cells_offset,it=0,i=0,velocity_comp="coordinates")

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

print("line 161",time.time() - start_time)

#modify to use sampling plane
for iv in np.arange(2,len(Variables)):
    Variable = Variables[iv]
    if Variable == "Ux":
        Ux_it = df["RtVAvgxh_[m/s]"][tstart_OF_idx:tend_OF_idx]
        dq["Ux"] = Ux_it

    elif Variable == "IA":
        IA_it = []
        print("IA calcs",len(np.arange(tstart_sample_idx,tend_sample_idx)))
        with Pool() as pool:
            for IA_i in pool.imap(IA_it_offset, np.arange(tstart_sample_idx,tend_sample_idx)):
                IA_it.append(IA_i)
                print(len(IA_it),time.time()-start_time)
        dq["IA"] = IA_it

    elif Variable == "MR" or Variable == "Theta":
        signaly = df["RtAeroMyh_[N-m]"][tstart_OF_idx:tend_OF_idx]
        signalz = df["RtAeroMzh_[N-m]"][tstart_OF_idx:tend_OF_idx]
        
        if Variable == "MR":
            signal = np.sqrt( np.square(signaly) + np.square(signalz) ) 
        elif Variable == "Theta": 
            signal = np.arctan2(signalz,signaly)
        dq[Variable] = signal  

    else:
        txt = "{0}_{1}".format(Variable,units[iv])
        signal = df[txt][tstart_OF_idx:tend_OF_idx]
        dq[Variable] = signal


dw = pd.DataFrame(dict([(key, pd.Series(value)) for key, value in dq.items()]))

dw.to_csv("../post_processing/out.csv")

print("line 205",time.time() - start_time)