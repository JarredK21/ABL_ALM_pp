from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
import time
from multiprocessing import Pool
from scipy import interpolate


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
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
        mag_fluc_horz_vel.extend(mag_fluc_horz_vel_i)
    mag_fluc_horz_vel = np.array(mag_fluc_horz_vel)
    return mag_fluc_horz_vel


start_time = time.time()

in_dir = "./"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)

#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38200)
t_end = np.searchsorted(precursor.variables["time"],39201)
u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 65")


a = Dataset("./sampling_r_-63.0.nc")

p = a.groups["p_r"]

#time options
Time = np.array(a.variables["time"])
tstart = 38200
tstart_idx = np.searchsorted(Time,tstart)
tend = 39201
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points

normal = 29

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

print("line 105")


u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

u[u<0]=0; v[v<0]=0 #remove negative velocities


#fluctuating streamwise velocity
with Pool() as pool:
    u_pri = []
    for u_pri_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_pri.append(u_pri_it)
        print(len(u_pri),time.time()-start_time)
u = np.array(u_pri)
del u_pri; del v
print("line 123")

w = np.array(p.variables["velocityz"][tstart_idx:tend_idx])


print("line 128",time.time()-start_time)

#find vmin and vmax for isocontour plots            
#min and max over data
                    

cminu = math.floor(np.min(u))
cmaxu = math.ceil(np.max(u))

nlevs = int((cmaxu-cminu)/2)
if nlevs>abs(cminu) or nlevs>cmaxu:
    nlevs = min([abs(cminu),cmaxu])+1

levs_min = np.linspace(cminu,0,nlevs,dtype=int); levs_max = np.linspace(0,cmaxu,nlevs,dtype=int)
levels_u = np.concatenate((levs_min,levs_max[1:]))
    
print("line 144",levels_u)

nlevs = int(-0.7-cminu)
levels_neg = np.linspace(cminu,-0.7,nlevs)

print(levels_neg)

cminw = math.floor(np.min(w))
cmaxw = math.ceil(np.max(w))

nlevs = int((cmaxw-cminw)/2)
if nlevs>abs(cminw) or nlevs>cmaxw:
    nlevs = min([abs(cminw),cmaxw])+1

levs_min = np.linspace(cminw,0,nlevs,dtype=int); levs_max = np.linspace(0,cmaxw,nlevs,dtype=int)
levels_w = np.concatenate((levs_min,levs_max[1:]))
    
print("line 161",levels_w)

                
nlevs = int(cmaxw-0.43)
levels_pos = np.linspace(0.43,cmaxw,nlevs)

print(levels_pos)


folder = out_dir+"Rotor_Plane_velz_Fluc_horz_vel_contours/"

isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)

def Update(it):

    U = u[it] #streamwise velocity time step it

    W = w[it] #vertical velocity time step it
    
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)


    u_plane = U.reshape(y,x)
    w_plane = W.reshape(y,x)
    X,Y = np.meshgrid(ys,zs)

    Zu = u_plane
    Zw = w_plane

    T = Time[it]

    #plotting vertical velocity with negative streamwise velocity contours
    fig = plt.figure(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = plt.contourf(X,Y,Zw,levels=levels_w, cmap=cm.coolwarm,vmin=cminw,vmax=cmaxw)

    #<-0.7 blue, <0.7 red
    CS = plt.contour(X, Y, Zu, levels=levels_neg, colors='b')
    plt.clabel(CS, fontsize=18, inline=True)



    plt.xlabel("y' axis (rotor frame of reference) [m]")
    plt.ylabel("z' axis (rotor frame of reference) [m]")


    cb = plt.colorbar(cs)

    #define titles and filenames for movie
    Title = "Rotor Plane. \nVerical velocity\nFluctuating Streamwise velocity contours [m/s]\nOffset = -63m, Time = {}[s]".format(round(T,4))
    filename = "Rotor_Vertical_vel_Fluc_horz_vel_contours_{}.png".format(Time_idx)

    plt.title(Title)
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return T

with Pool() as pool:
    for T in pool.imap(Update,Time_steps):

        print(T,time.time()-start_time)

time.sleep(30)


folder = out_dir+"Rotor_Plane_Fluc_horz_vel_velz_contours/"

isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)

def Update_2(it):

    U = u[it] #streamwise velocity time step it

    W = w[it] #vertical velocity time step it
    
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)


    u_plane = U.reshape(y,x)
    w_plane = W.reshape(y,x)
    X,Y = np.meshgrid(ys,zs)

    Zu = u_plane
    Zw = w_plane

    T = Time[it]

    #plotting vertical velocity with negative streamwise velocity contours
    fig = plt.figure(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = plt.contourf(X,Y,Zu,levels=levels_u, cmap=cm.coolwarm,vmin=cminu,vmax=cmaxu)

    #<-0.7 blue, <0.7 red
    CS = plt.contour(X, Y, Zw, levels=levels_pos, colors='r')
    plt.clabel(CS, fontsize=18, inline=True)



    plt.xlabel("y' axis (rotor frame of reference) [m]")
    plt.ylabel("z' axis (rotor frame of reference) [m]")


    cb = plt.colorbar(cs)

    #define titles and filenames for movie
    Title = "Rotor Plane. \nFluctuating Streamwise velocity\nVertical velocity contours [m/s]\nOffset = -63m, Time = {}[s]".format(round(T,4))
    filename = "Rotor_Fluc_horz_vel_vertical_vel_contours_{}.png".format(Time_idx)

    plt.title(Title)
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return T


with Pool() as pool:
    for T in pool.imap(Update_2,Time_steps):

        print(T,time.time()-start_time)

time.sleep(30)