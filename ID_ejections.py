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


def isInside(x, y):
     
    if ((x - 2) * (x - 2) +
        (y - 2) * (y - 2) <= 1 * 1):
        return True
    else:
        return False
    

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
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 61", time.time()-start_time)


#Iy, Iz
data = Dataset("Dataset.nc")
groups = data.groups["5.5"]
Iy = np.array((groups.variables["Iy"]))
Iz = np.array((groups.variables["Iz"]))



#directories
in_dir = "./"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)



a = Dataset("./sampling_r_-5.5.nc")

p = a.groups["p_r"]

#time options
Time = np.array(a.variables["time"])
tstart = 38000
tstart_idx = np.searchsorted(Time,tstart)
tend = 39200
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



#velocity field
u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

u[u<0]=0; v[v<0] #remove negative velocities

u = np.subtract(u,np.mean(u))
v = np.subtract(v,np.mean(v))

with Pool() as pool:
    u_hvel = []
    for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u = np.array(u_hvel); del u_hvel; del v

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

    
print("line 155",levels)

folder = out_dir+"Rotor_Plane_Fluctutating_horz_-5.5/"
isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)


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



    u_plane = U.reshape(y,x)
    X,Y = np.meshgrid(ys,zs)

    Z = u_plane

    T = Time[it]

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    cb = plt.colorbar(cs)

    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=2)
    ax.add_artist(Drawing_uncolored_circle)


    for t in np.arange(0,len(thresholds)):
        storage = np.zeros(len(ys))
        for j in np.arange(0,len(ys)):
            for k in np.arange(0,len(zs)-1):

                if u_plane[k+1,j] > thresholds[t]:
                    storage[j] = zs[k]
                    break

        ax.plot(ys,storage,linewidth=4,label="{}m/s".format(thresholds[t]))


    plt.xlabel("y' axis (rotor frame of reference) [m]")
    plt.ylabel("z' axis (rotor frame of reference) [m]")
    ax.legend(loc="upper right")


    #define titles and filenames for movie
    Title = "Rotor Plane. \nFluctuating horizontal velocity [m/s]: Offset = -5.5m, Time = {0}[s]".format(round(T,4))
    filename = "Rotor_Fluc_Horz_-5.5_{0}.png".format(Time_idx)
        

    plt.title(Title)
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return T


def Update_data(it):

    U = u[it] #velocity time step it

    u_plane = U.reshape(y,x)

    h = []
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)-1):

            if u_plane[k+1,j] > thresholds[t]:

                break
        
        #is coordinate inside rotor disk
        cc = isInside(ys[j],zs[k])
        if cc == True:
            z = np.min( np.roots([1,-180,(90**2-63**2+(ys[j]-2560)**2)]) )
            h.append(zs[k]-z) #height from coordinate zs to coordinate z on rotor disk
        #is coordinate above rotor disk so it is still covering it
        elif ys[j] > 2497 and ys[j] < 2623 and zs[k] > 90:
            z = np.roots([1,-180,(90**2-63**2+(ys[j]-2560)**2)])
            h.append(z[0]-z[1]) #height

    #integrate over area covering rotor disk
    A = 0
    delta_y = ys[1] - ys[0]
    for i in np.arange(0,len(h)-1):
        A+=((h[i+1] + h[i])/2)*delta_y

    if A == 0:
        prop = 0.0
    else:
        prop = A/(np.pi*1**2) #proportion of rotor disk covered at threshold t

    return it, prop


#thresholds to plot
thresholds = [-0.7,-2.0,-5.0]

with Pool() as pool:
    for T in pool.imap(Update,Time_steps):

        print(T,time.time()-start_time)


#create netcdf file
ncfile = Dataset(in_dir+"Thresholding_Dataset.nc",mode="w",format='NETCDF4')
ncfile.title = "Threshold data sampling output"

#create global dimensions
sampling_dim = ncfile.createDimension("sampling",None)

Time_sampling = ncfile.createVariable("Time", np.float64, ('sampling',),zlib=True)
Time_sampling[:] = Time

#thresholds to output data
thresholds = np.arange(-12.0,-0.0,2)
thresholds = np.append(thresholds,-0.7)

threshold_label = np.arange(12.0,0.0,-2)
threshold_label = np.append(threshold_label,0.7)


for t in np.arange(0,len(thresholds)):

    group = ncfile.createGroup("{}".format(threshold_label[t]))

    Iy_data = group.createVariable("Iy", np.float64, ('sampling'),zlib=True)
    Iz_data = group.createVariable("Iz", np.float64, ('sampling'),zlib=True)
    P_data = group.createVariable("P", np.float64, ('sampling'),zlib=True)

    Iy_it = []
    Iz_it = []
    P_it = []

    with Pool() as pool:
        for it,P_i in pool.imap(Update_data,Time):

            if P_i != 0:
                Iy_it.append(Iy[it])
                Iz_it.append(Iz[it])
                P_it.append(P_i)
            else:
                Iy_it.append(np.nan)
                Iz_it.append(np.nan)
                P_it.append(P_i)

            print(it,time.time()-start_time)

        Iy_data[:] = np.array(Iy_it); del Iy_it
        Iz_data[:] = np.array(Iz_it); del Iz_it 
        P_data[:] = np.array(P_it); del P_it

    print(ncfile.groups)


print(ncfile)
ncfile.close()