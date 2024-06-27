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
from random import randint


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


def isInside(x, y):
     
    if ((x - 2560) * (x - 2560) +
        (y - 90) * (y - 90) < 63 * 63):
        return True
    else:
        return False
    

def ejection_check_1(X,Y):

    X = np.round(X,3); Y = np.round(Y,3)

    X = X.tolist(); Y = Y.tolist()

    if X.count(round(np.min(ys),3)) > 1 or X.count(round(np.max(ys),3)) > 1:
        return False
    elif round(np.max(zs),3) in Y:
        return False
    elif round(np.min(zs),3) in Y or round(np.min(ys),3) in X or round(np.max(ys),3) in X:
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
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 61", time.time()-start_time)


#directories
in_dir = "./"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)

a = Dataset("./sampling_r_-63.0.nc")

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
p = a.groups["p_r"]; del a

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

dy = (max(ys) - min(ys))/x
dz = (max(zs) - min(zs))/y
dA = dy * dz

#velocity field
u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
del p

u[u<0]=0; v[v<0]=0 #remove negative velocities

with Pool() as pool:
    u_pri = []
    for u_pri_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_pri.append(u_pri_it)
        print(len(u_pri),time.time()-start_time)
u_pri = np.array(u_pri); del u; del v


print("line 139",time.time()-start_time)

#find vmin and vmax for isocontour plots            
#min and max over data
cmin = math.floor(np.min(u_pri))
cmax = math.ceil(np.max(u_pri))


nlevs = int((cmax-cmin)/2)
if nlevs>abs(cmin) or nlevs>cmax:
    nlevs = min([abs(cmin),cmax])+1

levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
levels = np.concatenate((levs_min,levs_max[1:]))

print("line 155",levels)

levels_neg = np.linspace(-5.0,-1.4,4)
print("line 159", levels_neg)


colours = ["y","g","r","k"]


folder = out_dir+"Rotor_Plane_Fluctutating_horz_-63.0_surface_flucs_2/"
isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)


#options
plot_thresholds = True
output_data = False


def Update(it):

    U_pri = u_pri[it] #velocity time step it
    
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    Z = U_pri.reshape(y,x)
    X,Y = np.meshgrid(ys,zs)

    T = Time[it]

    CZ = plt.contour(X,Y,Z, levels=levels_neg)

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    cb = plt.colorbar(cs)

    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)


    for idx in np.arange(0,len(levels_neg)):

        threshold = round(levels_neg[idx],2)

        lines = CZ.allsegs[idx] #plot only threshold velocity

        #only plot certain lines
        for line in lines:
            X, Y = line[:,0], line[:,1]

            C = ejection_check_1(X,Y)

            if C == True:
                plt.plot(X,Y,"-",color=colours[idx],linewidth=3)

        xl = [-1,-2];yl = [-1,-2]
        plt.plot(xl,yl,"-",color=colours[idx],label="{}m/s".format(threshold))
    
    plt.xlim([np.min(ys),np.max(ys)]);plt.ylim([np.min(zs),np.max(zs)])
    plt.xlabel("y' axis (rotor frame of reference) [m]",fontsize=40)
    plt.ylabel("z' axis (rotor frame of reference) [m]",fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    plt.legend(loc="upper right")


    #define titles and filenames for movie
    Title = "Rotor Plane\nFluctuating Streamwise velocity [m/s]: Offset = -63.0m, Time = {0}[s]".format(round(T,4))
    filename = "Rotor_Fluc_Horz_-63.0_{0}.png".format(Time_idx)
        

    plt.title(Title)
    plt.tight_layout()
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return T


def Update_data(it):

    #algorithm for ejections

    U_pri = u_pri[it] #velocity time step it

    u_plane = U_pri.reshape(y,x)

    H = []
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)-1):

            if u_plane[k+1,j] > threshold:

                break

        #is coordinate inside rotor disk
        cc = isInside(ys[j],zs[k])
        if cc == True:
            z = np.min( np.roots([1,-180,(90**2-63**2+(ys[j]-2560)**2)]) )
            H.append(zs[k]-z) #height from coordinate zs to coordinate z on rotor disk

        #is coordinate above rotor disk so it is still covering it
        elif ys[j] > 2497 and ys[j] < 2623 and zs[k] > 90:
            z = np.roots([1,-180,(90**2-63**2+(ys[j]-2560)**2)])
            H.append(z[0]-z[1]) #height


    #average height
    if len(H) > 0:
        H_avg = np.average(H)
    else:
        H_avg = 0

    #integrate over area covering rotor disk
    A = 0
    for i in np.arange(0,len(H)-1):
        A+=((H[i+1] + H[i])/2)*dy


    return A, H_avg


if plot_thresholds == True:

    with Pool() as pool:
        for T in pool.imap(Update,Time_steps):

            print(T,time.time()-start_time)


# if output_data == True:
#     #create netcdf file
#     ncfile = Dataset(in_dir+"Thresholding_Dataset.nc",mode="w",format='NETCDF4')
#     ncfile.title = "Threshold data sampling output"

#     #create global dimensions
#     sampling_dim = ncfile.createDimension("sampling",None)

#     Time_sampling = ncfile.createVariable("Time", np.float64, ('sampling',),zlib=True)
#     Time_sampling[:] = Time

#     #thresholds to output data
#     thresholds = np.linspace(cmin,-0.7,8)

    
#     for threshold in thresholds:

#         print("line 293",threshold)

#         group = ncfile.createGroup("{}".format(round(abs(threshold),1)))

#         A_ejection = group.createVariable("Area_ejection", np.float64, ('sampling'),zlib=True)
#         H_average = group.createVariable("Average_height", np.float64, ('sampling'),zlib=True)

#         A_ej = []
#         H_avg_array = []

#         ix = 1
#         with Pool() as pool:
#             for A_adv_it, H_avg_it in pool.imap(Update_data,Time_steps):

#                 A_ej.append(A_adv_it)
#                 H_avg_array.append(H_avg_it)

#                 print(ix,time.time()-start_time)

#                 ix+=1

#         A_ejection[:] = np.array(A_ej); del A_ej
#         H_average[:] = np.array(H_avg_array); del H_avg_array

#         print(ncfile.groups)


#     print(ncfile)
#     ncfile.close()