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


def Horizontal_velocity():
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    mag_fluc_horz_vel = []
    for i in np.arange(0,len(zs)):
        u_i = u[i*x:(i+1)*x]; v_i = v[i*x:(i+1)*x]
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
    

start_time = time.time()

#defining twist angles with height from precursor
precursor = Dataset("../../ABL_precursor_2_restart/abl_statistics70000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38000)
t_end = np.searchsorted(precursor.variables["time"],39200)
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
in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)



a = Dataset(in_dir+"sampling_r_-63.0_0.nc")


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
u = np.array(p.variables["velocityx"])
v = np.array(p.variables["velocityy"])
del p

u[u<0]=0; v[v<0]=0 #remove negative velocities

u_pri = Horizontal_velocity()
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


folder = out_dir+"test_ID_ejections/"
isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)


#options
plot_thresholds = True
output_data = True


def Update():

    U_pri = u_pri #velocity time step it
    

    Time_idx = 0000



    Z = U_pri.reshape(y,x)
    X,Y = np.meshgrid(ys,zs)

    T = 0.0

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    cb = plt.colorbar(cs)

    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)

    for t in np.arange(0,len(thresholds)):
        storage = np.zeros(len(ys))
        for j in np.arange(0,len(ys)):
            for k in np.arange(0,len(zs)-1):

                if Z[k+1,j] > thresholds[t]:
                    storage[j] = zs[k]
                    break

        ax.plot(ys,storage,linewidth=2,label="{}m/s".format(thresholds[t]))


    plt.xlabel("y' axis (rotor frame of reference) [m]",fontsize=40)
    plt.ylabel("z' axis (rotor frame of reference) [m]",fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)
    ax.legend(loc="upper right")


    #define titles and filenames for movie
    Title = "Rotor Plane. \nFluctuating horizontal velocity [m/s]: Offset = -63.0m, Time = {0}[s]".format(round(T,4))
    filename = "Rotor_Fluc_Horz_-63.0_{0}.png".format(Time_idx)
        

    plt.title(Title)
    plt.tight_layout()
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return T



def Update_data():

    #algorithm for ejections

    U_pri = u_pri #velocity time step it

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
    #thresholds to plot
    thresholds = [-0.7,-2.0,-5.0]

    z_avg = Update()
    print(z_avg)


if output_data == True:


    #create netcdf file
    ncfile = Dataset(folder+"Thresholding_Dataset.nc",mode="w",format='NETCDF4')
    ncfile.title = "Threshold data sampling output"

    #create global dimensions
    sampling_dim = ncfile.createDimension("sampling",None)

    #thresholds to output data
    thresholds = np.linspace(cmin,-0.7,8)

    for threshold in thresholds:
        
        print("line 293",threshold)

        group = ncfile.createGroup("{}".format(round(abs(threshold),1)))

        A_ejection = group.createVariable("Area_ejection", np.float64, ('sampling'),zlib=True)
        H_average = group.createVariable("Average_height", np.float64, ('sampling'),zlib=True)

        A_ej = []
        H_avg_array = []

        A_adv_it, H_avg_it = Update_data()
        print(A_adv_it)
        print(H_avg_it)
        A_ej.append(A_adv_it)
        H_avg_array.append(H_avg_it)


        A_ejection[:] = np.array(A_ej); del A_ej
        H_average[:] = np.array(H_avg_array); del H_avg_array

        print(ncfile.groups)


    print(ncfile)
    ncfile.close()