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
    if normal == "z":
        height = offset
        twist_h = f(height)
        mag_horz_vel = u[it]*np.cos(twist_h) + v[it]*np.sin(twist_h)
    else:
        mag_horz_vel = []
        for i in np.arange(0,len(zs_r)):
            u_i = u[it,i*x_r:(i+1)*x_r]; v_i = v[it,i*x_r:(i+1)*x_r]
            if zs_r[i] < h[0]:
                twist_h = f(h[0])
            elif zs_r[i] > h[-1]:
                twist_h = f(h[-1])
            else:
                twist_h = f(zs_r[i])
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
groups = data.groups["63.0"]
Iy = np.array((groups.variables["Iy"]))
Iz = np.array((groups.variables["Iz"]))
del data; del groups


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
tstart = 38000
tstart_idx = np.searchsorted(Time,tstart)
tend = 39200
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]
del a


#rotor data
p = a.groups["p_r"]

x_r = p.ijk_dims[0] #no. data points
y_r = p.ijk_dims[1] #no. data points


normal = 29

#define plotting axes
coordinates = np.array(p.variables["coordinates"])


xo = coordinates[0:x_r,0]
yo = coordinates[0:x_r,1]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-normal)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
xs_r = xs + rotor_coordiates[0]
ys_r = ys + rotor_coordiates[1]
zs_r = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y_r)



#velocity field
u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
del p

u[u<0]=0; v[v<0] #remove negative velocities

u = np.subtract(u,np.mean(u))
v = np.subtract(v,np.mean(v))

with Pool() as pool:
    u_hvel = []
    for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u_r = np.array(u_hvel); del u_hvel; del v

print("line 139",time.time()-start_time)

#find vmin and vmax for isocontour plots            
#min and max over data
cmin_r = math.floor(np.min(u_r))
cmax_r = math.ceil(np.max(u_r))


nlevs = int((cmax_r-cmin_r)/2)
if nlevs>abs(cmin_r) or nlevs>cmax_r:
    nlevs = min([abs(cmin_r),cmax_r])+1

levs_min = np.linspace(cmin_r,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax_r,nlevs,dtype=int)
levels_r = np.concatenate((levs_min,levs_max[1:]))

print("line 155",levels_r)


#horizontal plane 22.5m
offset = 22.5

a = Dataset("sampling_l_{}.nc".format(offset))

p = a.groups["p_l"]

x_22 = p.ijk_dims[0] #no. data points
y_22 = p.ijk_dims[1] #no. data points

normal = "z"

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xo = coordinates[0:x_22,0]
yo = coordinates[0:x_22,1]

xs_22 = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x_22)
ys_22 = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y_22)
zs_22 = 0

del a

u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
u[u<0] = 0; v[v<0]=0
u = np.subtract(u,np.mean(u)); v = np.subtract(v,np.mean(v))

with Pool() as pool:
    u_hvel = []
    for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u_22 = np.array(u_hvel); del u_hvel; del v

print("line 139",time.time()-start_time)


#horizontal plane 90m
offset = 85

a = Dataset("sampling_l_{}.nc".format(offset))

p = a.groups["p_l"]

x_85 = p.ijk_dims[0] #no. data points
y_85 = p.ijk_dims[1] #no. data points

normal = "z"

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xo = coordinates[0:x_85,0]
yo = coordinates[0:x_85,1]

xs_85 = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x_85)
ys_85 = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y_85)
zs_85 = 0

del a

u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
u[u<0] = 0; v[v<0]=0
u = np.subtract(u,np.mean(u)); v = np.subtract(v,np.mean(v))

with Pool() as pool:
    u_hvel = []
    for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u_85 = np.array(u_hvel); del u_hvel; del v

print("line 245",time.time()-start_time)



#horizontal plane 142.5m
offset = 142.5

a = Dataset("sampling_l_{}.nc".format(offset))

p = a.groups["p_l"]

x_142 = p.ijk_dims[0] #no. data points
y_142 = p.ijk_dims[1] #no. data points

normal = "z"

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xo = coordinates[0:x_142,0]
yo = coordinates[0:x_142,1]

xs_142 = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x_142)
ys_142 = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y_142)
zs_142 = 0

del a

u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
u[u<0] = 0; v[v<0]=0
u = np.subtract(u,np.mean(u)); v = np.subtract(v,np.mean(v))

with Pool() as pool:
    u_hvel = []
    for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u_142 = np.array(u_hvel); del u_hvel; del v

print("line 139",time.time()-start_time)




folder = out_dir+"Rotor_Plane_Fluctutating_horz_-63.0/"
isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)


#options
plot_thresholds = False
output_data = True


def Update(it):

    U = u_r[it] #velocity time step it
    
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)



    u_plane = U.reshape(y_r,x_r)
    X,Y = np.meshgrid(ys_r,zs_r)

    Z = u_plane

    T = Time[it]

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(X,Y,Z,levels=levels_r, cmap=cm.coolwarm,vmin=cmin_r,vmax=cmax_r)

    cb = plt.colorbar(cs)

    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)


    for t in np.arange(0,len(thresholds)):
        storage = np.zeros(len(ys_r))
        for j in np.arange(0,len(ys_r)):
            for k in np.arange(0,len(zs_r)-1):

                if u_plane[k+1,j] > thresholds[t]:
                    storage[j] = zs_r[k]
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


def Update_data(it):

    #algorithm for ejections
    if thresholds[t] <= -0.7:

        U = u_r[it] #velocity time step it

        u_plane = U.reshape(y_r,x_r)

        H = []
        for j in np.arange(0,len(ys_r)):
            for k in np.arange(0,len(zs_r)-1):

                if u_plane[k+1,j] > thresholds[t]:

                    break
            
            if it == 0:
                H_temp[1].append(zs_r[k])
                dH_dt.append(0.0)
            else:
                H_temp[0] = H_temp[1]
                H_temp[1] = []
                H_temp[1].append(zs_r[k])
                dH_dt.append(zs_r[k]-H_temp[0])/dt

            #is coordinate inside rotor disk
            cc = isInside(ys_r[j],zs_r[k])
            if cc == True:
                z = np.min( np.roots([1,-180,(90**2-63**2+(ys_r[j]-2560)**2)]) )
                #check if ejection is growing                
                if dH_dt[it] > 0.0:
                    H.append(zs_r[k]-z) #height from coordinate zs to coordinate z on rotor disk
                    it_ej = it
                else:
                    it_ej = np.nan

            #is coordinate above rotor disk so it is still covering it
            elif ys_r[j] > 2497 and ys_r[j] < 2623 and zs_r[k] > 90:
                z = np.roots([1,-180,(90**2-63**2+(ys_r[j]-2560)**2)])

                #check if ejection is continuously growing                
                if dH_dt[0][it] > 0.0:
                    H.append(z[0]-z[1]) #height
                    it_ej = it
                else:
                    it_ej = np.nan

            #coordinate is not inside or covering rotor disk
            else:
                it_ej = np.nan


        #average height
        H_avg = np.average(H)

        #integrate over area covering rotor disk
        A = 0
        delta_y = ys_r[1] - ys_r[0]
        for i in np.arange(0,len(H)-1):
            A+=((H[i+1] + H[i])/2)*delta_y

        if A == 0:
            prop_ej = 0.0
        else:
            prop_ej = A/(np.pi*63**2) #proportion of rotor disk covered at threshold t



    #check interpolation step
    U_85 = u_85[it]
    u_plane = U_85.reshape(x_85,y_85)
    rotor_array_x_85 = np.linspace(2555,2555,403)
    rotor_array_y_85 = np.linspace(2497,2623,403)
    f = interpolate.interp2d(xs_85,ys_85,u_plane)
    u_rotor_85 = f(rotor_array_x_85,rotor_array_y_85)

    prop_adv_pos = 0.0
    prop_adv_neg = 0.0
    for i in np.arange(1,len(u_rotor_85)):

        if thresholds[t] <= -0.7 and u_rotor_85[i] <= thresholds[t]:

            #check if ejection height and y coordinate
            idx = np.searchsorted(ys_r,rotor_array_y_85[i],side="left")
            if np.average([H_temp[idx-1],H_temp[idx]]) < 90:
                it_adv = it
                prop_adv_neg+=(rotor_array_y_85[i]-rotor_array_y_85[i-1])

        elif thresholds[t] >= 0.7 and u_rotor_85[i] >= thresholds[t]:
            it_adv = it
            prop_adv_pos+=(rotor_array_y_85[i]-rotor_array_y_85[i-1])

        else:
            it_adv = np.nan

    prop_adv_pos = prop_adv_pos/126
    prop_adv_neg = prop_adv_neg/126
    prop_adv = [prop_adv_pos,prop_adv_neg]


    return it_ej, prop_ej, it_adv, prop_adv, H_avg


if plot_thresholds == True:
    #thresholds to plot
    thresholds = [-0.7,-2.0,-5.0]

    with Pool() as pool:
        for T in pool.imap(Update,Time_steps):

            print(T,time.time()-start_time)


if output_data == True:
    #create netcdf file
    ncfile = Dataset(in_dir+"Thresholding_Dataset.nc",mode="w",format='NETCDF4')
    ncfile.title = "Threshold data sampling output"

    #create global dimensions
    sampling_dim = ncfile.createDimension("sampling",None)

    Time_sampling = ncfile.createVariable("Time", np.float64, ('sampling',),zlib=True)
    Time_sampling[:] = Time

    #thresholds to output data
    thresholds = np.linspace(cmin_r,-0.7,8)
    thresholds = np.append(thresholds,-0.7)
    thresholds = np.concatenate(thresholds,np.linspace(0.7,cmax_r,8))

    #update labels
    threshold_label = np.arange(12.0,0.0,-2)
    threshold_label = np.append(threshold_label,0.7)

    
    for t in np.arange(0,len(thresholds)):

        print("line 293",thresholds[t])

        group = ncfile.createGroup("{}".format(threshold_label[t]))

        Time_ejection = group.createVariable("Time_ejection", np.float64, ('sampling'),zlib=True)
        Time_advection = group.createVariable("Time_advection", np.float64, ('sampling'),zlib=True)
        P_ejection = group.createVariable("P_ejection", np.float64, ('sampling'),zlib=True)
        P_advection = group.createVariable("P_advection", np.float64, ('sampling'),zlib=True)
        H_average = group.createVariable("Average_height", np.float64, ('sampling'),zlib=True)

        Time_ej = []
        Time_adv = []
        P_ej = []
        P_adv = []
        H_avg_array = []
        dH_dt = []
        H_temp = [[],[]]

        ix = 1
        with Pool() as pool:
            for itx_ej, P_ej_it, itx_adv, P_adv_it, H_avg_it in pool.imap(Update_data,Time_steps):
                
                if itx_ej == np.nan:
                    Time_ej.append(itx_ej)
                elif itx_adv == np.nan:
                    Time_adv.append(itx_adv)

                P_ej.append(P_ej_it)
                P_adv.append(P_adv_it)

                H_avg_array.append(H_avg_it)

                print(ix,time.time()-start_time)

                ix+=1

            Time_ejection[:] = np.array(Time_ej); del Time_ej
            P_ejection[:] = np.array(P_ej); del P_ej
            Time_advection[:] = np.array(Time_adv); del Time_adv
            P_advection[:] = np.array(P_adv); del P_adv
            H_average[:] = np.array(H_avg_array); del H_avg_array

        print(ncfile.groups)


    print(ncfile)
    ncfile.close()