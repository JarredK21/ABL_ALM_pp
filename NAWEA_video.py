from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import math
import time
from multiprocessing import Pool
from scipy.signal import butter,filtfilt
from scipy import interpolate
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Circle


def blade_positions(it):

    R = 63
    Az = f_Az(Time_sampling[it])
    Y = [2560]; Y2 = [2560]; Y3 = [2560]
    Z = [90]; Z2 = [90]; Z3 = [90]

    Y.append(Y[0]+R*np.sin(Az))
    Z.append(Z[0]+R*np.cos(Az))

    Az2 = Az+(2*np.pi)/3
    if Az2 > 2*np.pi:
        Az2 -= (2*np.pi)
    
    Az3 = Az-(2*np.pi)/3
    if Az2 < 0:
        Az2 += (2*np.pi)

    Y2.append(Y2[0]+R*np.sin(Az2))
    Z2.append(Z2[0]+R*np.cos(Az2))

    Y3.append(Y3[0]+R*np.sin(Az3))
    Z3.append(Z3[0]+R*np.cos(Az3))

    return Y, Z, Y2, Z2, Y3, Z3


def Update(it):

    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    fig = plt.figure(figsize=(14,8),constrained_layout=True)
    gs = fig.add_gridspec(2, 2)
    f3_ax1 = fig.add_subplot(gs[1, :1])#bottom left
    f3_ax2 = fig.add_subplot(gs[1, 1:])#bottom right
    f3_ax3 = fig.add_subplot(gs[0, :1])#top left
    f3_ax4 = fig.add_subplot(gs[0, 1:])#top right

    #bottom left plot
    U_r = u_r[it] #velocity time step it

    u_plane = U_r.reshape(y_r,x_r)
    X,Y = np.meshgrid(ys_r,zs_r)

    Z = u_plane

    cs = f3_ax1.contourf(X,Y,Z,levels=levels_r, cmap=cm.coolwarm,vmin=cmin_r,vmax=cmax_r)
    f3_ax1.set_xlabel("Y' axis (rotor frame of reference) [m]")
    f3_ax1.set_ylabel("Z' axis (rotor frame of reference) [m]")

    divider = make_axes_locatable(f3_ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    cb = fig.colorbar(cs,cax=cax)


    YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

    f3_ax1.plot(YB1,ZB1,color="k",linewidth = 0.5)
    f3_ax1.plot(YB2,ZB2,color="k",linewidth = 0.5)
    f3_ax1.plot(YB3,ZB3,color="k",linewidth = 0.5)  
    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=0.5)
    f3_ax1.add_artist(Drawing_uncolored_circle)

    Title = "63m upwind of Rotor Plane. \nTotal Horizontal velocity [m/s]: Time = {}[s]".format(round(Time_sampling[it],4))

    f3_ax1.set_title(Title)


    #bottom right plot
    W_r = w_r[it] #velocity time step it

    u_plane = W_r.reshape(y_r,x_r)
    X,Y = np.meshgrid(ys_r,zs_r)

    Z = u_plane

    cs = f3_ax2.contourf(X,Y,Z,levels=levels_w, cmap=cm.coolwarm,vmin=cmin_w,vmax=cmax_w)
    f3_ax2.set_xlabel("Y' axis (rotor frame of reference) [m]")
    f3_ax2.set_ylabel("Z' axis (rotor frame of reference) [m]")

    divider = make_axes_locatable(f3_ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    cb = fig.colorbar(cs,cax=cax)


    YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

    f3_ax2.plot(YB1,ZB1,color="k",linewidth = 0.5)
    f3_ax2.plot(YB2,ZB2,color="k",linewidth = 0.5)
    f3_ax2.plot(YB3,ZB3,color="k",linewidth = 0.5)  
    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=0.5)
    f3_ax2.add_artist(Drawing_uncolored_circle)

    Title = "63m upwind of Rotor Plane. \nTotal Vertical velocity [m/s]: Time = {}[s]".format(round(Time_sampling[it],4))

    f3_ax2.set_title(Title)


    #top left
    U_l = u_fluc[it]

    u_plane = U_l.reshape(x_l,y_l)
    X,Y = np.meshgrid(xs_l,ys_l)

    Z = u_plane

    cz = f3_ax3.contourf(X,Y,Z,levels=levels_l, cmap=cm.coolwarm,vmin=cmin_l,vmax=cmax_l)
    f3_ax3.set_xlabel("X axis [m]")
    f3_ax3.set_ylabel("Y axis [m]")

    x = [2524.5,2585.5]; y = [2615.1,2504.9]
    f3_ax3.plot(x,y,linewidth=1.0,color="k")

    divider = make_axes_locatable(f3_ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cd = plt.colorbar(cz, cax=cax)

    Title = "Horizontal Plane hub height. \nFluctuating Horizontal velocity [m/s]: Time = {}[s]".format(round(Time_sampling[it],4))

    f3_ax3.set_title(Title)


    #top right
    U_l = u_fluc[it]

    u_plane = U_l.reshape(x_l,y_l)
    X,Y = np.meshgrid(xs_l,ys_l)

    Z = u_plane

    cz = f3_ax4.contourf(X,Y,Z,levels=levels_l, cmap=cm.coolwarm,vmin=cmin_l,vmax=cmax_l)
    f3_ax4.set_xlabel("X axis [m]")
    f3_ax4.set_ylabel("Y axis [m]")

    f3_ax4.set_xlim([2000,3000]); f3_ax4.set_ylim([2000,3000])

    x = [2524.5,2585.5]; y = [2615.1,2504.9]
    f3_ax4.plot(x,y,linewidth=1.0,color="k")

    divider = make_axes_locatable(f3_ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cd = plt.colorbar(cz, cax=cax)

    Title = "Horizontal Plane hub height. \nFluctuating Horizontal velocity [m/s]: Time = {}[s]".format(round(Time_sampling[it],4))

    f3_ax4.set_title(Title)

    plt.tight_layout()
    plt.savefig(out_dir+"NAWEA_plot_{}.png".format(Time_idx))
    plt.cla()
    cb.remove()
    cd.remove()
    plt.close(fig)

    return Time_idx



def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(u,v,twist,x,normal,zs,h,height):
    if normal == "z":
        h_idx = np.searchsorted(h,height)
        mag_horz_vel = np.add( np.multiply(u,np.cos(twist[h_idx])) , np.multiply( v,np.sin(twist[h_idx])) )
    else:
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



start_time = time.time()

out_dir = "new_vid_plots/"
a = Dataset("Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])

Azimuth = np.array(a.variables["Azimuth"])
Azimuth = np.radians(Azimuth)

f_Az = interpolate.interp1d(Time_OF,Azimuth)

print("line 263", time.time()-start_time)


#defining twist angles with height from precursor
precursor = Dataset("abl_statistics60000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],32300)
t_end = np.searchsorted(precursor.variables["time"],33500)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
del precursor


#rotor disk data
a = Dataset("sampling_r_-63.0.nc")

Time_sampling = np.array(a.variables["time"])
Time_start = 32500; Time_end = 33701
Time_start_idx = np.searchsorted(Time_sampling,Time_start); Time_end_idx = np.searchsorted(Time_sampling,Time_end)
Time_sampling = Time_sampling[Time_start_idx:Time_end_idx]; Time_sampling = Time_sampling - Time_sampling[0]

p = a.groups["p_r"]

x_r = p.ijk_dims[0] #no. data points
y_r = p.ijk_dims[1] #no. data points

normal = int(np.degrees(np.arccos(p.axis3[0])))

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
xs = xs + rotor_coordiates[0]
ys_r = ys + rotor_coordiates[1]
zs_r = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y_r)

del a

u = np.array(p.variables["velocityx"][Time_start_idx:Time_end_idx])
v = np.array(p.variables["velocityy"][Time_start_idx:Time_end_idx])
u_r = Horizontal_velocity(u,v,twist,x_r,normal,zs_r,h,height=90); del u; del v; 
w_r = np.array(p.variables["velocityz"][Time_start_idx:Time_end_idx]); del p

cmin_r = 0
cmax_r = math.ceil(np.max(u_r))
cmin_w = math.floor(np.min(w_r))
cmax_w = math.ceil(np.max(w_r))

nlevs = (cmax_r-cmin_r)
levels_r = np.linspace(cmin_r,cmax_r,nlevs,dtype=int)
print("line 317",cmin_r,cmax_r)

nlevs = (cmax_w-cmin_w)
levels_w = np.linspace(cmin_w,cmax_w,nlevs,dtype=int)


#longitudinal plane data
a = Dataset("sampling_l_85.nc")

p = a.groups["p_l"]

x_l = p.ijk_dims[0] #no. data points
y_l = p.ijk_dims[1] #no. data points

normal = "z"

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xo = coordinates[0:x_l,0]
yo = coordinates[0:x_l,1]

xs_l = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x_l)
ys_l = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y_l)
zs_l = 0

del a

u = np.array(p.variables["velocityx"][Time_start_idx:Time_end_idx])
v = np.array(p.variables["velocityy"][Time_start_idx:Time_end_idx])
u_l = Horizontal_velocity(u,v,twist,x_l,normal,zs_l,h,height=90); del u; del v; del p
u_mean = np.mean(u_l)
u_fluc = np.subtract(u_l,u_mean)

cmin_l = math.floor(np.min(u_fluc))
cmax_l = math.ceil(np.max(u_fluc))

nlevs = (cmax_l-cmin_l)
levels_l = np.linspace(cmin_l,cmax_l,nlevs,dtype=int)
print("line 349",cmin_l,cmax_l)

Time_steps = np.arange(0,len(Time_sampling))

with Pool() as pool:
    for T in pool.imap(Update,Time_steps):

        print(T,time.time()-start_time)