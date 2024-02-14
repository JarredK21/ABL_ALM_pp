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
import os


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

    #bottom right plot
    U_r = u_r[it] #velocity time step it

    u_plane = U_r.reshape(y_r,x_r)
    X,Y = np.meshgrid(ys_r,zs_r)

    Z = u_plane

    cs = f3_ax2.contourf(X,Y,Z,levels=levels_r, cmap=cm.coolwarm,vmin=cmin_r,vmax=cmax_r)

    if contours == True:
        CS = f3_ax2.contour(X, Y, Z, levels=levels_r_neg, colors='b',linewidth=0.7)  # Negative contours default to dashed.
        f3_ax2.clabel(CS, fontsize=12, inline=True)

    f3_ax2.set_xlabel("y' axis (rotor frame of reference) [m]")
    f3_ax2.set_ylabel("z' axis (rotor frame of reference) [m]")

    divider = make_axes_locatable(f3_ax2)
    cax = divider.append_axes('right', size='5%', pad=0.05)

    cb = fig.colorbar(cs,cax=cax)

    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=0.5)
    f3_ax2.add_artist(Drawing_uncolored_circle)

    Title = "{}m from tower centerline. \nFluctuating Horizontal velocity [m/s]: Time = {}[s]".format(plane,round(Time_sampling[it],4))

    f3_ax2.set_title(Title)


    #bottom left plot
    U_l = u_22[it]

    u_plane = U_l.reshape(x_22,y_22)
    X,Y = np.meshgrid(xs_22,ys_22)

    Z = u_plane

    cz = f3_ax1.contourf(X,Y,Z,levels=levels_22, cmap=cm.coolwarm,vmin=cmin_22,vmax=cmax_22)

    if contours == True:
        CS = f3_ax1.contour(X, Y, Z, levels=levels_22_neg, colors='b',linewidth=0.7)  # Negative contours default to dashed.
        f3_ax1.clabel(CS, fontsize=12, inline=True)
        CS = f3_ax1.contour(X, Y, Z, levels=levels_22_pos, colors='r',linewidth=0.7)  # Negative contours default to dashed.
        f3_ax1.clabel(CS, fontsize=12, inline=True)

    f3_ax1.set_xlabel("x axis [m]")
    f3_ax1.set_ylabel("y axis [m]")

    f3_ax1.set_xlim([2000,3000]); f3_ax1.set_ylim([2000,3000])

    x = [2524.5,2585.5]; y = [2615.1,2504.9]
    f3_ax1.plot(x,y,linewidth=1.0,color="k")

    divider = make_axes_locatable(f3_ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cd = plt.colorbar(cz, cax=cax)

    Title = "Horizontal Plane 27.5m from surface. \nFluctuating Horizontal velocity [m/s]: Time = {}[s]".format(round(Time_sampling[it],4))

    f3_ax1.set_title(Title)


    #top left
    U_t = u_t[it]

    u_plane = U_t.reshape(y_t,x_t)
    X,Y = np.meshgrid(xs_t,zs_t)

    Z = u_plane

    cz = f3_ax3.contourf(X,Y,Z,levels=levels_t, cmap=cm.coolwarm,vmin=cmin_t,vmax=cmax_t)

    if contours == True:
        CS = f3_ax3.contour(X, Y, Z, levels=levels_t_neg, colors='b',linewidth=0.7)  # Negative contours default to dashed.
        f3_ax3.clabel(CS, fontsize=12, inline=True)

    f3_ax3.set_xlabel("x' axis [m]")
    f3_ax3.set_ylabel("z' axis [m]")

    f3_ax3.set_xlim([2000,3000]); f3_ax3.set_ylim([0,300])

    x = [2555,2555]; y = [27,153]
    f3_ax3.plot(x,y,linewidth=1.0,color="k")

    divider = make_axes_locatable(f3_ax3)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cd = plt.colorbar(cz, cax=cax)

    Title = "Transverse Plane center of rotor disk. \nFluctuating Horizontal velocity [m/s]: Time = {}[s]".format(round(Time_sampling[it],4))

    f3_ax3.set_title(Title)


    #top right
    U_l = u_85[it]

    u_plane = U_l.reshape(x_85,y_85)
    X,Y = np.meshgrid(xs_85,ys_85)

    Z = u_plane

    cz = f3_ax4.contourf(X,Y,Z,levels=levels_85, cmap=cm.coolwarm,vmin=cmin_85,vmax=cmax_85)

    if contours == True:
        CZ = f3_ax4.contour(X, Y, Z, levels=levels_85_neg, colors='b',linewidth=0.7)  # Negative contours default to dashed.
        f3_ax4.clabel(CZ, fontsize=12, inline=True)
        CZ = f3_ax4.contour(X, Y, Z, levels=levels_85_pos, colors='r',linewidth=0.7)  # Negative contours default to dashed.
        f3_ax4.clabel(CZ, fontsize=12, inline=True)

    f3_ax4.set_xlabel("x axis [m]")
    f3_ax4.set_ylabel("y axis [m]")

    f3_ax4.set_xlim([2000,3000]); f3_ax4.set_ylim([2000,3000])

    x = [2524.5,2585.5]; y = [2615.1,2504.9]
    f3_ax4.plot(x,y,linewidth=1.0,color="k")

    divider = make_axes_locatable(f3_ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cd = plt.colorbar(cz, cax=cax)

    Title = "Horizontal plane 90m from surface. \nFluctuating Horizontal velocity [m/s]: Time = {}[s]".format(round(Time_sampling[it],4))

    f3_ax4.set_title(Title)

    plt.tight_layout()
    if contours == True:
        plt.savefig(out_dir+"contour_plot_{}.png".format(Time_idx))
    else:
        plt.savefig(out_dir+"combined_plot_{}.png".format(Time_idx))
    plt.cla()
    cb.remove()
    cd.remove()
    plt.close(fig)

    return Time_idx


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


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


def level_calc(cmin,cmax):
    nlevs = int((cmax-cmin)/2)
    if nlevs>abs(cmin) or nlevs>cmax:
        nlevs = min([abs(cmin),cmax])+1

    levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
    levels = np.concatenate((levs_min,levs_max[1:]))

    return levels


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


start_time = time.time()

#plane options -5.5 or -63.0
plane = -5.5

contours = True

out_dir = "horizontal_plane_plots/"

isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)

#defining twist angles with height from precursor
precursor = Dataset("abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38000)
t_end = np.searchsorted(precursor.variables["time"],39200)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
del precursor

print("line 207")


#rotor disk data
a = Dataset("sampling_r_{}.nc".format(plane))

Time_sampling = np.array(a.variables["time"])

Time_start = 38000; Time_end = 39200
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
u[u<0] = 0; v[v<0] = 0
u = np.subtract(u,np.mean(u)); v = np.subtract(v,np.mean(v))
u_r = Horizontal_velocity(u,v,twist,x_r,normal,zs_r,h,height=90); del u; del v; 



cmin_r = math.floor(np.min(u_r))
cmax_r = math.ceil(np.max(u_r))

levels_r = level_calc(cmin_r,cmax_r)
print("line 297", levels_r)

nlevs = np.int(-0.7-cmin_r)
levels_r_neg = np.linspace(cmin_r,-0.7,nlevs)

print("lind 303", levels_r_neg)


#horizontal plane 90m data
a = Dataset("sampling_l_85.nc")

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

u = np.array(p.variables["velocityx"][Time_start_idx:Time_end_idx])
v = np.array(p.variables["velocityy"][Time_start_idx:Time_end_idx])
u[u<0] = 0; v[v<0] = 0
u = np.subtract(u,np.mean(u)); v = np.subtract(v,np.mean(v))
u_85 = Horizontal_velocity(u,v,twist,x_85,normal,zs_85,h,height=90); del u; del v; del p

cmin_85 = math.floor(np.min(u_85))
cmax_85 = math.ceil(np.max(u_85))

levels_85 = level_calc(cmin_85,cmax_85)
print("line 366", levels_85)

nlevs = int(-0.7-cmin_85)
levels_85_neg = np.linspace(cmin_85,-0.7,nlevs)
nlevs = int(cmax_85-0.7)
levels_85_pos = np.linspace(0.7,cmax_85,nlevs)
print("line 344",levels_85_neg,levels_85_pos)


#horizontal plane 22.5m
#longitudinal plane data
a = Dataset("sampling_l_22.5.nc")

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

u = np.array(p.variables["velocityx"][Time_start_idx:Time_end_idx])
v = np.array(p.variables["velocityy"][Time_start_idx:Time_end_idx])
u_l = Horizontal_velocity(u,v,twist,x_22,normal,zs_22,h,height=90); del u; del v; del p
u_l[u_l<0] = 0
u_mean = np.mean(u_l)
u_22 = np.subtract(u_l,u_mean)

cmin_22 = math.floor(np.min(u_22))
cmax_22 = math.ceil(np.max(u_22))

levels_22 = level_calc(cmin_22,cmax_22)
print("line 403", levels_22)

nlevs = int(-0.7-cmin_22)
levels_22_neg = np.linspace(cmin_22,-0.7,nlevs)
nlevs = int(cmax_22-0.7)
levels_22_pos = np.linspace(0.7,cmax_22,nlevs)
print("line 344",levels_22_neg,levels_22_pos)


#transverse plane
a = Dataset("sampling_t_0.0.nc")

p = a.groups["p_t"]

x_t = p.ijk_dims[0] #no. data points
y_t = p.ijk_dims[1] #no. data points

normal = 29

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xo = coordinates[0:x_t,0]
yo = coordinates[0:x_t,1]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-normal)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
xs_t = xs + rotor_coordiates[0]
ys_t = ys + rotor_coordiates[1]
zs_t = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y_t)

del a

u = np.array(p.variables["velocityx"][Time_start_idx:Time_end_idx])
v = np.array(p.variables["velocityy"][Time_start_idx:Time_end_idx])
u[u<0] = 0; v[v<0] = 0
u = np.subtract(u,np.mean(u)); v = np.subtract(v,np.mean(v))
u_t = Horizontal_velocity(u,v,twist,x_t,normal,zs_t,h,height=90); del u; del v; 



cmin_t = math.floor(np.min(u_t))
cmax_t = math.ceil(np.max(u_t))

levels_t = level_calc(cmin_t,cmax_t)
print("line 297", levels_t)

nlevs = np.int(-0.7-cmin_t)
levels_t_neg = np.linspace(cmin_t,-0.7,nlevs)

print("lind 303", levels_t_neg)



Time_steps = np.arange(0,len(Time_sampling))

with Pool() as pool:
    for T in pool.imap(Update,Time_steps):

        print(T,time.time()-start_time)