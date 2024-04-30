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
    f3_ax3 = fig.add_subplot(gs[0, :1],projection="polar")#top left
    f3_ax4 = fig.add_subplot(gs[0, 1:])#top right


    #bottom left plot
    f3_ax1.plot(Time,Iy[:it],"-b")

    f3_ax1.set_xlabel("Time [s]")
    f3_ax1.set_ylabel("Asymmetry around y axis [$m^4/s$]")

    divider = make_axes_locatable(f3_ax1)
    cax = divider.append_axes('right', size='5%', pad=0.05)


    #bottom right plot
    f3_ax2.plot(Time,Iz[:it],"-r")
        
    f3_ax2.set_xlabel("Time [s]")
    f3_ax2.set_ylabel("Asymmetry around z axis [$m^4/s$]")


    #top left
    c = f3_ax3.scatter(Theta_FB_LPF[it], FBR_LPF[it]/np.max(FBR_LPF), c="k", s=20)
    d = f3_ax3.scatter(Theta[it],I[it]/np.max(I),c="m",s=20)
    f3_ax3.arrow(0, 0, Theta_FB_LPF[it], FBR_LPF[it]/np.max(FBR_LPF), length_includes_head=True)
    f3_ax3.arrow(0, 0, Theta[it], I[it]/np.max(I), length_includes_head=True)
    f3_ax3.set_ylim([0,1])
    f3_ax3.legend(["Bearing force vector", "Asymmetry vector"],loc="lower right")
    f3_ax3.set_title("Time = {}s".format(round(Time[it],4)), va='bottom')


    #top right
    U = u[it]

    Z = U.reshape(y,x)
    X,Y = np.meshgrid(ys,zs)


    cz = f3_ax4.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    CZ = f3_ax4.contour(X, Y, Z, levels=levels, colors='k',linewidth=0.7)  # Negative contours default to dashed.
    f3_ax4.clabel(CZ, fontsize=9, inline=True)

    f3_ax4.set_xlabel("y' axis [m]")
    f3_ax4.set_ylabel("z' axis [m]")

    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=0.5)
    f3_ax4.add_artist(Drawing_uncolored_circle)

    divider = make_axes_locatable(f3_ax4)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cd = plt.colorbar(cz, cax=cax)

    Title = "Rotor plane -66m from Rotor. \nFluctuating Streamwise velocity [m/s]: Time = {}[s]".format(round(Time[it],4))

    f3_ax4.set_title(Title)

    plt.tight_layout()
    plt.savefig(out_dir+"combined_plot_{}.png".format(Time_idx))

    plt.cla()
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


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


def low_pass_filter(signal, cutoff,dt):  
    
    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal



start_time = time.time()

#plane options -5.5 or -63.0
plane = -63.0

out_dir = "ISOplots/combined_plots_Asymmetry/"


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
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 207")


#rotor disk data
a = Dataset("sampling_r_{}.nc".format(plane))

Time = np.array(a.variables["time"])
Time = Time - Time[0]
Time_steps = np.arange(0,len(Time))


p = a.groups["p_r"]

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points

normal = int(np.degrees(np.arccos(p.axis3[0])))

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

del a

#velocity field
u = np.array(p.variables["velocityx"])
v = np.array(p.variables["velocityy"])
del p

u[u<0]=0; v[v<0]=0 #remove negative velocities

#fluctuating streamwise velocity
with Pool() as pool:
    u_pri = []
    for u_fluc_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_pri.append(u_fluc_hvel_it)
        print(len(u_pri),time.time()-start_time)
u = np.array(u); del u_pri; del v


cmin = math.floor(np.min(u))
cmax = math.ceil(np.max(u))

nlevs = int((cmax-cmin)/2)
if nlevs>abs(cmin) or nlevs>cmax:
    nlevs = min([abs(cmin),cmax])+1

levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
levels = np.concatenate((levs_min,levs_max[1:]))

print("line 304", levels)


a = Dataset("Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

dt_OF = Time_OF[1] - Time_OF[0]
dt_sampling = Time_sampling[1] - Time_sampling[0]


cutoff = 0.3
LSShftFys = np.array(a.variables["LSShftFys"])
LSShftFzs = np.array(a.variables["LSShftFzs"])
LSSTipMys = np.array(a.variables["LSSTipMys"])
LSSTipMzs = np.array(a.variables["LSSTipMzs"])

f = interpolate.interp1d(Time_OF,LSShftFys)
LSShftFys = f(Time)
LSShftFys_LPF = low_pass_filter(LSShftFys,0.3,dt_sampling)
f = interpolate.interp1d(Time_OF,LSShftFzs)
LSShftFzs = f(Time)
LSShftFzs_LPF = low_pass_filter(LSShftFzs,0.3,dt=dt_sampling)
f = interpolate.interp1d(Time_OF,LSSTipMys)
LSSTipMys = f(Time)
LSSTipMys_LPF = low_pass_filter(LSSTipMys,0.3,dt=dt_sampling)
f = interpolate.interp1d(Time_OF,LSSTipMzs)
LSSTipMzs = f(Time)
LSSTipMzs_LPF = low_pass_filter(LSSTipMzs,0.3,dt=dt_sampling)

L1 = 1.912; L2 = 2.09; L = L1 + L2

FBMy_LPF = LSSTipMzs_LPF/L2; FBFy_LPF = -LSShftFys_LPF*((L1+L2)/L2)
FBMz_LPF = -LSSTipMys_LPF/L2; FBFz_LPF = -LSShftFzs_LPF*((L1+L2)/L2)

FBy_LPF = FBMy_LPF + FBFy_LPF; FBz_LPF = FBMz_LPF + FBFz_LPF
FBR_LPF = np.sqrt(np.add(np.square(FBy_LPF),np.square(FBz_LPF)))
Theta_FB_LPF = np.degrees(np.arctan2(FBz_LPF,FBy_LPF))
Theta_FB_LPF = theta_360(Theta_FB_LPF)
Theta_FB_LPF = np.radians(np.array(Theta_FB_LPF))


offset = "63.0"
group = a.groups["{}".format(offset)]
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = -np.array(group.variables["Iz"])
Iy = low_pass_filter(Iy,cutoff,dt_sampling)
Iz = low_pass_filter(Iz,cutoff,dt_sampling)

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

Theta = np.degrees(np.arctan2(Iz,Iy))
Theta = theta_360(Theta)
Theta = np.radians(np.array(Theta))


time_shift = 4.78
time_shift_idx = np.searchsorted(Time_OF,time_shift)

Time = Time[:-time_shift_idx]
I = I[:-time_shift_idx]
Iy = Iy[:-time_shift_idx]
Iz = Iz[:-time_shift_idx]
FBR_LPF = FBR_LPF[time_shift_idx:]
Theta_FB_LPF = Theta_FB_LPF[time_shift_idx:]
u = u[:-time_shift_idx]

Time_steps = np.arange(0,len(Time))

with Pool() as pool:
    for T in pool.imap(Update,Time_steps):

        print(T,time.time()-start_time)