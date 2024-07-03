import numpy as np
import matplotlib.pyplot as plt
import pyFAST.input_output as io
from scipy import interpolate
from netCDF4 import Dataset
from multiprocessing import Pool
from scipy.signal import butter,filtfilt


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


def low_pass_filter(signal, cutoff,dt):

    fs = 1/dt     # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal



#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38000)
u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v


a = Dataset("./sampling_r_-63.0.nc")

#time options
Time = np.array(a.variables["time"])
tstart = 38000
tstart_idx = np.searchsorted(Time,tstart)
tend = 39201
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]
Time = Time-Time[0]


#rotor data
p = a.groups["p_r"]; del a

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points


normal = 29

#define plotting axes
x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points

#find normal

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
u = np.array(p.variables["velocityx"])
v = np.array(p.variables["velocityy"])

u[u<0]=0; v[v<0]=0 #remove negative velocities

with Pool() as pool:
    u_pri = []
    for u_fluc_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_pri.append(u_fluc_hvel_it)
        print(len(u_pri))
u = np.array(u_pri); del u_pri; del v
print(np.shape(u))


#define plotting axes
coordinates = np.array(p.variables["coordinates"])

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points

xo = coordinates[:,0]
yo = coordinates[:,1]
zo = coordinates[:,2]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-29)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
zs = zo - rotor_coordiates[2]

print(len(ys),len(zs))


def Ux_it_offset(it):

    Ux_rotor = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            Ux_rotor.append(u[it,ijk])
        ijk+=1
    return np.average(Ux_rotor)

Ux_avg = []
print("Ux calcs")
with Pool() as pool:
    cc = 1
    for Ux_it in pool.imap(Ux_it_offset, Time_steps):
        Ux_avg.append(Ux_it)
        print(cc)
        cc+=1

Ux_avg = np.array(Ux_avg)


del u; del p


df = io.fast_output_file.FASTOutputFile("NREL_5MW_Main.out").toDataFrame()


Time_OF = np.array(df["Time_[s]"])
dt = Time_OF[1]- Time_OF[0]

Azimuth = np.radians(np.array(df["Azimuth_[deg]"]))

RtAeroFyh = np.array(df["RtAeroFyh_[N]"])
RtAeroFzh = np.array(df["RtAeroFzh_[N]"])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(df["RtAeroMyh_[N-m]"])
RtAeroMzh = np.array(df["RtAeroMzh_[N-m]"])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

#Filtering FBR aero
LPF_1_FBR = low_pass_filter(FBR,0.3,dt)


plt.rcParams.update({'font.size': 18})

Time_shift = 4.78
Time_shift_idx = np.searchsorted(Time,Time_shift)

fig,ax=plt.subplots(figsize=(14,8),sharex=True)
ax.plot(Time[:-Time_shift_idx],Ux_avg[:-Time_shift_idx],"-b")
ax.set_ylabel("Rotor averaged\nfluctuating streamwise velocity [m/s]")
ax2=ax.twinx()
Time_shift_idx = np.searchsorted(Time_OF,Time_shift)
ax2.plot(Time_OF[:-Time_shift_idx],LPF_1_FBR[Time_shift_idx:],"-r")
ax2.set_ylabel("LPF (0.3Hz)\nMagnitude Main Bearing force vector [kN]")
ax.axhline(y=0.7,linestyle="--",color="k")
ax.axhline(y=-0.7,linestyle="--",color="k")
fig.supxlabel("Time [s]")
ax.grid()
plt.tight_layout()
plt.savefig("LPF_FBR_Ux_avg.png")
plt.close()