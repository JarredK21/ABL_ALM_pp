import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.signal import butter,filtfilt


def low_pass_filter(signal, cutoff):  
    
    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])

Azimuth = np.radians(np.array(a.variables["Azimuth"]))

RtAeroFyh = np.array(a.variables["RtAeroFyh"])
RtAeroFzh = np.array(a.variables["RtAeroFzh"])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)


RtAeroMyh = np.array(a.variables["RtAeroMyh"])
RtAeroMzh = np.array(a.variables["RtAeroMzh"])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

L1 = 1.912; L2 = 2.09


Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Rel_Aero_FBy = np.true_divide(np.square(Aero_FBy),np.square(Aero_FBR))
Rel_Aero_FBz = np.true_divide(np.square(Aero_FBz),np.square(Aero_FBR))
add_Aero_RelFB = np.add(Rel_Aero_FBy,Rel_Aero_FBz)
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))

Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

offset = "5.5"
group = a.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Uz = np.array(group.variables["Uz"])
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

in_dir = "../../ABL_precursor_2_restart/"


#loads statisitcs data
data = Dataset(in_dir+"abl_statistics70000.nc")
Mean_profiles = data.groups["mean_profiles"]


Time_2 = np.array(data.variables["time"])
tstart = 38000
tstart_idx = np.searchsorted(Time_2,tstart)
tend = 39200
tend_idx = np.searchsorted(Time_2,tend)

z_2 = np.array(Mean_profiles.variables["h"])
u_2 = np.array(Mean_profiles.variables["u"])
v_2 = np.array(Mean_profiles.variables["v"])
hvelmag_2 = []
for u_i, v_i in zip(u_2,v_2):
    hvelmag_2.append(np.add( np.multiply(np.cos(np.radians(29)),u_i), np.multiply(np.sin(np.radians(29)),v_i) ))
hvelmag_2 = np.array(hvelmag_2)
#hub height
z_hub = 90
z_hub_idx = np.searchsorted(z_2,z_hub)
hvelmag_hub_2 = hvelmag_2[:,z_hub_idx]
glob_hvelmag_hub = np.average(hvelmag_hub_2[tstart_idx:tend_idx])

fig,(ax1,ax2) = plt.subplots(2,figsize=(14,8),sharex=True)

ax1.plot(Time_sampling,Iy)
#ax1.axhline(np.mean(Ux),linestyle="--")
ax1.set_ylabel("Asymmetry around z axis",fontsize=14)
#ax1.yaxis.label.set_color("blue")

ax2.plot(Time_sampling, Iz)
ax2.set_ylabel("Bearing Force y component",fontsize=14)
#ax2.yaxis.label.set_color("red")

ax1.grid()
ax2.grid()
plt.xlabel("Time [s]",fontsize=16)
plt.suptitle("offset = -{}m".format(offset))
plt.tight_layout()
plt.show()