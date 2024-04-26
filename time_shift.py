from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def energy_contents_check(Var,e_fft,signal,dt):

    E = (1/dt)*np.sum(e_fft)

    q = np.sum(np.square(signal))

    E2 = q

    print(Var, E, E2, abs(E2/E))    


def temporal_spectra(signal,dt,Var):

    fs =1/dt
    n = len(signal) 
    if n%2==0:
        nhalf = int(n/2+1)
    else:
        nhalf = int((n+1)/2)
    frq = np.arange(nhalf)*fs/n
    Y   = np.fft.fft(signal)
    PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
    PSD[1:-1] = PSD[1:-1]*2


    energy_contents_check(Var,PSD,signal,dt)

    return frq, PSD


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


in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

out_dir = in_dir+"correlations/"

a = Dataset(in_dir+"Dataset.nc")

Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]


offset = "63.0"
group = a.groups["{}".format(offset)]
IA_63 = np.array(group.variables["IA"])
Ux_63 = np.array(group.variables["Ux"])

offset = "0.0"
group = a.groups["{}".format(offset)]
IA_5_5 = np.array(group.variables["IA"])
Ux_5_5 = np.array(group.variables["Ux"])

dt = round((Time_sampling[1] - Time_sampling[0]),2)

Time_shift = np.arange(4.0,6.5,dt)

corr = []
for shift in Time_shift:

    time_shift_idx = np.searchsorted(Time_sampling,shift)

    corr.append(correlation_coef(IA_5_5[time_shift_idx:],IA_63[:-time_shift_idx]))

max_idx = corr.index(max(corr))
max_Time_shift = Time_shift[max_idx]

fig = plt.figure(figsize=(14,8))
plt.plot(Time_shift,corr)
plt.ylabel("Correlation coefficient",fontsize=16)
plt.xlabel("Time shift [s]",fontsize=16)
plt.title("peak in correlation for Asymmetry parameter = {}".format(round(max_Time_shift,2)))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"IA_time_shift.png")
plt.cla()

corr = []
for shift in Time_shift:

    time_shift_idx = np.searchsorted(Time_sampling,shift)

    corr.append(correlation_coef(Ux_5_5[time_shift_idx:],Ux_63[:-time_shift_idx]))

max_idx = corr.index(max(corr))
max_Time_shift = Time_shift[max_idx]

fig = plt.figure(figsize=(14,8))
plt.plot(Time_sampling,Ux_5_5-np.mean(Ux_5_5),"r")
plt.plot(Time_sampling,Ux_63-np.mean(Ux_63),"b")

fig = plt.figure(figsize=(14,8))
plt.plot(Time_sampling[:-time_shift_idx],Ux_5_5[time_shift_idx:]-np.mean(Ux_5_5[time_shift_idx:]),"r")
plt.plot(Time_sampling[:-time_shift_idx],Ux_63[:-time_shift_idx]-np.mean(Ux_63[:-time_shift_idx]),"b")
plt.show()

fig = plt.figure(figsize=(14,8))
plt.plot(Time_shift,corr)
plt.ylabel("Correlation coefficient",fontsize=16)
plt.xlabel("Time shift [s]",fontsize=16)
plt.title("peak in correlation for Rotor averaged horizontal velocity = {}".format(round(max_Time_shift,2)))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ux_time_shift.png")
plt.cla()

# Time_OF = np.array(a.variables["time_OF"])
# dt_OF = Time_OF[1] - Time_OF[0]

# Azimuth = np.radians(np.array(a.variables["Azimuth"]))

# RtAeroFyh = np.array(a.variables["RtAeroFyh"])
# RtAeroFzh = np.array(a.variables["RtAeroFzh"])

# RtAeroFys = []; RtAeroFzs = []
# for i in np.arange(0,len(Time_OF)):
#     RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
#     RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
# RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)


# RtAeroMyh = np.array(a.variables["RtAeroMyh"])
# RtAeroMzh = np.array(a.variables["RtAeroMzh"])

# RtAeroMys = []; RtAeroMzs = []
# for i in np.arange(0,len(Time_OF)):
#     RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
#     RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
# RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

# L1 = 1.912; L2 = 2.09


# Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
# Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

# Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

# Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
# Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,Aero_FBR)

# frq_FBR, PSD_FBR = temporal_spectra(Aero_FBR,dt_OF,Var="FBR")
# fig = plt.figure(figsize=(14,8))
# plt.loglog(frq_FBR,PSD_FBR)
# plt.grid()

# plt.show()