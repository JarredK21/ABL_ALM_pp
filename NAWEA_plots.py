import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset


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



in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"
out_dir = "../../NAWEA_23/post_processing/plots_3/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]
#Time_end = 250

dt = Time_OF[1] - Time_OF[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])
LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

frq, RtAeroMxh_FFT = temporal_spectra(RtAeroMxh/1000,dt,Var="Torque")
frq, LSSTipMR_FFT = temporal_spectra(LSSTipMR, dt, Var="LSSTipMR")

RtAeroMxh_LPF_low = low_pass_filter(RtAeroMxh,0.1)
RtAeroMxh_LPF_3p = low_pass_filter(RtAeroMxh,1)
RtAeroMxh_LPF_40 = low_pass_filter(RtAeroMxh, 40)

group = a.groups["0.0"]
Ux = np.array(group.variables["Ux"])
IA = np.array(group.variables["IA"])
Uy = np.array(group.variables["Uy"])
Uz = np.array(group.variables["Uz"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

f = interpolate.interp1d(Time_sampling,Ux)
Ux = f(Time_OF)

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF, RtAeroMxh/1000)
# plt.ylabel("Rotor Torque [kN-m]",fontsize=16)
# plt.xlabel("Time [s]",fontsize=16)
# plt.tight_layout()
# plt.savefig(out_dir+"Torque.png")

fig = plt.figure(figsize=(14,8))
plt.plot(frq, LSSTipMR_FFT)
plt.yscale("log")
plt.xscale("log")
plt.ylabel("Power spectral density",fontsize=16)
plt.xlabel("Time [s]",fontsize=16)
plt.title("Rotor Out-of-plane bending Moment [$(kN-m)^2$]",fontsize=18)
plt.tight_layout()
plt.savefig(out_dir+"OOPBM_FFT.png")

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF, RtAeroMxh_LPF_low/1000)
# plt.ylabel("Rotor Torque [kN-m]",fontsize=16)
# plt.xlabel("Time [s]",fontsize=16)
# plt.title("Low pass filtered 0.1Hz",fontsize=18)
# plt.tight_layout()
# plt.savefig(out_dir+"Torque_LPF_low.png")

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF, RtAeroMxh_LPF_3p/1000)
# plt.ylabel("Rotor Torque [kN-m]",fontsize=16)
# plt.xlabel("Time [s]",fontsize=16)
# plt.title("Low pass filtered 1Hz",fontsize=18)
# plt.tight_layout()
# plt.savefig(out_dir+"Torque_LPF_3p.png")

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF, RtAeroMxh_LPF_40/1000)
# plt.ylabel("Rotor Torque [kN-m]",fontsize=16)
# plt.xlabel("Time [s]",fontsize=16)
# plt.title("Low pass filtered 40Hz",fontsize=18)
# plt.tight_layout()
# plt.savefig(out_dir+"Torque_LPF_40.png")
