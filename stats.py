from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
import matplotlib.pyplot as plt


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


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


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]
dt_sampling = Time_sampling[1] - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

time_shift_idx = np.searchsorted(Time_OF,4.78)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

Azimuth = np.radians(np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx]))

RtAeroFyh = np.array(a.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(a.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)

RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)


LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])


L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))


Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))


print(np.mean(Aero_FBy)/1000)
print(np.std(Aero_FBy)/1000)
print(np.mean(Aero_FBz)/1000)
print(np.std(Aero_FBz)/1000)

print(np.mean(Aero_FBMy)/1000)
print(np.std(Aero_FBMy)/1000)
print(np.mean(Aero_FBFy)/1000)
print(np.std(Aero_FBFy)/1000)

print(np.mean(Aero_FBMz)/1000)
print(np.std(Aero_FBMz)/1000)
print(np.mean(Aero_FBFz)/1000)
print(np.std(Aero_FBFz)/1000)


print(correlation_coef(RtAeroMzs,Aero_FBy))
print(correlation_coef(RtAeroFys,Aero_FBy))

print(correlation_coef(RtAeroMys,Aero_FBz))
print(correlation_coef(RtAeroFzs,Aero_FBz))

LPF_My = low_pass_filter(RtAeroMys,cutoff=0.3)
LPF_Mz = low_pass_filter(RtAeroMzs,cutoff=0.3)

print(correlation_coef(LPF_My,LPF_Mz))

fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time_OF,LPF_My/1000,'-b')
ax.set_ylabel("Aerodynamic rotor moment component y [kN-m]",fontsize=14)
ax.yaxis.label.set_color('blue')
ax2 = ax.twinx()
ax2.plot(Time_OF,LPF_Mz/1000,"-r")
ax2.set_ylabel("Aerodynamic rotor moment component z [kN-m]",fontsize=14)
ax2.yaxis.label.set_color('red')
ax.set_xlabel("Time [s]",fontsize=16)
ax.axhline(y=np.mean(RtAeroMzs/1000),linestyle="--",color="k")
ax2.axhline(y=np.mean(RtAeroMzs/1000),linestyle="--",color="k")
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"FW&M_plots/My_Mz.png")
plt.close()

# offset = "5.5"
# group = a.groups["{}".format(offset)]
# Ux_1 = np.array(group.variables["Ux"])
# Uz_1 = np.array(group.variables["Uz"])
# IA_1 = np.array(group.variables["IA"])
# Iy_1 = np.array(group.variables["Iy"])
# Iz_1 = np.array(group.variables["Iz"])

offset = "63.0"
group = a.groups["{}".format(offset)]
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))


print("Iy_Iz",correlation_coef(Iy,Iz))

fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time_sampling,Iy,'-b')
ax.set_ylabel("Asymmetry vector component y [$m^4/s$]",fontsize=14)
ax.yaxis.label.set_color('blue')
ax2 = ax.twinx()
ax2.plot(Time_sampling,Iz,"-r")
ax2.set_ylabel("Asymmetry vector component z [$m^4/s$]",fontsize=14)
ax2.yaxis.label.set_color('red')
ax.set_xlabel("Time [s]",fontsize=16)
ax.axhline(y=np.mean(Iy),linestyle="--",color="k")
ax2.axhline(y=np.mean(Iz),linestyle="--",color="k")
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"FW&M_plots/Iy_Iz.png")
plt.close()

# offset = "5.5"
# group = a.groups["{}".format(offset)]
# Ux = np.array(group.variables["Ux"])
# Uz = np.array(group.variables["Uz"])
# IA = np.array(group.variables["IA"])
# Iy = np.array(group.variables["Iy"])
# Iz = np.array(group.variables["Iz"])

# f = interpolate.interp1d(Time_sampling,Ux)
# Ux = f(Time_OF)

# f = interpolate.interp1d(Time_sampling,Uz)
# Uz = f(Time_OF)

# f = interpolate.interp1d(Time_sampling,IA)
# IA = f(Time_OF)

# f = interpolate.interp1d(Time_sampling,Iy)
# Iy = f(Time_OF)

# f = interpolate.interp1d(Time_sampling,Iz)
# Iz = f(Time_OF)

cutoff = 0.3
f = interpolate.interp1d(Time_sampling,I)
I_interp = f(Time_OF)
I_LPF = low_pass_filter(I_interp,cutoff)

LPF_Aero_FBR = low_pass_filter(Aero_FBR,cutoff)


cc = round(correlation_coef(I_LPF[:-time_shift_idx],LPF_Aero_FBR[time_shift_idx:]/1000),2)
fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time_OF[:-time_shift_idx],LPF_Aero_FBR[time_shift_idx:]/1000,'-b')
ax.set_ylabel("Magnitude main bearing force [$kN$]",fontsize=14)
ax.yaxis.label.set_color('blue')
ax2 = ax.twinx()
ax2.plot(Time_OF[:-time_shift_idx],I_LPF[:-time_shift_idx],"-r")
ax2.set_ylabel("Asymmetry parameter [$m^4/s$]",fontsize=14)
ax2.yaxis.label.set_color('red')
plt.title("Correlation coefficient {}".format(cc),fontsize=16)
ax.set_xlabel("Time [s]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"FW&M_plots/LPF_I_FBR.png")
plt.close()

cc = round(correlation_coef(I_interp[:-time_shift_idx],Aero_FBR[time_shift_idx:]/1000),2)
fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time_OF[:-time_shift_idx],Aero_FBR[time_shift_idx:]/1000,'-b')
ax.set_ylabel("Magnitude main bearing force [$kN$]",fontsize=14)
ax.yaxis.label.set_color('blue')
ax2 = ax.twinx()
ax2.plot(Time_OF[:-time_shift_idx],I_interp[:-time_shift_idx],"-r")
ax2.set_ylabel("Asymmetry parameter [$m^4/s$]",fontsize=14)
ax2.yaxis.label.set_color('red')
plt.title("Correlation coefficient {}".format(cc),fontsize=16)
ax.set_xlabel("Time [s]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"FW&M_plots/I_FBR.png")
plt.close()

frq_I,PSD_I = temporal_spectra(I,dt_sampling,Var="IA")
frq_FBR,PSD_FBR = temporal_spectra(Aero_FBR,dt,Var="FBR")

fig,ax = plt.subplots(figsize=(14,8))

ax.loglog(frq_I,PSD_I,'-b')
ax.set_ylabel("PSD Asymmetry Vector [$m^4/s$]",fontsize=14)
ax.yaxis.label.set_color('blue')
ax2 = ax.twinx()
ax2.loglog(frq_FBR,PSD_FBR,"-r")
ax2.set_ylabel("PSD Magnitude main bearing force [$kN$]",fontsize=14)
ax2.yaxis.label.set_color('red')
ax.set_xlabel("Frequency [Hz]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"FW&M_plots/Spectra_I_FBR.png")
plt.close()
