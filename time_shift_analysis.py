from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import interpolate

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


def low_pass_filter(signal, cutoff,dt):  
    
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

out_dir = in_dir+"Time_shift_analysis/"

a = Dataset(in_dir+"Dataset.nc")

b = Dataset(in_dir+"Dataset_2.nc")

Time_sampling = np.array(a.variables["Time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]
dt_sampling = round((Time_sampling[1] - Time_sampling[0]),2)

Time_OF = np.array(a.variables["Time_OF"])
dt_OF = Time_OF[1] - Time_OF[0]

Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
Offset_1 = Rotor_avg_vars.groups["5.5"]
Offset_2 = Rotor_avg_vars.groups["63.0"]

Ux_5 = np.array(Offset_1.variables["Ux"])
Iy_5 = np.array(Offset_1.variables["Iy"])
Iz_5 = np.array(Offset_1.variables["Iz"])
I_5 = np.sqrt(np.add(np.square(Iy_5),np.square(Iz_5)))


Ux_63 = np.array(Offset_2.variables["Ux"])
Iy_63 = np.array(Offset_2.variables["Iy"])
Iz_63 = np.array(Offset_2.variables["Iz"])
I_63 = np.sqrt(np.add(np.square(Iy_63),np.square(Iz_63)))

Ux_63_2 = np.array(Offset_2.variables["Ux"])
Iy_63_2 = np.array(Offset_2.variables["Iy"])
Iz_63_2 = np.array(Offset_2.variables["Iz"])
I_63_2 = np.sqrt(np.add(np.square(Iy_63),np.square(Iz_63)))

Time_shift = np.arange(4.0,6.5,dt_sampling)

corr_Ux = []
corr_I = []
for shift in Time_shift:

    time_shift_idx = np.searchsorted(Time_sampling,shift)

    corr_Ux.append(correlation_coef(Ux_5[time_shift_idx:],Ux_63[:-time_shift_idx]))
    corr_I.append(correlation_coef(I_5[time_shift_idx:],I_63[:-time_shift_idx]))

max_idx = corr_Ux.index(max(corr_Ux))
max_Time_shift_Ux = Time_shift[max_idx]

max_idx = corr_I.index(max(corr_I))
max_Time_shift_I = Time_shift[max_idx]

fig = plt.figure(figsize=(14,8))
plt.plot(Time_shift,corr_Ux)
plt.ylabel("Correlation coefficient",fontsize=16)
plt.xlabel("Time shift [s]",fontsize=16)
plt.title("peak in correlation for Rotor averaged velocity = {}".format(round(max_Time_shift_Ux,2)))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ux_time_shift.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.plot(Time_shift,corr_I)
plt.ylabel("Correlation coefficient",fontsize=16)
plt.xlabel("Time shift [s]",fontsize=16)
plt.title("peak in correlation for Magnitude in Asymmetry vector = {}".format(round(max_Time_shift_I,2)))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"I_time_shift.png")
plt.close()

time_shift = 4.6
Time_shift_idx = np.searchsorted(Time_sampling,time_shift)

OF_vars = a.groups["OpenFAST_Variables"]
Azimuth = np.array(OF_vars.variables["Azimuth"])
Mx = np.array(OF_vars.variables["LSShftMxa"])
My = np.array(OF_vars.variables["LSSTipMys"])
Mz = np.array(OF_vars.variables["LSSTipMys"])
MR = np.sqrt(np.add(np.square(My),np.square(Mz)))

RtAeroMyh = np.array(OF_vars.variables["RtAeroMyh"])
RtAeroMzh = np.array(OF_vars.variables["RtAeroMzh"])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000
MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

Mx_LPF = low_pass_filter(Mx,0.3,dt_OF)
f = interpolate.interp1d(Time_OF,Mx_LPF)
Mx_LPF_interp = f(Time_sampling)

Time_start_idx = np.searchsorted(Time_sampling,200)
Mx_LPF_interp = Mx_LPF_interp[Time_start_idx:]

Ux_63_LPF = low_pass_filter(Ux_63,0.3,dt_sampling)
Ux_63_LPF = Ux_63_LPF[Time_start_idx:]

Ux_63_2_LPF = low_pass_filter(Ux_63_2,0.3,dt_sampling)
Ux_63_2_LPF = Ux_63_2_LPF[Time_start_idx:]

Ux_5_LPF = low_pass_filter(Ux_5,0.3,dt_sampling)
Ux_5_LPF = Ux_5_LPF[Time_start_idx:]

f = interpolate.interp1d(Time_OF,low_pass_filter(MR,0.3,dt_OF))
MR_LPF_interp = f(Time_sampling)
MR_LPF_interp = MR_LPF_interp[Time_start_idx:]

I_63_LPF = low_pass_filter(I_63,0.3,dt_sampling)
I_63_LPF = I_63_LPF[Time_start_idx:]

I_63_2_LPF = low_pass_filter(I_63_2,0.3,dt_sampling)
I_63_2_LPF = I_63_2_LPF[Time_start_idx:]

I_5_LPF = low_pass_filter(I_5,0.3,dt_sampling)
I_5_LPF = I_5_LPF[Time_start_idx:]

Time_sampling = Time_sampling[Time_start_idx:]

cc = round(correlation_coef(Mx_LPF_interp[Time_shift_idx:],Ux_63_LPF[:-Time_shift_idx]),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling[:-Time_shift_idx],Mx_LPF_interp[Time_shift_idx:],"-r")
ax.set_ylabel("Rotor Torque LPF 0.3Hz [kN-m]")
ax2=ax.twinx()
ax2.plot(Time_sampling[:-Time_shift_idx],Ux_63_LPF[:-Time_shift_idx],"-b")
ax2.set_ylabel("Rotor averaged velocity LPF 0.3Hz\n1/2D in front, Time shifted 4.6s [m/s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ux_cc_Mx.png")
plt.close()

cc = round(correlation_coef(MR_LPF_interp[Time_shift_idx:],I_63_LPF[:-Time_shift_idx]),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling[:-Time_shift_idx],MR_LPF_interp[Time_shift_idx:],"-r")
ax.set_ylabel("Rotor OOPBM LPF 0.3Hz [kN-m]")
ax2=ax.twinx()
ax2.plot(Time_sampling[:-Time_shift_idx],I_63_LPF[:-Time_shift_idx],"-b")
ax2.set_ylabel("Magnitude Asymmetry vector LPF 0.3Hz\n1/2D in front, Time shifted 4.6s [m/s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"I_cc_MR.png")
plt.close()

cc = round(correlation_coef(Mx_LPF_interp,Ux_5_LPF),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling,Mx_LPF_interp,"-r")
ax.set_ylabel("Rotor Torque LPF 0.3Hz [kN-m]")
ax2=ax.twinx()
ax2.plot(Time_sampling,Ux_5_LPF,"-b")
ax2.set_ylabel("Rotor averaged velocity LPF 0.3Hz [m/s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ux_cc_Mx_2.png")
plt.close()

cc = round(correlation_coef(MR_LPF_interp,I_5_LPF),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling,MR_LPF_interp,"-r")
ax.set_ylabel("Rotor OOPBM LPF 0.3Hz [kN-m]")
ax2=ax.twinx()
ax2.plot(Time_sampling,I_5_LPF,"-b")
ax2.set_ylabel("Magnitude Asymmetry vector LPF 0.3Hz [m/s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"I_cc_MR_2.png")
plt.close()

cc = round(correlation_coef(Mx_LPF_interp[Time_shift_idx:],Ux_63_2_LPF[:-Time_shift_idx]),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling[:-Time_shift_idx],Mx_LPF_interp[Time_shift_idx:],"-r")
ax.set_ylabel("Rotor Torque LPF 0.3Hz [kN-m]")
ax2=ax.twinx()
ax2.plot(Time_sampling[:-Time_shift_idx],Ux_63_2_LPF[:-Time_shift_idx],"-b")
ax2.set_ylabel("Rotor averaged velocity LPF 0.3Hz\n1/2D in front, Time shifted 4.6s\nOnly using 70-80% of blade span [m/s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ux_cc_Mx_3.png")
plt.close()

cc = round(correlation_coef(MR_LPF_interp[Time_shift_idx:],I_63_2_LPF[:-Time_shift_idx]),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling[:-Time_shift_idx],MR_LPF_interp[Time_shift_idx:],"-r")
ax.set_ylabel("Rotor OOPBM LPF 0.3Hz [kN-m]")
ax2=ax.twinx()
ax2.plot(Time_sampling[:-Time_shift_idx],I_63_2_LPF[:-Time_shift_idx],"-b")
ax2.set_ylabel("Magnitude Asymmetry vector LPF 0.3Hz\n1/2D in front, Time shifted 4.6s\nOnly using 70-80% of blade span [m/s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"I_cc_MR_3.png")
plt.close()

Time_sampling = np.array(a.variables["Time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Rotor_grads = a.groups["Rotor_Gradients"]
drUx = np.array(Rotor_grads.variables["drUx"])

Rotor_grads = b.groups["Rotor_Gradients"]
drUx_2 = np.array(Rotor_grads.variables["drUx"])

cc = round(correlation_coef(drUx,drUx_2),2)
fig = plt.figure(figsize=(14,8))
plt.plot(Time_sampling,drUx,"-r",label="0-100% blade span")
plt.plot(Time_sampling,drUx_2,"-b",label="70-80% blade span")
plt.xlabel("Time [s]")
plt.ylabel("Rotor Averaged Magnitude velocity gradient [1/s]")
plt.title("correlation coefficient = {}".format(cc))
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"drUx_comp.png")
plt.close()