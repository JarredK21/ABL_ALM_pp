import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt


def correlation_coef(x,y):

    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt


def low_pass_filter(signal, cutoff,dt):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


in_dir = "../../NREL_5MW_3.4.1/Steady_Rigid_blades_shear_0.066/"
out_dir = in_dir+"plots/"
df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

Time = np.array(df["Time_[s]"])

time_start = 200; time_end = 1201

time_start_idx = np.searchsorted(Time,time_start); time_end_idx = np.searchsorted(Time,time_end)

Time = Time[time_start_idx:time_end_idx]

dt = Time[1] - Time[0]

My = np.array(df["LSSTipMys_[kN-m]"][time_start_idx:time_end_idx])
Mz = np.array(df["LSSTipMzs_[kN-m]"][time_start_idx:time_end_idx])

My = low_pass_filter(My,1.0,dt)
Mz = low_pass_filter(Mz,1.0,dt)

corr = correlation_coef(My,Mz)

fig = plt.figure(figsize=(14,8))
plt.plot(Time,My,"r")
plt.plot(Time,Mz,"b")
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("Rotor moments [kN-m]\nlow pass filtered 1.0Hz",fontsize=16)
plt.legend(["$\widetilde{M}_y$","$\widetilde{M}_z$"],fontsize=14)
plt.xlim([200,240])
plt.title("Correlation = {}".format(round(corr,2)),fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"My_Mz.png")
plt.close()


dMy_dt = dt_calc(My,dt)
dMz_dt = dt_calc(Mz,dt)

fig = plt.figure(figsize=(14,8))
plt.plot(Time[:-1],dMy_dt,"r")
plt.plot(Time[:-1],dMz_dt,"b")
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("Time derivative Rotor moments [kN-m/s]",fontsize=16)
plt.legend(["$d\widetilde{M}_y/dt$","$d\widetilde{M}_z/dt$"],fontsize=14,loc="upper right")
plt.xlim([200,240])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"dMy_dMz_dt.png")
plt.close()

zero_crossings_My = np.where(np.diff(np.sign(dMy_dt)))[0]
zero_crossings_Mz = np.where(np.diff(np.sign(dMz_dt)))[0]
zero_crossings_Mz = zero_crossings_Mz[1:]

time_shift = []
for i,j in zip(zero_crossings_My,zero_crossings_Mz):

    time_shift.append(Time[j]-Time[i])

fig = plt.figure(figsize=(14,8))
plt.plot(time_shift)
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("Time shift [s]",fontsize=16)
plt.grid()
plt.axhline(np.mean(time_shift),linewidth=1,color="r")
plt.tight_layout()
plt.savefig(out_dir+"time_shift.png")
plt.close()