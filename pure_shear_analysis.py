import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
from netCDF4 import Dataset
from scipy import interpolate


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

df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

Time = np.array(df["Time_[s]"])

time_start = 200; time_end = 1201

time_start_idx = np.searchsorted(Time,time_start); time_end_idx = np.searchsorted(Time,time_end)

Time = Time[time_start_idx:time_end_idx]

dt = Time[1] - Time[0]

My = np.array(df["LSSTipMys_[kN-m]"][time_start_idx:time_end_idx])
Mz = np.array(df["LSSTipMzs_[kN-m]"][time_start_idx:time_end_idx])

My_LPF = low_pass_filter(My,1.0,dt)
Mz_LPF = low_pass_filter(Mz,1.0,dt)

corr = correlation_coef(My,Mz)

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
out_dir = in_dir+"Time_shift_analysis/"

fig = plt.figure(figsize=(14,8))
plt.plot(Time,My,"r")
plt.plot(Time,Mz,"b")
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("OpenFAST Rotor moments [kN-m]",fontsize=16)
plt.legend(["$\widetilde{M}_y$","$\widetilde{M}_z$"],fontsize=14)
plt.xlim([200,240])
plt.title("Steady shear profile $alpha$ = 0.066\nCorrelation = {}".format(round(corr,2)),fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"OF_data_My_Mz.png")
plt.close()


dMy_dt = dt_calc(My_LPF,dt)
dMz_dt = dt_calc(Mz_LPF,dt)

fig = plt.figure(figsize=(14,8))
plt.plot(Time[:-1],dMy_dt,"r")
plt.plot(Time[:-1],dMz_dt,"b")
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("OpenFAST Rotor moments Time derivatives [kN-m/s]",fontsize=16)
plt.legend(["$d\widetilde{M}_y/dt$","$d\widetilde{M}_z/dt$"],fontsize=14,loc="upper right")
plt.title("Steady shear profile $alpha$ = 0.066\nLow pass filer 1.0Hz")
plt.xlim([200,240])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"OF_data_dMy_dMz_dt.png")
plt.close()

zero_crossings_My = np.where(np.diff(np.sign(dMy_dt)))[0]
zero_crossings_Mz = np.where(np.diff(np.sign(dMz_dt)))[0]
zero_crossings_Mz = zero_crossings_Mz[1:]

time_shift = []
for i,j in zip(zero_crossings_My,zero_crossings_Mz):

    time_shift.append(Time[j]-Time[i])

time_temp = np.linspace(0,1200,len(time_shift))
fig = plt.figure(figsize=(14,8))
plt.plot(time_temp,time_shift)
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("Time shift [s] between OpenFAST Rotor moments",fontsize=16)
plt.grid()
plt.axhline(np.mean(time_shift),linewidth=1,color="r")
plt.legend(["Time shift between My and Mz", "Mean time shift between My and Mz"])
plt.title("{}s average time shift ~ 90deg phase shift between My and Mz zero crossings\nSteady shear profile $alpha$ = 0.066\nLow pass filtered 1.0Hz".format(np.round(np.mean(time_shift),2)))
plt.tight_layout()
plt.savefig(out_dir+"OF_data_My_Mz_time_shift.png")
plt.close()


df = Dataset(in_dir+"Dataset.nc")

Time = np.array(df.variables["time_OF"])

time_start = 200; time_end = 1201

time_start_idx = np.searchsorted(Time,time_start); time_end_idx = np.searchsorted(Time,time_end)

Time = Time[time_start_idx:time_end_idx]

dt = Time[1] - Time[0]


LSSTipMys = np.array(df.variables["LSSTipMys"][time_start_idx:time_end_idx])
LSSTipMys_LPF = low_pass_filter(LSSTipMys,1.0,dt)
LSSTipMzs = np.array(df.variables["LSSTipMzs"][time_start_idx:time_end_idx])
LSSTipMzs_LPF = low_pass_filter(LSSTipMzs,1.0,dt)

Time_sampling = np.array(df.variables["time_sampling"])

offset = "5.5"
group = df.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

Iz = -Iz

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))


corr = correlation_coef(My,Mz)

fig = plt.figure(figsize=(14,8))
plt.plot(Time,LSSTipMys,"r")
plt.plot(Time,LSSTipMzs,"b")
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("Rotor moments [kN-m]",fontsize=16)
plt.legend(["$\widetilde{M}_y$","$\widetilde{M}_z$"],fontsize=14)
plt.xlim([200,240])
plt.title("LES-Turbine\nCorrelation = {}".format(round(corr,2)),fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"LES_data_My_Mz.png")
plt.close()


dMy_dt = dt_calc(LSSTipMys_LPF,dt)
dMz_dt = dt_calc(LSSTipMzs_LPF,dt)

fig = plt.figure(figsize=(14,8))
plt.plot(Time[:-1],dMy_dt,"r")
plt.plot(Time[:-1],dMz_dt,"b")
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("Time derivative Rotor moments [kN-m/s]",fontsize=16)
plt.legend(["$d\widetilde{M}_y/dt$","$d\widetilde{M}_z/dt$"],fontsize=14,loc="upper right")
plt.title("LES-Turbine")
plt.xlim([200,240])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"LES_data_dMy_dMz_dt.png")
plt.close()

zero_crossings_My = np.where(np.diff(np.sign(dMy_dt)))[0]
zero_crossings_Mz = np.where(np.diff(np.sign(dMz_dt)))[0]
zero_crossings_Mz = zero_crossings_Mz[1:]

time_shift = []
for i,j in zip(zero_crossings_My,zero_crossings_Mz):

    time_shift.append(Time[j]-Time[i])

time_temp = np.linspace(0,1200,len(time_shift))
fig = plt.figure(figsize=(14,8))
plt.plot(time_temp,time_shift)
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("Time shift [s] between LES-Turbine Rotor moments",fontsize=16)
plt.grid()
plt.axhline(np.mean(time_shift),linewidth=1,color="r")
plt.axhline(0.413,linewidth=1,linestyle="--",color="k")
plt.legend(["Time shift beween My and Mz", "Mean time shift between My and Mz", "Phase shift = 90deg"])
plt.title("{}s average time shift between My and Mz zero crossings\n LES-Turbine".format(np.round(np.mean(time_shift),2)))
plt.tight_layout()
plt.savefig(out_dir+"LES_data_My_Mz_time_shift.png")
plt.close()


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.plot(time_temp,time_shift)
ax1.axhline(y=0.413,linestyle="--",color="k")
ax1.grid()
ax1.set_title("Time shift [s] between LES-Turbine Rotor moments")
ax2.plot(Time_sampling,Ux)
ax2.axhline(y=np.mean(Ux),linestyle="--",color="k")
ax2.grid()
ax2.set_title("Rotor averaged velocity [m/s]")

fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"LES_data_Ux_time_shift.png")
plt.close()
