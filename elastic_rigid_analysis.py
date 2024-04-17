import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
from netCDF4 import Dataset
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


in_dir = "../../NREL_5MW_3.4.1/"

out_dir = in_dir + "Steady_Elastic_blades_shear_0.085/plots/"

df_rigid = io.fast_output_file.FASTOutputFile(in_dir+"Steady_Rigid_blades_shear_0.085/NREL_5MW_Main.out").toDataFrame()

df_elastic = io.fast_output_file.FASTOutputFile(in_dir+"Steady_Elastic_blades_shear_0.085/NREL_5MW_Main.out").toDataFrame()

Time = np.array(df_rigid["Time_[s]"])

dt = Time[1] - Time[0]

Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)
Time_end = 400; Time_end_idx = np.searchsorted(Time,Time_end)


R_Fx = np.array(df_rigid["LSShftFxa_[kN]"])
R_Fy = np.array(df_rigid["LSShftFys_[kN]"])
R_Fz = np.array(df_rigid["LSShftFzs_[kN]"])
R_Mx = np.array(df_rigid["LSShftMxa_[kN-m]"])
R_My = np.array(df_rigid["LSSTipMys_[kN-m]"])
R_Mz = np.array(df_rigid["LSSTipMzs_[kN-m]"])

E_Fx = np.array(df_elastic["LSShftFxa_[kN]"])
E_Fy = np.array(df_elastic["LSShftFys_[kN]"])
E_Fz = np.array(df_elastic["LSShftFzs_[kN]"])
E_Mx = np.array(df_elastic["LSShftMxa_[kN-m]"])
E_My = np.array(df_elastic["LSSTipMys_[kN-m]"])
E_Mz = np.array(df_elastic["LSSTipMzs_[kN-m]"])


R_Fx_frq, R_Fx_PSD = temporal_spectra(R_Fx,dt,Var="R_Fx")
R_Fy_frq, R_Fy_PSD = temporal_spectra(R_Fy,dt,Var="R_Fy")
R_Fz_frq, R_Fz_PSD = temporal_spectra(R_Fz,dt,Var="R_Fz")
R_Mx_frq, R_Mx_PSD = temporal_spectra(R_Mx,dt,Var="R_Mx")
R_My_frq, R_My_PSD = temporal_spectra(R_My,dt,Var="R_My")
R_Mz_frq, R_Mz_PSD = temporal_spectra(R_Mz,dt,Var="R_Mz")

E_Fx_frq, E_Fx_PSD = temporal_spectra(E_Fx,dt,Var="E_Fx")
E_Fy_frq, E_Fy_PSD = temporal_spectra(E_Fy,dt,Var="E_Fy")
E_Fz_frq, E_Fz_PSD = temporal_spectra(E_Fz,dt,Var="E_Fz")
E_Mx_frq, E_Mx_PSD = temporal_spectra(E_Mx,dt,Var="E_Mx")
E_My_frq, E_My_PSD = temporal_spectra(E_My,dt,Var="E_My")
E_Mz_frq, E_Mz_PSD = temporal_spectra(E_Mz,dt,Var="E_Mz")


plt.rcParams.update({'font.size': 16})


cc = correlation_coef(R_My,R_Mz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time[Time_start_idx:Time_end_idx],R_My[Time_start_idx:Time_end_idx],"-b")
plt.plot(Time[Time_start_idx:Time_end_idx],R_Mz[Time_start_idx:Time_end_idx],"-r")
plt.xlabel("Time [s]")
plt.ylabel("Rigid blades - Rotor Moments [kN-m]")
plt.legend(["y component", "z component"])
plt.title("correlation coefficient 1200s = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"R_My_Mz.png")
plt.close()


cc = correlation_coef(R_Fx,E_Fx)
fig = plt.figure(figsize=(14,8))
plt.plot(Time[Time_start_idx:Time_end_idx],R_Fx[Time_start_idx:Time_end_idx],"-b")
plt.plot(Time[Time_start_idx:Time_end_idx],E_Fx[Time_start_idx:Time_end_idx],"-r")
plt.xlabel("Time [s]")
plt.ylabel("Thrust [kN]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.title("correlation coefficient 1200s = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Fx.png")
plt.close()

cc = correlation_coef(R_Fy,E_Fy)
fig = plt.figure(figsize=(14,8))
plt.plot(Time[Time_start_idx:Time_end_idx],R_Fy[Time_start_idx:Time_end_idx],"-b")
plt.plot(Time[Time_start_idx:Time_end_idx],E_Fy[Time_start_idx:Time_end_idx],"-r")
plt.xlabel("Time [s]")
plt.ylabel("Rotor Force y component [kN]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.title("correlation coefficient 1200s = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Fy.png")
plt.close()

cc = correlation_coef(R_Fz,E_Fz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time[Time_start_idx:Time_end_idx],R_Fz[Time_start_idx:Time_end_idx],"-b")
plt.plot(Time[Time_start_idx:Time_end_idx],E_Fz[Time_start_idx:Time_end_idx],"-r")
plt.xlabel("Time [s]")
plt.ylabel("Rotor Force z component [kN]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.title("correlation coefficient 1200s = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Fz.png")
plt.close()

cc = correlation_coef(R_Mx,E_Mx)
fig = plt.figure(figsize=(14,8))
plt.plot(Time[Time_start_idx:Time_end_idx],R_Mx[Time_start_idx:Time_end_idx],"-b")
plt.plot(Time[Time_start_idx:Time_end_idx],E_Mx[Time_start_idx:Time_end_idx],"-r")
plt.xlabel("Time [s]")
plt.ylabel("Torque [kN-m]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.title("correlation coefficient 1200s = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Mx.png")
plt.close()


cc = correlation_coef(R_My,E_My)
fig = plt.figure(figsize=(14,8))
plt.plot(Time[Time_start_idx:Time_end_idx],R_My[Time_start_idx:Time_end_idx],"-b")
plt.plot(Time[Time_start_idx:Time_end_idx],E_My[Time_start_idx:Time_end_idx],"-r")
plt.xlabel("Time [s]")
plt.ylabel("Rotor Moment y component [kN-m]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.title("correlation coefficient 1200s = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"My.png")
plt.close()

cc= correlation_coef(R_Mz,E_Mz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time[Time_start_idx:Time_end_idx],R_Mz[Time_start_idx:Time_end_idx],"-b")
plt.plot(Time[Time_start_idx:Time_end_idx],E_Mz[Time_start_idx:Time_end_idx],"-r")
plt.xlabel("Time [s]")
plt.ylabel("Rotor Moment z component [kN-m]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.title("correlation coefficient 1200s = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Mz.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_Fx_frq,R_Fx_PSD,"-b")
plt.loglog(E_Fx_frq,E_Fx_PSD,"-r")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor Force x component [kN]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Fx.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_Fy_frq,R_Fy_PSD,"-b")
plt.loglog(E_Fy_frq,E_Fy_PSD,"-r")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor Force y component [kN]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Fy.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_Fz_frq,R_Fz_PSD,"-b")
plt.loglog(E_Fz_frq,E_Fz_PSD,"-r")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor Force z component [kN]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Fz.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_Mx_frq,R_Mx_PSD,"-b")
plt.loglog(E_Mx_frq,E_Mx_PSD,"-r")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor Moment x component [kN-m]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Mx.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_My_frq,R_My_PSD,"-b")
plt.loglog(E_My_frq,E_My_PSD,"-r")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor Moment y component [kN-m]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_My.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_Mz_frq,R_Mz_PSD,"-b")
plt.loglog(E_Mz_frq,E_Mz_PSD,"-r")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor Moment z component [kN]")
plt.legend(["Rigid blades", "Deformable blades"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Mz.png")
plt.close()