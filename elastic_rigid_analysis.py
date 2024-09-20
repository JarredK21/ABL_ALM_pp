import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter,filtfilt
from netCDF4 import Dataset
from scipy import interpolate
from scipy.fft import fft, fftfreq, fftshift,ifft


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


def hard_filter(signal,cutoff,dt,filter_type):

    N = len(signal)
    spectrum = fft(signal)
    F = fftfreq(N,dt)
    if filter_type=="lowpass":
        spectrum_filter = spectrum*(np.absolute(F)<cutoff)
    elif filter_type=="highpass":
        spectrum_filter = spectrum*(np.absolute(F)>cutoff)
    elif filter_type=="bandpass":
        spectrum_filter = spectrum*(np.absolute(F)>cutoff[0])
        spectrum_filter = spectrum_filter*(np.absolute(F)<cutoff[1])
        

    spectrum_filter = ifft(spectrum_filter)

    return np.real(spectrum_filter)


def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    dX = X[1] - X[0]
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    print(np.sum(P)*dX)
    return P,X


def moments(y):
    mu = np.mean(y)
    std = np.std(y)
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    return mu, std, skewness,kurotsis



in_dir = "../../NREL_5MW_MCBL_E_CRPM/post_processing/"

out_dir = in_dir+"Rotor_Var_plots/"

df_elastic = Dataset(in_dir+"Dataset.nc")

Time = np.array(df_elastic["Time_OF"])

dt_OF = Time[1] - Time[0]

Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)

OF_Vars = df_elastic.groups["OpenFAST_Variables"]

E_Fx = np.array(OF_Vars["LSShftFxa"][Time_start_idx:])
E_Fy = np.array(OF_Vars["LSShftFys"][Time_start_idx:])
E_Fz = np.array(OF_Vars["LSShftFzs"][Time_start_idx:])
E_Mx = np.array(OF_Vars["LSShftMxa"][Time_start_idx:])
E_My = np.array(OF_Vars["LSSTipMys"][Time_start_idx:])
E_Mz = np.array(OF_Vars["LSSTipMzs"][Time_start_idx:])
E_MR = np.sqrt(np.add(np.square(E_My),np.square(E_Mz)))

moms = moments(E_Fx)
E_Fx_moms = []
for m in moms:
    E_Fx_moms.append(round(m,2))

moms = moments(E_Fy)
E_Fy_moms = []
for m in moms:
    E_Fy_moms.append(round(m,2))

moms = moments(E_Fz)
E_Fz_moms = []
for m in moms:
    E_Fz_moms.append(round(m,2))

moms = moments(E_Mx)
E_Mx_moms = []
for m in moms:
    E_Mx_moms.append(round(m,2))

moms = moments(E_My)
E_My_moms = []
for m in moms:
    E_My_moms.append(round(m,2))

moms = moments(E_Mz)
E_Mz_moms = []
for m in moms:
    E_Mz_moms.append(round(m,2))

moms = moments(E_MR)
E_MR_moms = []
for m in moms:
    E_MR_moms.append(round(m,2))

E_Fx_frq, E_Fx_PSD = temporal_spectra(E_Fx,dt_OF,Var="E_Fx")
E_Fy_frq, E_Fy_PSD = temporal_spectra(E_Fy,dt_OF,Var="E_Fy")
E_Fz_frq, E_Fz_PSD = temporal_spectra(E_Fz,dt_OF,Var="E_Fz")
E_Mx_frq, E_Mx_PSD = temporal_spectra(E_Mx,dt_OF,Var="E_Mx")
E_My_frq, E_My_PSD = temporal_spectra(E_My,dt_OF,Var="E_My")
E_Mz_frq, E_Mz_PSD = temporal_spectra(E_Mz,dt_OF,Var="E_Mz")
E_MR_frq, E_MR_PSD = temporal_spectra(E_MR,dt_OF,Var="E_MR")

E_LPF_My = hard_filter(E_My,0.1,dt_OF,"lowpass")
E_BPF_My = hard_filter(E_My,[0.3,0.9],dt_OF,"bandpass")
E_HPF_My = hard_filter(E_My,[1.5,40],dt_OF,"bandpass")

moms = moments(E_LPF_My)
E_LPF_My_moms = []
for m in moms:
    E_LPF_My_moms.append(round(m,2))

moms = moments(E_BPF_My)
E_BPF_My_moms = []
for m in moms:
    E_BPF_My_moms.append(round(m,2))

moms = moments(E_HPF_My)
E_HPF_My_moms = []
for m in moms:
    E_HPF_My_moms.append(round(m,2))

E_LPF_Mz = hard_filter(E_Mz,0.1,dt_OF,"lowpass")
E_BPF_Mz = hard_filter(E_Mz,[0.3,0.9],dt_OF,"bandpass")
E_HPF_Mz = hard_filter(E_Mz,[1.5,40],dt_OF,"bandpass")

moms = moments(E_LPF_Mz)
E_LPF_Mz_moms = []
for m in moms:
    E_LPF_Mz_moms.append(round(m,2))

moms = moments(E_BPF_Mz)
E_BPF_Mz_moms = []
for m in moms:
    E_BPF_Mz_moms.append(round(m,2))

moms = moments(E_HPF_Mz)
E_HPF_Mz_moms = []
for m in moms:
    E_HPF_Mz_moms.append(round(m,2))

E_LPF_MR = hard_filter(E_MR,0.1,dt_OF,"lowpass")
E_BPF_MR = hard_filter(E_MR,[0.3,0.9],dt_OF,"bandpass")
E_HPF_MR = hard_filter(E_MR,[1.5,40],dt_OF,"bandpass")

moms = moments(E_LPF_MR)
E_LPF_MR_moms = []
for m in moms:
    E_LPF_MR_moms.append(round(m,2))

moms = moments(E_BPF_MR)
E_BPF_MR_moms = []
for m in moms:
    E_BPF_MR_moms.append(round(m,2))

moms = moments(E_HPF_MR)
E_HPF_MR_moms = []
for m in moms:
    E_HPF_MR_moms.append(round(m,2))


df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

E_FLx = np.array(df["B1N016FLx_[kN]"][Time_start_idx:])#flapwise
E_FLy = np.array(df["B1N016FLy_[kN]"][Time_start_idx:])#edgewise
E_FLz = np.array(df["B1N016FLy_[kN]"][Time_start_idx:])#axial
E_MLx = np.array(df["B1N016MLx_[kN-m]"][Time_start_idx:])#edgewise
E_MLy = np.array(df["B1N016MLy_[kN-m]"][Time_start_idx:])#flapwise
E_MLz = np.array(df["B1N016MLz_[kN-m]"][Time_start_idx:])#pitching

moms = moments(E_FLx)
E_FLx_moms = []
for m in moms:
    E_FLx_moms.append(round(m,2))

moms = moments(E_FLy)
E_FLy_moms = []
for m in moms:
    E_FLy_moms.append(round(m,2))

moms = moments(E_FLz)
E_FLz_moms = []
for m in moms:
    E_FLz_moms.append(round(m,2))

moms = moments(E_MLx)
E_MLx_moms = []
for m in moms:
    E_MLx_moms.append(round(m,2))

moms = moments(E_MLy)
E_MLy_moms = []
for m in moms:
    E_MLy_moms.append(round(m,2))

moms = moments(E_MLz)
E_MLz_moms = []
for m in moms:
    E_MLz_moms.append(round(m,2))

E_FLx_frq, E_FLx_PSD = temporal_spectra(E_FLx,dt_OF,Var="E_Fx")
E_FLy_frq, E_FLy_PSD = temporal_spectra(E_FLy,dt_OF,Var="E_Fy")
E_FLz_frq, E_FLz_PSD = temporal_spectra(E_FLz,dt_OF,Var="E_Fz")
E_MLx_frq, E_MLx_PSD = temporal_spectra(E_MLx,dt_OF,Var="E_Mx")
E_MLy_frq, E_MLy_PSD = temporal_spectra(E_MLy,dt_OF,Var="E_My")
E_MLz_frq, E_MLz_PSD = temporal_spectra(E_MLz,dt_OF,Var="E_Mz")



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"


df_rigid = Dataset(in_dir+"Dataset.nc")

Time = np.array(df_rigid["Time_OF"])

dt_OF = Time[1] - Time[0]

Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)

Time = Time[Time_start_idx:]

OF_Vars = df_rigid.groups["OpenFAST_Variables"]

R_Fx = np.array(OF_Vars["LSShftFxa"][Time_start_idx:])
R_Fy = np.array(OF_Vars["LSShftFys"][Time_start_idx:])
R_Fz = np.array(OF_Vars["LSShftFzs"][Time_start_idx:])
R_Mx = np.array(OF_Vars["LSShftMxa"][Time_start_idx:])
R_My = np.array(OF_Vars["LSSTipMys"][Time_start_idx:])
R_Mz = np.array(OF_Vars["LSSTipMzs"][Time_start_idx:])
R_MR = np.sqrt(np.add(np.square(R_My),np.square(R_Mz)))


moms = moments(R_Fx)
R_Fx_moms = []
for m in moms:
    R_Fx_moms.append(round(m,2))

moms = moments(R_Fy)
R_Fy_moms = []
for m in moms:
    R_Fy_moms.append(round(m,2))

moms = moments(R_Fz)
R_Fz_moms = []
for m in moms:
    R_Fz_moms.append(round(m,2))

moms = moments(R_Mx)
R_Mx_moms = []
for m in moms:
    R_Mx_moms.append(round(m,2))

moms = moments(R_My)
R_My_moms = []
for m in moms:
    R_My_moms.append(round(m,2))

moms = moments(R_Mz)
R_Mz_moms = []
for m in moms:
    R_Mz_moms.append(round(m,2))

moms = moments(R_MR)
R_MR_moms = []
for m in moms:
    R_MR_moms.append(round(m,2))

R_Fx_frq, R_Fx_PSD = temporal_spectra(R_Fx,dt_OF,Var="E_Fx")
R_Fy_frq, R_Fy_PSD = temporal_spectra(R_Fy,dt_OF,Var="E_Fy")
R_Fz_frq, R_Fz_PSD = temporal_spectra(R_Fz,dt_OF,Var="E_Fz")
R_Mx_frq, R_Mx_PSD = temporal_spectra(R_Mx,dt_OF,Var="E_Mx")
R_My_frq, R_My_PSD = temporal_spectra(R_My,dt_OF,Var="E_My")
R_Mz_frq, R_Mz_PSD = temporal_spectra(R_Mz,dt_OF,Var="E_Mz")
R_MR_frq, R_MR_PSD = temporal_spectra(R_MR,dt_OF,Var="E_MR")

R_LPF_My = hard_filter(R_My,0.1,dt_OF,"lowpass")
R_BPF_My = hard_filter(R_My,[0.3,0.9],dt_OF,"bandpass")
R_HPF_My = hard_filter(R_My,[1.5,40],dt_OF,"bandpass")

moms = moments(R_LPF_My)
R_LPF_My_moms = []
for m in moms:
    R_LPF_My_moms.append(round(m,2))

moms = moments(R_BPF_My)
R_BPF_My_moms = []
for m in moms:
    R_BPF_My_moms.append(round(m,2))

moms = moments(R_HPF_My)
R_HPF_My_moms = []
for m in moms:
    R_HPF_My_moms.append(round(m,2))

R_LPF_Mz = hard_filter(R_Mz,0.1,dt_OF,"lowpass")
R_BPF_Mz = hard_filter(R_Mz,[0.3,0.9],dt_OF,"bandpass")
R_HPF_Mz = hard_filter(R_Mz,[1.5,40],dt_OF,"bandpass")

moms = moments(R_LPF_Mz)
R_LPF_Mz_moms = []
for m in moms:
    R_LPF_Mz_moms.append(round(m,2))

moms = moments(R_BPF_Mz)
R_BPF_Mz_moms = []
for m in moms:
    R_BPF_Mz_moms.append(round(m,2))

moms = moments(R_HPF_Mz)
R_HPF_Mz_moms = []
for m in moms:
    R_HPF_Mz_moms.append(round(m,2))

R_LPF_MR = hard_filter(R_MR,0.1,dt_OF,"lowpass")
R_BPF_MR = hard_filter(R_MR,[0.3,0.9],dt_OF,"bandpass")
R_HPF_MR = hard_filter(R_MR,[1.5,40],dt_OF,"bandpass")

moms = moments(R_LPF_MR)
R_LPF_MR_moms = []
for m in moms:
    R_LPF_MR_moms.append(round(m,2))

moms = moments(R_BPF_MR)
R_BPF_MR_moms = []
for m in moms:
    R_BPF_MR_moms.append(round(m,2))

moms = moments(R_HPF_MR)
R_HPF_MR_moms = []
for m in moms:
    R_HPF_MR_moms.append(round(m,2))

df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

R_FLx = np.array(df["B1N016FLx_[kN]"][Time_start_idx:])#flapwise
R_FLy = np.array(df["B1N016FLy_[kN]"][Time_start_idx:])#edgewise
R_FLz = np.array(df["B1N016FLy_[kN]"][Time_start_idx:])#axial
R_MLx = np.array(df["B1N016MLx_[kN-m]"][Time_start_idx:])#edgewise
R_MLy = np.array(df["B1N016MLy_[kN-m]"][Time_start_idx:])#flapwise
R_MLz = np.array(df["B1N016MLz_[kN-m]"][Time_start_idx:])#pitching

moms = moments(R_FLx)
R_FLx_moms = []
for m in moms:
    R_FLx_moms.append(round(m,2))

moms = moments(R_FLy)
R_FLy_moms = []
for m in moms:
    R_FLy_moms.append(round(m,2))

moms = moments(R_FLz)
R_FLz_moms = []
for m in moms:
    R_FLz_moms.append(round(m,2))

moms = moments(R_MLx)
R_MLx_moms = []
for m in moms:
    R_MLx_moms.append(round(m,2))

moms = moments(R_MLy)
R_MLy_moms = []
for m in moms:
    R_MLy_moms.append(round(m,2))

moms = moments(R_MLz)
R_MLz_moms = []
for m in moms:
    R_MLz_moms.append(round(m,2))

R_FLx_frq, R_FLx_PSD = temporal_spectra(R_FLx,dt_OF,Var="R_Fx")
R_FLy_frq, R_FLy_PSD = temporal_spectra(R_FLy,dt_OF,Var="R_Fy")
R_FLz_frq, R_FLz_PSD = temporal_spectra(R_FLz,dt_OF,Var="R_Fz")
R_MLx_frq, R_MLx_PSD = temporal_spectra(R_MLx,dt_OF,Var="R_Mx")
R_MLy_frq, R_MLy_PSD = temporal_spectra(R_MLy,dt_OF,Var="R_My")
R_MLz_frq, R_MLz_PSD = temporal_spectra(R_MLz,dt_OF,Var="R_Mz")


plt.rcParams.update({'font.size': 16})


cc = correlation_coef(R_Fx,E_Fx)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_Fx,"-r")
plt.plot(Time,E_Fx,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Rotor Thrust [kN]")
plt.legend(["Rigid rotor\n{}".format(R_Fx_moms), "Elastic Rotor\n{}".format(E_Fx_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Fx.png")
plt.close()


cc = correlation_coef(R_Fy,E_Fy)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_Fy,"-r")
plt.plot(Time,E_Fy,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Rotor force y component [kN]")
plt.legend(["Rigid rotor\n{}".format(R_Fy_moms), "Elastic Rotor\n{}".format(E_Fy_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Fy.png")
plt.close()

cc = correlation_coef(R_Fz,E_Fz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_Fz,"-r")
plt.plot(Time,E_Fz,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Rotor force z component [kN]")
plt.legend(["Rigid rotor\n{}".format(R_Fz_moms), "Elastic Rotor\n{}".format(E_Fz_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Fz.png")
plt.close()


cc = correlation_coef(R_Mx,E_Mx)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_Mx,"-r")
plt.plot(Time,E_Mx,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Rotor Torque [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_Mx_moms), "Elastic Rotor\n{}".format(E_Mx_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Mx.png")
plt.close()


cc = correlation_coef(R_My,E_My)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_My,"-r")
plt.plot(Time,E_My,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Rotor moment y component [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_My_moms), "Elastic Rotor\n{}".format(E_My_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"My.png")
plt.close()

cc = correlation_coef(R_LPF_My,E_LPF_My)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_LPF_My,"-r")
plt.plot(Time,E_LPF_My,"-b")
plt.xlabel("Time [s]")
plt.ylabel("LPF Rotor moment y component [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_LPF_My_moms), "Elastic Rotor\n{}".format(E_LPF_My_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"LPF_My.png")
plt.close()

cc = correlation_coef(R_BPF_My,E_BPF_My)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_BPF_My,"-r")
plt.plot(Time,E_BPF_My,"-b")
plt.xlabel("Time [s]")
plt.ylabel("BPF Rotor moment y component [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_BPF_My_moms), "Elastic Rotor\n{}".format(E_BPF_My_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"BPF_My.png")
plt.close()

cc = correlation_coef(R_HPF_My,E_HPF_My)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_HPF_My,"-r")
plt.plot(Time,E_HPF_My,"-b")
plt.xlabel("Time [s]")
plt.ylabel("HPF Rotor moment y component [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_HPF_My_moms), "Elastic Rotor\n{}".format(E_HPF_My_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"HPF_My.png")
plt.close()

cc = correlation_coef(R_Mz,E_Mz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_Mz,"-r")
plt.plot(Time,E_Mz,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Rotor moment z component [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_Mz_moms), "Elastic Rotor\n{}".format(E_Mz_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Mz.png")
plt.close()

cc = correlation_coef(R_LPF_Mz,E_LPF_Mz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_LPF_Mz,"-r")
plt.plot(Time,E_LPF_Mz,"-b")
plt.xlabel("Time [s]")
plt.ylabel("LPF Rotor moment z component [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_LPF_Mz_moms), "Elastic Rotor\n{}".format(E_LPF_Mz_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"LPF_Mz.png")
plt.close()

cc = correlation_coef(R_BPF_Mz,E_BPF_Mz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_BPF_Mz,"-r")
plt.plot(Time,E_BPF_Mz,"-b")
plt.xlabel("Time [s]")
plt.ylabel("BPF Rotor moment z component [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_BPF_Mz_moms), "Elastic Rotor\n{}".format(E_BPF_Mz_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"BPF_Mz.png")
plt.close()

cc = correlation_coef(R_HPF_Mz,E_HPF_Mz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_HPF_Mz,"-r")
plt.plot(Time,E_HPF_Mz,"-b")
plt.xlabel("Time [s]")
plt.ylabel("HPF Rotor moment z component [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_HPF_Mz_moms), "Elastic Rotor\n{}".format(E_HPF_Mz_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"HPF_Mz.png")
plt.close()

cc = correlation_coef(R_MR,E_MR)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_MR,"-r")
plt.plot(Time,E_MR,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Rotor out-of-plane bending moment [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_MR_moms), "Elastic Rotor\n{}".format(E_MR_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"MR.png")
plt.close()

cc = correlation_coef(R_LPF_MR,E_LPF_MR)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_LPF_MR,"-r")
plt.plot(Time,E_LPF_MR,"-b")
plt.xlabel("Time [s]")
plt.ylabel("LPF Rotor out-of-plane bending moment [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_LPF_MR_moms), "Elastic Rotor\n{}".format(E_LPF_MR_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"LPF_MR.png")
plt.close()

cc = correlation_coef(R_BPF_MR,E_BPF_MR)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_BPF_MR,"-r")
plt.plot(Time,E_BPF_MR,"-b")
plt.xlabel("Time [s]")
plt.ylabel("BPF Rotor out-of-plane bending moment [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_BPF_MR_moms), "Elastic Rotor\n{}".format(E_BPF_MR_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"BPF_MR.png")
plt.close()

cc = correlation_coef(R_HPF_MR,E_HPF_MR)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_HPF_MR,"-r")
plt.plot(Time,E_HPF_MR,"-b")
plt.xlabel("Time [s]")
plt.ylabel("HPF Rotor out-of-plane bending moment [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_HPF_MR_moms), "Elastic Rotor\n{}".format(E_HPF_MR_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"HPF_MR.png")
plt.close()


fig = plt.figure(figsize=(14,8))
plt.loglog(R_Mx_frq,R_Mx_PSD,"-r")
plt.loglog(E_Mx_frq,E_Mx_PSD,"-b")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor torque PSD [kN-m]")
plt.legend(["Rigid rotor", "Elastic Rotor"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Mx.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_MR_frq,R_MR_PSD,"-r")
plt.loglog(E_MR_frq,E_MR_PSD,"-b")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor out-of-plane bending moment PSD [kN-m]")
plt.legend(["Rigid rotor", "Elastic Rotor"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_MR.png")
plt.close()


cc = correlation_coef(R_FLx,E_FLx)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_FLx,"-r")
plt.plot(Time,E_FLx,"-b")
plt.xlabel("Time [s]")
plt.ylabel("flapwise force 75% span [kN]")
plt.legend(["Rigid rotor\n{}".format(R_FLx_moms), "Elastic Rotor\n{}".format(E_FLx_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"FLx.png")
plt.close()


cc = correlation_coef(R_FLy,E_FLy)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_FLy,"-r")
plt.plot(Time,E_FLy,"-b")
plt.xlabel("Time [s]")
plt.ylabel("edgewise force 75% span [kN]")
plt.legend(["Rigid rotor\n{}".format(R_FLy_moms), "Elastic Rotor\n{}".format(E_FLy_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"FLy.png")
plt.close()

cc = correlation_coef(R_FLz,E_FLz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_FLz,"-r")
plt.plot(Time,E_FLz,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Axial force 75% span [kN]")
plt.legend(["Rigid rotor\n{}".format(R_FLz_moms), "Elastic Rotor\n{}".format(E_FLz_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"FLz.png")
plt.close()


cc = correlation_coef(R_MLx,E_MLx)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_MLx,"-r")
plt.plot(Time,E_MLx,"-b")
plt.xlabel("Time [s]")
plt.ylabel("edgewise moment 75% span [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_MLx_moms), "Elastic Rotor\n{}".format(E_MLx_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"MLx.png")
plt.close()


cc = correlation_coef(R_MLy,E_MLy)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_MLy,"-r")
plt.plot(Time,E_MLy,"-b")
plt.xlabel("Time [s]")
plt.ylabel("flapwise moment 75% span [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_MLy_moms), "Elastic Rotor\n{}".format(E_MLy_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"MLy.png")
plt.close()

cc = correlation_coef(R_MLz,E_MLz)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,R_MLz,"-r")
plt.plot(Time,E_MLz,"-b")
plt.xlabel("Time [s]")
plt.ylabel("Pitching moment 75% span [kN-m]")
plt.legend(["Rigid rotor\n{}".format(R_MLz_moms), "Elastic Rotor\n{}".format(E_MLz_moms)])
plt.title("Correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"MLz.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_MLx_frq,R_MLx_PSD,"-r")
plt.loglog(E_MLx_frq,E_MLx_PSD,"-b")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Edgewise moment 75% span PSD [kN-m]")
plt.legend(["Rigid rotor", "Elastic Rotor"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_MLx.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_MLy_frq,R_MLy_PSD,"-r")
plt.loglog(E_MLy_frq,E_MLy_PSD,"-b")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Flap moment 75% span PSD [kN-m]")
plt.legend(["Rigid rotor", "Elastic Rotor"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_MLy.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.loglog(R_MLz_frq,R_MLz_PSD,"-r")
plt.loglog(E_MLz_frq,E_MLz_PSD,"-b")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Pitching moment 75% span PSD [kN-m]")
plt.legend(["Rigid rotor", "Elastic Rotor"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_MLz.png")
plt.close()