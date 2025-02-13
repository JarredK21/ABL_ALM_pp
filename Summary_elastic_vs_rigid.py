from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pyFAST.input_output as io
from scipy.fft import fft, fftfreq, fftshift,ifft
from multiprocessing import Pool
import time
from scipy import interpolate
from scipy.signal import butter,filtfilt


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


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def tranform_fixed_frame(y,z,Theta):

    Y = y*np.cos(Theta) - z*np.sin(Theta)
    Z = y*np.sin(Theta) + z*np.cos(Theta)

    return Y,Z


def hard_filter(signal,cutoff,dt,filter_type):

    N = len(signal)
    spectrum = np.fft.fft(signal)
    F = np.fft.fftfreq(N,dt)
    #F = (1/(dt*N)) * np.arange(N)
    if filter_type=="lowpass":
        spectrum_filter = spectrum*(np.abs(F)<cutoff)
    elif filter_type=="highpass":
        spectrum_filter = spectrum*(np.abs(F)>cutoff)
    elif filter_type=="bandpass":
        spectrum_filter = spectrum*(np.abs(F)>cutoff[0])
        spectrum_filter = spectrum_filter*(np.abs(F)<cutoff[1])
        

    spectrum_filter = np.fft.ifft(spectrum_filter)

    return np.real(spectrum_filter)


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(y)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt



def probability_dist(y,N):
    std = np.std(y)
    if N=="default":
        N=20
    bin_width = std/N
    x = np.arange(np.min(y),np.max(y)+bin_width,bin_width)
    dx = x[1]-x[0]
    P = []
    X = []
    for i in np.arange(0,len(x)-1):
        p = 0
        for yi in y:
            if yi >= x[i] and yi <= x[i+1]:
                p+=1
        P.append(p/(dx*len(y)))
        X.append((x[i+1]+x[i])/2)

    print(np.sum(P)*dx)

    return P,X



def moments(y):
    mu = np.mean(y)
    std = np.std(y)
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    return round(mu,2), round(std,2), round(skewness,2),round(kurotsis,2)



in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_E_CRPM/76000/"

df = Dataset(in_dir+"Dataset.nc")
Time_OF = np.array(df.variables["Time_OF"])
dt_OF = Time_OF[1] - Time_OF[0]
Tstart_idx = np.searchsorted(Time_OF,Time_OF[0]+200)
Time_OF = Time_OF[Tstart_idx:]

Time_sampling = np.array(df.variables["Time_sampling"])
Tstart_idx_sampling = np.searchsorted(Time_sampling,Time_sampling[0]+200)
Time_sampling = Time_sampling[Tstart_idx_sampling:]

OF_vars = df.groups["OpenFAST_Variables"]

Azimuth = np.radians(np.array(OF_vars.variables["Azimuth"][Tstart_idx:]))

RtAeroFxh_E = np.array(OF_vars.variables["RtAeroFxh"][Tstart_idx:])/1000

RtAeroFyh = np.array(OF_vars.variables["RtAeroFyh"][Tstart_idx:])/1000
RtAeroFzh = np.array(OF_vars.variables["RtAeroFzh"][Tstart_idx:])/1000

RtAeroFys_E, RtAeroFzs_E = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

RtAeroFR_E = np.sqrt( np.add(np.square(RtAeroFys_E), np.square(RtAeroFzs_E)) ) 

RtAeroMxh_E = np.array(OF_vars.variables["RtAeroMxh"][Tstart_idx:])/1000

RtAeroMyh = np.array(OF_vars.variables["RtAeroMyh"][Tstart_idx:])/1000
RtAeroMzh = np.array(OF_vars.variables["RtAeroMzh"][Tstart_idx:])/1000

RtAeroMys_E, RtAeroMzs_E = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

RtAeroMR_E = np.sqrt( np.add(np.square(RtAeroMys_E), np.square(RtAeroMzs_E)) ) 


#LPFMH = hard_filter(RtAeroMR_E,0.3,dt_OF,"lowpass")
# BPFMH = hard_filter(RtAeroMR_E,[0.3,0.9],dt_OF,"bandpass")
# HPFMH = hard_filter(RtAeroMR_E,[1.5,40],dt_OF,"bandpass")

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"Three_frequency_analysis/MH_all_freqs/"
# times = np.arange(200,1300,100)
# for j in np.arange(0,len(times)-1):
#     it_1 = np.searchsorted(Time_OF,times[j])
#     it_2 = np.searchsorted(Time_OF,times[j+1])

#     fig = plt.figure(figsize=(14,8))
#     plt.plot(Time_OF[it_1:it_2],RtAeroMR_E[it_1:it_2],"-k",label="Total $|M_H|$")
#     plt.plot(Time_OF[it_1:it_2],LPFMH[it_1:it_2],"-g",label="LPF 0.3Hz $|M_H|$")
#     plt.plot(Time_OF[it_1:it_2],BPFMH[it_1:it_2],"-r",label="BPF 0.3-0.9Hz $|M_H|$")
#     plt.plot(Time_OF[it_1:it_2],HPFMH[it_1:it_2]-1000,"-b",label="HPF 1.5Hz $|M_H|$\noffset:-1000kN-m")
#     plt.legend()
#     plt.grid()
#     plt.xlabel("Time [s]")
#     plt.ylabel("Aerodynamic out-of-plane bending moment [kN-m]")
#     plt.tight_layout()
#     plt.savefig(out_dir+"{}_{}.png".format(times[j],times[j+1]))
#     plt.close()

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs_E/L2; FBFy = -RtAeroFys_E*((L1+L2)/L2)
FBMz = -RtAeroMys_E/L2; FBFz = -RtAeroFzs_E*((L1+L2)/L2)

Aero_FBy_E = -(FBMy + FBFy); Aero_FBz_E = -(FBMz + FBFz)

Aero_FBR_E = np.sqrt(np.add(np.square(Aero_FBy_E),np.square(Aero_FBz_E)))


LSShftFxa_E = np.array(OF_vars.variables["LSShftFxa"][Tstart_idx:])

LSShftFys_E = np.array(OF_vars.variables["LSShftFys"][Tstart_idx:])
LSShftFzs_E = np.array(OF_vars.variables["LSShftFzs"][Tstart_idx:])

LSShftMxa_E = np.array(OF_vars.variables["LSShftMxa"][Tstart_idx:])

LSSTipMys_E = np.array(OF_vars.variables["LSSTipMys"][Tstart_idx:])

LSSTipMzs_E = np.array(OF_vars.variables["LSSTipMzs"][Tstart_idx:])

Elasto_MR_E = np.sqrt(np.add(np.square(LSSTipMys_E),np.square(LSSTipMzs_E)))

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs_E/L2; FBFy = -LSShftFys_E*((L1+L2)/L2)
FBMz = -LSSTipMys_E/L2; FBFz = -LSShftFzs_E*((L1+L2)/L2)



Elasto_FBy_E = -(FBMy + FBFy); Elasto_FBz_E = -(FBMz + FBFz)

Elasto_FBR_E = np.sqrt(np.add(np.square(Elasto_FBy_E),np.square(Elasto_FBz_E)))

# Rotor_avg_vars = df.groups["Rotor_Avg_Variables"]
# Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]

# Ux_E = np.array(Rotor_avg_vars.variables["Ux"][Tstart_idx_sampling:])
# Iy_E = np.array(Rotor_avg_vars.variables["Iy"][Tstart_idx_sampling:])
# Iz_E = np.array(Rotor_avg_vars.variables["Iz"][Tstart_idx_sampling:])
# IE = np.sqrt(np.add(np.square(Iy_E),np.square(Iz_E)))

# f = interpolate.interp1d(Time_OF,RtAeroMxh_E)
# RtAeroMxh_interp = f(Time_sampling)
# print(correlation_coef(Ux_E,RtAeroMxh_interp))
# f = interpolate.interp1d(Time_OF,RtAeroFxh_E)
# RtAeroFxh_interp = f(Time_sampling)
# print(correlation_coef(Ux_E,RtAeroFxh_interp))
# f = interpolate.interp1d(Time_OF,RtAeroMys_E)
# RtAeroMys_interp = f(Time_sampling)
# print(correlation_coef(Iy_E,RtAeroMys_interp))
# f = interpolate.interp1d(Time_OF,RtAeroMzs_E)
# RtAeroMzs_interp = f(Time_sampling)
# print(correlation_coef(Iz_E,RtAeroMzs_interp))
# f = interpolate.interp1d(Time_OF,RtAeroMR_E)
# RtAeroMR_interp = f(Time_sampling)
# print(correlation_coef(IE,RtAeroMR_interp))
# f = interpolate.interp1d(Time_OF,Aero_FBR_E)
# Aero_FBR_interp = f(Time_sampling)
# print(correlation_coef(IE,Aero_FBR_interp))


# # #blade asymmetry
# df = Dataset(in_dir+"WTG01a.nc")

# # Time_steps = np.arange(0,len(Time_OF)-1)
# # Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:])

# uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
# vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1_E = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
# vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2_E = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
# vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3_E = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)


# df_E = Dataset(in_dir+"WTG01b.nc")
# WT_E = df_E.groups["WTG01"]

# Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]


# xo = np.array(WT_E.variables["xyz"][Tstart_idx:,225,0])
# yo = np.array(WT_E.variables["xyz"][Tstart_idx:,225,1])


# x_trans = xo - Rotor_coordinates[0]
# y_trans = yo - Rotor_coordinates[1]

# phi = np.radians(-29.29)
# xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
# ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))

# xs_E = xs + Rotor_coordinates[0]
# ys_E = ys + Rotor_coordinates[1]



# R = np.linspace(0,63,300)

# dr = R[1]-R[0]
# IyE = []
# IzE = []
# ix=0
# with Pool() as pool:
#     for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,Time_steps):
#         IyE.append(Iy_it); IzE.append(Iz_it)
#         print(ix)
#         ix+=1

# IE = np.sqrt(np.add(np.square(IyE),np.square(IzE)))

# print(correlation_coef(IyE,RtAeroMys_E[:-1]))
# print(correlation_coef(IzE,RtAeroMzs_E[:-1]))
# print(correlation_coef(IE,RtAeroMR_E[:-1]))
# print(correlation_coef(IE,Aero_FBR_E[:-1]))


in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

df = Dataset(in_dir+"Dataset.nc")

OF_vars = df.groups["OpenFAST_Variables"]

RtAeroFxh_R = np.array(OF_vars.variables["RtAeroFxh"][Tstart_idx:])/1000

RtAeroFyh = np.array(OF_vars.variables["RtAeroFyh"][Tstart_idx:])/1000
RtAeroFzh = np.array(OF_vars.variables["RtAeroFzh"][Tstart_idx:])/1000

RtAeroFys_R, RtAeroFzs_R = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

RtAeroFR_R = np.sqrt( np.add(np.square(RtAeroFys_R), np.square(RtAeroFzs_R)) ) 

RtAeroMxh_R = np.array(OF_vars.variables["RtAeroMxh"][Tstart_idx:])/1000

RtAeroMyh = np.array(OF_vars.variables["RtAeroMyh"][Tstart_idx:])/1000
RtAeroMzh = np.array(OF_vars.variables["RtAeroMzh"][Tstart_idx:])/1000

RtAeroMys_R, RtAeroMzs_R = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

RtAeroMR_R = np.sqrt( np.add(np.square(RtAeroMys_R), np.square(RtAeroMzs_R)) ) 

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs_R/L2; FBFy = -RtAeroFys_R*((L1+L2)/L2)
FBMz = -RtAeroMys_R/L2; FBFz = -RtAeroFzs_R*((L1+L2)/L2)

Aero_FBy_R = -(FBMy + FBFy); Aero_FBz_R = -(FBMz + FBFz)

Aero_FBR_R = np.sqrt(np.add(np.square(Aero_FBy_R),np.square(Aero_FBz_R)))

#rigid case
LSShftFxa_R = np.array(OF_vars.variables["LSShftFxa"][Tstart_idx:])

LSShftFys_R = np.array(OF_vars.variables["LSShftFys"][Tstart_idx:])
LSShftFzs_R = np.array(OF_vars.variables["LSShftFzs"][Tstart_idx:])

LSShftMxa_R = np.array(OF_vars.variables["LSShftMxa"][Tstart_idx:])

LSSTipMys_R = np.array(OF_vars.variables["LSSTipMys"][Tstart_idx:])
LSSTipMzs_R = np.array(OF_vars.variables["LSSTipMzs"][Tstart_idx:])

Elasto_MR_R = np.sqrt(np.add(np.square(LSSTipMys_R),np.square(LSSTipMzs_R)))

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs_R/L2; FBFy = -LSShftFys_R*((L1+L2)/L2)
FBMz = -LSSTipMys_R/L2; FBFz = -LSShftFzs_R*((L1+L2)/L2)

Elasto_FBy_R = -(FBMy + FBFy); Elasto_FBz_R = -(FBMz + FBFz)

Elasto_FBR_R = np.sqrt(np.add(np.square(Elasto_FBy_R),np.square(Elasto_FBz_R)))


# Rotor_avg_vars = df.groups["Rotor_Avg_Variables"]
# Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]

# Ux_R = np.array(Rotor_avg_vars.variables["Ux"][Tstart_idx_sampling:])
# Iy_R = np.array(Rotor_avg_vars.variables["Iy"][Tstart_idx_sampling:])
# Iz_R = np.array(Rotor_avg_vars.variables["Iz"][Tstart_idx_sampling:])
# IR = np.sqrt(np.add(np.square(Iy_R),np.square(Iz_R)))


# FH_E = RtAeroFzs_E*((L1+L2)/L2); MH_E = RtAeroMys_E/L2
# M_F = round(np.mean(FH_E),2); S_F = round(np.std(FH_E),2)
# M_M = round(np.mean(MH_E),2); S_M = round(np.std(MH_E),2)

# out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,MH_E,"-b",label="$M_{H,y}/L_2$:"+"{}, {}".format(M_M,S_M))
# plt.plot(Time_OF,FH_E,"-r",label="$F_{H,z}L/L_2$:"+"{}, {}".format(M_F,S_F))
# plt.xlabel("Time [s]")
# plt.ylabel("Aerodynamic contributions to $F_{B,z}$ [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"FH_MH.png")
# plt.close(fig)

# #steady shear inflow elastic case
#in_dir="../../NREL_5MW_3.4.1/Steady_Elastic_blades_shear_0.085/"
# df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

# Time_S = np.array(df["Time_[s]"])
# dt_S = Time_S[1] - Time_S[0]
# Tstart_idx_S = np.searchsorted(Time_S,Time_S[0]+200)
# LSShftFxa_S = np.array(df["LSShftFxa_[kN]"][Tstart_idx_S:])

# LSShftFys_S = np.array(df["LSShftFys_[kN]"][Tstart_idx_S:])
# LSShftFzs_S = np.array(df["LSShftFzs_[kN]"][Tstart_idx_S:])

# LSShftMxa_S = np.array(df["LSShftMxa_[kN-m]"][Tstart_idx_S:])

# LSSTipMys_S = np.array(df["LSSTipMys_[kN-m]"][Tstart_idx_S:])
# LSSTipMzs_S = np.array(df["LSSTipMzs_[kN-m]"][Tstart_idx_S:])

# Elasto_MR_S = np.sqrt(np.add(np.square(LSSTipMys_S),np.square(LSSTipMzs_S)))

# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = LSSTipMzs_S/L2; FBFy = -LSShftFys_S*((L1+L2)/L2)
# FBMz = -LSSTipMys_S/L2; FBFz = -LSShftFzs_S*((L1+L2)/L2)

# Elasto_FBy_S = -(FBMy + FBFy); Elasto_FBz_S = -(FBMz + FBFz)

# Elasto_FBR_S = np.sqrt(np.add(np.square(Elasto_FBy_S),np.square(Elasto_FBz_S)))

# f = interpolate.interp1d(Time_OF,RtAeroMxh_R)
# RtAeroMxh_interp = f(Time_sampling)
# print(correlation_coef(Ux_R,RtAeroMxh_interp))
# f = interpolate.interp1d(Time_OF,RtAeroFxh_R)
# RtAeroFxh_interp = f(Time_sampling)
# print(correlation_coef(Ux_R,RtAeroFxh_interp))
# f = interpolate.interp1d(Time_OF,RtAeroMys_R)
# RtAeroMys_interp = f(Time_sampling)
# print(correlation_coef(Iy_R,RtAeroMys_interp))
# f = interpolate.interp1d(Time_OF,RtAeroMzs_R)
# RtAeroMzs_interp = f(Time_sampling)
# print(correlation_coef(Iz_R,RtAeroMzs_interp))
# f = interpolate.interp1d(Time_OF,RtAeroMR_R)
# RtAeroMR_interp = f(Time_sampling)
# print(correlation_coef(IR,RtAeroMR_interp))
# f = interpolate.interp1d(Time_OF,Aero_FBR_R)
# Aero_FBR_interp = f(Time_sampling)
# print(correlation_coef(IR,Aero_FBR_interp))

# #blade asymmetry
# df = Dataset(in_dir+"WTG01a.nc")

# # Time_steps = np.arange(0,len(Time_OF)-1)
# # Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:])

# uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
# vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1_R = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
# vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2_R = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
# vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3_R = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)


# df_R = Dataset(in_dir+"WTG01b.nc")
# WT_R = df_R.groups["WTG01"]

# Rotor_coordinates = [np.float64(WT_R.variables["xyz"][0,0,0]),np.float64(WT_R.variables["xyz"][0,0,1]),np.float64(WT_R.variables["xyz"][0,0,2])]


# xo = np.array(WT_R.variables["xyz"][Tstart_idx:,225,0])
# yo = np.array(WT_R.variables["xyz"][Tstart_idx:,225,1])


# x_trans = xo - Rotor_coordinates[0]
# y_trans = yo - Rotor_coordinates[1]

# phi = np.radians(-29.29)
# xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
# ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))

# xs_R = xs + Rotor_coordinates[0]
# ys_R = ys + Rotor_coordinates[1]

# R = np.linspace(0,63,300)

# dr = R[1]-R[0]
# IyR = []
# IzR = []
# ix=0
# with Pool() as pool:
#     for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,Time_steps):
#         IyR.append(Iy_it); IzR.append(Iz_it)
#         print(ix)
#         ix+=1

# IR = np.sqrt(np.add(np.square(IyR),np.square(IzR)))

# print(correlation_coef(IyR,RtAeroMys_R[:-1]))
# print(correlation_coef(IzR,RtAeroMzs_R[:-1]))
# print(correlation_coef(IR,RtAeroMR_R[:-1]))
# print(correlation_coef(IR,Aero_FBR_R[:-1]))


#Comparing mean and standard deviation analysis
Aero_Array = [Aero_FBR_E,Aero_FBR_R,Aero_FBy_E,Aero_FBy_R,Aero_FBz_E,Aero_FBz_R,RtAeroMR_E,RtAeroMR_R,RtAeroMys_E,RtAeroMys_R,RtAeroMzs_E,RtAeroMzs_R,RtAeroFys_E,RtAeroFys_R,
              RtAeroFzs_E,RtAeroFzs_R,RtAeroFxh_E,RtAeroFxh_R,RtAeroMxh_E,RtAeroMxh_R]

Elasto_Array = [Elasto_FBR_E,Elasto_FBR_R,Elasto_FBy_E,Elasto_FBy_R,Elasto_FBz_E,Elasto_FBz_R,Elasto_MR_E,Elasto_MR_R,LSSTipMys_E,LSSTipMys_R,
                LSSTipMzs_E,LSSTipMzs_R,LSShftFys_E,LSShftFys_R,LSShftFzs_E,LSShftFzs_R,LSShftFxa_E,LSShftFxa_R,LSShftMxa_E,LSShftMxa_R]

xlabel = np.array(["$|F_B|_E$","$|F_B|_R$", "$F_{B,y,E}$","$F_{B,y,R}$", "$F_{B,z,E}$","$F_{B,z,R}$", "$|M_H|_E$","$|M_H|_R$", "$M_{H,y,E}$","$M_{H,y,R}$", 
                    "$M_{H,z,E}$","$M_{H,z,R}$", "$F_{H,y,E}$","$F_{H,y,R}$", "$F_{H,z,E}$","$F_{H,z,R}$", "$F_{H,x,E}$","$F_{H,x,R}$", "$M_{H,x,E}$","$M_{H,x,R}$"])


Aero_mean_array = []
Aero_std_array = []
Elasto_mean_array = []
Elasto_std_array = []
for i in np.arange(0,len(Aero_Array)):
    Aero_mean_array.append(np.mean(Aero_Array[i]))
    Aero_std_array.append(np.std(Aero_Array[i]))

    Elasto_mean_array.append(np.mean(Elasto_Array[i]))
    Elasto_std_array.append(np.std(Elasto_Array[i]))
    print(np.std(Aero_Array[i]))
    print(np.std(Elasto_Array[i]))

mean_Aero_Elasto_array = []
std_Aero_Elasto_array = []
for i in np.arange(0,len(Aero_Array)):
    if xlabel[i] == "$F_{H,y,E}$" or xlabel[i] == "$F_{H,z,E}$":
        mean_Aero_Elasto_array.append(np.mean(Elasto_Array[i])-np.mean(Aero_Array[i]))
        std_Aero_Elasto_array.append(0.0)
    else:
        mean_Aero_Elasto_array.append(np.mean(Elasto_Array[i])-np.mean(Aero_Array[i]))

        std_Aero_Elasto_array.append(((np.std(Elasto_Array[i])-np.std(Aero_Array[i]))/np.std(Aero_Array[i]))*100)


mean_diff_Aero_Elasto = []
for i in np.arange(0,len(mean_Aero_Elasto_array),2):
    mean_diff_Aero_Elasto.append(mean_Aero_Elasto_array[i]-mean_Aero_Elasto_array[i+1])


Aero_mean_change = []; Aero_mean_perc_change = []
Aero_std_change = []; Aero_std_perc_change = []
Elasto_mean_change = []; Elasto_mean_perc_change = []
Elasto_std_change = []; Elasto_std_perc_change = []
for i in np.arange(0,len(Aero_mean_array),2):
    if xlabel[i] == "$F_{H,y,E}$" or xlabel[i] == "$F_{H,z,E}$":
        Aero_mean_perc_change.append(0.0); Elasto_mean_perc_change.append(0.0)
        Aero_std_perc_change.append(0.0); Elasto_std_perc_change.append(0.0)
        Aero_mean_change.append(abs(Aero_mean_array[i])-abs(Aero_mean_array[i+1]))
        Elasto_mean_change.append(abs(Elasto_mean_array[i])-abs(Elasto_mean_array[i+1]))
        Aero_std_change.append(Aero_std_array[i]-Aero_std_array[i+1])
        Elasto_std_change.append(Elasto_std_array[i]-Elasto_std_array[i+1])
    else:
        Aero_mean_change.append(abs(Aero_mean_array[i])-abs(Aero_mean_array[i+1]))
        Aero_mean_perc_change.append(((abs(Aero_mean_array[i])-abs(Aero_mean_array[i+1]))/abs(Aero_mean_array[i+1]))*100)
        Aero_std_change.append(Aero_std_array[i]-Aero_std_array[i+1])
        Aero_std_perc_change.append(((Aero_std_array[i]-Aero_std_array[i+1])/Aero_std_array[i+1])*100)

        Elasto_mean_change.append(abs(Elasto_mean_array[i])-abs(Elasto_mean_array[i+1]))
        Elasto_mean_perc_change.append(((abs(Elasto_mean_array[i])-abs(Elasto_mean_array[i+1]))/abs(Elasto_mean_array[i+1]))*100)
        Elasto_std_change.append(Elasto_std_array[i]-Elasto_std_array[i+1])
        Elasto_std_perc_change.append(((Elasto_std_array[i]-Elasto_std_array[i+1])/Elasto_std_array[i+1])*100)




xlabel = np.array(["$|F_B|_E$\n[kN]","$|F_B|_R$\n[kN]", "$F_{B,y,E}$\n[kN]","$F_{B,y,R}$\n[kN]", "$F_{B,z,E}$\n[kN]","$F_{B,z,R}$\n[kN]", 
                   "$|M_H|_E$\n[kN-m]","$|M_H|_R$\n[kN-m]", "$M_{H,y,E}$\n[kN-m]","$M_{H,y,R}$\n[kN-m]", "$M_{H,z,E}$\n[kN-m]","$M_{H,z,R}$\n[kN-m]", 
                   "$F_{H,y,E}$\n[kN]","$F_{H,y,R}$\n[kN]", "$F_{H,z,E}$\n[kN]","$F_{H,z,R}$\n[kN]", "$F_{H,x,E}$\n[kN]","$F_{H,x,R}$\n[kN]", "$M_{H,x,E}$\n[kN-m]","$M_{H,x,R}$\n[kN-m]"])
colors = ["b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r"]
out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_mean_array,color=colors)
ax1.axhline(y=0.0,color="k")
ax1.errorbar(xlabel,Aero_mean_array,yerr=Aero_std_array,fmt = "o",color="k",capsize=10)
ax1.set_title("Aerodynamic (AeroDyn) variables")
ax2.bar(xlabel,Elasto_mean_array,color=colors)
ax2.axhline(y=0.0,color="k")
ax2.errorbar(xlabel,Elasto_mean_array,yerr=Elasto_std_array,fmt = "o",color="k",capsize=10)
ax2.set_title("Aerodynamic+blade deformations (Elastodyn) variables")
plt.tight_layout()
plt.savefig("../../Thesis/Figures/summary_bar_pairs.png")
plt.close(fig)

# fig,ax1 = plt.subplots(figsize=(18,8))
# ax1.bar(xlabel,mean_Aero_Elasto_array)
# ax1.axhline(y=0.0,color="k")
# ax1.set_title("Elasto-Aero")
# plt.tight_layout()
# plt.savefig(out_dir+"summary_bar_pairs_mean_change_aero_elasto.png")
# plt.close(fig)

# fig,ax1 = plt.subplots(figsize=(18,8))
# ax1.bar(xlabel,std_Aero_Elasto_array)
# ax1.axhline(y=0.0,color="k")
# ax1.set_title("Elasto-Aero")
# ax1.set_ylabel("Percentage change in the standard deviation [%]")
# ax1.axhline(y=10.0,color="k",linestyle="--")
# plt.tight_layout()
# plt.savefig(out_dir+"summary_bar_pairs_std_change_aero_elasto.png")
# plt.close(fig)

xlabel = np.array(["$|F_B|$\n[kN]", "$F_{B,y}$\n[kN]", "$F_{B,z}$\n[kN]", "$|M_H|$\n[kN-m]", "$M_{H,y}$\n[kN-m]","$M_{H,z}$\n[kN-m]", 
                   "$F_{H,y}$\n[kN]", "$F_{H,z}$\n[kN]", "$F_{H,x}$\n[kN]", "$M_{H,x}$\n[kN-m]"])
# fig,ax1 = plt.subplots(figsize=(18,8))
# ax1.bar(xlabel,mean_diff_Aero_Elasto)
# ax1.axhline(y=0.0,color="k")
# ax1.set_title("(Elasto-Aero)E - (Elasto-Aero)R")
# plt.tight_layout()
# plt.savefig(out_dir+"summary_bar_pairs_diff_change_aero_elasto.png")
# plt.close(fig)

plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_mean_change)
ax1.axhline(y=0.0,color="k")
ax1.set_title("Aerodynamic (AeroDyn) variables")
ax2.bar(xlabel,Elasto_mean_change)
ax2.axhline(y=0.0,color="k")
ax2.set_title("Aerodynamic+blade deformation (ElastoDyn) variables")
fig.supylabel("Change in average value\n(Elastic-Rigid)")
plt.tight_layout()
plt.savefig("../../Thesis/Figures/summary_bar_mean_change.png")
plt.close(fig)

plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_mean_perc_change)
ax1.axhline(y=0.0,color="k")
ax1.axhline(y=10.0,color="k",linestyle="--")
ax1.axhline(y=-10.0,color="k",linestyle="--")
ax1.set_title("Aerodynamic (AeroDyn) variables")
ax2.bar(xlabel,Elasto_mean_perc_change)
ax2.axhline(y=0.0,color="k")
ax2.axhline(y=10.0,color="k",linestyle="--")
ax2.axhline(y=-10.0,color="k",linestyle="--")
fig.supylabel("Percentage change in average value [%]\n((Elastic-Rigid)/Rigid)x100")
ax2.set_title("Aerodynamic+blade deformations (ElastoDyn) variables")
plt.tight_layout()
plt.savefig("../../Thesis/Figures/summary_bar_mean_perc_change.png")
plt.close(fig)


# plt.rcParams['font.size'] = 16
# fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
# ax1.bar(xlabel,Aero_std_change)
# ax1.axhline(y=0.0,color="k")
# ax1.set_title("Aerodynamic variables (Elastic-Rigid)")
# ax2.bar(xlabel,Elasto_std_change)
# ax2.axhline(y=0.0,color="k")
# ax2.set_title("Elastodyn variables (Elastic-Rigid)")
# fig.supylabel("Change in standard deviation")
# plt.tight_layout()
# plt.savefig(out_dir+"summary_bar_std_change.png")
# plt.close(fig)

plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_std_perc_change)
ax1.axhline(y=0.0,color="k")
ax1.axhline(y=-10.0,color="k",linestyle="--")
ax1.set_title("Aerodynamic (AeroDyn) variables")
ax2.bar(xlabel,Elasto_std_perc_change)
ax2.axhline(y=0.0,color="k")
ax2.axhline(y=-10.0,color="k",linestyle="--")
fig.supylabel("Percentage change in standard deviation [%]\n((Elastic-Rigid)/Rigid)x100")
ax2.set_title("Aerodynamic+blade deformations (ElastoDyn) variables")
plt.tight_layout()
plt.savefig("../../Thesis/Figures/summary_bar_std_perc_change.png")
plt.close(fig)


# #comparing spectra analysis
out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Rotor_Var_plots/"
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(Elasto_MR_E,dt_OF,"MR_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# frq,PSD = temporal_spectra(Elasto_MR_R,dt_OF,"MR_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Out-of-plane bending moment [kN-m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_MR.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(Elasto_MR_R,dt_OF,"MR_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# frq,PSD = temporal_spectra(Elasto_MR_E,dt_OF,"MR_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Out-of-plane bending moment [kN-m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_MR_2.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(Elasto_MR_E,dt_OF,"MR_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# frq,PSD = temporal_spectra(Elasto_MR_S,dt_S,"MR_S")
# plt.loglog(frq,PSD,"-r",label="Rigid Steady Shear inflow")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Out-of-plane bending moment [kN-m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_MR_SSI.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSSTipMys_R,dt_OF,"My_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# frq,PSD = temporal_spectra(LSSTipMys_E,dt_OF,"My_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Hub moment y component [kN-m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_My.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSSTipMzs_E,dt_OF,"Mz_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# frq,PSD = temporal_spectra(LSSTipMzs_R,dt_OF,"Mz_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Hub moment z component [kN-m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_Mz.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSShftFxa_R,dt_OF,"Fx_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# frq,PSD = temporal_spectra(LSShftFxa_E,dt_OF,"Fx_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Hub force x component [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_Fx.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSShftFys_E,dt_OF,"Fy_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# frq,PSD = temporal_spectra(LSShftFys_R,dt_OF,"Fy_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Hub force y component [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_Fy.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSShftFys_E,dt_OF,"Fy_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# frq,PSD = temporal_spectra(LSShftFys_S,dt_S,"Fy_S")
# plt.loglog(frq,PSD,"-r",label="Rigid Steady Shear inflow")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Hub force y component [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_Fy_SSI.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSShftFzs_E,dt_OF,"Fz_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# frq,PSD = temporal_spectra(LSShftFzs_R,dt_OF,"Fz_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Hub force z component [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_Fz.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSShftFzs_E,dt_OF,"Fz_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# frq,PSD = temporal_spectra(LSShftFzs_S,dt_S,"Fz_S")
# plt.loglog(frq,PSD,"-r",label="Rigid Steady Shear inflow")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Hub force z component [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_Fz_SSI.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSSTipMys_E/L2,dt_OF,"My_E")
# plt.loglog(frq,PSD,"-b",label="$M_{H,y}L_2$")
# frq,PSD = temporal_spectra(LSShftFzs_E*((L1+L2)/L2),dt_OF,"Fz_E")
# plt.loglog(frq,PSD,"-r",label="$F_{H,z}L/L_2$")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Elastic bearing force z contributions [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_FBz_contributions.png")
# plt.close(fig)


# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(LSSTipMzs_E/L2,dt_OF,"Mz_E")
# plt.loglog(frq,PSD,"-b",label="$M_{H,z}L_2$")
# frq,PSD = temporal_spectra(LSShftFys_E*((L1+L2)/L2),dt_OF,"Fy_E")
# plt.loglog(frq,PSD,"-r",label="$F_{H,y}L/L_2$")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Elastic bearing force y contributions [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_FBy_contributions.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(Elasto_FBR_R,dt_OF,"FBR_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# frq,PSD = temporal_spectra(Elasto_FBR_E,dt_OF,"FBR_E")
# plt.loglog(frq,PSD,"-b",label="Derform")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Bearing force magnitude [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_FBR.png")
# plt.close(fig)



# in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"


# df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()
# cols = np.array(df.columns)
# Time = np.array(df["Time_[s]"])
# dt = Time[1] - Time[0]
# Tstart_idx = np.searchsorted(Time,Time[0]+200)
# Time = Time[Tstart_idx:]

# RootFxb1_E = np.array(df["RootFxb1_[kN]"][Tstart_idx:])
# RootFxb2_E = np.array(df["RootFxb2_[kN]"][Tstart_idx:])
# RootFxb3_E = np.array(df["RootFxb3_[kN]"][Tstart_idx:])
# RootFyb1_E = np.array(df["RootFyb1_[kN]"][Tstart_idx:])
# RootFzb1_E = np.array(df["RootFzc1_[kN]"][Tstart_idx:])

# in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"


# df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

# RootFxb1_R = np.array(df["RootFxb1_[kN]"][Tstart_idx:])
# RootFxb2_R = np.array(df["RootFxb2_[kN]"][Tstart_idx:])
# RootFxb3_R = np.array(df["RootFxb3_[kN]"][Tstart_idx:])
# RootFyb1_R = np.array(df["RootFyb1_[kN]"][Tstart_idx:])
# RootFzb1_R = np.array(df["RootFzc1_[kN]"][Tstart_idx:])

# out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Rotor_Var_plots/"
# plt.rcParams['font.size'] = 16

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(RootFxb1_R,dt,"Fxb_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# frq,PSD = temporal_spectra(RootFxb1_E,dt,"Fxb_E")
# plt.loglog(frq,PSD,"-b",label="Deform")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Root force xb [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_RootFxb.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(RootFxb1_E,dt,"Fxb_E")
# plt.loglog(frq,PSD,"-b",label="Deform")
# frq,PSD = temporal_spectra(RootFxb1_R,dt,"Fxb_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Root force xb [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_RootFxb_2.png")
# plt.close(fig)


# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(RootFyb1_R,dt,"Fyb_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# frq,PSD = temporal_spectra(RootFyb1_E,dt,"Fyb_E")
# plt.loglog(frq,PSD,"-b",label="Deform")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Root force yb [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_RootFyb.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(RootFyb1_E,dt,"Fyb_E")
# plt.loglog(frq,PSD,"-b",label="Deform")
# frq,PSD = temporal_spectra(RootFyb1_R,dt,"Fyb_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Root force yb [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_RootFyb_2.png")
# plt.close(fig)


# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(RootFzb1_E,dt,"Fzb_E")
# plt.loglog(frq,PSD,"-b",label="Deform")
# frq,PSD = temporal_spectra(RootFzb1_R,dt,"Fzb_R")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Root force zb [kN]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_RootFzb.png")
# plt.close(fig)


# # #Three frequency analysis
# WR = 1079.1
# FBR_E = hard_filter(Elasto_FBR_E,40,dt_OF,"lowpass")
# LPF_FBR_E = hard_filter(Elasto_FBR_E,0.3,dt_OF,"lowpass")
# LPF_2_FBR_E = hard_filter(Elasto_FBR_E,0.9,dt_OF,"lowpass")
# BPF_FBR_E = hard_filter(Elasto_FBR_E,[0.3,0.9],dt_OF,"bandpass")
# HPF_FBR_E = hard_filter(Elasto_FBR_E,[1.5,40],dt_OF,"bandpass")

# FBR_R = hard_filter(Elasto_FBR_R,40,dt_OF,"lowpass")
# LPF_FBR_R = hard_filter(Elasto_FBR_R,0.3,dt_OF,"lowpass")
# LPF_2_FBR_R = hard_filter(Elasto_FBR_R,0.9,dt_OF,"lowpass")
# BPF_FBR_R = hard_filter(Elasto_FBR_R,[0.3,0.9],dt_OF,"bandpass")
# HPF_FBR_R = hard_filter(Elasto_FBR_R,[1.5,40],dt_OF,"bandpass")


# dFBR_E = np.array(dt_calc(FBR_E,dt_OF))
# zero_crossings_index_FBR_E = np.where(np.diff(np.sign(dFBR_E)))[0]
# dLPF_FBR_E = np.array(dt_calc(LPF_FBR_E,dt_OF))
# zero_crossings_index_LPF_FBR_E = np.where(np.diff(np.sign(dLPF_FBR_E)))[0]
# dBPF_FBR_E = np.array(dt_calc(BPF_FBR_E,dt_OF))
# zero_crossings_index_BPF_FBR_E = np.where(np.diff(np.sign(dBPF_FBR_E)))[0]
# dHPF_FBR_E = np.array(dt_calc(HPF_FBR_E,dt_OF))
# zero_crossings_index_HPF_FBR_E = np.where(np.diff(np.sign(dHPF_FBR_E)))[0]

# dFBR_R = np.array(dt_calc(FBR_R,dt_OF))
# zero_crossings_index_FBR_R = np.where(np.diff(np.sign(dFBR_R)))[0]
# dLPF_FBR_R = np.array(dt_calc(LPF_FBR_R,dt_OF))
# zero_crossings_index_LPF_FBR_R = np.where(np.diff(np.sign(dLPF_FBR_R)))[0]
# dBPF_FBR_R = np.array(dt_calc(BPF_FBR_R,dt_OF))
# zero_crossings_index_BPF_FBR_R = np.where(np.diff(np.sign(dBPF_FBR_R)))[0]
# dHPF_FBR_R = np.array(dt_calc(HPF_FBR_R,dt_OF))
# zero_crossings_index_HPF_FBR_R = np.where(np.diff(np.sign(dHPF_FBR_R)))[0]

# # frq,PSD = temporal_spectra(LPF_FBR_E,dt_OF,"LPF FBR")
# # plt.loglog(frq,PSD)

# #jump analysis elastic rotor total signal
# dF_E = []
# dt_E = []
# T_E = []
# for i in np.arange(0,len(zero_crossings_index_FBR_E)-1):
#     it_1 = zero_crossings_index_FBR_E[i]
#     it_2 = zero_crossings_index_FBR_E[i+1]
#     T_E.append(Time_OF[it_1])
#     dt_E.append(Time_OF[it_2]-Time_OF[it_1])
#     dF_E.append(FBR_E[it_2]-FBR_E[it_1])

# dF_E = np.true_divide(dF_E,(WR*((L1+L2)/L2)))

# #Low pass filtered
# dF_LPF_E = []
# dt_LPF_E = []
# T_LPF_E = []
# for i in np.arange(0,len(zero_crossings_index_LPF_FBR_E)-1):
#     it_1 = zero_crossings_index_LPF_FBR_E[i]
#     it_2 = zero_crossings_index_LPF_FBR_E[i+1]
#     T_LPF_E.append(Time_OF[it_1])
#     dt_LPF_E.append(Time_OF[it_2]-Time_OF[it_1])
#     dF_LPF_E.append(LPF_FBR_E[it_2]-LPF_FBR_E[it_1])

# dF_LPF_E = np.true_divide(dF_LPF_E,(WR*((L1+L2)/L2)))

# # fig,ax = plt.subplots()
# # ax.plot(Time_OF,LPF_FBR_E)
# # ax2=ax.twinx()
# # ax2.scatter(T_LPF_E,dt_LPF_E)
# # plt.show()

# #band pass filtered
# dF_BPF_E = []
# dt_BPF_E = []
# T_BPF_E = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR_E)-1):
#     it_1 = zero_crossings_index_BPF_FBR_E[i]
#     it_2 = zero_crossings_index_BPF_FBR_E[i+1]
#     T_BPF_E.append(Time_OF[it_1])
#     dt_BPF_E.append(Time_OF[it_2]-Time_OF[it_1])
#     dF_BPF_E.append(BPF_FBR_E[it_2]-BPF_FBR_E[it_1])

# dF_BPF_E = np.true_divide(dF_BPF_E,(WR*((L1+L2)/L2)))

# #high pass filtered
# dF_HPF_E = []
# dt_HPF_E = []
# T_HPF_E = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR_E)-1):
#     it_1 = zero_crossings_index_HPF_FBR_E[i]
#     it_2 = zero_crossings_index_HPF_FBR_E[i+1]
#     T_HPF_E.append(Time_OF[it_1])
#     dt_HPF_E.append(Time_OF[it_2]-Time_OF[it_1])
#     dF_HPF_E.append(HPF_FBR_E[it_2]-HPF_FBR_E[it_1])

# dF_HPF_E = np.true_divide(dF_HPF_E,(WR*((L1+L2)/L2)))


# #jump analysis rigid rotor total signal
# dF_R = []
# dt_R = []
# T_R = []
# for i in np.arange(0,len(zero_crossings_index_FBR_R)-1):
#     it_1 = zero_crossings_index_FBR_R[i]
#     it_2 = zero_crossings_index_FBR_R[i+1]
#     T_R.append(Time_OF[it_1])
#     dt_R.append(Time_OF[it_2]-Time_OF[it_1])
#     dF_R.append(FBR_R[it_2]-FBR_R[it_1])

# dF_R = np.true_divide(dF_R,(WR*((L1+L2)/L2)))

# #Low pass filtered
# dF_LPF_R = []
# dt_LPF_R = []
# T_LPF_R = []
# for i in np.arange(0,len(zero_crossings_index_LPF_FBR_R)-1):
#     it_1 = zero_crossings_index_LPF_FBR_R[i]
#     it_2 = zero_crossings_index_LPF_FBR_R[i+1]
#     T_LPF_R.append(Time_OF[it_1])
#     dt_LPF_R.append(Time_OF[it_2]-Time_OF[it_1])
#     dF_LPF_R.append(LPF_FBR_R[it_2]-LPF_FBR_R[it_1])

# dF_LPF_R = np.true_divide(dF_LPF_R,(WR*((L1+L2)/L2)))

# #band pass filtered
# dF_BPF_R = []
# dt_BPF_R = []
# T_BPF_R = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR_R)-1):
#     it_1 = zero_crossings_index_BPF_FBR_R[i]
#     it_2 = zero_crossings_index_BPF_FBR_R[i+1]
#     T_BPF_R.append(Time_OF[it_1])
#     dt_BPF_R.append(Time_OF[it_2]-Time_OF[it_1])
#     dF_BPF_R.append(BPF_FBR_R[it_2]-BPF_FBR_R[it_1])

# dF_BPF_R = np.true_divide(dF_BPF_R,(WR*((L1+L2)/L2)))

# #high pass filtered
# dF_HPF_R = []
# dt_HPF_R = []
# T_HPF_R = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR_R)-1):
#     it_1 = zero_crossings_index_HPF_FBR_R[i]
#     it_2 = zero_crossings_index_HPF_FBR_R[i+1]
#     T_HPF_R.append(Time_OF[it_1])
#     dt_HPF_R.append(Time_OF[it_2]-Time_OF[it_1])
#     dF_HPF_R.append(HPF_FBR_R[it_2]-HPF_FBR_R[it_1])

# dF_HPF_R = np.true_divide(dF_HPF_R,(WR*((L1+L2)/L2)))



# # #Figures 3 freq analysis
# N=2
# out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Three_frequency_analysis/"
# plt.rcParams['font.size'] = 16
# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8),sharey=False)
# P,X = probability_dist(dF_R,N)
# ax1.plot(X,P,"-r",label="Rigid: {}".format(moments(dF_R)))
# P,X = probability_dist(dF_E,N)
# ax1.plot(X,P,"-b",label="Deform: {}".format(moments(dF_E)))
# ax1.set_xlabel("$\\Delta F/(W_RL/L_2)$ [-]")
# ax1.set_yscale("log")
# ax1.grid()
# ax1.legend()
# P,X = probability_dist(dt_R,N)
# ax2.plot(X,P,"-r",label="Rigid: {}".format(moments(dt_R)))
# P,X = probability_dist(dt_E,N)
# ax2.plot(X,P,"-b",label="Deform: {}".format(moments(dt_E)))
# ax2.set_xlabel("$\\Delta \\tau$ [s]")
# ax2.set_yscale("log")
# ax2.grid()
# ax2.legend()
# fig.supylabel("log() Probability [-]")
# plt.tight_layout()
# plt.savefig(out_dir+"dF_dt.png")
# plt.close(fig)


# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8),sharey=False)
# P,X = probability_dist(dF_LPF_R,N)
# ax1.plot(X,P,"-r",label="Rigid: {}".format(moments(dF_LPF_R)))
# P,X = probability_dist(dF_LPF_E,N)
# ax1.plot(X,P,"-b",label="Deform: {}".format(moments(dF_LPF_E)))
# ax1.set_xlabel("LPF (0.3Hz): $\\Delta F/(W_RL/L_2)$ [-]")
# ax1.grid()
# ax1.set_yscale("log")
# ax1.legend()
# P,X = probability_dist(dt_LPF_R,N)
# ax2.plot(X,P,"-r",label="Rigid: {}".format(moments(dt_LPF_R)))
# P,X = probability_dist(dt_LPF_E,N)
# ax2.plot(X,P,"-b",label="Deform: {}".format(moments(dt_LPF_E)))
# ax2.set_xlabel("$\\Delta \\tau$ [s]")
# ax2.grid()
# ax2.legend()
# ax2.set_yscale("log")
# fig.supylabel("log() Probability [-]")
# plt.tight_layout()
# plt.savefig(out_dir+"LPF_dF_dt.png")
# plt.close(fig)


# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8),sharey=False)
# P,X = probability_dist(dF_BPF_R,N)
# ax1.plot(X,P,"-r",label="Rigid: {}".format(moments(dF_BPF_R)))
# P,X = probability_dist(dF_BPF_E,N)
# ax1.plot(X,P,"-b",label="Deform: {}".format(moments(dF_BPF_E)))
# ax1.set_xlabel("BPF (0.3-0.9Hz): $\\Delta F/(W_RL/L_2)$ [-]")
# ax1.grid()
# ax1.legend()
# ax1.set_yscale("log")
# P,X = probability_dist(dt_BPF_R,N)
# ax2.plot(X,P,"-r",label="Rigid: {}".format(moments(dt_BPF_R)))
# P,X = probability_dist(dt_BPF_E,N)
# ax2.plot(X,P,"-b",label="Deform: {}".format(moments(dt_BPF_E)))
# ax2.set_xlabel("$\\Delta \\tau$ [s]")
# ax2.grid()
# ax2.legend()
# ax2.set_yscale("log")
# fig.supylabel("log() Probability [-]")
# plt.tight_layout()
# plt.savefig(out_dir+"BPF_dF_dt.png")
# plt.close(fig)

# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8),sharey=False)
# P,X = probability_dist(dF_HPF_R,N)
# ax1.plot(X,P,"-r",label="Rigid: {}".format(moments(dF_HPF_R)))
# P,X = probability_dist(dF_HPF_E,N)
# ax1.plot(X,P,"-b",label="Deform: {}".format(moments(dF_HPF_E)))
# ax1.set_xlabel("HPF (1.5-40Hz): $\\Delta F/(W_RL/L_2)$ [-]")
# ax1.grid()
# ax1.legend()
# ax1.set_yscale("log")
# P,X = probability_dist(dt_HPF_R,N)
# ax2.plot(X,P,"-r",label="Rigid: {}".format(moments(dt_HPF_R)))
# P,X = probability_dist(dt_HPF_E,N)
# ax2.plot(X,P,"-b",label="Deform: {}".format(moments(dt_HPF_E)))
# ax2.set_xlabel("$\\Delta \\tau$ [s]")
# ax2.grid()
# ax2.legend()
# ax2.set_yscale("log")
# fig.supylabel("log() Probability [-]")
# plt.tight_layout()
# plt.savefig(out_dir+"HPF_dF_dt.png")
# plt.close(fig)



# #FBR calc
# dF_diff_E = []
# for i in np.arange(0,len(zero_crossings_index_FBR_E)-1):

#     it_1 = zero_crossings_index_FBR_E[i]
#     it_2 = zero_crossings_index_FBR_E[i+1]

   
#     dF_diff_E.append((abs(FBR_E[it_2] - FBR_E[it_1]) - abs(LPF_2_FBR_E[it_2] - LPF_2_FBR_E[it_1])))


# dF_diff_R = []
# for i in np.arange(0,len(zero_crossings_index_FBR_R)-1):

#     it_1 = zero_crossings_index_FBR_R[i]
#     it_2 = zero_crossings_index_FBR_R[i+1]

   
#     dF_diff_R.append((abs(FBR_R[it_2] - FBR_R[it_1]) - abs(LPF_2_FBR_R[it_2] - LPF_2_FBR_R[it_1])))


# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(dF_diff_R,N)
# plt.plot(X,P,"-r",label="Rigid: {}".format(moments(dF_diff_R)))
# P,X = probability_dist(dF_diff_E,N)
# plt.plot(X,P,"-b",label="Deform: {}".format(moments(dF_diff_E)))
# plt.xlabel("$| \\Delta F_{total}|-| \\Delta F_{LPF+BPF}|$ [kN]")
# plt.ylabel("log() Probability [-]")
# plt.yscale("log")
# plt.legend()
# plt.title("Contribution from HPF to total $F_B$")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"dF_diff.png")
# plt.close(fig)


# E = []
# R = []
# thresholds = np.arange(0.5,1.5,0.1)
# for threshold in thresholds:
#     filtered = np.abs(dF_R) >= threshold
#     R.append(sum(np.abs(dF_R) >= threshold))
#     E.append(sum(np.abs(dF_E) >= threshold))

# fig = plt.figure(figsize=(14,8))
# plt.plot(thresholds,R,"-or",label="Rigid")
# plt.plot(thresholds,E,"-ob",label="Deform")
# plt.xlabel("$|\\Delta F/(W_RL/L_2)|$ [-]")
# plt.ylabel("Number of jumps exceed threshold [-]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"threshold_events.png")
# plt.close(fig)

# fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8),sharex=True)
# # ax1.plot(Time_OF[1:],hvelB1_E[:,225],"-b",label="Deform")
# # ax1.plot(Time_OF[1:],hvelB1_R[:,225],"-r",label="Rigid")
# # ax1.legend()
# # ax1.grid()
# # ax1.set_title("B1 streamwise velocity at 75% span [m/s]")
# # ax2.plot(Time_OF[1:],hvelB2_E[:,225],"-b",label="Deform")
# # ax2.plot(Time_OF[1:],hvelB2_R[:,225],"-r",label="Rigid")
# # ax2.legend()
# # ax2.grid()
# # ax2.set_title("B2 streamwise velocity at 75% span [m/s]")
# # ax3.plot(Time_OF[1:],hvelB3_E[:,225],"-b",label="Deform")
# # ax3.plot(Time_OF[1:],hvelB3_R[:,225],"-r",label="Rigid")
# # ax3.set_title("B3 streamwise velocity at 75% span [m/s]")
# # ax3.legend()
# # ax3.grid()
# # fig.supxlabel("Time [s]")
# # plt.tight_layout()

# xD = np.subtract(xs_E,xs_R)
# vD = dt_calc(xD,dt_OF)
# vdiff = np.subtract(hvelB1_E[:,225],hvelB1_R[:,225])
# out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
# plt.rcParams['font.size'] = 16
# cc = round(correlation_coef(vdiff[:-1],vD),2)
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF[1:],vdiff,"-b",label="Difference elastic-rigid")
# plt.plot(Time_OF[:-2],vD,"-r",label="Deformation velocity")
# plt.xlabel("Time [s]")
# plt.ylabel("Streamwise velocity at 75% span [m/s]")
# plt.title("correlation coefficent = {}".format(cc))
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.show()
# # plt.savefig(out_dir+"Deformation_velocity.png")
# # plt.close()