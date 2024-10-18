from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pyFAST.input_output as io
from scipy.fft import fft, fftfreq, fftshift,ifft
from multiprocessing import Pool
import time
from scipy import interpolate



def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def tranform_fixed_frame(y,z,Theta):

    Y = y*np.cos(Theta) - z*np.sin(Theta)
    Z = y*np.sin(Theta) + z*np.cos(Theta)

    return Y,Z


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



in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

df = Dataset(in_dir+"Dataset.nc")
Time_OF = np.array(df.variables["Time_OF"])
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

RtAeroMxh_E = np.array(OF_vars.variables["RtAeroMxh"][Tstart_idx:])/1000

RtAeroMyh = np.array(OF_vars.variables["RtAeroMyh"][Tstart_idx:])/1000
RtAeroMzh = np.array(OF_vars.variables["RtAeroMzh"][Tstart_idx:])/1000

RtAeroMys_E, RtAeroMzs_E = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

RtAeroMR_E = np.sqrt( np.add(np.square(RtAeroMys_E), np.square(RtAeroMzs_E)) ) 

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

Rotor_avg_vars = df.groups["Rotor_Avg_Variables"]
Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]

Ux_E = np.array(Rotor_avg_vars.variables["Ux"][Tstart_idx_sampling:])
Iy_E = np.array(Rotor_avg_vars.variables["Iy"][Tstart_idx_sampling:])
Iz_E = np.array(Rotor_avg_vars.variables["Iz"][Tstart_idx_sampling:])
IE = np.sqrt(np.add(np.square(Iy_E),np.square(Iz_E)))

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


# #blade asymmetry
# df = Dataset(in_dir+"WTG01a.nc")

# Time_steps = np.arange(0,len(Time_OF)-1)
# Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:])

# uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
# vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1 = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
# vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2 = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
# vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3 = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)

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


in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df = Dataset(in_dir+"Dataset.nc")

OF_vars = df.groups["OpenFAST_Variables"]

RtAeroFxh_R = np.array(OF_vars.variables["RtAeroFxh"][Tstart_idx:])/1000

RtAeroFyh = np.array(OF_vars.variables["RtAeroFyh"][Tstart_idx:])/1000
RtAeroFzh = np.array(OF_vars.variables["RtAeroFzh"][Tstart_idx:])/1000

RtAeroFys_R, RtAeroFzs_R = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

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


Rotor_avg_vars = df.groups["Rotor_Avg_Variables"]
Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]

Ux_R = np.array(Rotor_avg_vars.variables["Ux"][Tstart_idx_sampling:])
Iy_R = np.array(Rotor_avg_vars.variables["Iy"][Tstart_idx_sampling:])
Iz_R = np.array(Rotor_avg_vars.variables["Iz"][Tstart_idx_sampling:])
IR = np.sqrt(np.add(np.square(Iy_R),np.square(Iz_R)))

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

# Time_steps = np.arange(0,len(Time_OF)-1)
# Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:])

# uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
# vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1 = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
# vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2 = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
# vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3 = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)

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




xlabel = np.array(["$|F_B|_E$","$|F_B|_R$", "$F_{B,y,E}$","$F_{B,y,R}$", "$F_{B,z,E}$","$F_{B,z,R}$", "$|M_H|_E$","$|M_H|_R$", "$M_{H,y,E}$","$M_{H,y,R}$", 
                    "$M_{H,z,E}$","$M_{H,z,R}$", "$F_{H,y,E}$","$F_{H,y,R}$", "$F_{H,z,E}$","$F_{H,z,R}$", "$F_{H,x,E}$","$F_{H,x,R}$", "$M_{H,x,E}$","$M_{H,x,R}$"])
colors = ["b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r"]
out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_mean_array,color=colors)
ax1.axhline(y=0.0,color="k")
ax1.errorbar(xlabel,Aero_mean_array,yerr=Aero_std_array,fmt = "o",color="k",capsize=10)
ax1.set_title("Aerodynamic variables")
ax2.bar(xlabel,Elasto_mean_array,color=colors)
ax2.axhline(y=0.0,color="k")
ax2.errorbar(xlabel,Elasto_mean_array,yerr=Elasto_std_array,fmt = "o",color="k",capsize=10)
ax2.set_title("Elastodyn variables")
plt.tight_layout()
plt.savefig(out_dir+"summary_bar_pairs.png")
plt.close(fig)

fig,ax1 = plt.subplots(figsize=(18,8))
ax1.bar(xlabel,mean_Aero_Elasto_array)
ax1.axhline(y=0.0,color="k")
ax1.set_title("Elasto-Aero")
plt.tight_layout()
plt.savefig(out_dir+"summary_bar_pairs_mean_change_aero_elasto.png")
plt.close(fig)

fig,ax1 = plt.subplots(figsize=(18,8))
ax1.bar(xlabel,std_Aero_Elasto_array)
ax1.axhline(y=0.0,color="k")
ax1.set_title("Elasto-Aero")
ax1.set_ylabel("Percentage change in the standard deviation [%]")
ax1.axhline(y=10.0,color="k",linestyle="--")
plt.tight_layout()
plt.savefig(out_dir+"summary_bar_pairs_std_change_aero_elasto.png")
plt.close(fig)

xlabel = np.array(["$|F_B|$", "$F_{B,y}$", "$F_{B,z}$", "$|M_H|$", "$M_{H,y}$","$M_{H,z}$", "$F_{H,y}$", "$F_{H,z}$", "$F_{H,x}$", "$M_{H,x}$"])
fig,ax1 = plt.subplots(figsize=(18,8))
ax1.bar(xlabel,mean_diff_Aero_Elasto)
ax1.axhline(y=0.0,color="k")
ax1.set_title("(Elasto-Aero)E - (Elasto-Aero)R")
plt.tight_layout()
plt.savefig(out_dir+"summary_bar_pairs_diff_change_aero_elasto.png")
plt.close(fig)

plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_mean_change)
ax1.axhline(y=0.0,color="k")
ax1.set_title("Aerodynamic variables (Elastic-Rigid)")
ax2.bar(xlabel,Elasto_mean_change)
ax2.axhline(y=0.0,color="k")
ax2.set_title("Elastodyn variables (Elastic-Rigid)")
fig.supylabel("Change in mean value")
plt.tight_layout()
plt.savefig(out_dir+"summary_bar_mean_change.png")
plt.close(fig)

plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_mean_perc_change)
ax1.axhline(y=0.0,color="k")
ax1.axhline(y=10.0,color="k",linestyle="--")
ax1.axhline(y=-10.0,color="k",linestyle="--")
ax1.set_title("Aerodynamic variables (Elastic-Rigid)/Rigid *100")
ax2.bar(xlabel,Elasto_mean_perc_change)
ax2.axhline(y=0.0,color="k")
ax2.axhline(y=10.0,color="k",linestyle="--")
ax2.axhline(y=-10.0,color="k",linestyle="--")
fig.supylabel("Percentage change in mean value [%]")
ax2.set_title("Elastodyn variables (Elastic-Rigid)/Rigid *100")
plt.tight_layout()
plt.savefig(out_dir+"summary_bar_mean_perc_change.png")
plt.close(fig)

plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_std_change)
ax1.axhline(y=0.0,color="k")
ax1.set_title("Aerodynamic variables (Elastic-Rigid)")
ax2.bar(xlabel,Elasto_std_change)
ax2.axhline(y=0.0,color="k")
ax2.set_title("Elastodyn variables (Elastic-Rigid)")
fig.supylabel("Change in standard deviation")
plt.tight_layout()
plt.savefig(out_dir+"summary_bar_std_change.png")
plt.close(fig)

plt.rcParams['font.size'] = 16
fig,(ax1,ax2) = plt.subplots(2,1,figsize=(18,8))
ax1.bar(xlabel,Aero_std_perc_change)
ax1.axhline(y=0.0,color="k")
ax1.axhline(y=10.0,color="k",linestyle="--")
ax1.axhline(y=-10.0,color="k",linestyle="--")
ax1.set_title("Aerodynamic variables (Elastic-Rigid)/Rigid *100")
ax2.bar(xlabel,Elasto_std_perc_change)
ax2.axhline(y=0.0,color="k")
ax2.axhline(y=10.0,color="k",linestyle="--")
ax2.axhline(y=-10.0,color="k",linestyle="--")
fig.supylabel("Percentage change in standard deviation [%]")
ax2.set_title("Elastodyn variables (Elastic-Rigid)/Rigid *100")
plt.tight_layout()
plt.savefig(out_dir+"summary_bar_std_perc_change.png")
plt.close(fig)