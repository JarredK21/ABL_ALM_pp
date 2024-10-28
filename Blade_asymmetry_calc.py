from netCDF4 import Dataset
import numpy as np
from scipy.fft import fft, fftfreq, fftshift,ifft
from scipy import interpolate
from multiprocessing import Pool
import matplotlib.pyplot as plt

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


def tranform_fixed_frame(y,z,Theta):

    Y = y*np.cos(Theta) - z*np.sin(Theta)
    Z = y*np.sin(Theta) + z*np.cos(Theta)

    return Y,Z



def actuator_asymmetry_calc(it):

    xo = np.array(WT.variables["xyz"][it,1:301,0])
    yo = np.array(WT.variables["xyz"][it,1:301,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB1 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB1 = np.array(WT.variables["xyz"][it,1:301,2]) - Rotor_coordinates[2]

    xo = np.array(WT.variables["xyz"][it,301:601,0])
    yo = np.array(WT.variables["xyz"][it,301:601,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB2 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB2 = np.array(WT.variables["xyz"][it,301:601,2]) - Rotor_coordinates[2]


    xo = np.array(WT.variables["xyz"][it,601:901,0])
    yo = np.array(WT.variables["xyz"][it,601:901,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB3 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB3 = np.array(WT.variables["xyz"][it,601:901,2]) - Rotor_coordinates[2]


    IyB1 = np.sum(hvelB1[it]*zB1)*dr
    IzB1 = np.sum(hvelB1[it]*yB1)*dr

    IyB2 = np.sum(hvelB2[it]*zB2)*dr
    IzB2 = np.sum(hvelB2[it]*yB2)*dr


    IyB3 = np.sum(hvelB3[it]*zB3)*dr
    IzB3 = np.sum(hvelB3[it]*yB3)*dr

    return IyB1+IyB2+IyB3, IzB1+IzB2+IzB3



in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

df = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df.variables["Time_OF"])
dt_OF = Time_OF[1] - Time_OF[0]
Tstart_OF_idx = np.searchsorted(Time_OF,Time_OF[0]+200)
Time_steps = np.arange(Tstart_OF_idx,len(Time_OF)-1)
Time_OF = Time_OF[Tstart_OF_idx:]

Time_sampling = np.array(df.variables["Time_sampling"])
dt_sample = Time_sampling[1] - Time_sampling[0]
Tstart_sample_idx = np.searchsorted(Time_sampling,Time_sampling[0]+200)
Time_sampling = Time_sampling[Tstart_sample_idx:]

Rotor_avg_vars = df.groups["Rotor_Avg_Variables"]
Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]

Iy = np.array(Rotor_avg_vars.variables["Iy"][Tstart_sample_idx:])
Iz = -np.array(Rotor_avg_vars.variables["Iz"][Tstart_sample_idx:])
Iy_LPF = hard_filter(Iy,0.3,dt_sample,"lowpass")
Iz_LPF = hard_filter(Iz,0.3,dt_sample,"lowpass")
I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
I_LPF = np.sqrt(np.add(np.square(Iy_LPF),np.square(Iz_LPF)))

OF_vars = df.groups["OpenFAST_Variables"]

Azimuth = np.radians(np.array(OF_vars.variables["Azimuth"][Tstart_OF_idx:]))

RtAeroFxh = np.array(OF_vars.variables["RtAeroFxh"][Tstart_OF_idx:])/1000

RtAeroFyh = np.array(OF_vars.variables["RtAeroFyh"][Tstart_OF_idx:])/1000
RtAeroFzh = np.array(OF_vars.variables["RtAeroFzh"][Tstart_OF_idx:])/1000

RtAeroFys, RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

RtAeroMxh = np.array(OF_vars.variables["RtAeroMxh"][Tstart_OF_idx:])/1000

RtAeroMyh = np.array(OF_vars.variables["RtAeroMyh"][Tstart_OF_idx:])/1000
RtAeroMzh = np.array(OF_vars.variables["RtAeroMzh"][Tstart_OF_idx:])/1000

RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
RtAeroMys_LPF = hard_filter(RtAeroMys,0.3,dt_OF,"lowpass")
RtAeroMzs_LPF = hard_filter(RtAeroMzs,0.3,dt_OF,"lowpass")

RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 
RtAeroMR_LPF = np.sqrt( np.add(np.square(RtAeroMys_LPF), np.square(RtAeroMzs_LPF)) ) 

f = interpolate.interp1d(Time_OF,RtAeroMys_LPF)
RtAeroMys_interp = f(Time_sampling)
print(correlation_coef(Iy_LPF,RtAeroMys_interp))

f = interpolate.interp1d(Time_OF,RtAeroMzs_LPF)
RtAeroMzs_interp = f(Time_sampling)
print(correlation_coef(Iz_LPF,RtAeroMzs_interp))

f = interpolate.interp1d(Time_OF,RtAeroMR_LPF)
RtAeroMR_interp = f(Time_sampling)
print(correlation_coef(I_LPF,RtAeroMR_interp))



df_WT = Dataset(in_dir+"WTG01b.nc")

WT = df_WT.groups["WTG01"]


Rotor_coordinates = [np.float64(WT.variables["xyz"][0,0,0]),np.float64(WT.variables["xyz"][0,0,1]),np.float64(WT.variables["xyz"][0,0,2])]


df = Dataset(in_dir+"WTG01a.nc")
uvelB1 = np.array(df.variables["uvel"][:,1:301])
vvelB1 = np.array(df.variables["vvel"][:,1:301])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
hvelB1 = np.add(np.cos(np.radians(29.29))*uvelB1, np.sin(np.radians(29.29))*vvelB1)
uvelB2 = np.array(df.variables["uvel"][:,301:601])
vvelB2 = np.array(df.variables["vvel"][:,301:601])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
hvelB2 = np.add(np.cos(np.radians(29.29))*uvelB2, np.sin(np.radians(29.29))*vvelB2)
uvelB3 = np.array(df.variables["uvel"][:,601:901])
vvelB3 = np.array(df.variables["vvel"][:,601:901])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
hvelB3 = np.add(np.cos(np.radians(29.29))*uvelB3, np.sin(np.radians(29.29))*vvelB3)


R = np.linspace(0,63,300)
dr = R[1] - R[0]

Iy = []
Iz = []
ix=0
with Pool() as pool:
    for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,Time_steps):
        Iy.append(Iy_it); Iz.append(Iz_it)
        print(ix)
        ix+=1
Iy = np.array(Iy); Iz = -np.array(Iz)
I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))


print(correlation_coef(Iy,RtAeroMys[:-1]))

print(correlation_coef(Iz,RtAeroMzs[:-1]))

print(correlation_coef(I,RtAeroMR[:-1]))

out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Rotor_Var_plots/"
plt.rcParams['font.size'] = 16
fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(I,dt_OF,"IB")
ax.loglog(frq,PSD,"-b")
ax.set_ylabel("Blade Asymmetry [$m^3/s$]")
ax.grid()
ax2=ax.twinx()
frq,PSD = temporal_spectra(RtAeroMR,dt_OF,"MR")
ax2.loglog(frq,PSD,"-r")
ax2.set_ylabel("Out-of-plane bending moment [kN-m]")
fig.supxlabel("Frequency [Hz]")
plt.tight_layout()
plt.savefig(out_dir+"Spectra_IB_MR.png")
plt.close(fig)

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(RtAeroMR,dt_OF,"MR")
ax.loglog(frq,PSD,"-r")
ax.set_ylabel("Out-of-plane bending moment [kN-m]")
ax.grid()
ax2=ax.twinx()
frq,PSD = temporal_spectra(I,dt_OF,"IB")
ax2.loglog(frq,PSD,"-b")
ax2.set_ylabel("Blade Asymmetry [$m^3/s$]")
fig.supxlabel("Frequency [Hz]")
plt.tight_layout()
plt.savefig(out_dir+"Spectra_IB_MR_2.png")
plt.close(fig)