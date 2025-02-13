from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate


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


def tranform_fixed_frame(y,z,Theta):

    Y = y*np.cos(Theta) - z*np.sin(Theta)
    Z = y*np.sin(Theta) + z*np.cos(Theta)

    return Y,Z


def dt_calc(u,dt):
    #compute time derivative using first order forward difference
    d_dt = []
    for i in np.arange(0,len(u)-1,1):
        d_dt.append( (u[i+1]-u[i])/dt )

    return d_dt


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


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


def LPF_standard_dev(signal):

    N = len(signal)
    signal_LPF = np.array(hard_filter(signal,0.3,dt,"lowpass"))
    sigma = np.sqrt(np.sum(np.square(np.subtract(signal,signal_LPF)))/N)

    return sigma


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360




in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
out_dir=in_dir+"Jims_plots/"


df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["Time_OF"])
dt = Time_OF[1] - Time_OF[0]

OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]
print(OpenFAST_vars)

Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"]))

#Torque
RtAeroMxa = np.array(OpenFAST_vars.variables["RtAeroMxh"])/1000

RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"])/1000
RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"])/1000
RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"])/1000
RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"])/1000

RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

#OOPBM
L1 = 1.912; L2 = 2.09

OOPBM = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

#FB
FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FBR = np.degrees(np.arctan2(FBz,FBy))
Theta_FBR = theta_360(Theta_FBR)


#LPF signals
LPF_Mx = np.array(hard_filter(RtAeroMxa,0.3,dt,"lowpass"))

LPF_OOPBM = np.array(hard_filter(OOPBM,0.3,dt,"lowpass"))
BPF_OOPBM = np.array(hard_filter(OOPBM,[0.3,0.9],dt,"bandpass"))
HPF_OOPBM = np.array(hard_filter(OOPBM,[1.5,40],dt,"bandpass"))
LPF_BPF_OOPBM = np.array(hard_filter(OOPBM,0.9,dt,"lowpass"))

LPF_FBR = np.array(hard_filter(FBR,0.3,dt,"lowpass"))
BPF_FBR = np.array(hard_filter(FBR,[0.3,0.9],dt,"bandpass"))
HPF_FBR = np.array(hard_filter(FBR,[1.5,40],dt,"bandpass"))
LPF_BPF_FBR = np.array(hard_filter(FBR,0.9,dt,"lowpass"))

Time_start_idx = np.searchsorted(Time_OF,200); Time_end_idx = np.searchsorted(Time_OF,240)
plt.rcParams['font.size'] = 24
fig = plt.figure(figsize=(28,12))
plt.plot(Time_OF[Time_start_idx:Time_end_idx],FBR[Time_start_idx:Time_end_idx],"-k",label="$F_{B,\perp}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],LPF_BPF_FBR[Time_start_idx:Time_end_idx],"-g",label="LPF 0.9Hz $F_{B,\perp}$")
plt.xlabel("Time [s]")
plt.ylabel("Main Bearing radial force magnitude [kN]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"LPF_BPF_FBR.png")
plt.close()

Time_start_idx = np.searchsorted(Time_OF,200); Time_end_idx = np.searchsorted(Time_OF,300)
fig = plt.figure(figsize=(28,12))
plt.plot(Time_OF[Time_start_idx:Time_end_idx],FBR[Time_start_idx:Time_end_idx],"-k",label="$F_{B,\perp}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],LPF_FBR[Time_start_idx:Time_end_idx],"-g",label="LPF 0.3Hz $F_{B,\perp}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],BPF_FBR[Time_start_idx:Time_end_idx],"-r",label="BPF 0.3-0.9Hz $F_{B,\perp}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],HPF_FBR[Time_start_idx:Time_end_idx]-1000,"-b",label="HPF 1.5-40Hz $F_{B,\perp}$\noffset=-1000kN")
plt.legend()
plt.xlabel("Time [s]")
plt.ylabel("Main Bearing radial force magnitude [kN]")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"FBR_3_freqs.png")
plt.close()


Time_start_idx = np.searchsorted(Time_OF,200); Time_end_idx = np.searchsorted(Time_OF,240)
fig = plt.figure(figsize=(28,12))
plt.plot(Time_OF[Time_start_idx:Time_end_idx],OOPBM[Time_start_idx:Time_end_idx],"-k",label="$M_{H,\perp}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],LPF_BPF_OOPBM[Time_start_idx:Time_end_idx],"-g",label="LPF 0.9Hz $M_{H,\perp}$")
plt.xlabel("Time [s]")
plt.ylabel("OOPBM magnitude [kN-m]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"LPF_BPF_OOPBM.png")
plt.close()

Time_start_idx = np.searchsorted(Time_OF,200); Time_end_idx = np.searchsorted(Time_OF,300)
fig = plt.figure(figsize=(28,12))
plt.plot(Time_OF[Time_start_idx:Time_end_idx],OOPBM[Time_start_idx:Time_end_idx],"-k",label="$M_{H,\perp}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],LPF_OOPBM[Time_start_idx:Time_end_idx],"-g",label="LPF 0.3Hz $M_{H,\perp}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],BPF_OOPBM[Time_start_idx:Time_end_idx],"-r",label="BPF 0.3-0.9Hz $M_{H,\perp}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],HPF_OOPBM[Time_start_idx:Time_end_idx]-1000,"-b",label="HPF 1.5-40Hz $M_{H,\perp}$\noffset=-1000kN")
plt.xlabel("Time [s]")
plt.ylabel("OOPBM magnitude [kN-m]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"OOPBM_3_freqs.png")
plt.close()


#Statistics over 1000s
print(np.average(OOPBM)); print(np.std(OOPBM)); print(LPF_standard_dev(OOPBM))
print(np.average(RtAeroMxa)); print(np.std(RtAeroMxa)); print(LPF_standard_dev(RtAeroMxa))
print(np.average(FBR)); print(np.std(FBR)); print(LPF_standard_dev(FBR))


#plots over 800-1000s

Time_start = 800
Time_start_idx = np.searchsorted(Time_OF,Time_start)

Time_end = 1000
Time_end_idx = np.searchsorted(Time_OF,Time_end)

plt.rcParams['font.size'] = 24
fig = plt.figure(figsize=(28,12))
plt.plot(Time_OF[Time_start_idx:Time_end_idx], OOPBM[Time_start_idx:Time_end_idx])
plt.axhline(y=np.average(OOPBM),linestyle="--",color="k",label="$\langle \widetilde{M}_{H,\perp ,mod} \\rangle _{1000s}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],LPF_OOPBM[Time_start_idx:Time_end_idx],"-g",label="LPF $\widetilde{M}_{H,\perp ,mod}$")
plt.xlabel("Time [s]")
plt.ylabel("Aerodynamic OOPBM magnitude [kN-m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"OOPBM.png")
plt.close()

fig = plt.figure(figsize=(28,12))
plt.plot(Time_OF[Time_start_idx:Time_end_idx], RtAeroMxa[Time_start_idx:Time_end_idx])
plt.axhline(y=np.average(RtAeroMxa),linestyle="--",color="k",label="$\langle \widetilde{M}_{H,x} \\rangle _{1000s}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],LPF_Mx[Time_start_idx:Time_end_idx],"-g",label="LPF $\widetilde{M}_{H,x}$")
plt.xlabel("Time [s]")
plt.ylabel("Torque [kN-m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"Torque.png")
plt.close()

fig = plt.figure(figsize=(28,12))
plt.plot(Time_OF[Time_start_idx:Time_end_idx], FBR[Time_start_idx:Time_end_idx])
plt.axhline(y=np.average(FBR),linestyle="--",color="k",label="$\langle \widetilde{F}_{B,\perp} \\rangle _{1000s}$")
plt.plot(Time_OF[Time_start_idx:Time_end_idx],LPF_FBR[Time_start_idx:Time_end_idx],"-g",label="LPF $\widetilde{F}_{B,\perp}$")
plt.xlabel("Time [s]")
plt.ylabel("Aerodynamic main bearing radial force magnitude [kN]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"FBR.png")
plt.close()


#trajectories

MR = np.sqrt(np.add(np.square(RtAeroMys/L2),np.square(RtAeroMzs/L2)))
MTheta = np.degrees(np.arctan2(RtAeroMys/L2,-RtAeroMzs/L2))
MTheta = theta_360(MTheta)

FR = np.sqrt(np.add(np.square(RtAeroFys*((L1+L2)/L2)),np.square(RtAeroFzs*((L1+L2)/L2))))
FTheta = np.degrees(np.arctan2(RtAeroFzs*((L1+L2)/L2),RtAeroFys*((L1+L2)/L2)))
FTheta = theta_360(FTheta)

Time_start = 200
Time_start_idx = np.searchsorted(Time_OF,Time_start)

Time_end = 240
Time_end_idx = np.searchsorted(Time_OF,Time_end)

plt.rcParams['font.size'] = 12
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(projection='polar')
ax.plot(np.radians(Theta_FBR[Time_start_idx:Time_end_idx]),FBR[Time_start_idx:Time_end_idx],"-r",label="$\widetilde{\mathbf{F}}_{B,\perp}$")
ax.plot(np.radians(MTheta[Time_start_idx:Time_end_idx]),MR[Time_start_idx:Time_end_idx],"-b",label="$\widetilde{\mathbf{M}}_{H,\perp,mod}(1/L_2)$")
ax.plot(np.radians(FTheta[Time_start_idx:Time_end_idx]),FR[Time_start_idx:Time_end_idx],"-g",label="$\widetilde{\mathbf{F}}_{H,\perp}(L/L_2)$")

ax.legend()
ax.set_title("Vector trajectories [kN]\n200-240s")
plt.savefig(out_dir+"FR_MR_FBR_trajectory.png")
plt.close(fig)