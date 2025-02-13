from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor

def energy_contents_check(Var,e_fft,signal,dt):

    E = np.sum(e_fft)

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
    PSD = (abs(Y[range(nhalf)])**2) /n # PSD
    PSD[1:-1] = PSD[1:-1]*2


    energy_contents_check(Var,PSD,signal,dt)

    return frq, PSD

def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist



#in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
df = Dataset("sampling70000.nc")
Time = np.array(df.variables["time"])
dt = Time[1]-Time[0]
Start_time_idx = np.searchsorted(Time,38000)
Time = Time[Start_time_idx:]
p_r = df.groups["p_r"]


coordinates = np.array(p_r.variables["coordinates"][:1228800])

ymid = (np.min(coordinates[:,1]) + np.max(coordinates[:,1]))/2

heights = [20,35,50,65,80]
for height in heights:

    ic = 0
    for coo in coordinates:
        if floor(ymid) <= coo[1] < ceil(ymid) and height <= coo[2] < height+1:
            print(coo)
            print(ic)
            break
        ic+=1


    velocityx = np.sqrt( np.add( np.square(p_r.variables["velocityx"][Start_time_idx:,ic]), np.square(p_r.variables["velocityy"][Start_time_idx:,ic]) ))

    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time,velocityx)
    plt.ylabel("u veloicity component [m/s]")
    plt.xlabel("Time [s]")
    plt.grid()
    plt.title("Center of rotor. Height = {}m".format(height))
    plt.tight_layout()
    plt.savefig("plots/LES_u_{}.png".format(height))
    plt.close(fig)


    upri = velocityx - np.average(velocityx)
    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(upri,dt,"upri")
    plt.loglog(frq,PSD)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD - fluctuating u velocity component [$m^2/s^2$]")
    plt.ylim(bottom=1e-07)
    plt.grid()
    plt.tight_layout()
    plt.savefig("plots/PSD_LES_u_{}.png".format(height))
    plt.close(fig)