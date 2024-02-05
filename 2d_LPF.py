from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import netCDF4 as nc
from scipy.fft import fft2, fftfreq, fftshift
import numpy as np
from multiprocessing import Pool
import math
import pandas as pd


def butterwort_low_pass_filer(f):

    M,N = f.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 9e-03 #cut off frequency

    delx = 10
    freqs = fftshift(fftfreq(x, d=delx)) #mirror frequencies are equal and opposite sign in the Re, Im are zero.
    freq2d = np.array(np.meshgrid( freqs, freqs)) #mirror frequencies are equal in the Re, Im are zero
    freqs2d = np.sqrt( freq2d[0]**2 + freq2d[1]**2)

    for u in range(M):
        for v in range(N):
            D = freqs2d[u,v]
            if D >= D0:
                H[u,v] = 0
            else:
                H[u,v] = 1

    return H


def Horizontal_velocity(it):
    mag_horz_vel = u[it]*np.cos(np.radians(29)) + v[it]*np.sin(np.radians(29))
    return mag_horz_vel


def two_dim_LPF(it):

    U = u[it].reshape(x,y)

    Z = U.reshape(x,y)
    X,Y = np.meshgrid(xs,ys)

    fig = plt.figure(figsize=(50,30))
    cs = plt.contourf(X,Y,Z, cmap=cm.coolwarm)
    cb = plt.colorbar(cs)

    #FFT
    ufft = np.fft.fftshift(np.fft.fft2(U))


    #multiply filter
    H = butterwort_low_pass_filer(U)
    ufft_filt = ufft * H

    #IFFT
    ufft_filt_shift = np.fft.ifftshift(ufft_filt)
    iufft_filt = np.real(np.fft.ifft2(ufft_filt_shift))

    Z = iufft_filt.reshape(x,y)
    X,Y = np.meshgrid(xs,ys)

    fig = plt.figure(figsize=(50,30))
    cz = plt.contourf(X,Y,Z, cmap=cm.coolwarm)
    cd = plt.colorbar(cz)

    plt.show()

    return iufft_filt.flatten()


in_dir = "../../ABL_precursor_2/"
out_dir = in_dir+"plots/"

df = pd.read_csv(in_dir+"spectral_data_uu.csv")
freq_uu_1d = df["freq"]
mean_uu = df["mean"]
del df
df = pd.read_csv(in_dir+"spectral_data_ww.csv")
freq_ww_1d = df["freq"]
mean_ww = df["mean"]


fig = plt.figure(figsize=(14,8))
plt.rcParams.update({'font.size': 16})
plt.loglog(freq_uu_1d, mean_uu,"-r")
plt.loglog(freq_ww_1d, mean_ww,"b")
plt.loglog(freq_uu_1d, 1e-06* freq_uu_1d**(-5./3.),"--k")
plt.ylim([1e-09, 1])
plt.xlabel('k - Wave number [1/m]')
plt.ylabel('(k) - Power spectral density')
plt.title("Energy spectra at z = 90m")
plt.grid()
plt.legend(["u","w", "$k^{-5/3}$"])
plt.savefig(out_dir+"Spacial_spectra_90m.png")
plt.close(fig)

a = Dataset(in_dir+"sampling_l_85.nc")

p = a.groups["p_l"]

#time options
Time = np.array(a.variables["time"])
tstart = 32700
tstart_idx = np.searchsorted(Time,tstart)
tend = 33700
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]


x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points


#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xs = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x)
ys = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y)
zs = 0


col_names = []
for it in Time_steps:
    col_names.append(str(Time[it]))

LPF_data_uu =  pd.DataFrame(data=None, columns=col_names)
LPF_data_ww = pd.DataFrame(data=None, columns=col_names)

#velocity field
velocity_field_u = True
velocity_field_w = True


if velocity_field_u == True:
    u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
    v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
    with Pool() as pool:
        u_hvel = []
        for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
            
            u_hvel.append(u_hvel_it)
            print(len(u_hvel))
    u = np.array(u_hvel); del u_hvel; del v


    ix = 0
    with Pool() as pool:
        for iufft in pool.imap(two_dim_LPF, Time_steps):
            LPF_data_uu["{}".format(Time[ix])] = iufft
            ix+=1
            print(ix)

    LPF_data_uu.to_csv(in_dir+'LPF_data_uu.csv',index=False); del u; del LPF_data_uu


#vertical velocity
if velocity_field_w == True:
    u = np.array(p.variables["velocityz"][tstart_idx:tend_idx])
    ix = 0
    with Pool() as pool:
        for iufft in pool.imap(two_dim_LPF, Time_steps):
            LPF_data_ww["{}".format(Time[ix])] = iufft
            ix+=1
            print(ix)

    LPF_data_ww.to_csv(in_dir+'LPF_data_ww.csv',index=False); del u; del LPF_data_ww