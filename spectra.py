# coding: utf-8
import netCDF4 as nc
from scipy.fft import fft2, fftfreq, fftshift
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import math


def two_d_spectra(it):

    if velocity == "u":
        U = u[it]*np.cos(np.radians(29)) + v[it]*np.sin(np.radians(29))


    U = U.reshape(x,y)

    ufft = np.fft.fftshift(np.fft.fft2(U - np.mean(U)))

    e_uvfft = (abs(ufft)/(x*y))**2

    freqs = fftshift(fftfreq(x, d=delx)) #mirror frequencies are equal and opposite sign in the Re, Im are zero.
    freq2d = np.array(np.meshgrid( freqs, freqs)) #mirror frequencies are equal in the Re, Im are zero
    freqs2d = np.sqrt( freq2d[0]**2 + freq2d[1]**2)
    max_freq = np.sqrt( 2.0 * freqs[-1]**2)

    freq1d = np.linspace(0, max_freq, int(x/2))[1:]
    freq1d_bins = np.r_[ [0], 0.5*(freq1d[1:] + freq1d[:-1]), [freq1d[-1]+100] ]

    e_1d = np.zeros(int(x/2) -1)

    for j,f in enumerate(freq1d):
        ff = np.where( (freqs2d >= freq1d_bins[j]) & (freqs2d < freq1d_bins[j+1]) )
        e_1d[j] = np.sum(e_uvfft[ff])

        fac = 1

    E = fac*np.sum(e_1d)

    u_pri = u - np.mean(u)
    q2_uu = np.sum(np.square(u_pri))
    
    A = (x*y)
    E2_uu = (1/A)*q2_uu

    delta_fft = np.sum(E) - np.sum(E2_uu)

    print("check summation over shells = ", delta_fft)


    return freq1d, e_1d
    

#main
#note sampling files need to exisit in all dirs otherwise error will occur
in_dir = "../../ABL_precursor_2/"

a = nc.Dataset(in_dir+"sampling_l_85.nc") #import sampling data
p = a.groups["p_l"] #Planer sampler
delx = 10.0

#time options
Time = np.array(a.variables["time"])
tstart = 32500
tstart_idx = np.searchsorted(Time,tstart)
tend = 33700
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]

y = p.ijk_dims[1] #no. y data points
x = p.ijk_dims[0] #no. x data points

col_names = []
for it in Time_steps:
    col_names.append(str(Time[it]))

velocities = ["u", "w"]

for velocity in velocities:

    if velocity == "u":
        spectra_data_uu =  pd.DataFrame(data=None, columns=col_names)
        u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
        v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

        ix = 0
        with Pool() as pool:
            for e,f in pool.imap(two_d_spectra, Time_steps):
                spectra_data_uu["{}".format(Time[ix])] = e
                ix+=1
                print(ix)

        spectra_uu_mean = spectra_data_uu.mean(axis=1)
        spectra_data_uu["mean"] = spectra_uu_mean

        spectra_data_uu['freqs'] = f

        spectra_data_uu.to_csv(in_dir+'spectral_data_uu_2.csv',index=False)


    if velocity == "w":
        spectra_data_ww =  pd.DataFrame(data=None, columns=col_names)
        u = np.array(p.variables["velocityw"][tstart_idx:tend_idx])

        ix = 0
        with Pool() as pool:
            for e,f in pool.imap(two_d_spectra, Time_steps):
                spectra_data_ww["{}".format(Time[ix])] = e
                ix+=1
                print(ix)

        spectra_ww_mean = spectra_data_ww.mean(axis=1)
        spectra_data_ww["mean"] = spectra_ww_mean

        spectra_data_ww['freqs'] = f

        spectra_data_ww.to_csv(in_dir+'spectral_data_ww_2.csv',index=False)


fig = plt.figure(figsize=(14,8))
plt.loglog(spectra_data_uu["freqs"], spectra_uu_mean,"-r")
plt.loglog(spectra_data_ww["freqs"], spectra_ww_mean,"b")
plt.loglog(spectra_data_uu["freqs"], 1e-06* spectra_data_uu["freqs"]**(-5./3.),"--k")
plt.ylim([1e-09, 1])
plt.xlabel('k - Wave number [1/m]')
plt.ylabel('(k) - Power spectral density')
plt.title("Energy spectra at z = 90m")
plt.grid()
plt.legend(["u","w", "$k^{-5/3}$"])
plt.savefig(in_dir+"Spacial_spectra_90m.png")
plt.close(fig)