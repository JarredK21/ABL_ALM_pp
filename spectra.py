# coding: utf-8
import netCDF4 as nc
from scipy.fft import fft2, fftfreq, fftshift
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import math


def Horizontal_velocity(it):
    mag_horz_vel = u[it]*np.cos(np.radians(29)) + v[it]*np.sin(np.radians(29))
    return mag_horz_vel


def calc_variance_spectra(u, u_mean, x, y):
    u = u.reshape(x,y)

    ufft = np.fft.fftshift(np.fft.fft2(u - u_mean))

    e_uvfft = (abs(ufft)/(x*y))**2

    return e_uvfft


#define and sum over shells
def energy_over_shells(x, delx, e):

    freqs = fftshift(fftfreq(x, d=delx)) #mirror frequencies are equal and opposite sign in the Re, Im are zero.
    freq2d = np.array(np.meshgrid( freqs, freqs)) #mirror frequencies are equal in the Re, Im are zero
    freqs2d = np.sqrt( freq2d[0]**2 + freq2d[1]**2)
    max_freq = np.sqrt( 2.0 * freqs[-1]**2)

    freq1d = np.linspace(0, max_freq, int(x/2))[1:]
    freq1d_bins = np.r_[ [0], 0.5*(freq1d[1:] + freq1d[:-1]), [freq1d[-1]+100] ]

    e_1d = np.zeros(int(x/2) -1)

    for j,f in enumerate(freq1d):
        ff = np.where( (freqs2d >= freq1d_bins[j]) & (freqs2d < freq1d_bins[j+1]) )
        e_1d[j] = np.sum(e[ff])

    return freq1d, e_1d



def energy_contents_check(e_fft, u, x, y, e_1d, velocity):

    #check total TKE
    fac = 1/((x*y)**2)
    #fac = 1

    E = fac*np.sum(e_fft)

    u_pri = u - np.mean(u)
    q2_uu = np.sum(np.square(u_pri))
    
    A = (x*y)
    E2_uu = (1/A)*q2_uu

    delta_fft = np.sum(e_fft) - np.sum(e_1d)

    if velocity == "u":
        print("check summation over shells uu = ", delta_fft)
        print(E, E2_uu, "uu = ",(abs(E2_uu)/E))
    elif velocity == "w":
        print("check summation over shells ww = ", delta_fft)
        print(E, E2_uu, "ww = ",(abs(E2_uu)/E))


def temporal_energy_contents_check(Var,e_fft,signal,dt):

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


    temporal_energy_contents_check(Var,PSD,signal,dt)

    return frq, PSD


spacial_spectra_plot = False
temporal_spectra_plot = True

if spacial_spectra_plot == True:
    #how to low pass filter spacial data

    #main
    #note sampling files need to exisit in all dirs otherwise error will occur
    in_dir = "../../ABL_precursor_2/"
    out_dir = in_dir+"plots/"

    a = nc.Dataset(in_dir+"sampling_l_85.nc") #import sampling data
    p = a.groups["p_l"] #Planer sampler
    delx = 10.0

    y = p.ijk_dims[1] #no. y data points
    x = p.ijk_dims[0] #no. x data points

    spectra_data_uu =  pd.DataFrame(data=None, columns=["85"])
    spectra_data_ww =  pd.DataFrame(data=None, columns=["85"])

    #times and averaging??
    u = np.array(p.variables["velocityx"]); v = np.array(p.variables["velocityy"]); w = np.array(p.variables["velocityw"])
    u = Horizontal_velocity(0); del v

    u_mean = np.mean(u); w_mean = np.array(w)

    #energy components
    e_uufft = calc_variance_spectra(u, u_mean, x, y)
    e_wwfft = calc_variance_spectra(w, w_mean, x, y)


    #calculate energy over shells
    freq_uu_1d, e_uu_1d = energy_over_shells(x, delx, e_uufft)
    freq_ww_1d, e_ww_1d = energy_over_shells(x, delx, e_wwfft)

        

    # check energy contents
    energy_contents_check(e_uufft, u, x, y, delx, e_uu_1d, velocity="u") #Energy in uu variance
    energy_contents_check(e_wwfft, w, x, y, delx, e_ww_1d, velocity="w") #Energy in ww variance

    #export data to csv
    spectra_data_uu["85"] = e_uu_1d
    spectra_data_ww["85"] = e_ww_1d

    spectra_data_uu['freq'] = freq_uu_1d
    spectra_data_ww['freq'] = freq_ww_1d


    spectra_data_uu.to_csv(in_dir+'spectral_data_uu.csv',index=False)
    spectra_data_ww.to_csv(in_dir+'spectral_data_ww.csv',index=False)

    fig = plt.figure(figsize=(14,8))
    plt.loglog(freq_uu_1d, e_uu_1d,"-r")
    plt.loglog(freq_ww_1d, e_ww_1d,"b")
    plt.loglog(freq_uu_1d, 1e-06* freq_uu_1d**(-5./3.),"--k")
    plt.ylim([1e-09, 1])
    plt.xlabel('k - Wave number [1/m]')
    plt.ylabel('(k) - Power spectral density')
    plt.title("Energy spectra at z = 90m")
    plt.grid()
    plt.legend(["u","w", "$k^{-5/3}$"])
    plt.savefig(out_dir+"Spacial_spectra_90m.png")
    plt.close(fig)

