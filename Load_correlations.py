import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy.stats import pearsonr
import glob 
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
import math


dir = "../../../jarred/NAWEA_23/post_processing/plots/"

offsets = [-63.0, -31.5, 0.0]

Variables = ["Time_OF","Time_sample","Ux_{}".format(offsets[2]),"IA_{}".format(offsets[2]),"RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[s]","[s]", "[m/s]","[$m^4/s$]","[N]","[N-m]","[N-m]","[degrees]"]
Ylabels = ["Time","Time","$<Ux'>_{rotor}$ rotor averaged velocity","Asymmery Parameter","Rotor Thrust", "Rotor Torque",
            "Out-of-plane bending moment","Angle Out-of-plane bending moment"]

compare_correlations = False
compare_time_series = False
compare_FFT = True


#needs fixing fft.shift()
def low_pass_filter2(signal, cutoff):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def low_pass_filter(signal,cutoff,dt):
    
    fs =1/dt
    n = len(signal) 
    if n%2==0:
        nhalf = int(n/2+1)
    else:
        nhalf = int((n+1)/2)
    frq = np.arange(nhalf)*fs/n
    Y   = np.fft.fft(signal)
    m = np.searchsorted(frq,cutoff)
    lc = nhalf-m; hc = nhalf+m
    mask = np.concatenate(( np.zeros(lc-1),np.ones(hc-lc),np.zeros(n-hc+1) ))
    Y_LP = np.multiply(Y,mask)
    signal_LP = np.fft.ifft(Y_LP)
    signal_LP = np.real(signal_LP)

    return signal_LP


def remove_nan(Var):

    signal = df[Var]
    new_signal = list()
    for element in signal:
        if not math.isnan(element):
            new_signal.append(element)
    return new_signal


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

#compare correlations
if compare_correlations == True:
    for j in np.arange(2,4,1):
        for i in np.arange(4,len(Variables)-1):

            fig,ax = plt.subplots(figsize=(14,8))
            Var = Variables[i]
            unit = units[i]
            Ylabel = Ylabels[i]
            corr_var = Variables[j]

            df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out.csv")

            time_OF = remove_nan("Time_OF")
            time_sample = remove_nan("Time_sample")
            time_sample[0] = time_OF[0]
            time_sample[-1] = time_OF[-1]

            dt = time_OF[1] - time_OF[0]

            correlation_variable = remove_nan(Var = corr_var)

            if corr_var[0:2] == "IA":
                f = interpolate.interp1d(time_sample,correlation_variable)
                correlation_variable = f(time_OF)

            Theta = remove_nan(Var = "Theta")

            signal = remove_nan(Var)

            if Var == "MR":
                cutoff = 0.5*(12.1/60)
                signal_LP = low_pass_filter(signal,cutoff,dt)
                corr, _ = pearsonr(correlation_variable,signal)
            else:
                cutoff = 0.5*(12.1/60)*3
                #signal_LP = low_pass_filter(signal,cutoff,dt)
                signal_LP = low_pass_filter2(signal, cutoff)
                #LP_diff = signal_LP-signal_LP2
                corr, _ = pearsonr(correlation_variable, signal)


            ax.plot(time_OF,signal,'-b')
            ax.plot(time_OF,signal_LP,"-r")
            ax.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)

            ax2=ax.twinx()
            if corr_var[0:2] == "IA":
                ax2.plot(time_OF,correlation_variable,"--k")
                ax2.set_ylabel("IA - Asymmetry Parameter [$m^4/s$]",fontsize=16)
                plt.title("Correlating IA at {0}m from turbine, with {1}".format(offsets[2],Ylabel),fontsize=18)
                ax.legend(["-","Correlation with IA = {0}".format(np.round(corr,2))])
            elif corr_var[0:2] == "Ux":
                ax2.plot(time_OF,correlation_variable,"--k")
                ax2.set_ylabel("$<Ux'>_{rotor}$ - Rotor averaged normal Velocity [m/s]",fontsize=16)
                plt.title("Correlating $<Ux'>rotor$ at {0}m from turbine, with {1}".format(offsets[2],Ylabel),fontsize=18)
                ax.legend(["-","Correlation with $<Ux'>rotor$ = {0}".format(np.round(corr,2))])

            ax.set_xlabel("Time [s]",fontsize=16)
            plt.tight_layout()
            plt.savefig(dir+"corr_{0}_{1}_{2}.png".format(offsets[2],corr_var[0:2],Var))
            plt.close(fig)

Variables = ["Time_OF","Time_sample","Ux_{}".format(offsets[2]),"RtAeroFxh","RtAeroMxh","IA_{}".format(offsets[2]),"MR","Theta"]
units = ["[s]","[s]", "[m/s]","[$m^4/s$]","[N]","[N-m]","[N-m]","[degrees]"]
Ylabels = ["Time","Time","$<Ux'>_{rotor}$ rotor averaged velocity","Rotor Thrust", "Rotor Torque","Asymmery Parameter",
            "Out-of-plane bending moment","Angle Out-of-plane bending moment"]


if compare_time_series == True:
    #comparing time series
    fig, axs = plt.subplots(6,1,figsize=(32,24))
    plt.rcParams.update({'font.size': 16})
    for i in np.arange(2,len(Variables)):

        Var = Variables[i]
        unit = units[i]
        Ylabel = Ylabels[i]

        df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out.csv")

        time_OF = remove_nan("Time_OF")
        time_sample = remove_nan("Time_sample")
        time_sample[0] = time_OF[0]
        time_sample[-1] = time_OF[-1]

        dt = time_OF[1] - time_OF[0]

        signal = remove_nan(Var)

        if Var[0:2] == "IA":
            f = interpolate.interp1d(time_sample, signal)
            signal = f(time_OF)
        elif Var == "Theta":
            signal = np.multiply((180/np.pi),signal)

        
        axs = axs.ravel()

        j=i-2

        axs[j].plot(time_OF,signal)
        axs[j].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=18)

    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(dir+"joint_vars.png")
    plt.close(fig)


#comparing spectra
if compare_FFT == True:
    fig, axs = plt.subplots(3,2,figsize=(32,24))
    plt.rcParams.update({'font.size': 16})
    for i in np.arange(2,len(Variables)):

        Var = Variables[i]
        unit = units[i]
        Ylabel = Ylabels[i]

        df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out.csv")

        time_OF = remove_nan("Time_OF")
        time_sample = remove_nan("Time_sample")
        time_sample[0] = time_OF[0]
        time_sample[-1] = time_OF[-1]

        dt = time_OF[1] - time_OF[0]

        signal = remove_nan(Var)

        if Var[0:2] == "IA":
            f = interpolate.interp1d(time_sample, signal)
            signal = f(time_OF)
        elif Var == "Theta":
            signal = np.multiply((180/np.pi),signal)

        
        frq, FFT_signal = temporal_spectra(signal,dt,Var)

        plt.plot()
        
        axs = axs.ravel()

        j=i-2

        axs[j].plot(frq,FFT_signal)
        axs[j].set_yscale('log')
        axs[j].set_xscale('log')
        axs[j].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=18)

        frq_int = [1/60, 1/30, 12.1/60, (12.1/60)*3]
        frq_label = ["60s eddy passage", "30s eddy passage", "1P", "3P"]
        y_FFT = FFT_signal[0]+1e+03
        for l in np.arange(0,len(frq_int)):
            axs[j].axvline(frq_int[l])
            axs[j].text(frq_int[l],y_FFT, frq_label[l])


    fig.supxlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.savefig(dir+"joint_vars_FFT.png")
    plt.close(fig)