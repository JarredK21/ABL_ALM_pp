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


dir = "../../../jarred/NAWEA_23/post_processing/plots2/"

offsets = [-63.0, -31.5, 0.0]

Variables = ["Time_OF","Time_sample","Ux_{}".format(offsets[2]),"IA_{}".format(offsets[2]),"RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[s]","[s]", "[m/s]","[$m^4/s$]","[N]","[N-m]","[N-m]","[degrees]"]
Ylabels = ["Time","Time","$<Ux'>_{rotor}$ rotor averaged velocity","Asymmery Parameter","Rotor Thrust", "Rotor Torque",
            "Out-of-plane bending moment","Angle Out-of-plane bending moment"]

compare_total_correlations = False
compare_LP_correlations = False
compare_time_series = False
compare_FFT = False


def low_pass_filter2(signal, cutoff):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


#needs fixing fft.shift()
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


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r

#compare total signal correlations
if compare_total_correlations == True:
    for j in np.arange(2,len(Variables)-1,1):
        for i in np.arange(2,len(Variables)-1):

            fig,ax = plt.subplots(figsize=(14,8))
            Var = Variables[i]
            unit = units[i]
            Ylabel = Ylabels[i]
            corr_var = Variables[j]
            Y2_label = Ylabels[j]

            df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out.csv")

            time_OF = remove_nan("Time_OF")
            time_sample = remove_nan("Time_sample")
            time_sample[0] = time_OF[0]
            time_sample[-1] = time_OF[-1]

            dt = time_OF[1] - time_OF[0]

            correlation_variable = remove_nan(Var = corr_var)

            Theta = remove_nan(Var = "Theta")

            signal = remove_nan(Var)

            if corr_var[0:2] == "IA":
                f = interpolate.interp1d(time_sample,correlation_variable)
                correlation_variable = f(time_OF)
            
            if Var[0:2] == "IA":
                f = interpolate.interp1d(time_sample,signal)
                signal = f(time_OF)


            if Var == "MR":
                cutoff = 0.5*(12.1/60)
                signal_LP = low_pass_filter2(signal,cutoff)
            elif correlation_variable == "MR":
                cutoff = 0.5*(12.1/60)
                corr_signal_LP = low_pass_filter2(correlation_variable,cutoff)
            else:
                cutoff = 0.5*(12.1/60)*3
                #signal_LP = low_pass_filter(signal,cutoff,dt)
                signal_LP = low_pass_filter2(signal, cutoff)
                corr_signal_LP = low_pass_filter2(correlation_variable,cutoff)
                #LP_diff = signal_LP-signal_LP2

            signal_mean = np.mean(signal)
            corr_signal_mean = np.mean(correlation_variable)
            
            corr = correlation_coef(correlation_variable,signal)
            #ticks = np.arange(int(min(time_OF)), int(max(time_OF))+10,10)

            ax.plot(time_OF,signal,'-b')
            ax.plot(time_OF,signal_LP,"-r")
            #ax.set_xticks(ticks)
            ax.tick_params(axis='y', labelsize=12)
            ax.axhline(signal_mean,color="b",linestyle="--")
            ax.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=20)

            ax2=ax.twinx()
            ax2.plot(time_OF,correlation_variable,"-k")
            ax2.plot(time_OF,corr_signal_LP,"-y")
            #ax.set_xticks(ticks)
            ax2.tick_params(axis='y', labelsize=12)
            ax2.axhline(corr_signal_mean,color="k",linestyle="--")
            ax2.set_ylabel("{}".format(Y2_label),fontsize=20)
            #plt.title("Correlating {0} at {1}m from turbine, with {2}".format(Y2_label,offsets[2],Ylabel),fontsize=18)
            #ax.legend(["Total {}".format(Ylabel),"Low pass filtered {}".format(Ylabel), "Mean {}".format(Ylabel)],loc="upper left")
            #ax2.legend(["Total {0} Correlation = {1}".format(Y2_label,round(corr,2)),"Low pass filtered {}".format(Y2_label), "Mean {}".format(Y2_label)],loc="upper right")

            ax.set_xlabel("Time [s]",fontsize=16)
            ax.tick_params(axis='x', labelsize=12)
            plt.tight_layout()
            plt.savefig(dir+"corr_{0}_{1}_{2}.png".format(offsets[2],corr_var[0:2],Var))
            plt.close(fig)

df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out.csv")
Ux = remove_nan("Ux_0.0")
print(np.mean(Ux))

