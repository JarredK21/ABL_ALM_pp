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

in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"
out_dir = in_dir + "plots/"

offsets = [0.0, -63.0, -126.0]


Variables = ["Time_OF","Time_sample","RtVAvgxh","RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[s]","[s]", "[m/s]","[N]","[N-m]","[N-m]","[rads]"]
Ylabels = ["Time","Time","$<Ux'>_{blade}$ blade averaged velocity","Rotor Thrust", "Rotor Torque",
            "Out-of-plane bending moment","Angle Out-of-plane bending moment"]

for offset in offsets:
    txt = ["Ux_{0}".format(offset), "Uz_{0}".format(offset), "IA_{0}".format(offset)]
    unit = ["[m/s]", "[m/s]", "[$m^4/s$]"]
    Ylabel = ["$<Ux'>_{rotor}$ rotor averaged velocity at {0}m".format(offset), "Asymmetry parameter at {0}m".format(offset)]
    Variables.insert(len(Variables)-1,txt)
    units.insert(len(units)-1,unit)
    Ylabels.extend(len(Ylabels)-1,Ylabel)


compare_total_correlations = True
compare_LP_correlations = True
compare_time_series = True
compare_FFT = True



def low_pass_filter(signal, cutoff):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


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
    for j in np.arange(2,len(Variables)-5,1):
        for i in np.arange(2,len(Variables)-5,1):

            fig,ax = plt.subplots(figsize=(14,8))
            Var = Variables[i]
            unit = units[i]
            Ylabel = Ylabels[i]
            corr_var = Variables[j]
            Y2_label = Ylabels[j]

            df = pd.read_csv(in_dir+"out.csv")

            time_OF = remove_nan("Time_OF")
            time_sample = remove_nan("Time_sample")
            time_sample[0] = time_OF[0]
            time_sample[-1] = time_OF[-1]

            dt = time_OF[1] - time_OF[0]

            correlation_variable = remove_nan(Var = corr_var)

            Theta = remove_nan(Var = "Theta")

            signal = remove_nan(Var)

            if corr_var == "IA" or corr_var == "Ux":
                f = interpolate.interp1d(time_sample,correlation_variable)
                correlation_variable = f(time_OF)
            
            if Var == "IA" or Var == "Ux":
                f = interpolate.interp1d(time_sample,signal)
                signal = f(time_OF)


            if Var == "MR":
                cutoff = 0.5*(12.1/60)
                signal_LP = low_pass_filter(signal,cutoff)
            elif correlation_variable == "MR":
                cutoff = 0.5*(12.1/60)
                corr_signal_LP = low_pass_filter(correlation_variable,cutoff)
            else:
                cutoff = 0.5*(12.1/60)*3
                signal_LP = low_pass_filter(signal, cutoff)
                corr_signal_LP = low_pass_filter(correlation_variable,cutoff)


            signal_mean = np.mean(signal)
            corr_signal_mean = np.mean(correlation_variable)
            
            corr = correlation_coef(correlation_variable,signal)
            ticks = np.arange(int(min(time_OF)), int(max(time_OF))+10,10)

            ax.plot(time_OF,signal,'-b')
            ax.plot(time_OF,signal_LP,"-r")
            ax.set_xticks(ticks)
            ax.axhline(signal_mean,color="b",linestyle="--")
            ax.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)

            ax2=ax.twinx()
            ax2.plot(time_OF,correlation_variable,"-k")
            ax2.plot(time_OF,corr_signal_LP,"-y")
            ax.set_xticks(ticks)
            ax2.axhline(corr_signal_mean,color="k",linestyle="--")
            ax2.set_ylabel("{}".format(Y2_label),fontsize=16)
            plt.title("Correlating {0} with {1}".format(Y2_label,Ylabel),fontsize=18)
            ax.legend(["Total {}".format(Ylabel),"Low pass filtered {}".format(Ylabel), "Mean {}".format(Ylabel)],loc="upper left")
            ax2.legend(["Total {0} Correlation = {1}".format(Y2_label,round(corr,2)),"Low pass filtered {}".format(Y2_label), "Mean {}".format(Y2_label)],loc="upper right")

            ax.set_xlabel("Time [s]",fontsize=16)
            plt.tight_layout()
            plt.savefig(out_dir+"corr_{0}_{1}.png".format(corr_var,Var))
            plt.close(fig)



#compare LPF signal correlations
if compare_LP_correlations == True:
    for j in np.arange(2,len(Variables)-5,1):
        for i in np.arange(2,len(Variables)-5,1):

            fig,ax = plt.subplots(figsize=(14,8))
            Var = Variables[i]
            unit = units[i]
            Ylabel = Ylabels[i]
            corr_var = Variables[j]
            Y2_label = Ylabels[j]

            df = pd.read_csv(in_dir+"out.csv")

            time_OF = remove_nan("Time_OF")
            time_sample = remove_nan("Time_sample")
            time_sample[0] = time_OF[0]
            time_sample[-1] = time_OF[-1]

            dt = time_OF[1] - time_OF[0]

            correlation_variable = remove_nan(Var = corr_var)

            Theta = remove_nan(Var = "Theta")

            signal = remove_nan(Var)

            if corr_var == "IA" or corr_var == "Ux":
                f = interpolate.interp1d(time_sample,correlation_variable)
                correlation_variable = f(time_OF)
            
            if Var == "IA" or Var == "Ux":
                f = interpolate.interp1d(time_sample,signal)
                signal = f(time_OF)


            if Var == "MR":
                cutoff = 0.5*(12.1/60)
                signal_LP = low_pass_filter(signal,cutoff)
            elif correlation_variable == "MR":
                cutoff = 0.5*(12.1/60)
                corr_signal_LP = low_pass_filter(correlation_variable,cutoff)
            else:
                cutoff = 0.5*(12.1/60)*3
                signal_LP = low_pass_filter(signal, cutoff)
                corr_signal_LP = low_pass_filter(correlation_variable,cutoff)
            
            corr_LP = correlation_coef(corr_signal_LP,signal_LP)

            signal_mean = np.mean(signal)
            corr_signal_mean = np.mean(correlation_variable)

            ticks = np.arange(int(min(time_OF)), int(max(time_OF))+10,10)

            ax.plot(time_OF,signal_LP,"-b")
            ax.set_xticks(ticks)
            ax.set_ylabel("Low pass filtered {0} {1}".format(Ylabel,unit),fontsize=16)

            ax2=ax.twinx()
            ax2.plot(time_OF,corr_signal_LP,"-r")
            ax.set_xticks(ticks)
            ax2.set_ylabel("Low pass filtered {}".format(Y2_label),fontsize=16)
            plt.title("Correlating {0} with {1}".format(Y2_label,Ylabel),fontsize=18)
            ax.legend(["Low pass filtered {}".format(Ylabel)],loc="upper left")
            ax2.legend(["Low pass filtered {0} Correlation = {1}".format(Y2_label,round(corr_LP,2))],loc="upper right")

            ax.set_xlabel("Time [s]",fontsize=16)
            plt.tight_layout()
            plt.savefig(out_dir+"LPF_corr_{0}_{1}.png".format(corr_var,Var))
            plt.close(fig)



if compare_time_series == True:
    #comparing time series
    fig, axs = plt.subplots(6,1,figsize=(32,24))
    plt.rcParams.update({'font.size': 16})
    for i in np.arange(2,len(Variables)-5):

        Var = Variables[i]
        unit = units[i]
        Ylabel = Ylabels[i]

        df = pd.read_csv(in_dir+"out.csv")

        time_OF = remove_nan("Time_OF")
        time_sample = remove_nan("Time_sample")
        time_sample[0] = time_OF[0]
        time_sample[-1] = time_OF[-1]

        dt = time_OF[1] - time_OF[0]

        signal = remove_nan(Var)

        if Var == "IA" or Var == "Ux":
            f = interpolate.interp1d(time_sample, signal)
            signal = f(time_OF)
        elif Var == "Theta":
            signal = np.multiply((180/np.pi),signal)

        ticks = np.arange(int(min(time_OF)), int(max(time_OF))+10,10)

        axs = axs.ravel()

        j=i-2

        axs[j].plot(time_OF,signal)
        axs[j].set_xticks(ticks)
        axs[j].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=18)

    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"joint_vars.png")
    plt.close(fig)


#comparing spectra
if compare_FFT == True:
    fig, axs = plt.subplots(3,2,figsize=(32,24))
    plt.rcParams.update({'font.size': 16})
    for i in np.arange(2,len(Variables)):

        Var = Variables[i]
        unit = units[i]
        Ylabel = Ylabels[i]

        df = pd.read_csv(in_dir+"out.csv")

        time_OF = remove_nan("Time_OF")
        time_sample = remove_nan("Time_sample")
        time_sample[0] = time_OF[0]
        time_sample[-1] = time_OF[-1]

        dt = time_OF[1] - time_OF[0]

        signal = remove_nan(Var)

        if Var == "IA" or Var == "Ux":
            f = interpolate.interp1d(time_sample, signal)
            signal = f(time_OF)
        elif Var == "Theta":
            signal = np.multiply((180/np.pi),signal)

        
        frq, FFT_signal = temporal_spectra(signal,dt,Var)
        
        axs = axs.ravel()

        j=i-2

        axs[j].plot(frq,FFT_signal)
        axs[j].set_yscale('log')
        axs[j].set_xscale('log')
        axs[j].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=18)

        frq_int = [1/60, 1/30, 12.1/60, (12.1/60)*3]
        frq_label = ["60s", "30s", "1P", "3P"]
        if Var == "Theta":
            y_FFT = 1e-04
        else:
            y_FFT = FFT_signal[0]+1e+03
        for l in np.arange(0,len(frq_int)):
            axs[j].axvline(frq_int[l])
            axs[j].text(frq_int[l],y_FFT, frq_label[l])


    fig.supxlabel("Frequency [Hz]")
    plt.tight_layout()
    plt.savefig(out_dir+"joint_vars_FFT.png")
    plt.close(fig)