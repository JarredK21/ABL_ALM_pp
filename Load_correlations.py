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

offsets = [0.0, 63.0]

a = Dataset(in_dir+"Dataset.nc")

ic = 1
for offset in offsets:

    Time_start = 90

    Time_OF = np.array(a.variables["time_OF"])
    Time_sampling = np.array(a.variables["time_sampling"])
    Time_sampling[0] = Time_OF[0]
    Time_sampling[-1] = Time_OF[-1]
    dt = Time_OF[1] - Time_OF[0]

    Time_start_idx = np.searchsorted(Time_OF,Time_start)

    #RtAeroVxh = np.array(a.variables["RtAeroVxh"][Time_start_idx:])
    RtAeroFxh = np.array(a.variables["RtAeroFxh"][Time_start_idx:])
    RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:])
    MR = np.array(a.variables["RtAeroMrh"][Time_start_idx:])
    Theta = np.array(a.variables["Theta"][Time_start_idx:])


    group = a.groups["{}".format(offset)]

    Ux = np.array(group.variables["Ux"])
    Uz = np.array(group.variables["Uz"])
    IA = np.array(group.variables["IA"])
    Uh = np.array(group.variables["HV"])

    f = interpolate.interp1d(Time_sampling,Ux)
    Ux = f(Time_OF)
    Ux = Ux[Time_start_idx:]

    f = interpolate.interp1d(Time_sampling,Uz)
    Uz = f(Time_OF)
    Uz = Uz[Time_start_idx:]

    f = interpolate.interp1d(Time_sampling,IA)
    IA = f(Time_OF)
    IA = IA[Time_start_idx:]

    f = interpolate.interp1d(Time_sampling,Uh)
    Uh = f(Time_OF)
    Uh = Uh[Time_start_idx:]

    Time_OF = Time_OF[Time_start_idx:]

    h_vars = [RtAeroFxh, RtAeroMxh, MR, Ux, Uz, IA, Uh]

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,Ux)
    plt.xlabel("Time [s]")
    plt.ylabel("$<Ux'>_{rotor}$ rotor averaged horizontal velocity")
    plt.savefig(in_dir+"velocity_comp.png")


    Variables = ["RtAeroFxh","RtAeroMxh","MR","Ux","Uz","IA", "Uh"]
    units = ["[N]","[N-m]","[N-m]","[m/s]", "[m/s]", "[$m^4/s$]", "[m/s]"]
    Ylabels = ["Rotor Thrust", "Rotor Torque","Out-of-plane bending moment",
               "$<Ux'>_{rotor}$ rotor averaged horizontal velocity", "$<Uz'>_{rotor}$ rotor averaged vertical velocity", 
               "Asymmetry parameter", "Local hub height velocity"]


    #plotting options
    compare_total_correlations = False
    compare_LP_correlations = False
    compare_time_series = False
    compare_FFT = False
    velocity_comp = True

    out_dir = in_dir + "plots{}/".format(ic)
    os.makedirs(out_dir)

    def low_pass_filter(signal, cutoff):  

        fs = 1/dt     # sample rate, Hz      
        nyq = 0.5 * fs  # Nyquist Frequency      
        order = 3  # sin wave can be approx represented as 3rd order polynomial

        normal_cutoff = cutoff / nyq
        # Get the filter coefficients 
        b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

        low_pass_signal = filtfilt(b, a, signal)

        return low_pass_signal


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
        for j in np.arange(0,len(h_vars),1):
            for i in np.arange(0,len(h_vars),1):

                fig,ax = plt.subplots(figsize=(14,8))
                Var = Variables[i]
                unit = units[i]
                Ylabel = Ylabels[i]
                corr_var = Variables[j]
                Y2_label = Ylabels[j]
                
                correlation_variable = h_vars[j]
                signal = h_vars[i]


                if Var == "MR":
                    cutoff = 0.5*(12.1/60)
                    signal_LP = low_pass_filter(signal,cutoff)
                elif corr_var == "MR":
                    cutoff = 0.5*(12.1/60)
                    corr_signal_LP = low_pass_filter(correlation_variable,cutoff)
                else:
                    cutoff = 0.5*(12.1/60)*3
                    signal_LP = low_pass_filter(signal, cutoff)
                    corr_signal_LP = low_pass_filter(correlation_variable,cutoff)


                signal_mean = np.mean(signal)
                corr_signal_mean = np.mean(correlation_variable)
                
                corr = correlation_coef(correlation_variable,signal)
                ticks = np.arange(int(min(Time_OF)), int(max(Time_OF))+10,10)

                ax.plot(Time_OF,signal,'-b')
                ax.plot(Time_OF,signal_LP,"-r")
                ax.set_xticks(ticks)
                ax.axhline(signal_mean,color="b",linestyle="--")
                ax.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)

                ax2=ax.twinx()
                ax2.plot(Time_OF,correlation_variable,"-k")
                ax2.plot(Time_OF,corr_signal_LP,"-y")
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
        for j in np.arange(0,len(h_vars),1):
            for i in np.arange(0,len(h_vars),1):

                fig,ax = plt.subplots(figsize=(14,8))
                Var = Variables[i]
                unit = units[i]
                Ylabel = Ylabels[i]
                corr_var = Variables[j]
                Y2_label = Ylabels[j]


                correlation_variable = h_vars[j]
                signal = h_vars[i]

                if Var == "MR":
                    cutoff = 0.5*(12.1/60)
                    signal_LP = low_pass_filter(signal,cutoff)
                elif corr_var == "MR":
                    cutoff = 0.5*(12.1/60)
                    corr_signal_LP = low_pass_filter(correlation_variable,cutoff)
                else:
                    cutoff = 0.5*(12.1/60)*3
                    signal_LP = low_pass_filter(signal, cutoff)
                    corr_signal_LP = low_pass_filter(correlation_variable,cutoff)
                
                corr_LP = correlation_coef(corr_signal_LP,signal_LP)

                signal_mean = np.mean(signal)
                corr_signal_mean = np.mean(correlation_variable)

                ticks = np.arange(int(min(Time_OF)), int(max(Time_OF))+10,10)

                ax.plot(Time_OF,signal_LP,"-b")
                ax.set_xticks(ticks)
                ax.set_ylabel("Low pass filtered {0} {1}".format(Ylabel,unit),fontsize=16)

                ax2=ax.twinx()
                ax2.plot(Time_OF,corr_signal_LP,"-r")
                ax.set_xticks(ticks)
                ax2.set_ylabel("Low pass filtered {}".format(Y2_label),fontsize=16)
                plt.title("Correlating {0} with {1}".format(Y2_label,Ylabel),fontsize=18)
                ax.legend(["Low pass filtered {}".format(Ylabel)],loc="upper left")
                ax2.legend(["Low pass filtered {0} Correlation = {1}".format(Y2_label,round(corr_LP,2))],loc="upper right")

                ax.set_xlabel("Time [s]",fontsize=16)
                plt.tight_layout()
                plt.savefig(out_dir+"LPF_corr_{0}_{1}.png".format(corr_var,Var))
                plt.close(fig)

    h_vars = [Ux,RtAeroFxh, RtAeroMxh,IA,MR,Theta]
    units = ["[m/s]","[N]","[N-m]","[$m^4/s$]","[N-m]","[rads]"]
    Ylabels = ["$<Ux'>_{Rotor}$ rotor averaged horizontal velocity","Rotor Thrust", "Rotor Torque", "Asymmetry Parameter",
                "Out-of-plane bending moment","Angle Out-of-plane bending moment"]


    if compare_time_series == True:
        #comparing time series
        fig, axs = plt.subplots(6,1,figsize=(32,24))
        plt.rcParams.update({'font.size': 16})
        for i in np.arange(0,len(h_vars)):

            Var = Variables[i]
            unit = units[i]
            Ylabel = Ylabels[i]

            signal = h_vars[i]

            ticks = np.arange(int(min(Time_OF)), int(max(Time_OF))+10,10)

            axs = axs.ravel()

            axs[i].plot(Time_OF,signal)
            if i == 2:
                axs[i].set_ylim([3e+06,8e+06])
            elif i == 1:
                axs[i].set_ylim([6e+05,10e+05])
            axs[i].set_xticks(ticks)
            axs[i].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=18)

        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(out_dir+"joint_vars.png")
        plt.close(fig)


    #comparing spectra
    if compare_FFT == True:
        fig, axs = plt.subplots(3,2,figsize=(32,24))
        plt.rcParams.update({'font.size': 16})
        for i in np.arange(0,len(h_vars)):

            Var = Variables[i]
            unit = units[i]
            Ylabel = Ylabels[i]

            signal = h_vars[i]
            
            frq, FFT_signal = temporal_spectra(signal,dt,Var)
            
            axs = axs.ravel()

            axs[i].plot(frq,FFT_signal)
            axs[i].set_yscale('log')
            axs[i].set_xscale('log')
            axs[i].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=18)

            frq_int = [1/60, 1/30, 12.1/60, (12.1/60)*3]
            frq_label = ["60s", "30s", "1P", "3P"]
            if Var == "Theta":
                y_FFT = 1e-04
            else:
                y_FFT = FFT_signal[0]+1e+03
            for l in np.arange(0,len(frq_int)):
                axs[i].axvline(frq_int[l])
                axs[i].text(frq_int[l],y_FFT, frq_label[l])


        fig.supxlabel("Frequency [Hz]")
        plt.tight_layout()
        plt.savefig(out_dir+"joint_vars_FFT.png")
        plt.close(fig)

    ic+=1