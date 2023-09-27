import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
import pandas

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


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


in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

offsets = [0.0,63.0]

a = Dataset(in_dir+"Dataset.nc")

for offset in offsets:

    Time_OF = np.array(a.variables["time_OF"])
    Time_sampling = np.array(a.variables["time_sampling"])
    Time_sampling = Time_sampling - Time_sampling[0]

    Time_start = 100
    Time_end = Time_sampling[-1]

    dt = Time_OF[1] - Time_OF[0]

    Time_start_idx = np.searchsorted(Time_OF,Time_start)
    Time_end_idx = np.searchsorted(Time_OF,Time_end)

    Time_OF = Time_OF[Time_start_idx:Time_end_idx]

    RtAeroFxh = np.array(a.variables["RtAeroFxh"][Time_start_idx:Time_end_idx])
    RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])
    RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
    RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])
    RtAeroMR = np.sqrt( np.add(np.square(RtAeroMyh), np.square(RtAeroMzh)) ) 
    Theta = np.array(a.variables["Theta"][Time_start_idx:Time_end_idx])

    LSSGagMys = np.array(a.variables["LSSGagMys"][Time_start_idx:Time_end_idx])
    LSSGagMzs = np.array(a.variables["LSSGagMzs"][Time_start_idx:Time_end_idx])
    LSSGagMR = np.sqrt( np.add(np.square(LSSGagMys), np.square(LSSGagMzs)) )

    LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
    LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
    LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )


    group = a.groups["{}".format(offset)]
    Ux = np.array(group.variables["Ux"])

    IA = np.array(group.variables["IA"])

    f = interpolate.interp1d(Time_sampling,Ux)
    Ux = f(Time_OF)

    f = interpolate.interp1d(Time_sampling,IA)
    IA = f(Time_OF)


    #plotting options
    plot_variables = False
    compare_total_correlations = False
    compare_LP_correlations = False
    compare_time_series = True
    compare_FFT = False

    out_dir = in_dir + "lineplots_{}/".format(offset)


    #plot variables#
    if plot_variables == True:
        Variables = ["Ux", "IA", "RtAeroFxh","RtAeroMxh","RtAeroMR","LSSTipMR","LSSGagMR"]
        units = ["[m/s]","[$m^4/s$]","[N]","[N-m]","[N-m]","[kN-m]","[kN-m]"]
        Ylabels = ["$<Ux'>_{Rotor}$ rotor averaged horizontal velocity","Asymmetry parameter","Rotor Thrust", "Rotor Torque", 
                   "Rotor Aerodyn Out-of-plane bending moment", "Rotor Elastodyn Rotor Out-of-plane bending moment",
                   "LSS Elastodyn Out-of-plane bending moment"]
        h_vars = [Ux, IA, RtAeroFxh, RtAeroMxh, RtAeroMR, LSSTipMR,LSSGagMR]

        for i in np.arange(0,len(h_vars)):
            cutoff = 0.5*(12.1/60)*3
            signal_LP = low_pass_filter(h_vars[i], cutoff)
            fig = plt.figure(figsize=(14,8))
            plt.plot(Time_OF,h_vars[i],"-b")
            plt.plot(Time_OF,signal_LP,"-r")
            plt.axhline(np.mean(h_vars[i]),linestyle="--",color="k")
            plt.xlabel("Time [s]",fontsize=16)
            plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
            plt.legend(["total signal", "Low pass filtered signal","mean signal"])
            plt.tight_layout()
            plt.savefig(out_dir+"{0}".format(Variables[i]))
            plt.close()


    #compare total signal correlations
    if compare_total_correlations == True:
        Variables = ["Ux", "IA", "RtAeroFxh","RtAeroMxh","RtAeroMR","LSSTipMR","LSSGagMR"]
        units = ["[m/s]","[$m^4/s$]","[N]","[N-m]","[N-m]","[kN-m]","[kN-m]"]
        Ylabels = ["$<Ux'>_{Rotor}$ rotor averaged horizontal velocity","Asymmetry parameter","Rotor Thrust", "Rotor Torque", 
                   "Rotor Aerodyn Out-of-plane bending moment", "Rotor Elastodyn Rotor Out-of-plane bending moment",
                   "LSS Elastodyn Out-of-plane bending moment"]
        h_vars = [Ux, IA, RtAeroFxh, RtAeroMxh, RtAeroMR, LSSTipMR,LSSGagMR]

        for j in np.arange(0,len(h_vars)):
            for i in np.arange(0,len(h_vars)):

                fig,ax = plt.subplots(figsize=(14,8))
                
                corr = correlation_coef(h_vars[j],h_vars[i])
                corr = round(corr,2)

                ax.plot(Time_OF,h_vars[j],'-b')
                ax.set_ylabel("{0} {1}".format(Ylabels[j],units[j]),fontsize=14)

                ax2=ax.twinx()
                ax2.plot(Time_OF,h_vars[i],"-r")
                ax2.set_ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=14)

                plt.title("Correlation: {0} with {1} = {2}".format(Ylabels[j],Ylabels[i],corr),fontsize=16)
                ax.set_xlabel("Time [s]",fontsize=16)
                plt.tight_layout()
                plt.savefig(out_dir+"corr_{0}_{1}.png".format(Variables[j],Variables[i]))
                plt.close(fig)



    #compare LPF signal correlations
    if compare_LP_correlations == True:
        Variables = ["Ux", "IA", "RtAeroFxh","RtAeroMxh","RtAeroMR","LSSTipMR","LSSGagMR"]
        units = ["[m/s]","[$m^4/s$]","[N]","[N-m]","[N-m]","[kN-m]","[kN-m]"]
        Ylabels = ["$<Ux'>_{Rotor}$ rotor averaged horizontal velocity","Asymmetry parameter","Rotor Thrust", "Rotor Torque", 
                   "Rotor Aerodyn Out-of-plane bending moment", "Rotor Elastodyn Rotor Out-of-plane bending moment",
                   "LSS Elastodyn Out-of-plane bending moment"]
        h_vars = [Ux, IA, RtAeroFxh, RtAeroMxh, RtAeroMR, LSSTipMR,LSSGagMR]

        for j in np.arange(0,len(h_vars),1):
            for i in np.arange(0,len(h_vars),1):
                
                fig,ax = plt.subplots(figsize=(14,8))

                cutoff = 0.5*(12.1/60)*3
                signal_LP_j = low_pass_filter(h_vars[j], cutoff)
                signal_LP_i = low_pass_filter(h_vars[i],cutoff)
                
                corr_LP = correlation_coef(signal_LP_j,signal_LP_i)
                corr_LP = round(corr_LP,2)

                ax.plot(Time_OF,signal_LP_j,"-b")
                ax.set_ylabel("Low pass filtered {0} {1}".format(Ylabels[j],units[j]),fontsize=14)

                ax2=ax.twinx()
                ax2.plot(Time_OF,signal_LP_i,"-r")
                ax2.set_ylabel("Low pass filtered {0} {1}".format(Ylabels[i], units[i]),fontsize=14)

                plt.title("Correlation: {0} with {1} = {2}".format(Ylabels[j],Ylabels[i],corr_LP),fontsize=16)

                ax.set_xlabel("Time [s]",fontsize=16)
                plt.tight_layout()
                plt.savefig(out_dir+"LPF_corr_{0}_{1}.png".format(Variables[j],Variables[i]))
                plt.close(fig)



    if compare_time_series == True:
        Variables = ["Ux","RtAeroFxh","RtAeroMxh","LSSTipMR","IA","Theta"]
        h_vars = [Ux,RtAeroFxh, RtAeroMxh,LSSTipMR,IA,Theta]
        units = ["[m/s]","[N]","[N-m]","[kN-m]","[$m^4/s$]","[rads]"]
        Ylabels = ["$<Ux'>_{Rotor}$ rotor averaged horizontal velocity","Rotor Thrust", "Rotor Torque",
                    "Tip Out-of-plane bending moment","Asymmetry parameter","Angle OOPBM"]
        
        #comparing time series
        fig, axs = plt.subplots(6,1,figsize=(32,24))
        plt.rcParams.update({'font.size': 16})
        for i in np.arange(0,len(h_vars)):

            unit = units[i]
            Ylabel = Ylabels[i]

            signal = h_vars[i]

            axs = axs.ravel()

            axs[i].plot(Time_OF,signal)

            axs[i].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=18)

        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(out_dir+"joint_vars.png")
        plt.close(fig)


    #comparing spectra
    if compare_FFT == True:
        Variables = ["Ux","RtAeroFxh","RtAeroMxh","RtAeroMR","LSSMR","IA"]
        h_vars = [Ux,RtAeroFxh, RtAeroMxh,RtAeroMR,LSSGagMR,IA]
        units = ["[m/s]","[N]","[N-m]","[N-m]","[kN-m]","[$m^4/s$]"]
        Ylabels = ["$<Ux'>_{Rotor}$ rotor averaged horizontal velocity","Rotor Thrust", "Rotor Torque",
                    "Rotor Out-of-plane bending moment","LSS Out-of-plane bending moment","Asymmetry parameter"]
        
        fig, axs = plt.subplots(3,2,figsize=(32,24))
        plt.rcParams.update({'font.size': 16})
        for i in np.arange(0,len(h_vars)):
            
            frq, FFT_signal = temporal_spectra(h_vars[i],dt,Variables[i])
            
            axs = axs.ravel()

            axs[i].plot(frq,FFT_signal)
            axs[i].set_yscale('log')
            axs[i].set_xscale('log')
            axs[i].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=18)

            frq_int = [1/60, 1/30, 12.1/60, (12.1/60)*3]
            frq_label = ["60s", "30s", "1P", "3P"]
            y_FFT = FFT_signal[0]+1e+03

            for l in np.arange(0,len(frq_int)):
                axs[i].axvline(frq_int[l])
                axs[i].text(frq_int[l],y_FFT, frq_label[l])


        fig.supxlabel("Frequency [Hz]")
        plt.tight_layout()
        plt.savefig(out_dir+"joint_vars_FFT.png")
        plt.close(fig)

