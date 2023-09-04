import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset

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

offsets = [0.0]

a = Dataset(in_dir+"Dataset.nc")

ic = 2
for offset in offsets:

    Time_start = 155

    Time_OF = np.array(a.variables["time_OF"])
    Time_sampling = np.array(a.variables["time_sampling"])
    Time_sampling[0] = Time_OF[0]
    Time_sampling[-1] = Time_OF[-1]
    dt = Time_OF[1] - Time_OF[0]

    Time_start_idx = np.searchsorted(Time_OF,Time_start)

    RtAeroFxh = np.array(a.variables["RtAeroFxh"][Time_start_idx:])
    RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:])
    MR = np.array(a.variables["RtAeroMrh"][Time_start_idx:])
    Theta = np.array(a.variables["Theta"][Time_start_idx:])
    RtAeroVxh = np.array(a.variables["RtAeroVxh"][Time_start_idx:])
    LSShftMys = np.array(a.variables["LSShftMys"][Time_start_idx:])
    LSShftMzs = np.array(a.variables["LSShftMzs"][Time_start_idx:])
    LSSMR = np.sqrt( np.add(np.square(LSShftMys), np.square(LSShftMzs)) ) 


    group = a.groups["{}".format(offset)]

    Ux = np.array(group.variables["Ux"])
    IA = np.array(group.variables["IA"])

    f = interpolate.interp1d(Time_sampling,Ux)
    Ux = f(Time_OF)
    Ux = Ux[Time_start_idx:]

    f = interpolate.interp1d(Time_sampling,IA)
    IA = f(Time_OF)
    IA = IA[Time_start_idx:]

    Time_OF = Time_OF[Time_start_idx:]


    #plotting options
    plot_variabes = True
    compare_total_correlations = True
    compare_LP_correlations = True
    compare_time_series = True
    compare_FFT = True

    out_dir = in_dir + "lineplots{}/".format(ic)


    #plot variables#
    if plot_variabes == True:
        Variables = ["RtAeroVxh","RtAeroFxh","RtAeroMxh","MR","Ux","IA","LSSMR"]
        units = ["[m/s]","[N]","[N-m]","[N-m]","[m/s]", "[$m^4/s$]","[kN-m]"]
        Ylabels = ["$<Ux'>_{blade}$ rotor averaged horizontal velocity","Rotor Thrust", "Rotor Torque",
                   "Rotor Out-of-plane bending moment","$<Ux'>_{rotor}$ rotor averaged horizontal velocity",
                   "Asymmetry parameter", "Low-speed shaft Out-of-plane bending moment"]
        h_vars = [RtAeroVxh, RtAeroFxh, RtAeroMxh, MR, Ux, IA, LSSMR]

        for i in np.arange(0,len(h_vars)):
            cutoff = 0.5*(12.1/60)*3
            signal_LP = low_pass_filter(h_vars[i], cutoff)
            fig = plt.figure(figsize=(14,8))
            plt.plot(Time_OF,h_vars[i],"-b")
            plt.plot(Time_OF,signal_LP,"-r")
            plt.axhline(np.mean(h_vars[i]),"--k")
            plt.xlabel("Time [s]")
            plt.ylabel("{0}".format(Ylabels[i]))
            plt.legend(["total signal", "Low pass filtered signal","mean signal"])
            plt.tight_layout()
            plt.savefig(out_dir+"{0}".format(Variables[i]))
            plt.close()


    #compare total signal correlations
    if compare_total_correlations == True:
        Variables = ["RtAeroVxh","RtAeroFxh","RtAeroMxh","MR","Ux","IA","LSSMR"]
        units = ["[m/s]","[N]","[N-m]","[N-m]","[m/s]", "[$m^4/s$]","[kN-m]"]
        Ylabels = ["$<Ux'>_{blade}$ rotor averaged horizontal velocity","Rotor Thrust", "Rotor Torque",
                   "Rotor Out-of-plane bending moment","$<Ux'>_{rotor}$ rotor averaged horizontal velocity",
                   "Asymmetry parameter", "Low-speed shaft Out-of-plane bending moment"]
        h_vars = [RtAeroVxh, RtAeroFxh, RtAeroMxh, MR, Ux, IA, LSSMR]

        for j in np.arange(0,len(h_vars)):
            for i in np.arange(0,len(h_vars)):

                fig,ax = plt.subplots(figsize=(14,8))
                
                corr = correlation_coef(h_vars[j],h_vars[i])

                ax.plot(Time_OF,h_vars[j],'-b')
                ax.set_xticks(ticks)
                ax.set_ylabel("{0} {1}".format(Ylabels[j],units[j]),fontsize=16)

                ax2=ax.twinx()
                ax2.plot(Time_OF,h_vars[i],"-r")
                ax2.set_ylabel("{}".format(Ylabels[i]),fontsize=16)

                plt.title("Correlation: {0} with {1} = {2}".format(Ylabels[j],Ylabels[i],corr),fontsize=18)
                ax.set_xlabel("Time [s]",fontsize=16)
                plt.tight_layout()
                plt.savefig(out_dir+"corr_{0}_{1}.png".format(Variables[j],Variables[i]))
                plt.close(fig)



    #compare LPF signal correlations
    if compare_LP_correlations == True:
        Variables = ["RtAeroVxh","RtAeroFxh","RtAeroMxh","MR","Ux","IA","LSSMR"]
        units = ["[m/s]","[N]","[N-m]","[N-m]","[m/s]", "[$m^4/s$]","[kN-m]"]
        Ylabels = ["$<Ux'>_{blade}$ rotor averaged horizontal velocity","Rotor Thrust", "Rotor Torque",
                "Rotor Out-of-plane bending moment","$<Ux'>_{rotor}$ rotor averaged horizontal velocity",
                "Asymmetry parameter", "Low-speed shaft Out-of-plane bending moment"]
        h_vars = [RtAeroVxh, RtAeroFxh, RtAeroMxh, MR, Ux, IA, LSSMR]

        for j in np.arange(0,len(h_vars),1):
            for i in np.arange(0,len(h_vars),1):
                
                fig,ax = plt.subplots(figsize=(14,8))

                cutoff = 0.5*(12.1/60)*3
                signal_LP_j = low_pass_filter(h_vars[j], cutoff)
                signal_LP_i = low_pass_filter(h_vars[i],cutoff)
                
                corr_LP = correlation_coef(signal_LP_j,signal_LP_i)

                ax.plot(Time_OF,signal_LP_j,"-b")
                ax.set_ylabel("Low pass filtered {0} {1}".format(Ylabels[j],units[j]),fontsize=16)

                ax2=ax.twinx()
                ax2.plot(Time_OF,signal_LP_i,"-r")
                ax2.set_ylabel("Low pass filtered {0} {1}".format(Ylabels[i], units[i]),fontsize=16)

                plt.title("Correlating {0} with {1}".format(Ylabels[j],Ylabel),fontsize=18)
                plt.title("Correlation: {0} with {1} = {2}".format(Ylabels[j],Ylabels[i],corr),fontsize=18)

                ax.set_xlabel("Time [s]",fontsize=16)
                plt.tight_layout()
                plt.savefig(out_dir+"LPF_corr_{0}_{1}.png".format(Variables[j],Variables[i]))
                plt.close(fig)



    if compare_time_series == True:
        Variables = ["Ux","RtAeroFxh","RtAeroMxh","MR","LSSMR""IA"]
        h_vars = [Ux,RtAeroFxh, RtAeroMxh,MR,LSSMR,IA]
        units = ["[m/s]","[N]","[N-m]","[N-m]","[kN-m]","[$m^4/s$]"]
        Ylabels = ["$<Ux'>_{Rotor}$ rotor averaged horizontal velocity","Rotor Thrust", "Rotor Torque",
                    "Rotor Out-of-plane bending moment","LSS Out-of-plane bending moment","Asymmetry parameter"]
        
        #comparing time series
        fig, axs = plt.subplots(6,1,figsize=(32,24))
        plt.rcParams.update({'font.size': 16})
        for i in np.arange(0,len(h_vars)):

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
        Variables = ["Ux","RtAeroFxh","RtAeroMxh","MR","LSSMR","IA"]
        h_vars = [Ux,RtAeroFxh,RtAeroMxh,MR,LSSMR,IA]
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

    ic+=1