import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
import pandas
import pyFAST.input_output as io

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

    Time_OF = np.array(a.variables["time_OF"])
    Time_sampling = np.array(a.variables["time_sampling"])
    Time_sampling = Time_sampling - Time_sampling[0]

    Time_start = 100
    #Time_end = Time_sampling[-1]
    Time_end = 250

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

    LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
    LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
    LSShftFR = np.sqrt( np.add(np.square(LSShftFys), np.square(LSShftFzs)) )

    df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()
    RtAeroFyh = np.array(df["RtAeroFyh_[N]"][Time_start_idx:Time_end_idx])
    RtAeroFzh = np.array(df["RtAeroFzh_[N]"][Time_start_idx:Time_end_idx])
    RtAeroFR = np.sqrt( np.add(np.square(RtAeroFyh), np.square(RtAeroFzh)) )

    xxx = np.add(np.square(RtAeroMyh/1000), np.square(RtAeroMzh/1000))
    yyy = np.add(np.square(LSSTipMys), np.square(LSSTipMzs))

    aaa = np.square(RtAeroMyh/1000); bbb = np.square(RtAeroMzh/1000)
    ccc = np.square(LSSTipMys); ddd = np.square(LSSTipMzs)


    L = 1.912

    C_LSSTipMys = LSSGagMys - L*LSShftFzs
    C_LSSTipMzs = LSSGagMzs + L*LSShftFys
    C_LSSTipMR = np.sqrt( np.add(np.square(C_LSSTipMys), np.square(C_LSSTipMzs)) )

    My_add = np.subtract(LSSTipMys, RtAeroMyh/1000)
    Mz_add = np.subtract(LSSTipMzs, RtAeroMzh/1000)
    MR_add = np.sqrt( np.add(np.square(My_add), np.square(Mz_add)) )

    MR_diff = np.subtract(LSSTipMR,RtAeroMR/1000)



    # group = a.groups["{}".format(offset)]
    # Ux = np.array(group.variables["Ux"])

    # #IA = np.array(group.variables["IA"])

    # f = interpolate.interp1d(Time_sampling,Ux)
    # Ux = f(Time_OF)

    #f = interpolate.interp1d(Time_sampling,IA)
    #IA = f(Time_OF)


    #plotting options
    plot_variables = False
    plot_FFT_OOPBM = True
    compare_total_OOPBM_correlations = False
    compare_FFT_OOPBM = False
    compare_OOPBM = False
    sys_LPF_OOPBM = False

    out_dir = in_dir + "OOPBM_lineplots/"


    #plot variables#
    if plot_variables == True:
        Variables = ["RtAeroFyh","RtAeroFzh"]
        units = ["[kN-m]","[kN-m]"]
        Ylabels = ["Rotor force y", "Rotor force z"]
        h_vars = [RtAeroFyh/1000, RtAeroFzh/1000]

        for i in np.arange(0,len(h_vars)):
            fig = plt.figure(figsize=(14,8))
            plt.plot(Time_OF,h_vars[i],"-b")
            plt.xlabel("Time [s]",fontsize=16)
            plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
            plt.tight_layout()
            plt.savefig(out_dir+"short_period_{0}".format(Variables[i]))
            plt.close()


    if plot_FFT_OOPBM == True:
        Variables = ["RtAeroFyh","RtAeroFzh"]
        units = ["[kN-m]","[kN-m]"]
        Ylabels = ["Rotor force y", "Rotor force z"]
        h_vars = [RtAeroFyh/1000, RtAeroFzh/1000]

        for i in np.arange(0,len(h_vars)):
            
            frq,FFT = temporal_spectra(h_vars[i],dt,Variables[i])

            fig = plt.figure(figsize=(14,8))
            plt.plot(frq,FFT)
            
            if Variables[i] != "MR_diff":
                frq_int = [1/60, 1/30, 12.1/60, (12.1/60)*3]
                frq_label = ["60s", "30s", "1P", "3P"]
                y_FFT = FFT[0]+1e+03

                for l in np.arange(0,len(frq_int)):
                    plt.axvline(frq_int[l])
                    plt.text(frq_int[l],y_FFT, frq_label[l])

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Frequency [Hz]",fontsize=14)
            plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=14)
            plt.tight_layout()
            plt.savefig(out_dir+"FFT_{0}.png".format(Variables[i]))
            plt.close(fig)



    #compare total signal correlations
    if compare_total_OOPBM_correlations == True:
        Variables = ["RtAeroMR^2","LSSTipMR^2"]
        units = ["$[kN-m]^2$","$[kN-m]^2$"]
        Ylabels = ["Rotor Aerodyn OOPBM squared","Rotor Elastodyn OOPBM squared"]
        h_vars = [xxx,yyy]

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


    if compare_FFT_OOPBM == True:
        Variables = ["RtAeroMR", "LSSTipMR"]
        units = ["[kN-m]","[kN-m]"]
        Ylabels = ["Rotor OOPBM", "Tip OOPBM"]
        h_vars = [RtAeroMR/1000, LSSTipMR]

        frq_i,FFT_i = temporal_spectra(h_vars[0],dt,Variables[0])
        frq_j,FFT_j = temporal_spectra(h_vars[1],dt,Variables[1])

        fig = plt.figure(figsize=(14,8))
        plt.plot(frq_i,FFT_i,"-b")
        plt.plot(frq_j,FFT_j,"-r")
        frq_int = [1/60, 1/30, 12.1/60, (12.1/60)*3]
        frq_label = ["60s", "30s", "1P", "3P"]
        y_FFT = FFT_i[0]+1e+03

        for l in np.arange(0,len(frq_int)):
            plt.axvline(frq_int[l])
            plt.text(frq_int[l],y_FFT, frq_label[l])

        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Frequency [Hz]",fontsize=14)
        plt.ylabel("{0} {1}".format(Ylabels[0],units[0]),fontsize=14)
        plt.legend([Variables[0],Variables[1]])
        plt.tight_layout()
        plt.savefig(out_dir+"FFT_{0}_{1}.png".format(Variables[0],Variables[1]))
        plt.close(fig)



    if compare_OOPBM == True:
        h_vars = [RtAeroMyh, LSSTipMys, RtAeroMzh, LSSTipMzs, RtAeroMR, LSSTipMR]
        units = ["[kN-m]", "[kN-m]", "[kN-m]","[kN-m]","[kN-m]","[kN-m]"]
        Ylabels = ["Rotor My", "Tip My", "Rotor Mz", "Tip Mz", "Rotor OOPBM","Tip OOPBM"]
        
        #comparing time series
        fig, axs = plt.subplots(6,2,figsize=(32,24))
        plt.rcParams.update({'font.size': 16})
        for i in np.arange(0,len(h_vars)):

            unit = units[i]
            Ylabel = Ylabels[i]

            signal = h_vars[i]

            axs = axs.ravel()

            axs[i].plot(Time_OF,signal)

            axs[i].set_title("{0} {1}".format(Ylabels[i],units[i]),fontsize=12)

        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(out_dir+"OOPBM_joint_vars_2.png")
        plt.close(fig)


    if sys_LPF_OOPBM == True:
        Variables = ["RtAeroFyh", "RtAeroFzh", "RtAeroMyh", "RtAeroMzh"]
        units = ["[kN-m]","[kN-m]", "[kN-m]", "[kN-m]"]
        Ylabels = ["Rotor force y", "Rotor force z", "Rotor moment y", "Rotor moment z"]
        h_vars = [RtAeroFyh/1000, RtAeroFzh/1000, RtAeroMyh/1000, RtAeroMzh/1000]

        Variables_2 = ["LSShftFys", "LSSshftFzs", "LSSTipMys", "LSSTipMzs"]
        units_2 = ["[kN-m]","[kN-m]", "[kN-m]", "[kN-m]"]
        Ylabels_2 = ["Tip force y", "Tip force z", "Tip moment y", "Tip moment z"]
        h_vars_2 = [LSShftFys, LSShftFzs, LSSTipMys, LSSTipMzs]

        cutoffs = [100, 10, round((12.1/60)*3,4), round(12.1/60,4)]

        for i in np.arange(0,len(h_vars)):

            for cutoff in cutoffs:

                signal_LP_1 = low_pass_filter(h_vars[i], cutoff)
                signal_LP_2 = low_pass_filter(h_vars_2[i],cutoff)
            
                fig, axs = plt.subplots(2,1,figsize=(14,8),sharex=True)
                
                corr = correlation_coef(signal_LP_1, signal_LP_2)
                corr = round(corr,2)

                axs = axs.ravel()

                axs[0].plot(Time_OF,signal_LP_1)
                axs[0].set_ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=14)
                axs[1].plot(Time_OF,signal_LP_2)
                axs[1].set_ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=14)


                plt.suptitle("Low pass filtered at {0}Hz. \nCorrelation: {1} with {2} = {3}".format(cutoff,Ylabels[i],Ylabels_2[i],corr),fontsize=16)
                plt.xlabel("Time [s]",fontsize=16)
                plt.tight_layout()
                plt.savefig(out_dir+"LPF_cutoff_{0}_corr_{1}_{2}.png".format(cutoff,Variables[i],Variables_2[i]))
                plt.close(fig)

    ic+=1



