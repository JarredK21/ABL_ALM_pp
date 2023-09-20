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


in_dir = "../../NREL_5MW_MCBL_R_CRPM_2/post_processing/"

offsets = [0.0]

a = Dataset(in_dir+"Dataset.nc")

ic = 2
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

    LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
    LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
    LSShftFR = np.sqrt( np.add(np.square(LSShftFys), np.square(LSShftFzs)) )

    df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()
    RtAeroFyh = np.array(df["RtAeroFyh_[N]"][Time_start_idx:Time_end_idx])
    RtAeroFzh = np.array(df["RtAeroFzh_[N]"][Time_start_idx:Time_end_idx])
    RtAeroFR = np.sqrt( np.add(np.square(RtAeroFyh), np.square(RtAeroFzh)) )


    L = 1.912

    C_LSSTipMys = LSSGagMys - L*LSShftFzs
    C_LSSTipMzs = LSSGagMzs + L*LSShftFys
    C_LSSTipMR = np.sqrt( np.add(np.square(C_LSSTipMys), np.square(C_LSSTipMzs)) )

    My_add = np.subtract(LSSTipMys, RtAeroMyh/1000)
    Mz_add = np.subtract(LSSTipMzs, RtAeroMzh/1000)
    MR_add = np.sqrt( np.add(np.square(My_add), np.square(Mz_add)) )



    group = a.groups["{}".format(offset)]
    Ux = np.array(group.variables["Ux"])

    #IA = np.array(group.variables["IA"])

    f = interpolate.interp1d(Time_sampling,Ux)
    Ux = f(Time_OF)

    #f = interpolate.interp1d(Time_sampling,IA)
    #IA = f(Time_OF)


    #plotting options
    plot_variables = True
    compare_total_OOPBM_correlations = False
    compare_FFT_OOPBM = True

    out_dir = in_dir + "OOPBM_lineplots/"


    #plot variables#
    if plot_variables == True:
        Variables = ["Mz_add"]
        units = ["[kN-m]"]
        Ylabels = ["Rotor Elastodyn additional Moment z direction"]
        h_vars = [Mz_add]

        for i in np.arange(0,len(h_vars)):
            fig = plt.figure(figsize=(14,8))
            plt.plot(Time_OF,h_vars[i],"-b")
            plt.xlabel("Time [s]",fontsize=16)
            plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
            plt.tight_layout()
            plt.savefig(out_dir+"{0}".format(Variables[i]))
            plt.close()



    #compare total signal correlations
    if compare_total_OOPBM_correlations == True:
        Variables = ["Mz_add"]
        units = ["[kN-m]"]
        Ylabels = ["Rotor Elastodyn additional Moment z direction"]
        h_vars = [Mz_add]

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
        Variables = ["Mz_add"]
        units = ["[kN-m]"]
        Ylabels = ["Rotor Elastodyn additional Moment z direction"]
        h_vars = [Mz_add]

        for j in np.arange(0,len(h_vars)):
            for i in np.arange(0,len(h_vars)):

                frq_i,FFT_i = temporal_spectra(h_vars[i],dt,Variables[i])
                frq_j,FFT_j = temporal_spectra(h_vars[j],dt,Variables[j])

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
                plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=14)
                plt.legend([Variables[i],Variables[j]])
                plt.tight_layout()
                plt.savefig(out_dir+"FFT_{0}_{1}.png".format(Variables[i],Variables[j]))
                plt.close(fig)

    ic+=1



