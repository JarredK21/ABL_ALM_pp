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


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    N = len(y)
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    dx = X[1]-X[0]
    P = []
    p = 0
    mu_3 = 0
    mu_4 = 0
    i = 0
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
        mu_3+=((y[i]-mu)**3)
        mu_4+=((y[i]-mu)**4)
        p+=(num/denom)*dx
        i+=1
    S = mu_3/((N-1)*sd**3)
    k = mu_4/(sd**4)
    print(p)
    return P,X, round(mu,2), round(sd,2),round(S,2),round(k,2)

in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

offsets = [0.0]

a = Dataset(in_dir+"OF_Dataset.nc")
b = Dataset(in_dir+"Dataset.nc")

ic = 2
for offset in offsets:

    out_dir = in_dir + "OOPBM_lineplots/"
    out_dir_LPF = in_dir + "Systematic_LPF/"
    out_dir_PDF = in_dir+"PDFs/"

    Time_OF = np.array(a.variables["time_OF"])
    Time_sampling = np.array(b.variables["time_sampling"])
    Time_sampling = Time_sampling - Time_sampling[0]

    Time_start = 200
    #Time_end = Time_sampling[-1]
    Time_end = 300

    dt = Time_OF[1] - Time_OF[0]

    Time_start_idx = np.searchsorted(Time_OF,Time_start)
    Time_end_idx = np.searchsorted(Time_OF,Time_end)

    Time_OF = Time_OF[Time_start_idx:Time_end_idx]

    Azimuth = np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx])
    Azimuth = np.radians(Azimuth)

    RtAeroFxh = np.array(a.variables["RtAeroFxh"][Time_start_idx:Time_end_idx])
    RtAeroFyh = np.array(a.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
    RtAeroFzh = np.array(a.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

    RtAeroFys = []; RtAeroFzs = []
    for i in np.arange(0,len(Time_OF)):
        RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
        RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
    RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)
    
    RtAeroFR = np.sqrt( np.add( np.square(RtAeroFys), np.square(RtAeroFzs) ) )

    RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])
    RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
    RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])
    
    RtAeroMys = []; RtAeroMzs = []
    for i in np.arange(0,len(Time_OF)):
        RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
        RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
    RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

    RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 

    LSSGagMys = np.array(a.variables["LSSGagMys"][Time_start_idx:Time_end_idx])
    LSSGagMzs = np.array(a.variables["LSSGagMzs"][Time_start_idx:Time_end_idx])
    LSSGagMR = np.sqrt( np.add(np.square(LSSGagMys), np.square(LSSGagMzs)) )

    LSShftMxa = np.array(a.variables["LSShftMxa"][Time_start_idx:Time_end_idx])
    LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
    LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
    LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

    LSShftFxa = np.array(a.variables["LSShftFxa"][Time_start_idx:Time_end_idx])
    LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
    LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
    LSShftFR = np.sqrt( np.add(np.square(LSShftFys), np.square(LSShftFzs)) )

    L = 1.912

    C_LSSTipMys = LSSGagMys - L*LSShftFzs
    C_LSSTipMzs = LSSGagMzs + L*LSShftFys
    C_LSSTipMR = np.sqrt( np.add(np.square(C_LSSTipMys), np.square(C_LSSTipMzs)) )

    Fy_add = np.subtract(LSShftFys,RtAeroFys/1000)
    Fz_add = np.subtract(LSShftFzs,RtAeroFzs/1000)
    My_add = np.subtract(LSSTipMys,RtAeroMys/1000)
    Mz_add = np.subtract(LSSTipMzs,RtAeroMzs/1000)
    MR_add = np.subtract(LSSTipMR,RtAeroMR/1000)

    Rel_LSSTipMys = np.true_divide(abs(LSSTipMys),LSSTipMR)
    Rel_LSSTipMzs = np.true_divide(abs(LSSTipMzs),LSSTipMR)
    Theta_LSSTipM = np.degrees(np.arctan2(LSSTipMzs,LSSTipMys))

    Rel_LSShftFys = np.true_divide(abs(LSShftFys),LSShftFR)
    Rel_LSShftFzs = np.true_divide(abs(LSShftFzs),LSShftFR)
    Theta_LSShftF = np.degrees(np.arctan2(LSShftFzs,LSShftFys))

    Rel_RtAeroFys = np.true_divide(abs(RtAeroFys/1000),RtAeroFR/1000)
    Rel_RtAeroFzs = np.true_divide(abs(RtAeroFzs/1000),RtAeroFR/1000)
    Theta_RtAeroF = np.degrees(np.arctan2(RtAeroFzs,RtAeroFys))

    Rel_RtAeroMys = np.true_divide(abs(RtAeroMys/1000),RtAeroMR/1000)
    Rel_RtAeroMzs = np.true_divide(abs(RtAeroMzs/1000),RtAeroMR/1000)
    Theta_RtAeroM = np.degrees(np.arctan2(RtAeroMzs,RtAeroMys))

    L1 = 1.912; L2 = 5

    FBy = -((np.add(LSSTipMys,LSShftFzs*(L1+L2))/L2))
    FBz = (np.subtract(LSSTipMzs,LSShftFys*(L1+L2))/L2)
    FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
    Rel_FBy = np.true_divide(abs(FBy),FBR)
    Rel_FBz = np.true_divide(abs(FBz),FBR)
    Theta_FB = np.degrees(np.arctan2(FBz,FBy))


    group = b.groups["{}".format(offset)]
    Ux = np.array(group.variables["Ux"])

    IA = np.array(group.variables["IA"])

    f = interpolate.interp1d(Time_sampling,Ux)
    Ux = f(Time_OF)

    f = interpolate.interp1d(Time_sampling,IA)
    IA = f(Time_OF)


    #plotting options
    plot_variables = False
    plot_FFT_OOPBM = False
    compare_total_OOPBM_correlations = False
    compare_FFT_OOPBM = False
    plot_sys_LPF = False
    plot_relative_contributions = True
    plot_PDF = False

    


    #plot variables#
    if plot_variables == True:

        Variables = ["RtAeroFxh", "RtAeroFys", "RtAeroFzs", "RtAeroMxh", "RtAeroMys", "RtAeroMzs", "RtAeroMR", 
                     "LSShftFxa", "LSShftFys","LSShftFzs", "LSShftMxa", "LSSTipMys", "LSSTipMzs", "LSSTipMR", 
                     "LSSGagMys", "LSSGagMzs", "LSSGagMR"]
        units = ["[kN]","[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN]","[kN-m]",
                 "[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]"]
        Ylabels = ["Rotor Aerodynamic Force x direction fixed frame of reference","Rotor Aerodynamic Force y direction fixed frame of reference",
                   "Rotor Aerodynamic Force z direction fixed frame of reference", "Rotor Aerodynamic Moment x direction fixed frame of reference",
                   "Rotor Aerodynamic Moment y direction fixed frame of reference", "Rotor Aerodynamic Moment z direction fixed frame of reference",
                   "Rotor Aerodynamic OOPBM fixed frame of reference", "Rotor Aeroelastic Force x direction fixed frame of reference",
                   "Rotor Aeroelastic Force y direction fixed frame of reference", "Rotor Aeroelastic Force z direction fixed frame of reference",
                   "Rotor Aeroelastic Moment x direction fixed frame of reference","Rotor Aeroelastic Moment y direction fixed frame of reference",
                   "Rotor Aeroelastic Moment z direction fixed frame of reference","Rotor Aeroelastic OOPBM fixed frame of reference",
                   "LSS Aeroelastic Moment y direction fixed frame of reference","LSS Aeroelastic Moment z direction fixed frame of reference",
                   "LSS Aeroelastic OOPBM fixed frame of reference"]
        h_vars = [RtAeroFxh/1000, RtAeroFys/1000, RtAeroFzs/1000, RtAeroMxh/1000, RtAeroMys/1000, 
                  RtAeroMzs/1000, RtAeroMR/1000, LSShftFxa, LSShftFys,
                  LSShftFzs, LSShftMxa, LSSTipMys, LSSTipMzs, LSSTipMR, LSSGagMys, LSSGagMzs, LSSGagMR]
        # Variables = ["LSShftFys-RtAeroFys", "LSShftFzs-RtAeroFzs", "LSSTipMys-RtAeroMys", "LSSTipMzs-RtAeroMzs", "LSSTipMR-RtAeroMR"]
        # Ylabels = ["LSShftFys-RtAeroFys", "LSShftFzs-RtAeroFzs", "LSSTipMys-RtAeroMys", "LSSTipMzss-RtAeroMzs", "LSSTipMR-RtAeroMR"]
        # units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]"]
        # h_vars = [Fy_add, Fz_add, My_add, Mz_add, MR_add]
        # Variables = ["LSSTipMys", "LSSTipMzs", "LSSTipMR"]
        # Ylabels = ["Aeroelastic Rotor Moment y", "Aeroelastic Rotor Moment z", "Aeroelastic Rotor OOPBM"]
        # units = ["[kN-m]","[kN-m]","[kN-m]"]
        # h_vars = [LSSTipMys, LSSTipMzs, LSSTipMR]

        for i in np.arange(0,len(h_vars)):
            cutoff = 40
            signal_LP = low_pass_filter(h_vars[i], cutoff)
            fig = plt.figure(figsize=(14,8))
            plt.plot(Time_OF,signal_LP,"-b")
            # plt.plot(Time_OF,signal_LP,"-r")
            plt.axhline(np.mean(signal_LP),linestyle="--",color="k")
            plt.xlabel("Time [s]",fontsize=16)
            plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
            #plt.legend(["total signal", "Low pass filtered signal","mean signal"])
            plt.tight_layout()
            plt.savefig(out_dir+"{0}".format(Variables[i]))
            plt.close()


    if plot_FFT_OOPBM == True:
        # Variables = ["RtAeroFxh", "RtAeroFys", "RtAeroFzs", "RtAeroMxh", "RtAeroMys", "RtAeroMzs", "RtAeroMR", 
        #              "LSShftFxa", "LSShftFys","LSShftFzs", "LSShftMxa", "LSSTipMys", "LSSTipMzs", "LSSTipMR", 
        #              "LSSGagMys", "LSSGagMzs", "LSSGagMR"]
        # units = ["[kN]","[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN]","[kN-m]",
        #          "[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]"]
        # Ylabels = ["Rotor Aerodynamic Force x direction fixed frame of reference","Rotor Aerodynamic Force y direction fixed frame of reference",
        #            "Rotor Aerodynamic Force z direction fixed frame of reference", "Rotor Aerodynamic Moment x direction fixed frame of reference",
        #            "Rotor Aerodynamic Moment y direction fixed frame of reference", "Rotor Aerodynamic Moment z direction fixed frame of reference",
        #            "Rotor Aerodynamic OOPBM fixed frame of reference", "Rotor Aeroelastic Force x direction fixed frame of reference",
        #            "Rotor Aeroelastic Force y direction fixed frame of reference", "Rotor Aeroelastic Force z direction fixed frame of reference",
        #            "Rotor Aeroelastic Moment x direction fixed frame of reference","Rotor Aeroelastic Moment y direction fixed frame of reference",
        #            "Rotor Aeroelastic Moment z direction fixed frame of reference","Rotor Aeroelastic OOPBM fixed frame of reference",
        #            "LSS Aeroelastic Moment y direction fixed frame of reference","LSS Aeroelastic Moment z direction fixed frame of reference",
        #            "LSS Aeroelastic OOPBM fixed frame of reference"]
        # h_vars = [RtAeroFxh/1000, RtAeroFys/1000, RtAeroFzs/1000, RtAeroMxh/1000, RtAeroMys/1000, 
        #           RtAeroMzs/1000, RtAeroMR/1000, LSShftFxa, LSShftFys,
        #           LSShftFzs, LSShftMxa, LSSTipMys, LSSTipMzs, LSSTipMR, LSSGagMys, LSSGagMzs, LSSGagMR]

        Variables = ["LSShftFys-RtAeroFys", "LSShftFzs-RtAeroFzs", "LSSTipMys-RtAeroMys", "LSSTipMzs-RtAeroMzs", "LSSTipMR-RtAeroMR"]
        Ylabels = ["LSShftFys-RtAeroFys", "LSShftFzs-RtAeroFzs", "LSSTipMys-RtAeroMys", "LSSTipMzss-RtAeroMzs", "LSSTipMR-RtAeroMR"]
        units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]"]
        h_vars = [Fy_add, Fz_add, My_add, Mz_add, MR_add]

        for i in np.arange(0,len(h_vars)):
            
            frq,FFT = temporal_spectra(h_vars[i],dt,Variables[i])

            fig = plt.figure(figsize=(14,8))
            plt.plot(frq,FFT)
            
            # if Variables[i] != "MR_diff":
            #     frq_int = [1/60, 1/30, 12.1/60, (12.1/60)*3]
            #     frq_label = ["60s", "30s", "1P", "3P"]
            #     y_FFT = FFT[0]+1e+03

            #     for l in np.arange(0,len(frq_int)):
            #         plt.axvline(frq_int[l])
            #         plt.text(frq_int[l],y_FFT, frq_label[l])

            plt.xscale("log")
            plt.yscale("log")
            plt.xlabel("Frequency [Hz]",fontsize=14)
            plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=14)
            plt.tight_layout()
            plt.savefig(out_dir+"FFT_{0}.png".format(Variables[i]))
            plt.close(fig)



    #compare total signal correlations
    if compare_total_OOPBM_correlations == True:
        Variables = [ ["RtAeroFxh", "LSShftFxa"],["RtAeroFys", "LSShftFys"],["RtAeroFzs", "LSShftFzs"], ["RtAeroMxh", "LSShftMxa"], 
                  ["RtAeroMys", "LSSTipMys"],["RtAeroMzs", "LSSTipMzs"], ["RtAeroMR", "LSSTipMR"] ]
        units = [ ["[kN]","[kN]"],["[kN]","[kN]"],["[kN]","[kN]"],["[kN-m]","[kN-m]"],["[kN-m]","[kN-m]"],["[kN-m]","[kN-m]"],["[kN-m]","[kN-m]"] ]
        Ylabels = [ ["Rotor Aerodynamic Force x direction fixed frame of reference","Rotor Aeroelastic Force x direction fixed frame of reference"],
        ["Rotor Aerodynamic Force y direction fixed frame of reference","Rotor Aeroelastic Force y direction fixed frame of reference"],
        ["Rotor Aerodynamic Force z direction fixed frame of reference","Rotor Aeroelastic Force z direction fixed frame of reference"],
        ["Rotor Aerodynamic Moment x direction fixed frame of reference","Rotor Aeroelastic Moment x direction fixed frame of reference"],
        ["Rotor Aerodynamic Moment y direction fixed frame of reference","Rotor Aeroelastic Moment y direction fixed frame of reference"],
        ["Rotor Aerodynamic Moment z direction fixed frame of reference","Rotor Aeroelastic Moment z direction fixed frame of reference"],
        ["Rotor Aerodynamic OOPBM fixed frame of reference","Rotor Aeroelastic OOPBM fixed frame of reference"] ]
        h_vars = [ [RtAeroFxh/1000, LSShftFxa],[RtAeroFys/1000, LSShftFys],[RtAeroFzs/1000, LSShftFzs], [RtAeroMxh/1000, LSShftMxa], 
                  [RtAeroMys/1000, LSSTipMys],[RtAeroMzs/1000, LSSTipMzs], [RtAeroMR/1000, LSSTipMR] ]

        for i in np.arange(0,len(h_vars)):

            h_var = h_vars[i]
            Ylabel = Ylabels[i]
            unit = units[i]
            Variable = Variables[i]

            fig,ax = plt.subplots(figsize=(14,8))
            
            corr = correlation_coef(h_var[0],h_var[1])
            corr = round(corr,2)

            ax.plot(Time_OF,h_var[0],'-b')
            ax.set_ylabel("{0} {1}".format(Ylabel[0],unit[0]),fontsize=14)

            ax2=ax.twinx()
            ax2.plot(Time_OF,h_var[1],"-r")
            ax2.set_ylabel("{0} {1}".format(Ylabel[1],unit[1]),fontsize=14)

            plt.title("Correlation: {0} with \n{1} = {2}".format(Ylabel[0],Ylabel[1],corr),fontsize=16)
            ax.set_xlabel("Time [s]",fontsize=16)
            plt.tight_layout()
            plt.savefig(out_dir+"corr_{0}_{1}.png".format(Variable[0],Variable[1]))
            plt.close(fig)


    if compare_FFT_OOPBM == True:
        Variables = [ ["RtAeroFxh", "LSShftFxa"],["RtAeroFys", "LSShftFys"],["RtAeroFzs", "LSShftFzs"], ["RtAeroMxh", "LSShftMxa"], 
                  ["RtAeroMys", "LSSTipMys"],["RtAeroMzs", "LSSTipMzs"], ["RtAeroMR", "LSSTipMR"] ]
        units = [ ["[kN]","[kN]"],["[kN]","[kN]"],["[kN]","[kN]"],["[kN-m]","[kN-m]"],["[kN-m]","[kN-m]"],["[kN-m]","[kN-m]"],["[kN-m]","[kN-m]"] ]
        Ylabels = [ ["Rotor Aerodynamic Force x direction fixed frame of reference","Rotor Aeroelastic Force x direction fixed frame of reference"],
        ["Rotor Aerodynamic Force y direction fixed frame of reference","Rotor Aeroelastic Force y direction fixed frame of reference"],
        ["Rotor Aerodynamic Force z direction fixed frame of reference","Rotor Aeroelastic Force z direction fixed frame of reference"],
        ["Rotor Aerodynamic Moment x direction fixed frame of reference","Rotor Aeroelastic Moment x direction fixed frame of reference"],
        ["Rotor Aerodynamic Moment y direction fixed frame of reference","Rotor Aeroelastic Moment y direction fixed frame of reference"],
        ["Rotor Aerodynamic Moment z direction fixed frame of reference","Rotor Aeroelastic Moment z direction fixed frame of reference"],
        ["Rotor Aerodynamic OOPBM fixed frame of reference","Rotor Aeroelastic OOPBM fixed frame of reference"] ]
        h_vars = [ [RtAeroFxh/1000, LSShftFxa],[RtAeroFys/1000, LSShftFys],[RtAeroFzs/1000, LSShftFzs], [RtAeroMxh/1000, LSShftMxa], 
                  [RtAeroMys/1000, LSSTipMys],[RtAeroMzs/1000, LSSTipMzs], [RtAeroMR/1000, LSSTipMR] ]

        for i in np.arange(0,len(h_vars)):
            h_var = h_vars[i]
            Ylabel = Ylabels[i]
            unit = units[i]
            Variable = Variables[i]

            frq_i,FFT_i = temporal_spectra(h_var[0],dt,Variable[0])
            frq_j,FFT_j = temporal_spectra(h_var[1],dt,Variable[1])

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
            plt.ylabel("{0} {1}".format(Ylabel[0],unit[0]),fontsize=14)
            plt.legend([Variable[0],Variable[1]])
            plt.tight_layout()
            plt.savefig(out_dir+"FFT_{0}_{1}.png".format(Variable[0],Variable[1]))
            plt.close(fig)


    if plot_sys_LPF == True:

        Variables = ["LSShftFys-RtAeroFys", "LSShftFzs-RtAeroFzs", "LSSTipMys-RtAeroMys", "LSSTipMzs-RtAeroMzs", "LSSTipMR-RtAeroMR"]
        Ylabels = ["LSShftFys-RtAeroFys", "LSShftFzs-RtAeroFzs", "LSSTipMys-RtAeroMys", "LSSTipMzss-RtAeroMzs", "LSSTipMR-RtAeroMR"]
        units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]"]
        h_vars = [Fy_add, Fz_add, My_add, Mz_add, MR_add]
        # Variables = ["LSShftFys","LSShftFzs", "LSSTipMys", "LSSTipMzs", "LSSTipMR"]
        # units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]"]
        # Ylabels = ["Rotor Aeroelastic Force y direction fixed frame of reference", "Rotor Aeroelastic Force z direction fixed frame of reference",
        #            "Rotor Aeroelastic Moment y direction fixed frame of reference","Rotor Aeroelastic Moment z direction fixed frame of reference",
        #            "Rotor Aeroelastic OOPBM fixed frame of reference"]
        # h_vars = [LSShftFys,LSShftFzs, LSSTipMys, LSSTipMzs, LSSTipMR]

        for i in np.arange(0,len(h_vars)):
            cutoffs = [40,10,3,2,1,0.3,0.1]
            for cutoff in cutoffs:
                signal_LP = low_pass_filter(h_vars[i], cutoff)
                fig = plt.figure(figsize=(14,8))
                plt.plot(Time_OF,signal_LP)
                plt.xlabel("Time [s]",fontsize=16)
                plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
                plt.tight_layout()
                plt.savefig(out_dir_LPF+"LPF_{0}_{1}Hz.png".format(Variables[i],cutoff))
                plt.close()


    if plot_relative_contributions == True:


        ylabel1 = "Relative contributions to the Bearing radial force from (blue) y and (red) z components"
        ylabel2 = "Magnitude of Bearing radial force [kN]"
        ylabel3 = "Angle of Bearing radial force [deg]"

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(14,8))
        ax1.plot(Time_OF, Rel_FBy,"b")
        ax1.plot(Time_OF,Rel_FBz,"r")
        ax1.set_title('{}'.format(ylabel1))
        ax2.plot(Time_OF, FBR)
        ax2.set_title("{}".format(ylabel2))
        ax3.plot(Time_OF,Theta_FB)
        ax3.set_title("{}".format(ylabel3))
        fig.supxlabel("Time [s]")
        plt.tight_layout()
        #plt.show()
        plt.savefig(out_dir+"short_Relative_plots_BearingF.png")


    if plot_PDF == True:

        Variables = ["LSShftFys","LSShftFzs", "LSSTipMys", "LSSTipMzs", "LSSTipMR"]
        units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]"]
        Ylabels = ["Rotor Aeroelastic Force y direction fixed frame of reference", "Rotor Aeroelastic Force z direction fixed frame of reference",
                   "Rotor Aeroelastic Moment y direction fixed frame of reference","Rotor Aeroelastic Moment z direction fixed frame of reference",
                   "Rotor Aeroelastic OOPBM fixed frame of reference"]
        h_vars = [LSShftFys,LSShftFzs,LSSTipMys, LSSTipMzs, LSSTipMR]
        Variables = ["LSShftFys-RtAeroFys", "LSShftFzs-RtAeroFzs", "LSSTipMys-RtAeroMys", "LSSTipMzs-RtAeroMzs", "LSSTipMR-RtAeroMR"]
        Ylabels = ["LSShftFys-RtAeroFys", "LSShftFzs-RtAeroFzs", "LSSTipMys-RtAeroMys", "LSSTipMzss-RtAeroMzs", "LSSTipMR-RtAeroMR"]
        units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]"]
        h_vars = [Fy_add, Fz_add, My_add, Mz_add, MR_add]

        for i in np.arange(0,len(h_vars)):
            cutoff = 40
            signal_LP = low_pass_filter(h_vars[i], cutoff)

            P,X,mu,std,S,k = probability_dist(signal_LP)

            txt = "mean = {0}{1}\nstandard deviation = {2}{1}\nSkewness = {3}\nKurtosis = {4}".format(mu,units[i],std,S,k)
            fig = plt.figure(figsize=(14,8))
            plt.plot(X,P)
            plt.ylabel("Probability",fontsize=16)
            plt.xlabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
            if Variables[i] == "LSShftFzs-RtAeroFzs":
                plt.text(-1078,1.5,txt)
            elif Variables[i] == "LSShftFzs":
                plt.text(-1060,0.035,txt)
            else:
                plt.text(np.max(X)-0.1*np.max(X),np.max(P)-0.1*np.max(P),txt,horizontalalignment="right",verticalalignment="top")
            plt.tight_layout()
            plt.savefig(out_dir_PDF+"PDF_{0}".format(Variables[i]))
            plt.close()

    ic+=1



