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

#a = Dataset(in_dir+"OF_Dataset.nc")
a = Dataset(in_dir+"Dataset.nc")

#plotting options
compare_variables = False
compare_FFT = False
plot_relative_contributions = False
compare_total_correlations = False
compare_LPF_correlations = True
plot_PDF = False


out_dir = in_dir + "Bearing_loads/"

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]
#Time_end = 300

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

LSShftMxa = np.array(a.variables["LSShftMxa"][Time_start_idx:Time_end_idx])
LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

LSShftFxa = np.array(a.variables["LSShftFxa"][Time_start_idx:Time_end_idx])
LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
LSShftFR = np.sqrt( np.add(np.square(LSShftFys), np.square(LSShftFzs)) )

LSSGagMys = np.array(a.variables["LSSGagMys"][Time_start_idx:Time_end_idx])
LSSGagMzs = np.array(a.variables["LSSGagMzs"][Time_start_idx:Time_end_idx])
LSSGagMR = np.sqrt( np.add(np.square(LSSGagMys), np.square(LSSGagMzs)) )

L1 = 1.912; L2 = 5

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz

FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Rel_FBy = np.true_divide(np.square(FBy),np.square(FBR))
Rel_FBz = np.true_divide(np.square(FBz),np.square(FBR))
add_RelFB = np.add(Rel_FBy,Rel_FBz)
Theta_FB = np.degrees(np.arctan2(FBz,FBy))


Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Rel_Aero_FBy = np.true_divide(np.square(Aero_FBy),np.square(Aero_FBR))
Rel_Aero_FBz = np.true_divide(np.square(Aero_FBz),np.square(Aero_FBR))
add_Aero_RelFB = np.add(Rel_Aero_FBy,Rel_Aero_FBz)
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))

group = a.groups["0.0"]
Ux = np.array(group.variables["Ux"])
IA = np.array(group.variables["IA"])
Uy = np.array(group.variables["Uy"])
Uz = np.array(group.variables["Uz"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

f = interpolate.interp1d(Time_sampling,Ux)
Ux = f(Time_OF)

f = interpolate.interp1d(Time_sampling,IA)
IA = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iy)
Iy = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iz)
Iz = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Uy)
Uy = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Uz)
Uz = f(Time_OF)

Uxz = np.sqrt(np.add(np.square(Ux),np.square(Uz)))

Uxyz = []
for i in np.arange(0,len(Time_OF)):
    Uxyz.append(np.sqrt(Ux[i]**2 + Uy[i]**2 + Uz[i]**2))


if compare_variables == True:

    Variables =["Bearing reaction force z comps"]
    units = [["[kN]", "[kN]","[kN]"]]
    Ylabels = [["[-M_y/L_2]","[$-F_z(L_1+L_2)/L_2$]","Bearing Force z"]]
    h_vars = [[Aero_FBMz/1000,Aero_FBFz/1000,Aero_FBz/1000]]

    for i in np.arange(0,len(h_vars)):
        h_var = h_vars[i]; unit = units[i]; ylabel = Ylabels[i]

        cutoff = 40
        signal_LP_0 = low_pass_filter(h_var[0], cutoff)
        signal_LP_1 = low_pass_filter(h_var[1], cutoff)
        signal_LP_2 = low_pass_filter(h_var[2], cutoff)


        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(14,8))
        ax1.plot(Time_OF, signal_LP_0)
        ax1.set_title('{} {}'.format(ylabel[0],unit[0]))
        ax2.plot(Time_OF, signal_LP_1)
        ax2.set_title("{} {}".format(ylabel[1],unit[1]))
        ax3.plot(Time_OF,signal_LP_2)
        ax3.set_title("{} {}".format(ylabel[2],unit[2]))
        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(in_dir+"Bearing_Aero_Loads/{}.png".format(Variables[i]))


if compare_FFT == True:
    Variables = ["Radial Bearing Force components", "Y Bearing Force components", "Z Bearing Force components"]
    units = [["[kN]", "[kN]", "[kN]"], ["[kN]", "[kN]", "[kN]"], ["[kN]", "[kN]", "[kN]"]]
    Ylabels = [["Bearing Force y direction", "Bearing Force z direction", "Magnitude Bearing Force"], 
               ["Force due to the moment contribution \nto the Bearing Force y direction",
                "Force due to the force contribution \nto the Bearing Force y direction","Bearing Force y direction"],
               ["Force due to the moment contribution \nto the Bearing Force z direction",
                "Force due to the force contribution \nto the Bearing Force z direction","Bearing Force z direction"]]
    h_vars = [[Aero_FBy/1000, Aero_FBz/1000, Aero_FBR/1000], [FBMy/1000, FBFy/1000, FBy/1000], [FBMz/1000, FBFz/1000, FBz/1000]]

    for i in np.arange(0,len(h_vars)):
        h_var = h_vars[i]; unit = units[i]; ylabel = Ylabels[i]

        cutoff = 40
        signal_LP_0 = low_pass_filter(h_var[0], cutoff)
        signal_LP_1 = low_pass_filter(h_var[1], cutoff)
        signal_LP_2 = low_pass_filter(h_var[2], cutoff)

        frq_0,FFT_0 = temporal_spectra(h_var[0],dt,Variables[i])
        frq_1,FFT_1 = temporal_spectra(h_var[1],dt,Variables[i])
        frq_2,FFT_2 = temporal_spectra(h_var[2],dt,Variables[i])

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(14,8))
        ax1.plot(frq_0, FFT_0)
        ax1.set_title('{} {}'.format(ylabel[0],unit[0]),fontsize=14)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.plot(frq_1, FFT_1)
        ax2.set_title("{} {}".format(ylabel[1],unit[1]),fontsize=14)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        ax3.plot(frq_2,FFT_2)
        ax3.set_title("{} {}".format(ylabel[2],unit[2]),fontsize=14)
        ax3.set_xscale("log")
        ax3.set_yscale("log")
        fig.supxlabel("Frequency [Hz]",fontsize=14)
        plt.tight_layout()
        plt.savefig(in_dir+"Bearing_Aero_Loads/FFT_{}.png".format(Variables[i]))
        plt.close()


if plot_relative_contributions == True:

    h_vars = [[Rel_Aero_FBy, Rel_Aero_FBz, Aero_FBR/1000, Theta_Aero_FB]]
    Ylabels = [["Relative contributions to the Radial Bearing Force (y blue) (z red)", "Bearing Radial Force", "Angle Bearing Radial Force"]]
    Variables = ["BearingF"]
    units = [["[-]","[kN]","[deg]"]]

    for i in np.arange(0,len(h_vars)):
        h_var = h_vars[i]
        ylabel = Ylabels[i]
        Variable = Variables[i]
        unit = units[i]

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(14,8))
        ax1.plot(Time_OF, h_var[0],"b")
        ax1.plot(Time_OF,h_var[1],"r")
        ax1.set_title('{}'.format(ylabel[0]))
        ax2.plot(Time_OF, h_var[2])
        ax2.set_title("{}".format(ylabel[1]))
        ax3.plot(Time_OF,h_var[3])
        ax3.set_title("{}".format(ylabel[2]))
        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(in_dir+"Bearing_Aero_Loads/Relative_{}.png".format(Variable))
        

if compare_total_correlations == True:
    # Variables = ["Iy", "Iz","Aero_FBR", "Aero_FBy", "Aero_FBz", "Aero_FBMz", "Aero_FBMy", "Aero_FBFy", "Aero_FBFz"]
    # units = ["[m/s]","[m/s]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]"]
    # Ylabels = ["Iy", "Iz","Aerodynamic Bearing Reaction Force", "Aerodynamic Bearing Reaction Force y", 
    #            "Aerodynamic Bearing Reaction Force z","[$-M_y/L_2$]", "[$M_z/L_2$]", "[$-F_y(L_1+L_2)/L_2$]","[$-F_z(L_1+L_2)/L_2$]"]
    # h_vars = [Iy, Iz, Aero_FBR/1000, Aero_FBy/1000, Aero_FBz/1000, Aero_FBMz/1000, Aero_FBMy/1000, Aero_FBFy/1000, Aero_FBFz/1000]
    Variables = ["IA", "RtAeroMR", "Aero_FBR", "Aero_FBy", "Aero_FBz", "Aero_FBMy", "Aero_FBMz", "Aero_FBFy","Aero_FBFz"]
    units = ["[$m^4/s$]","[kN-m]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]"]
    Ylabels = ["Asymmetry Parameter", "OOPBM","Magnitude Bearing Force", "Bearing Force y", "Bearing Force z",
               "[$M_z/L_2$]", "[$-M_y/L_2$]","[$-F_y(L_1+L_2)/L_2$]","[$-F_z(L_1+L_2)/L_2$]"]
    h_vars = [IA, RtAeroMR/1000, Aero_FBR/1000, Aero_FBy/1000, Aero_FBz/1000, Aero_FBMy/1000, Aero_FBMz/1000, Aero_FBFy/1000,
              Aero_FBFz/1000]

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
            plt.savefig(in_dir+"Aero_correlations/corr_{0}_{1}.png".format(Variables[j],Variables[i]))
            plt.close(fig)


if compare_LPF_correlations == True:
    # Variables = ["Iy", "Iz","Aero_FBR", "Aero_FBy", "Aero_FBz", "Aero_FBMz", "Aero_FBMy", "Aero_FBFy", "Aero_FBFz"]
    # units = ["[m/s]","[m/s]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]"]
    # Ylabels = ["Iy", "Iz","Aerodynamic Bearing Reaction Force", "Aerodynamic Bearing Reaction Force y", 
    #            "Aerodynamic Bearing Reaction Force z","[$-M_y/L_2$]", "[$M_z/L_2$]", "[$-F_y(L_1+L_2)/L_2$]","[$-F_z(L_1+L_2)/L_2$]"]
    # h_vars = [Iy, Iz, Aero_FBR/1000, Aero_FBy/1000, Aero_FBz/1000, Aero_FBMz/1000, Aero_FBMy/1000, Aero_FBFy/1000, Aero_FBFz/1000]
    Variables = ["IA", "RtAeroMR", "Aero_FBR", "Aero_FBy", "Aero_FBz", "Aero_FBMy", "Aero_FBMz", "Aero_FBFy","Aero_FBFz"]
    units = ["[$m^4/s$]","[kN-m]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]"]
    Ylabels = ["Asymmetry Parameter", "OOPBM","Magnitude Bearing Force", "Bearing Force y", "Bearing Force z",
               "[$M_z/L_2$]", "[$-M_y/L_2$]","[$-F_y(L_1+L_2)/L_2$]","[$-F_z(L_1+L_2)/L_2$]"]
    h_vars = [IA, RtAeroMR/1000, Aero_FBR/1000, Aero_FBy/1000, Aero_FBz/1000, Aero_FBMy/1000, Aero_FBMz/1000, Aero_FBFy/1000,
              Aero_FBFz/1000]

    for j in np.arange(0,len(h_vars)):
        for i in np.arange(0,len(h_vars)):

            cutoff = 0.1
            signal_LP_0 = low_pass_filter(h_vars[i], cutoff)
            signal_LP_1 = low_pass_filter(h_vars[j], cutoff)

            fig,ax = plt.subplots(figsize=(14,8))
            
            corr = correlation_coef(signal_LP_0,signal_LP_1)
            corr = round(corr,2)

            ax.plot(Time_OF,signal_LP_0,'-b')
            ax.set_ylabel("{0} {1}".format(Ylabels[j],units[j]),fontsize=14)

            ax2=ax.twinx()
            ax2.plot(Time_OF,signal_LP_1,"-r")
            ax2.set_ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=14)

            plt.title("Low passs filtered at 0.1Hz.\nCorrelation: {0} with {1} = {2}".format(Ylabels[j],Ylabels[i],corr),fontsize=16)
            ax.set_xlabel("Time [s]",fontsize=16)
            plt.tight_layout()
            plt.savefig(in_dir+"Aero_correlations/LPF_corr_{0}_{1}.png".format(Variables[j],Variables[i]))
            plt.close(fig)


if plot_PDF == True:

    Variables = ["Aero_FBR", "Aero_FBy", "Aero_FBz", "Aero_FBMz", "Aero_FBMy", "Aero_FBFy", "Aero_FBFz"]
    units = ["[kN]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]"]
    Ylabels = ["Aerodynamic Bearing Reaction Force", "Aerodynamic Bearing Reaction Force y", "Aerodynamic Bearing Reaction Force z",
               "[$-M_y/L_2$]", "[$M_z/L_2$]", "[$-F_y(L_1+L_2)/L_2$]","[$-F_z(L_1+L_2)/L_2$]"]
    h_vars = [Aero_FBR/1000, Aero_FBy/1000, Aero_FBz/1000, Aero_FBMz/1000, Aero_FBMy/1000, Aero_FBFy/1000, Aero_FBFz/1000]

    for i in np.arange(0,len(h_vars)):
        cutoff = 40
        signal_LP = low_pass_filter(h_vars[i], cutoff)

        P,X,mu,std,S,k = probability_dist(signal_LP)

        txt = "mean = {0}{1}\nstandard deviation = {2}{1}\nSkewness = {3}\nKurtosis = {4}".format(mu,units[i],std,S,k)
        fig = plt.figure(figsize=(14,8))
        plt.plot(X,P)
        plt.ylabel("Probability",fontsize=16)
        plt.xlabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
        if Variables[i] == "FBFz":
            plt.text(1525,np.max(P)-0.1*np.max(P),txt,horizontalalignment="right",verticalalignment="top")
        else:
            plt.text(np.max(X)-0.1*np.max(X),np.max(P)-0.1*np.max(P),txt,horizontalalignment="right",verticalalignment="top")
        plt.tight_layout()
        plt.savefig(in_dir+"Aero_PDFs/{0}".format(Variables[i]))
        plt.close()