import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
import pyFAST.input_output as io
from matplotlib.backends.backend_pdf import PdfPages

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


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    return P,X, round(mu,2), round(sd,2)


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


def moving_stats(x,y,Ylabelx,unitsx,Ylabely,unitsy,time_window):

    window = int(time_window/dt)
    df = pd.Series(x)
    var = df.rolling(window=window).std()
    corr = correlation_coef(var[window:],abs(y[window:])) 

    fig,ax1 = plt.subplots(figsize=(14,8))
    ax1.plot(Time_OF,x,"k")
    ax1.plot(Time_OF,var,"b")
    ax2 = ax1.twinx()
    ax2.plot(Time_OF,abs(y),"r")
    ax1.set_ylabel("{}{}".format(Ylabelx,unitsx),fontsize=16)
    ax1.yaxis.label.set_color("blue")
    ax1.legend(["{}".format(Ylabelx),"Variance"])
    ax2.set_ylabel("{}{}".format(Ylabely,unitsy),fontsize=16)
    ax2.yaxis.label.set_color("red")
    fig.suptitle("correlation = {}".format(corr),fontsize=16)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"variance_{}.png".format(i))
    plt.cla()


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

out_dir = in_dir + "correlations/"

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

Azimuth = np.radians(np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx]))

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

L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))


Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Rel_Aero_FBy = np.true_divide(np.square(Aero_FBy),np.square(Aero_FBR))
Rel_Aero_FBz = np.true_divide(np.square(Aero_FBz),np.square(Aero_FBR))
add_Aero_RelFB = np.add(Rel_Aero_FBy,Rel_Aero_FBz)
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))




offset = "5.5"
group = a.groups["{}".format(offset)]
Ux_1 = np.array(group.variables["Ux"])
Uz_1 = np.array(group.variables["Uz"])
IA_1 = np.array(group.variables["IA"])
Iy_1 = np.array(group.variables["Iy"])
Iz_1 = np.array(group.variables["Iz"])

offset = "63.0"
group = a.groups["{}".format(offset)]
Ux_2 = np.array(group.variables["Ux"])
Uz_2 = np.array(group.variables["Uz"])
IA_2 = np.array(group.variables["IA"])
Iy_2 = np.array(group.variables["Iy"])
Iz_2 = np.array(group.variables["Iz"])

offset = "5.5"
group = a.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Uz = np.array(group.variables["Uz"])
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

f = interpolate.interp1d(Time_sampling,Ux)
Ux = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Uz)
Uz = f(Time_OF)

f = interpolate.interp1d(Time_sampling,IA)
IA = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iy)
Iy = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iz)
Iz = f(Time_OF)

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

cc = correlation_coef(FBR,I)
print(cc)

#plotting options
time_vs_corr = False
range_vs_corr = False
spectra = False
plot_moving_stats = False
line_plots = False

if time_vs_corr == True:
    Variables_arr = [["Ux", "RtAeroMR"], ["AeroFBy", "AeroFBz"], ["RtAeroMys", "RtAeroMzs"]]
    units_arr = [["[m/s]", "[kN-m]"],["[kN]", "[kN]"], ["[kN-m]", "[kN-m]"], ["[$m^4/s$]", "[$m^4/s$]"]]
    Ylabels_arr = [["Rotor averaged horizontal velocity", "Aerodynamic Rotor out-of-plane bending moment"], 
                ["Aerodynamic Bearing force y component", "Aerodynamic Bearing force z component"], 
                    ["Aerodynamic Rotor moment y component", "Aerodynamic Rotor moment z component"]]
    h_vars_arr = [[Ux, RtAeroMR/1000],[Aero_FBy/1000, Aero_FBz/1000], [RtAeroMys/1000, RtAeroMzs/1000]]

    cutoff = 40
    for j in np.arange(0,len(h_vars_arr)):
        Variables = Variables_arr[j]
        Ylabels = Ylabels_arr[j]
        units = units_arr[j]
        h_vars = h_vars_arr[j]

        signal_LP_0 = low_pass_filter(h_vars[0], cutoff)
        signal_LP_1 = low_pass_filter(h_vars[1], cutoff)

        corr_t = []
        for i in np.arange(100,len(Time_OF)):
            corr_t.append(correlation_coef(signal_LP_0[:i],signal_LP_1[:i]))
        
        corr = correlation_coef(signal_LP_0,signal_LP_1)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time_OF[100:],corr_t)
        plt.xlabel("Time [s]",fontsize=16)
        plt.ylabel("Correlations coefficient",fontsize=16)
        plt.title("Correlation between {0} {1} \n {2} {3} = {4}".format(Ylabels[0],units[0],Ylabels[1],units[1],round(corr,2)),fontsize=16)
        plt.tight_layout()
        plt.savefig(in_dir+"correlations/{}_{}_{}.png".format(cutoff,Variables[0],Variables[1]))
        plt.cla()


if range_vs_corr == True:
    Variables_arr = [["Ux", "RtAeroMR"], ["AeroFBy", "AeroFBz"], ["RtAeroMys", "RtAeroMzs"]]
    units_arr = [["[m/s]", "[kN-m]"],["[kN]", "[kN]"], ["[kN-m]", "[kN-m]"]]
    Ylabels_arr = [["Rotor averaged horizontal velocity", "Aerodynamic Rotor out-of-plane bending moment"], 
                ["Aerodynamic Bearing force y component", "Aerodynamic Bearing force z component"], 
                    ["Aerodynamic Rotor moment y component", "Aerodynamic Rotor moment z component"]]
    h_vars_arr = [[Ux, RtAeroMR/1000],[Aero_FBy/1000, Aero_FBz/1000], [RtAeroMys/1000, RtAeroMzs/1000]]
    Time_steps = np.linspace(0,len(Time_OF),60,dtype=int)
    Time_range = np.linspace(np.min(Time_OF),np.max(Time_OF),60)

    cutoff = 40
    for j in np.arange(0,len(h_vars_arr)):
        Variables = Variables_arr[j]
        Ylabels = Ylabels_arr[j]
        units = units_arr[j]
        h_vars = h_vars_arr[j]

        signal_LP_0 = low_pass_filter(h_vars[0], cutoff)
        signal_LP_1 = low_pass_filter(h_vars[1], cutoff)

        corr_t = []
        for i in np.arange(0,len(Time_steps)-1):
            corr_t.append(correlation_coef(signal_LP_0[Time_steps[i]:Time_steps[i+1]],signal_LP_1[Time_steps[i]:Time_steps[i+1]]))
        
        corr = correlation_coef(signal_LP_0,signal_LP_1)

        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_range[1:],corr_t,"-r")

        Time_start_idx = np.searchsorted(Time_sampling,Time_start)
        Time_end_idx = np.searchsorted(Time_sampling,Time_end)

        Time_sampling = Time_sampling[Time_start_idx:Time_end_idx]
        dt_sampling = Time_sampling[1] - Time_sampling[0]
        Ux = group.variables["Ux"][Time_start_idx:Time_end_idx]

        ax2=ax.twinx()
        ax2.plot(Time_sampling,Ux,"-b")
        ax2.set_ylabel("Ux",fontsize=16)

        fig.supxlabel("Time [s]",fontsize=16)
        ax.set_ylabel("Correlations coefficient",fontsize=16)
        plt.suptitle("Correlation between {0} {1} \n {2} {3} = {4}".format(Ylabels[0],units[0],Ylabels[1],units[1],round(corr,2)),fontsize=16)
        plt.tight_layout()
        plt.savefig(in_dir+"correlations/{}_{}_{}_2.png".format(cutoff,Variables[0],Variables[1]))
        plt.cla()


        P,X,mu,std = probability_dist(corr_t)
        print(mu)
        fig = plt.figure(figsize=(14,8))
        plt.plot(X,P)
        plt.xlabel("Correlation coeefficient",fontsize=16)
        plt.ylabel("Probability",fontsize=16)
        plt.title("Correlation between {0} {1} \n {2} {3} = {4}".format(Ylabels[0],units[0],Ylabels[1],units[1],round(corr,2)),fontsize=16)
        plt.tight_layout()
        plt.savefig(in_dir+"correlations/PDF_{}_{}_{}.png".format(cutoff,Variables[0],Variables[1]))
        plt.cla()


if spectra == True:
    Variables = ["Ux", "RtAeroMR", "AeroFBy", "AeroFBz", "RtAeroMys", "RtAeroMzs"]
    Ylabels = ["Rotor averaged horizontal velocity", "Aerodynamic Rotor out-of-plane bending moment", 
                "Aerodynamic Bearing force y component", "Aerodynamic Bearing force z component", 
                    "Aerodynamic Rotor moment y component", "Aerodynamic Rotor moment z component"]
    h_vars = [Ux, RtAeroMR/1000,Aero_FBy/1000, Aero_FBz/1000, RtAeroMys/1000, RtAeroMzs/1000]


    cutoff = 40
    for j in np.arange(0,len(h_vars)):
        Variable = Variables[j]
        Ylabel = Ylabels[j]
        h_var = h_vars[j]

        signal_LP_0 = low_pass_filter(h_var, cutoff)
        frq,PSD = temporal_spectra(h_var,dt,Variable)

        fig = plt.figure(figsize=(14,8))
        plt.loglog(frq,PSD)
        plt.xlabel("Frequency [Hz]",fontsize=16)
        plt.ylabel("Power spectral density \n{}".format(Ylabel))
        plt.grid()
        plt.tight_layout()
        plt.savefig(in_dir+"correlations/spectra_{}.png".format(Variable))
        plt.cla()


if plot_moving_stats == True:
    Vars = [[Aero_FBR/1000,IA],[Aero_FBy/1000,Iz],[Aero_FBz/1000,Iy],[RtAeroMys/1000, Iy], [RtAeroMzs/1000, Iz]]
    Ylabelxs = ["Magnitude Aerodynamic Bearing force", "Aerodynamic Bearing force y component", "Aerodynamic Bearing force z component",
                "Aerodynamic rotor force y component", "Aerodynamic rotor force z component"]
    Ylabelys = ["Absolute Asymmetry parameter", "Absolute Asymmetry around z", "Absolute Asymmetry around y",
                "Absolute Asymmetry around y", "Absolute Asymmetry around z"]
    unitsxs = ["[kN]", "[kN]", "[kN]","[kN-m]", "[kN-m]"]
    unitsys = ["[$m^4/s$]","[$m^4/s$]","[$m^4/s$]","[$m^4/s$]","[$m^4/s$]"]

    for i in np.arange(0,len(Vars)):
        x = Vars[i][0]; y = Vars[i][1]
        Ylabelx = Ylabelxs[i]; Ylabely = Ylabelys[i]
        unitsx = unitsxs[i]; unitsy = unitsys[i]
        time_window = 10

        moving_stats(x,y,Ylabelx,unitsx,Ylabely,unitsy,time_window)


if line_plots == True:

    out_dir = in_dir+"line_plots/"

    #planar quantities -5.5m
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Ux_1)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Rotor averaged horizontal velocity [m/s]",fontsize=16)
    plt.title("offset: -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"1.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Uz_1)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Rotor averaged vertical velocity [m/s]",fontsize=16)
    plt.title("offset: -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"2.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,IA_1)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry parameter [$m^4/s$]",fontsize=16)
    plt.title("offset: -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"3.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iy_1)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry around y axis [$m^4/s$]",fontsize=16)
    plt.title("offset: -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"4.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iz_1)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry around z axis [$m^4/s$]",fontsize=16)
    plt.title("offset: -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"5.png")
    plt.close()

    #planar quantities -63.0m not time shifted
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Ux_2)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Rotor averaged horizontal velocity [m/s]",fontsize=16)
    plt.title("offset: -63.0m not time shifted",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"6.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Uz_2)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Rotor averaged vertical velocity [m/s]",fontsize=16)
    plt.title("offset: -63.0m not time shifted",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"7.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,IA_2)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry parameter [$m^4/s$]",fontsize=16)
    plt.title("offset: -63.0m not time shifted",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"8.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iy_2)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry around y axis [$m^4/s$]",fontsize=16)
    plt.title("offset: -63.0m not time shifted",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"9.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iz_2)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry around z axis [$m^4/s$]",fontsize=16)
    plt.title("offset: -63.0m not time shifted",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"10.png")
    plt.close()


    #planar quantities -63.0m time shifted equivalent to if it was measured at -5.5m
    time_shift = 4.87
    time_shift_idx = np.searchsorted(Time_sampling,time_shift)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling[:-time_shift_idx],Ux_2[:-time_shift_idx])
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Rotor averaged horizontal velocity [m/s]",fontsize=16)
    plt.title("offset: -63.0m time shifted 4.78s \nequavalent to -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"11.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling[:-time_shift_idx],Uz_2[:-time_shift_idx])
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Rotor averaged vertical velocity [m/s]",fontsize=16)
    plt.title("offset: -63.0m time shifted 4.78s \nequavalent to -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"12.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling[:-time_shift_idx],IA_2[:-time_shift_idx])
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry parameter [$m^4/s$]",fontsize=16)
    plt.title("offset: -63.0m time shifted 4.78s \nequavalent to -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"13.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling[:-time_shift_idx],Iy_2[:-time_shift_idx])
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry around y axis [$m^4/s$]",fontsize=16)
    plt.title("offset: -63.0m time shifted 4.78s \nequavalent to -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"14.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling[:-time_shift_idx],Iz_2[:-time_shift_idx])
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Asymmetry around z axis [$m^4/s$]",fontsize=16)
    plt.title("offset: -63.0m time shifted 4.78s \nequavalent to -5.5m",fontsize=16)
    plt.xticks(np.arange(0,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"15.png")
    plt.close()


    #openfast variables
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,Aero_FBR/1000)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Magnitude Aerodynamic Bearing force [kN]",fontsize=16)
    plt.xticks(np.arange(200,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"16.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,Aero_FBy/1000)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Aerodynamic Bearing force y component [kN]",fontsize=16)
    plt.xticks(np.arange(200,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"17.png")
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,Aero_FBz/1000)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Aerodynamic Bearing force z component [kN]",fontsize=16)
    plt.xticks(np.arange(200,1250,50))
    plt.tight_layout()
    plt.savefig(out_dir+"18.png")
    plt.close()

