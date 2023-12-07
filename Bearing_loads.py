import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
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


def dt_calc(u,time):
    dt = []
    for i in np. arange(0,len(time)-1):
        dt.append((u[i+1]-u[i])/(time[i+1]-time[i]))
    return np.array(dt)


in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

#plotting options
compare_variables = False
compare_FFT = False
plot_relative_contributions = False
compare_total_correlations = False
compare_LPF_correlations = False
plot_PDF = False
plot_derivative = False
plot_moving_stats = False
systematic_LPF_FFT = True
systematic_LPF = True

out_dir = in_dir + "Bearing_Loads_2/"

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

offset = "63.0"
group = a.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Uz = np.array(group.variables["Uz"])
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

# I_vec = np.sqrt( np.add( np.square(Iy), np.square(Iz) ) )

# Ux_dt = dt_calc(Ux,Time_sampling)
# Uz_dt = dt_calc(Uz,Time_sampling)
# Iy_dt = dt_calc(Iy,Time_sampling)

# fig, (ax2, ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
# # ax1.plot(Time_sampling,Ux)
# # ax1.grid()
# # ax1.set_title("Rotor averaged horizontal velocity [m/s]",fontsize=16)
# ax2.plot(Time_sampling,Iy)
# ax2.set_title("Asymmetry around y axis [$m^4/s$]",fontsize=16)
# ax2.grid()
# ax3.plot(Time_sampling,Iz)
# ax3.set_title("Asymmetry around z axis [$m^4/s$]",fontsize=16)
# ax3.grid()
# plt.xlabel("Time [s]",fontsize=16)
# plt.tight_layout()
# #plt.show()
# plt.savefig(in_dir+"velocity_correlations_2/Rel_Iy_Iz.png")
# plt.close()


# fig, (ax1, ax2) = plt.subplots(2,1,sharex=True, sharey=True)
# offsets = ["0.0", "63.0"]
# for offset in offsets:
#     group = a.groups["{}".format(offset)]
#     Ux = np.array(group.variables["Ux"])
#     IA = np.array(group.variables["IA"])
#     IB = np.array(group.variables["IB"])
#     Iy = np.array(group.variables["Iy"])
#     Iz = np.array(group.variables["Iz"])

#     if offset == "63.0":
#         time_shift_idx = np.searchsorted(Time_sampling,5.5)
#         ax1.plot(Time_sampling,Iy,"r-")
#         ax2.plot(Time_sampling,Iz,"r-")
#     else:
#         ax1.plot(Time_sampling,Iy,"b")
#         ax2.plot(Time_sampling,Iz,"b")

# ax1.axhline(y=(np.mean(Iy)+0.2e+06),linestyle="--",color="k")
# ax1.axhline(y=(np.mean(Iy)-0.2e+06),linestyle="--",color="k")
# ax2.axhline(y=(np.mean(Iz)+0.2e+06),linestyle="--",color="k")
# ax2.axhline(y=(np.mean(Iz)-0.2e+6),linestyle="--",color="k")
# plt.legend(offsets)
# plt.xlabel("Time [s]",fontsize=16)
# ax1.set_ylabel("Asymmtry around y axis [$m^4/s$]")
# ax2.set_ylabel("Asymmtry around z axis [$m^4/s$]")
# ax1.grid(); ax2.grid()
# plt.tight_layout()
# plt.show()


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



if compare_variables == True:

    Variables =["Bearing reaction force z comps"]
    units = [["[kN]", "[m^4/s]","[kN-m]","[kN-m]"]]
    Ylabels = [["FBR","IA","Mys","Mzs"]]
    h_vars = [[Aero_FBR/1000, IA, RtAeroMys/1000, RtAeroMzs/1000]]

    for i in np.arange(0,len(h_vars)):
        h_var = h_vars[i]; unit = units[i]; ylabel = Ylabels[i]

        cutoff = 40
        signal_LP_0 = low_pass_filter(h_var[0], cutoff)
        signal_LP_1 = low_pass_filter(h_var[1], cutoff)
        signal_LP_2 = low_pass_filter(h_var[2], cutoff)
        signal_LP_3 = low_pass_filter(h_var[3], cutoff)


        fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, 1, sharex=True, figsize=(14,8))
        ax1.plot(Time_OF, signal_LP_0)
        ax1.set_title('{} {}'.format(ylabel[0],unit[0]))
        ax2.plot(Time_OF, signal_LP_1)
        ax2.set_title("{} {}".format(ylabel[1],unit[1]))
        ax3.plot(Time_OF,signal_LP_2)
        ax3.set_title("{} {}".format(ylabel[2],unit[2]))
        ax4.plot(Time_OF,signal_LP_3)
        ax4.set_title("{} {}".format(ylabel[3],unit[3]))
        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.show()
        #plt.savefig(in_dir+"Bearing_Aero_Loads/{}.png".format(Variables[i]))


if compare_FFT == True:
    Variables = [["Iy", "Iz"]]
    units = [["[$m^4/s$]","[$m^4/s$]"]]
    Ylabels = [["Asymmetry in horizontal", "Asymmetry in vertical"]]
    h_vars = [[Iy,Iz]]



    for i in np.arange(0,len(h_vars)):
        h_var = h_vars[i]; unit = units[i]; ylabel = Ylabels[i]

        # cutoff = 40
        # signal_LP_0 = low_pass_filter(h_var[0], cutoff)
        # signal_LP_1 = low_pass_filter(h_var[1], cutoff)
        # signal_LP_2 = low_pass_filter(h_var[2], cutoff)

        dt = Time_sampling[1]-Time_sampling[0]

        frq_0,FFT_0 = temporal_spectra(h_var[0],dt,Variables[i])
        frq_1,FFT_1 = temporal_spectra(h_var[1],dt,Variables[i])
        #frq_2,FFT_2 = temporal_spectra(h_var[2],dt,Variables[i])

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(14,8)) #set to 2 change
        ax1.plot(frq_0, FFT_0)
        ax1.set_title('{} {}'.format(ylabel[0],unit[0]),fontsize=14)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax2.plot(frq_1, FFT_1)
        ax2.set_title("{} {}".format(ylabel[1],unit[1]),fontsize=14)
        ax2.set_xscale("log")
        ax2.set_yscale("log")
        # ax3.plot(frq_2,FFT_2)
        # ax3.set_title("{} {}".format(ylabel[2],unit[2]),fontsize=14)
        # ax3.set_xscale("log")
        # ax3.set_yscale("log")
        fig.supxlabel("Frequency [Hz]",fontsize=14)
        plt.tight_layout()
        plt.savefig(in_dir+"velocity_correlations_2/FFT_-63m_{}.png".format(Variables[i]))
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
    
    Variables = ["Iy", "Iz"]
    units = ["[]","[]"]
    Ylabels = ["Asymmetry in horizontal", "Asymmetry in vertical"]
    h_vars = [Iy,Iz]

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
    Variables = ["IA","RtAeroMR"]
    units = ["[$m^4/s$]", "[kN-m]"]
    Ylabels = ["Asymmetry parameter", "Aerodynamic OOPBM"]
    h_vars = [IA, RtAeroMR]

    for j in np.arange(0,len(h_vars)):
        for i in np.arange(0,len(h_vars)):

            cutoff = 40
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

            plt.title("Low passs filtered at {0}Hz.\nCorrelation: {1} with {2} = {3}".format(cutoff,Ylabels[j],Ylabels[i],corr),fontsize=16)
            ax.set_xlabel("Time [s]",fontsize=16)
            plt.tight_layout()
            plt.savefig(in_dir+"velocity_correlations_2/LPF_{0}_corr_{1}_{2}.png".format(cutoff,Variables[j],Variables[i]))
            plt.close(fig)


if plot_PDF == True:

    Variables = ["Iy","Iz"]
    units = ["[$m^4/s$]","[$m^4/s$]"]
    Ylabels = ["Asymmetry around y axis", "Asymmetry around z axis"]
    
    h_vars = [Iy,Iz]

    for i in np.arange(0,len(h_vars)):
        # cutoff = 40
        # signal_LP = low_pass_filter(h_vars[i], cutoff)

        P,X,mu,std,S,k = probability_dist(h_vars[i])

        txt = "mean = {0}{1}\nstandard deviation = {2}{1}".format(mu,units[i],std)
        print(Variables[i], txt)

        idx1 = np.searchsorted(X,mu-std)
        idx2 = np.searchsorted(X,mu+std)
        ymax1 = P[idx1]
        ymax2 = P[idx2]

        fig = plt.figure(figsize=(14,8))
        plt.plot(X,P)
        plt.axvline(mu,color="b")
        plt.axvline(mu-std, ymax=ymax1, color="r"); plt.axvline(mu+std, ymax=ymax2, color="r")
        plt.ylabel("Probability",fontsize=16)
        plt.xlabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=28)
        plt.tight_layout()
        plt.savefig(in_dir+"velocity_correlations_2/{0}".format(Variables[i]))
        plt.close()


if plot_derivative == True:

    Variables = ["IA"]
    units = ["[$m^4/s$]"]
    Ylabels = ["Asymmetry parameter"]
    h_vars = [IA]

    for i in np.arange(0,len(h_vars)):

        d_var_dt = []
        for j in np.arange(0,len(h_vars[i])-1):
            d_var_dt.append( (IA[j+1]-IA[j])/(Time_sampling[j+1]-Time_sampling[j]) )

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time_sampling[:-1],d_var_dt)
        plt.xlabel("Time [s]",fontsize=16)
        plt.ylabel("d{}/dt - Time derivative of {}".format(Variables[i], Ylabels[i]),fontsize=16)
        plt.tight_layout()
        plt.savefig(out_dir+"{}.png".format(Variables[i]))
        plt.close()

        dt = Time_sampling[1] - Time_sampling[0]

        frq,FFT = temporal_spectra(h_vars[i],dt,Variables[i])

        fig = plt.figure(figsize=(14,8))
        plt.plot(frq, FFT)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Frequency [Hz]",fontsize=16)
        plt.ylabel("PSD - d{}/dt - Time derivative of {}".format(Variables[i], Ylabels[i]),fontsize=16)
        plt.tight_layout()
        plt.savefig(out_dir+"FFT_{}.png".format(Variables[i]))
        plt.close()


if plot_moving_stats == True:


    Variables = ["FBR","Iy","Iz"]
    units = ["[kN]","[$m^4/s$]","[$m^4/s$]"]
    h_vars = [FBR,Iy,Iz]

    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(14,8))
        
    i = 0
    for ax in axs.ravel():

        
        # some sample data
        if i == 2 or i == 1:
            ts = pd.Series(h_vars[i],Time_sampling)
            #plot the time series
            ts.plot(style='b-',ax=ax)
            ts.rolling(window=250).mean().plot(style='k--',ax=ax)
            ts.rolling(window=250).std().plot(style='r-',ax=ax)
        else:
            ts = pd.Series(h_vars[i], index=Time_OF)
            #plot the time series
            ts.plot(style='b-',ax=ax)
            ts.rolling(window=25000).mean().plot(style='k--',ax=ax)
            ts.rolling(window=25000).std().plot(style='r-',ax=ax)

        ax.set_title('{} {}'.format(Variables[i],units[i]))

        i+=1

    fig.supxlabel("Time [s]")
    plt.xlim(left=200)
    plt.tight_layout()
    plt.savefig(in_dir+"velocity_correlations_2/moving_stats.png")
    plt.close()


if systematic_LPF == True:

    Time_shift_idx = np.searchsorted(Time_OF,Time_OF[0]+5.5)

    IA = IA[:-Time_shift_idx]; RtAeroMR = RtAeroMR[Time_shift_idx:]
    Time_OF = Time_OF[Time_shift_idx:]

    Variables = ["IA","RtAeroMR"]
    units = ["[$m^4/s$]", "[kN-m]"]
    Ylabels = ["Asymmetry parameter", "Aerodynamic OOPBM"]
    h_vars = [IA, RtAeroMR/1000]
    cutoffs = [40,1,0.3]

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))
    fig.suptitle("Systematic LPF of IA and OOPBM", fontsize=18, y=0.95)

    # loop through tickers and axes
    corrs = []
    for cutoff, ax in zip(cutoffs, axs.ravel()):

        signal_LP_0 = low_pass_filter(h_vars[0], cutoff)
        signal_LP_1 = low_pass_filter(h_vars[1], cutoff)

        frq_0,FFT_0 = temporal_spectra(signal_LP_0,dt,Variables[0])
        frq_1,FFT_1 = temporal_spectra(signal_LP_1,dt,Variables[1])
        
        corr = correlation_coef(signal_LP_0,signal_LP_1)
        corr = round(corr,2)

        # filter df for ticker and plot on specified axes
        ax2=ax.twinx()
        ax.plot(frq_0,FFT_0,color="r")
        ax2.plot(frq_1,FFT_1,color="b")
        ax.set_xscale("log");ax.set_yscale("log")
        ax2.set_xscale("log");ax2.set_yscale("log")
        ax.legend(["IA"],loc="upper right",fontsize=12); ax2.legend(["OOPBM"],loc="center right",fontsize=12)
        ax.grid()
        

        # chart formatting
        ax.set_title("LPF at {}Hz, cc = {}".format(cutoff, corr),fontsize=14)
    fig.supxlabel("Frequency [Hz]",fontsize=16)
    plt.savefig(in_dir+"velocity_correlations_2/FFT_sys_LPF_corr_{}_{}.png".format(Variables[0],Variables[1]))
    plt.close(fig)

    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(15, 12))
    fig.suptitle("Systematic LPF of IA and OOPBM", fontsize=18, y=0.95)


    # loop through tickers and axes
    corrs = []
    for cutoff, ax in zip(cutoffs, axs.ravel()):

        signal_LP_0 = low_pass_filter(h_vars[0], cutoff)
        signal_LP_1 = low_pass_filter(h_vars[1], cutoff)
        
        corr = correlation_coef(signal_LP_0,signal_LP_1)
        corr = round(corr,2)

        # filter df for ticker and plot on specified axes
        ax2=ax.twinx()
        ax.plot(Time_OF,signal_LP_0,color="r")
        ax2.plot(Time_OF,signal_LP_1,color="b")
        ax.legend(["IA"],loc="upper right",fontsize=12); ax2.legend(["OOPBM"],loc="center right",fontsize=12)
        ax.grid()
        

        # chart formatting
        ax.set_title("LPF at {}Hz, cc = {}".format(cutoff, corr),fontsize=14)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.savefig(in_dir+"velocity_correlations_2/sys_LPF_corr_{}_{}.png".format(Variables[0],Variables[1]))
    plt.close(fig)

    