import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
import pandas
import pyFAST.input_output as io

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


def start_stop_idx(y):

    for i in np.arange(1,len(Time)-1):

        if y[-i] == y[0]:
            break
    
    return i



in_dir = "../../NREL_5MW_3.4.1/Steady_Rigid_blades_shear_0.085/"

df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

Time = np.array(df["Time_[s]"])

Time_start = 10
Time_start_idx = np.searchsorted(Time,Time_start)

Time = Time[Time_start_idx:]

dt = Time[1] - Time[0]

Azimuth = np.array(df["Azimuth_[deg]"][Time_start_idx:])
Azimuth = np.radians(Azimuth)

RtAeroFyh = np.array(df["RtAeroFyh_[N]"][Time_start_idx:])
RtAeroFzh = np.array(df["RtAeroFzh_[N]"][Time_start_idx:])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)


RtAeroMyh = np.array(df["RtAeroMyh_[N-m]"][Time_start_idx:])
RtAeroMzh = np.array(df["RtAeroMzh_[N-m]"][Time_start_idx:])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 


LSSTipMys = np.array(df["LSSTipMys_[kN-m]"][Time_start_idx:])
LSSTipMzs = np.array(df["LSSTipMzs_[kN-m]"][Time_start_idx:])
LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

LSShftFys = np.array(df["LSShftFys_[kN]"][Time_start_idx:])
LSShftFzs = np.array(df["LSShftFzs_[kN]"][Time_start_idx:])

L = 1.912

Fy_add = np.subtract(LSShftFys,RtAeroFys/1000)
Fz_add = np.subtract(LSShftFzs,RtAeroFzs/1000)
My_add = np.subtract(LSSTipMys,RtAeroMys/1000)
Mz_add = np.subtract(LSSTipMzs,RtAeroMzs/1000)
MR_add = np.subtract(LSSTipMR,RtAeroMR/1000)

L1 = 1.912; L2 = 2.09

Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz
FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))

idx = len(Time)-1

vel_profile = [df["Wind1VelX_[m/s]"][0], df["Wind2VelX_[m/s]"][0], df["Wind3VelX_[m/s]"][0], df["Wind4VelX_[m/s]"][0], df["Wind5VelX_[m/s]"][0], 
               df["Wind6VelX_[m/s]"][0], df["Wind7VelX_[m/s]"][0], df["Wind8VelX_[m/s]"][0], df["Wind9VelX_[m/s]"][0]]
z = [7.5, 22.5, 37.5, 52.5, 82.5, 97.5, 112.5, 127.5, 157.5]


a = Dataset("../../ABL_precursor/post_processing/abl_statistics60000.nc")
mean_profiles = a.groups["mean_profiles"]
z_h = np.array(mean_profiles.variables["h"])
hvelmag = np.array(mean_profiles.variables["hvelmag"])

hvelmag = np.average(hvelmag,axis=0)

z_h = z_h[:11]
hvelmag = hvelmag[:11]


plt.plot(vel_profile,z,"bo-")
plt.plot(hvelmag,z_h,"r*-")
plt.axhline(90-63,linestyle="--",color="k"); plt.axhline(90+63,linestyle="--",color="k")
plt.xlabel("Horizontal velocity [m/s]")
plt.ylabel("Height from surface [m]")
plt.legend(["OpenFast profile", "ABL profile"])
plt.savefig(in_dir+"plots/velocity_profile.png")
plt.close()

plot_variables = True
plot_PDF = True
plot_FFT = True

if plot_variables == True:
    Variables = ["RtAeroFys", "RtAeroFzs", "RtAeroMys", "RtAeroMzs", "RtAeroMR", 
                    "LSShftFys","LSShftFzs", "LSSTipMys", "LSSTipMzs", "LSSTipMR",
                    "FBy", "FBz", "FBR"]
    units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN]"]
    Ylabels = ["Rotor Aerodynamic Force y direction fixed frame of reference","Rotor Aerodynamic Force z direction fixed frame of reference",
                "Rotor Aerodynamic Moment y direction fixed frame of reference", "Rotor Aerodynamic Moment z direction fixed frame of reference",
                "Rotor Aerodynamic OOPBM fixed frame of reference",
                "Rotor Aeroelastic Force y direction fixed frame of reference", "Rotor Aeroelastic Force z direction fixed frame of reference",
               "Rotor Aeroelastic Moment y direction fixed frame of reference",
                "Rotor Aeroelastic Moment z direction fixed frame of reference","Rotor Aeroelastic OOPBM fixed frame of reference",
                "LSS Aeroelastic Moment y direction fixed frame of reference","LSS Aeroelastic Moment z direction fixed frame of reference",
                "Bearing Force y direction", "Bearing Force z direction", "Bearing Force"]
    h_vars = [RtAeroFys/1000, RtAeroFzs/1000, RtAeroMys/1000, RtAeroMzs/1000, RtAeroMR/1000, LSShftFys,
                LSShftFzs, LSSTipMys, LSSTipMzs, LSSTipMR, Aero_FBy/1000, Aero_FBz/1000, FBR/1000]

    for i in np.arange(0,len(h_vars)):
        signal_LP = low_pass_filter(h_vars[i],cutoff=3)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,signal_LP)
        plt.axhline(np.mean(h_vars[i]),linestyle="--",color="k")
        plt.xlabel("Time [s]",fontsize=16)
        plt.ylabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
        plt.tight_layout()
        plt.savefig(in_dir+"plots/LPF_{0}".format(Variables[i]))
        plt.close()


if plot_PDF == True:

    Variables = ["RtAeroFys", "RtAeroFzs", "RtAeroMys", "RtAeroMzs", "RtAeroMR", 
                    "LSShftFys","LSShftFzs", "LSSTipMys", "LSSTipMzs", "LSSTipMR",
                    "FBy", "FBz", "FBR"]
    units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN]"]
    Ylabels = ["Rotor Aerodynamic Force y direction fixed frame of reference","Rotor Aerodynamic Force z direction fixed frame of reference",
                "Rotor Aerodynamic Moment y direction fixed frame of reference", "Rotor Aerodynamic Moment z direction fixed frame of reference",
                "Rotor Aerodynamic OOPBM fixed frame of reference",
                "Rotor Aeroelastic Force y direction fixed frame of reference", "Rotor Aeroelastic Force z direction fixed frame of reference",
               "Rotor Aeroelastic Moment y direction fixed frame of reference",
                "Rotor Aeroelastic Moment z direction fixed frame of reference","Rotor Aeroelastic OOPBM fixed frame of reference",
                "LSS Aeroelastic Moment y direction fixed frame of reference","LSS Aeroelastic Moment z direction fixed frame of reference",
                "Bearing Force y direction", "Bearing Force z direction", "Bearing Force"]
    
    h_vars = [RtAeroFys/1000, RtAeroFzs/1000, RtAeroMys/1000, RtAeroMzs/1000, RtAeroMR/1000, LSShftFys,
                LSShftFzs, LSSTipMys, LSSTipMzs, LSSTipMR, Aero_FBy/1000, Aero_FBz/1000, FBR/1000]

    for i in np.arange(0,len(h_vars)):

        signal_LP = low_pass_filter(h_vars[i],cutoff=3)

        P,X,mu,std,S,k = probability_dist(signal_LP)

        txt = "mean = {0}{1}\nstandard deviation = {2}{1}".format(mu,units[i],std)
        print(Variables[i], txt)
        fig = plt.figure(figsize=(14,8))
        plt.plot(X,P)
        plt.ylabel("Probability",fontsize=16)
        plt.xlabel("{0} {1}".format(Ylabels[i],units[i]),fontsize=16)
        plt.text(np.max(X)-0.1*np.max(X),np.max(P)-0.1*np.max(P),txt,horizontalalignment="right",verticalalignment="top",fontsize=12)
        plt.tight_layout()
        plt.savefig(in_dir+"plots/LPF_PDF_{0}".format(Variables[i]))
        plt.close()

if plot_FFT == True:
    Variables = ["RtAeroFys", "RtAeroFzs", "RtAeroMys", "RtAeroMzs", "RtAeroMR", 
                    "LSShftFys","LSShftFzs", "LSSTipMys", "LSSTipMzs", "LSSTipMR",
                    "FBy", "FBz", "FBR"]
    units = ["[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN]"]
    Ylabels = ["Rotor Aerodynamic Force y direction fixed frame of reference","Rotor Aerodynamic Force z direction fixed frame of reference",
                "Rotor Aerodynamic Moment y direction fixed frame of reference", "Rotor Aerodynamic Moment z direction fixed frame of reference",
                "Rotor Aerodynamic OOPBM fixed frame of reference",
                "Rotor Aeroelastic Force y direction fixed frame of reference", "Rotor Aeroelastic Force z direction fixed frame of reference",
            "Rotor Aeroelastic Moment y direction fixed frame of reference",
                "Rotor Aeroelastic Moment z direction fixed frame of reference","Rotor Aeroelastic OOPBM fixed frame of reference",
                "LSS Aeroelastic Moment y direction fixed frame of reference","LSS Aeroelastic Moment z direction fixed frame of reference",
                "Bearing Force y direction", "Bearing Force z direction", "Bearing Force"]
    
    h_vars = [RtAeroFys/1000, RtAeroFzs/1000, RtAeroMys/1000, RtAeroMzs/1000, RtAeroMR/1000, LSShftFys,
                LSShftFzs, LSSTipMys, LSSTipMzs, LSSTipMR, Aero_FBy/1000, Aero_FBz/1000, FBR/1000]

    for i in np.arange(0,len(h_vars)):
        frq,PSD = temporal_spectra(h_vars[i],dt,Variables[i])

        fig = plt.figure()
        plt.plot(frq, PSD)
        plt.axvline(0.6)
        plt.xscale("log")
        plt.yscale("log")
        plt.ylabel("{0}{1}".format(Ylabels[i], units[i]))
        plt.xlabel("Frequency [Hz]",fontsize=14)
        plt.tight_layout()
        plt.savefig(in_dir+"plots/FFT_{}.png".format(Variables[i]))
        plt.close()