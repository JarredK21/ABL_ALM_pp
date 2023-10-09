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


in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

a = Dataset(in_dir+"OF_Dataset.nc")

#plotting options
compare_variables = True
compare_FFT = True
plot_relative_contributions = True


out_dir = in_dir + "Bearing_loads/"

Time_OF = np.array(a.variables["time_OF"])

Time_start = 200
Time_end = 1900
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

L1 = 1.912; L2 = 5

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz

FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Rel_FBy = np.true_divide(np.square(FBy),np.square(FBR))
Rel_FBz = np.true_divide(np.square(FBz),np.square(FBR))
add_RelFB = np.add(Rel_FBy,Rel_FBz)
Theta_FB = np.degrees(np.arctan2(FBz,FBy))




if compare_variables == True:

    Variables = ["Radial Bearing Force components", "Y Bearing Force components", "Z Bearing Force components"]
    units = [["[kN]", "[kN]", "[kN]"], ["[kN]", "[kN]", "[kN]"], ["[kN]", "[kN]", "[kN]"]]
    Ylabels = [["Bearing Force y direction", "Bearing Force z direction", "Magnitude Bearing Force"], 
               ["Force due to the moment contribution \nto the Bearing Force y direction",
                "Force due to the force contribution \nto the Bearing Force y direction","Bearing Force y direction"],
               ["Force due to the moment contribution \nto the Bearing Force z direction",
                "Force due to the force contribution \nto the Bearing Force z direction","Bearing Force z direction"]]
    h_vars = [[FBy, FBz, FBR], [FBMy, FBFy, FBy], [FBMz, FBFz, FBz]]

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
        plt.savefig(in_dir+"Bearing_Loads/{}.png".format(Variables[i]))


if compare_FFT == True:
    Variables = ["Radial Bearing Force components", "Y Bearing Force components", "Z Bearing Force components"]
    units = [["[kN]", "[kN]", "[kN]"], ["[kN]", "[kN]", "[kN]"], ["[kN]", "[kN]", "[kN]"]]
    Ylabels = [["Bearing Force y direction", "Bearing Force z direction", "Magnitude Bearing Force"], 
               ["Force due to the moment contribution \nto the Bearing Force y direction",
                "Force due to the force contribution \nto the Bearing Force y direction","Bearing Force y direction"],
               ["Force due to the moment contribution \nto the Bearing Force z direction",
                "Force due to the force contribution \nto the Bearing Force z direction","Bearing Force z direction"]]
    h_vars = [[FBy, FBz, FBR], [FBMy, FBFy, FBy], [FBMz, FBFz, FBz]]

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
        plt.savefig(in_dir+"Bearing_Loads/FFT_{}.png".format(Variables[i]))
        plt.close()


if plot_relative_contributions == True:

    h_vars = [[Rel_FBy, Rel_FBz, FBR, Theta_FB]]
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
        plt.savefig(in_dir+"Bearing_Loads/Relative_{}.png".format(Variable))

