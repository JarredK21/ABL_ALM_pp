import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy.stats import pearsonr
import glob 
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
import math


dir = "../../../jarred/NAWEA_23/post_processing/plots/"

offsets = [-63.0, -31.5, 0.0]

Variables = ["Time_OF","Time_sample","Ux_{}".format(offsets[2]),"IA_{}".format(offsets[2]),"RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[s]","[s]", "[m/s]","[m^4/s]","[N]","[N-m]","[N-m]","[rads]"]
Ylabels = ["Time","Time","Ux' rotor averaged velocity","Asymmery Parameter","Rotor Thrust", "Rotor Torque",
            "Magnitude Out-of-plane bending moment","Angle Out-of-plane bending moment"]



def low_pass_filter(signal, cutoff):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def remove_nan(Var):

    signal = df[Var]
    new_signal = list()
    for element in signal:
        if not math.isnan(element):
            new_signal.append(element)
    return new_signal

for i in np.arange(4,len(Variables)-1):

    fig,ax = plt.subplots(figsize=(14,8))
    Var = Variables[i]
    unit = units[i]
    Ylabel = Ylabels[i]

    df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out2.csv")

    time_OF = remove_nan("Time_OF")
    time_sample = remove_nan("Time_sample")
    time_sample[0] = time_OF[0]
    time_sample[-1] = time_OF[-1]

    dt = time_OF[1] - time_OF[0]

    Ux = remove_nan(Var = "Ux_{}".format(offsets[2]))
    IA = remove_nan(Var = "IA_{}".format(offsets[2]))

    # f = interpolate.interp1d(time_sample, Ux)
    # Ux = f(time_OF)

    f = interpolate.interp1d(time_sample,IA)
    IA = f(time_OF)

    Theta = remove_nan(Var = "Theta")

    signal = remove_nan(Var)

    if Var == "MR":
        cutoff = 0.5*(12.1/60)
        signal_LP = low_pass_filter(signal,cutoff)
        corr, _ = pearsonr(IA,signal)
    else:
        cutoff = 0.5*(12.1/60)*3
        signal_LP = low_pass_filter(signal,cutoff)
        corr, _ = pearsonr(Ux, signal)


    ax.plot(time_OF,signal,'-b')
    ax.plot(time_OF,signal_LP,"-r")
    ax.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)

    ax2=ax.twinx()
    if Var == "MR":
        ax2.plot(time_OF,IA,"--k")
        ax2.set_ylabel("IA' - Asymmetry Parameter [$m^4/s$]",fontsize=16)
        plt.title("Correlating IA' at {0}m from turbine, with {1}".format(offsets[2],Ylabel),fontsize=18)
        ax.legend(["-","Correlation with IA' = {0}".format(np.round(corr,2))])
    else:
        ax2.plot(time_OF,Ux,"--k")
        ax2.set_ylabel("Ux' - Rotor averaged normal Velocity [m/s]",fontsize=16)
        plt.title("Correlating Ux' at {0}m from turbine, with {1}".format(offsets[2],Ylabel),fontsize=18)
        ax.legend(["-","Correlation with Ux' = {0}".format(np.round(corr,2))])

    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.savefig(dir+"corr_{0}_{1}.png".format(offsets[2],Var))
    plt.close(fig)



#comparing time series
fig, axs = plt.subplots(6,1,figsize=(32,24))
plt.rcParams.update({'font.size': 16})
for i in np.arange(2,len(Variables)):

    Var = Variables[i]
    unit = units[i]
    Ylabel = Ylabels[i]

    df = pd.read_csv("../../../jarred/NAWEA_23/post_processing/out2.csv")

    time_OF = remove_nan("Time_OF")
    time_sample = remove_nan("Time_sample")
    time_sample[0] = time_OF[0]
    time_sample[-1] = time_OF[-1]

    dt = time_OF[1] - time_OF[0]

    signal = remove_nan(Var)

    if Var[0:2] == "IA":
        f = interpolate.interp1d(time_sample, signal)
        signal = f(time_OF)

    
    axs = axs.ravel()

    j=i-2

    axs[j].plot(time_OF,signal)
    axs[j].set_title("{0}{1}".format(Ylabels[i],units[i]),fontsize=18)

fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(dir+"joint_vars.png")
plt.close(fig)

