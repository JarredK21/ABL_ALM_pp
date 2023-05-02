import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
import pyFAST.postpro as postpro
from scipy.fft import fft, fftfreq, fftshift
import pandas as pd
from scipy import interpolate
import math
from scipy.signal import butter,filtfilt
from scipy.stats import pearsonr

dir = "../../../jarred/ALM_sensitivity_analysis/"
cases = ["Ex1"]
dt_cases = [0.001]

time_start = [10] #time in seconds to remove from start of data - insert 0 if plot all time
time_end = [100] #time in seconds to plot upto - insert False if plot all time

Vars = ["RtAeroFxh", "RtAeroMxh", "AB1N041Alpha","Rt_OOPBM"]
units = ["[N]", "[N-m]","[deg]", "[N-m]"]
Ylabels = ["Rotor Thrust", "Rotor Torque","Angle of Attack", "Magnitude Out-of-plane bending moment"]



def low_pass_filter(signal, ix,cutoff):  

    fs = 1/dt_cases[ix]       # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal




def offset_data(p_h,no_cells_offset,i,velocity_comp,it):

    if velocity_comp =="coordinates":
        u = np.array(p_h.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_h.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice



def average_velocity(df, tstart, tend): #average velocity into vertical and streamwise components over rotor disk

    Ux = df["Wind1VelX_[m/s]"][tstart:tend]

    return Ux
    


            
for i in np.arange(0,len(Vars)):

    #fig = plt.figure(figsize=(14,8))
    fig,ax = plt.subplots(figsize=(14,8))
    Var = Vars[i]
    unit = units[i]
    Ylabel = Ylabels[i]
    ix = 0
    for case in cases:

        df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

        time = df["Time_[s]"]
        time = np.array(time)

        tstart = np.searchsorted(time[:],time_start[ix])
        tend = np.searchsorted(time[:],time_end[ix])

        Ux = average_velocity(df, tstart, tend)

        if Var == "Rt_OOPBM":
            txty = "{0}_{1}".format("RtAeroMyh",unit)
            txtz = "{0}_{1}".format("RtAeroMzh",unit)

            signaly = df[txty][tstart:tend]
            signalz = df[txtz][tstart:tend]

            signal = np.sqrt( np.square(signaly) + np.square(signalz) )

            cutoff = 0.5*(12.1/60)
        elif Var == "AB1N041Alpha":
            txt = "{0}_{1}".format(Var,unit)
            signal = df[txt][tstart:tend]

            cutoff = 0.5*(12.1/60)            
        else:
            txt = "{0}_{1}".format(Var,unit)
            signal = df[txt][tstart:tend]

            cutoff = 0.5*(12.1/60)*3
        

        low_pass_signal = low_pass_filter(signal, ix, cutoff)


        corr, _ = pearsonr(Ux, low_pass_signal)


        ax.plot(time[tstart:tend],signal,'-b')
        ax.plot(time[tstart:tend],low_pass_signal,"-r")
        ax2=ax.twinx()
        ax2.plot(time[tstart:tend],Ux,"--k")


        ix+=1 

    ax.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)
    ax2.set_ylabel("Ux [m/s]",fontsize=16)
    ax.set_xlabel("time [s]",fontsize=16)
    plt.title("5 levels of refinement, 54 actuator points",fontsize=18)
    ax.legend(["-","Correlation with Ux = {0}".format(np.round(corr,2))])
    plt.tight_layout()
    plt.savefig(dir+"{0}/post_processing/plots/low_pass_filtered_{1}.png".format(case,Var))
    plt.close(fig)