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



def average_velocity_asymmetry(tstart, tend,case): #average velocity into vertical and streamwise components over rotor disk

        
    sampling = glob.glob("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/sampling*".format(case))
    a = Dataset("./{}".format(sampling[0])) #script for correct sampling file
    p_h = a.groups["p_sw1"]

    no_cells = len(p_h.variables["coordinates"])
    no_offsets = len(p_h.offsets)
    no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

    offsets = p_h.offsets

    y = p_h.ijk_dims[0] #no. data points
    z = p_h.ijk_dims[1] #no. data points

    time_sampling = np.array( p_h.variables["time"][:] )
    tstart_idx = np.searchsorted(time_sampling,tstart)
    tend_idx = np.searchsorted(time_sampling,tend)

    time_sampling = time_sampling[tstart_idx:tend_idx]

    time_idx = np.arange(tstart_idx,tend_idx,1)

    rotor_coordinates = np.array([2560,2560,90])
    ly = 400
    Oy = 2560 - ly/2

    Oz = p_h.origin[2]
    lz = p_h.axis2[2]

    ys = np.linspace(Oy,Oy+ly,y) - rotor_coordinates[1]
    zs = np.linspace(Oz,Oz+lz,z) - rotor_coordinates[2]


    #create R,theta space over rotor
    R = np.linspace(0,63,500)
    Theta = np.arange(0,2*np.pi,(2*np.pi)/729)

    dR = R[1]-R[0]
    dTheta = Theta[1] - Theta[0]
    dA = (dTheta/2)*(dR**2)


    avg_rotor_field_offset = []
    for j in np.arange(0,no_offsets):
        IA_it = []
        avg_rotor_field_it = []
        for it in time_idx:

            velocityx = offset_data(p_h, no_cells_offset,j,it,velocity_comp="velocityx")
            velocityy = offset_data(p_h, no_cells_offset,j,it,velocity_comp="velocityy")

            hvelmag = np.add( np.multiply(velocityx,np.cos(np.radians(29))) , np.multiply( velocityy,np.sin(np.radians(29))) )
            hvelmag = hvelmag.reshape((z,y))

            f = interpolate.interp2d(ys,zs,hvelmag,kind="linear")

            IA = 0
            Ux_rotor = []
            for r in R:
                for theta in Theta:
                    
                    #average velocity
                    Y = r*np.cos(theta)
                    Z = r*np.sin(theta)

                    Ux =  f(Y,Z)

                    Ux_rotor.append(Ux)


                    #asymmetry
                    Y_0 = r*np.cos(theta)
                    Z_0 = r*np.sin(theta)

                    if theta + ((2*np.pi)/3) > (2*np.pi):
                        theta_1 = theta +(2*np.pi)/3 - (2*np.pi)
                    else:
                        theta_1 = theta + (2*np.pi)/3

                    Y_1 = r*np.cos(theta_1)
                    Z_1 = r*np.sin(theta_1)


                    if theta - ((2*np.pi)/3) < 0:
                        theta_2 = theta - ((2*np.pi)/3) + (2*np.pi)
                    else:
                        theta_2 = theta - ((2*np.pi)/3)

                    Y_2 = r*np.cos(theta_2)
                    Z_2 = r*np.sin(theta_2)

                    Ux_0 =  f(Y_0,Z_0)
                    Ux_1 =  f(Y_1,Z_1)
                    Ux_2 =  f(Y_2,Z_2)

                    delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

                    IA += r * delta_Ux * dA
            
            avg_rotor_field_it.append(np.average(Ux_rotor))

            IA_it.append(IA)

        avg_rotor_field_offset.append(avg_rotor_field_it)

    return avg_rotor_field_offset, no_offsets,offsets, time_sampling,IA_it


            
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

        Ux_offset_it,no_offsets,offsets,time_sampling,IA_it = average_velocity_asymmetry(tstart, tend,case) #unit test with data

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



        for j in np.arange(0,no_offsets):
            if Var == "AB1N041Alpha":
                corr, _ = pearsonr(Ux_offset_it[j], low_pass_signal)
            else:
                corr, _ = pearsonr(Ux_offset_it[j], signal)


            ax.plot(time[tstart:tend],signal,'-b')
            ax.plot(time[tstart:tend],low_pass_signal,"-r")
            ax2=ax.twinx()
            ax2.plot(time_sampling,Ux_offset_it[j],"--k")
            ax.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)
            ax2.set_ylabel("Ux [m/s]",fontsize=16)
            ax.set_xlabel("time [s]",fontsize=16)
            plt.title("Correlating Ux at {0}m from turbine, with {1}".format(offsets[j],Ylabel),fontsize=18)
            ax.legend(["-","Correlation with Ux = {0}".format(np.round(corr,2))])
            plt.tight_layout()
            plt.savefig(dir+"{0}/post_processing/plots/corr_{1}_{2}.png".format(case,offsets[j],Var))
            plt.close(fig)


        ix+=1 

