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


dir = "./post_processing/plots/"
dt = 0.0039

tstart = 50 #time in seconds to remove from start of data - insert 0 if plot all time
tend = 350 #time in seconds to plot upto - insert False if plot all time

Vars = ["RtAeroFxh", "RtAeroMxh", "AB1N041Alpha","Rt_OOPBM"]
units = ["[N]", "[N-m]","[deg]", "[N-m]"]
Ylabels = ["Rotor Thrust", "Rotor Torque","Angle of Attack", "Magnitude Out-of-plane bending moment"]



def low_pass_filter(signal, cutoff):  

    fs = 1/dt     # sample rate, Hz      
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


#calculate at each offset averaged rotor velocity and asymmetry parameter
sampling = glob.glob("./post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]

time_sampling = p_rotor.variables["time"]
tstart_idx = np.searchsorted(time_sampling,tstart)
tend_idx = np.searchsorted(time_sampling,tend)

time_sampling = time_sampling[tstart_idx:tend_idx]

no_cells = len(p_rotor.variables["coordinates"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset
offsets = p_rotor.offsets

y = p_rotor.ijk_dims[0] #no. data points
z = p_rotor.ijk_dims[1] #no. data points

rotor_coordinates = np.array([2560,2560,90])
ly = 400
Oy = 2560 - ly/2

Oz = p_rotor.origin[2]
lz = p_rotor.axis2[2]

ys = np.linspace(Oy,Oy+ly,y) - rotor_coordinates[1]
zs = np.linspace(Oz,Oz+lz,z) - rotor_coordinates[2]

#create R,theta space over rotor
R = np.linspace(1.5,63,500)
Theta = np.arange(0,2*np.pi,(2*np.pi)/729)    

dR = R[1]-R[0]
dTheta = Theta[1] - Theta[0]
dA = (dTheta/2)*(dR**2)

avg_rotor_it_offset = []
IA_it_offset = []
for i in np.arange(0,no_offsets):
    avg_rotor_it = []
    IA_it = []
    for it in np.arange(tstart_idx,tend_idx):

        velocityx = offset_data(p_rotor, no_cells_offset,i,it,velocity_comp="velocityx")
        velocityy = offset_data(p_rotor, no_cells_offset,i,it,velocity_comp="velocityy")

        hvelmag = np.add( np.multiply(velocityx,np.cos(np.radians(29))) , np.multiply( velocityy,np.sin(np.radians(29))) )
        hvelmag = hvelmag.reshape((z,y))

        f = interpolate.interp2d(ys,zs,hvelmag,kind="linear")

        Ux_rotor = []
        IA = 0
        for r in R:
            for theta in Theta:

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
        
        avg_rotor_it.append(np.average(Ux_rotor))
        IA_it.append(IA)

    avg_rotor_it_offset.append(avg_rotor_it)
    IA_it_offset.append(IA_it)



for i in np.arange(0,len(Vars)):

    fig,ax = plt.subplots(figsize=(14,8))
    Var = Vars[i]
    unit = units[i]
    Ylabel = Ylabels[i]

    df = io.fast_output_file.FASTOutputFile("./NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

    time = df["Time_[s]"]
    time = np.array(time)

    
    if Var == "Rt_OOPBM":
        txty = "{0}_{1}".format("RtAeroMyh",unit)
        txtz = "{0}_{1}".format("RtAeroMzh",unit)

        signaly = df[txty][tstart_idx:tend_idx]
        signalz = df[txtz][tstart_idx:tend_idx]

        signal = np.sqrt( np.square(signaly) + np.square(signalz) )

        cutoff = 0.5*(12.1/60)
    elif Var == "AB1N041Alpha":
        txt = "{0}_{1}".format(Var,unit)
        signal = df[txt][tstart_idx:tend_idx]

        cutoff = 0.5*(12.1/60)            
    else:
        txt = "{0}_{1}".format(Var,unit)
        signal = df[txt][tstart_idx:tend_idx]

        cutoff = 0.5*(12.1/60)*3
    

    low_pass_signal = low_pass_filter(signal, cutoff)



    for j in np.arange(0,no_offsets):
        if Var == "AB1N041Alpha":
            corr, _ = pearsonr(avg_rotor_it_offset[j], low_pass_signal)
        elif Var == "Rt_OOPBM":
            corr, _ = pearsonr(IA_it_offset[j],signal)
        else:
            corr, _ = pearsonr(avg_rotor_it_offset[j], signal)


        ax.plot(time[tstart_idx:tend_idx],signal,'-b')
        ax.plot(time[tstart_idx:tend_idx],low_pass_signal,"-r")
        ax.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)

        ax2=ax.twinx()
        if Var == "Rt_OOPBM":
            ax2.plot(time_sampling,IA_it_offset[j],"--k")
            ax2.set_ylabel("IA' - Asymmetry Parameter [$m^4/s$]",fontsize=16)
            plt.title("Correlating IA' at {0}m from turbine, with {1}".format(offsets[j],Ylabel),fontsize=18)
            ax.legend(["-","Correlation with IA' = {0}".format(np.round(corr,2))])
        else:
            ax2.plot(time_sampling,avg_rotor_it_offset[j],"--k")
            ax2.set_ylabel("Ux' - Rotor normal Velocity [m/s]",fontsize=16)
            plt.title("Correlating Ux' at {0}m from turbine, with {1}".format(offsets[j],Ylabel),fontsize=18)
            ax.legend(["-","Correlation with Ux' = {0}".format(np.round(corr,2))])

        ax.set_xlabel("time [s]",fontsize=16)
        plt.tight_layout()
        plt.savefig(dir+"corr_{0}_{1}.png".format(offsets[j],Var))
        plt.close(fig)

        #comparing time series
        fig, (ax1, ax2, ax3, ax4, ax5,ax6) = plt.subplots(6,figsize=(14,8))
        ax1.plot(time[tstart_idx:tend_idx], avg_rotor_it_offset[-1])
        ax1.set_ylabel("Ux' - Rotor normal Velocity [m/s]")
        ax2.plot(time[tstart_idx:tend_idx], df["RtAeroFxh_[N]"][tstart_idx:tend_idx])
        ax2.set_ylabel("Rotor Thrust [N]")
        ax3.plot(time[tstart_idx:tend_idx], df["RtAeroMxh_[N-m]"][tstart_idx:tend_idx])
        ax3.set_ylabel("Rotor torque [N-m]")
        ax4.plot(time_sampling,IA_it_offset[-1])
        ax4.set_ylabel("Asymmetry Parameter [$m^4/s$]")

        txty = "{0}_{1}".format("RtAeroMyh",unit)
        txtz = "{0}_{1}".format("RtAeroMzh",unit)

        signaly = df["RtAeroMyh_[N-m]"][tstart_idx:tend_idx]
        signalz = df["RtAeroMzh_[N-m]"][tstart_idx:tend_idx]

        MR = np.sqrt( np.square(signaly) + np.square(signalz) )
        ax5.plot(time[tstart_idx:tend_idx],MR)
        ax5.set_ylabel("Magnitude Out-of-plane bending moment [N-m]")

        Theta_OOP = np.arctan(np.true_divide(signalz,signaly))
        ax6.plot(time[tstart_idx:tend_idx],Theta_OOP)
        ax6.set_ylabel("Theta [Radians]")

        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.savefig("./post_processing/plots/joint_vars.png")
        plt.close(fig)

