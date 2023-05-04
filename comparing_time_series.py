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

Vars = ["IA","Rt_OOPBM","Theta","RtAeroMxh", "Avg_rotor_Ux"]
units = ["[$m^4/s$]","[N-m]", "[deg]","[N-m]", "[m/s]"]
Ylabels = ["Asmmetry Parameter","Magnitude Out-of-plane bending moment", "Angle (0 deg - Y axis)", "Rotor Torque", 
           "Average Rotor velocity Ux'"]




def offset_data(p_h,no_cells_offset,i,velocity_comp,it):

    if velocity_comp =="coordinates":
        u = np.array(p_h.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_h.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice



def Avg_rotor_Ux(case,tstart,tend):
    
    sampling = glob.glob("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/sampling*".format(case))
    a = Dataset("./{}".format(sampling[0])) #check is right file
    p_h = a.groups["p_sw1"]


    time_sampling = np.array( p_h.variables["time"][:] )
    tstart_idx = np.searchsorted(time_sampling,tstart)
    tend_idx = np.searchsorted(time_sampling,tend)

    time_sampling = time_sampling[tstart_idx:tend_idx]

    time_idx = np.arange(tstart_idx,tend_idx,1)

    no_cells = len(p_h.variables["coordinates"])
    no_offsets = len(p_h.offsets)
    no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

    y = p_h.ijk_dims[0] #no. data points
    z = p_h.ijk_dims[1] #no. data points

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

    Ux_it = []
    IA_it = []
    for it in time_idx:
        velocityx = offset_data(p_h, no_cells_offset,it,j=2,velocity_comp="velocityx")
        velocityy = offset_data(p_h, no_cells_offset,it,j=2,velocity_comp="velocityy")

        hvelmag = np.add( np.multiply(velocityx,np.cos(np.radians(29))) , np.multiply( velocityy,np.sin(np.radians(29))) )
        hvelmag = hvelmag.reshape((z,y))

        f = interpolate.interp2d(ys,zs,hvelmag,kind="linear")

        Ux_rotor = []
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
        
        Ux_it.append(np.average(Ux_rotor))

        IA_it.append(IA)

    return Ux_it, IA_it, time_sampling



ix = 0
for case in cases:

    df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

    time = df["Time_[s]"]
    time = np.array(time)

    tstart = np.searchsorted(time[:],time_start[ix])
    tend = np.searchsorted(time[:],time_end[ix])

    fig = plt.figure()
    for i in np.arange(0,len(Vars)):
        Var = Vars[i]
        unit = units[i]
        Ylabel = Ylabels[i]
        
        if Var == "Rt_OOPBM" or Var == "Theta":
            txty = "{0}_{1}".format("RtAeroMyh",unit)
            txtz = "{0}_{1}".format("RtAeroMzh",unit)

            signaly = df[txty][tstart:tend]
            signalz = df[txtz][tstart:tend]

            if Var == "Rt_OOPBM":
                signal = np.sqrt( np.square(signaly) + np.square(signalz) )
            elif Var == "Theta":
                signal = np.arctan(np.true_divide(signalz,signaly))
        elif Var == "Avg_rotor_Ux" or Var == "IA":
            Ux_it, IA_it, time_sampling = Avg_rotor_Ux(case,tstart,tend)
        else:
            txt = "{0}_{1}".format(Var,unit)
            signal = df[txt][tstart:tend]

        
        plt.subplot(2, 1, i,sharex=True)
        if Var == "Avg_rotor_Ux":
            plt.plot(time_sampling,Ux_it)
        elif Var == "IA":
            plt.plot(time_sampling,IA_it) 
        else:
            plt.plot(time,signal)

        plt.ylabel("{0} {1}".format(Ylabel,unit))

    plt.xlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(dir+"{0}/post_processing/plots/joint_vars.png".format(case))
    plt.close(fig)

    


    ix+=1