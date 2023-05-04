#from pyFAST.input_output import FASTOutputFile
#import pyFAST.input_output as io
import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
import pyFAST.postpro as postpro
from scipy.fft import fft, fftfreq, fftshift
import math


# cases = ["Ex1","Ex2","Ex3"]
# act_stations_cases = [54,47,59]
# dt_cases = [0.001,0.001,0.001]
# Titles = ["5 levels of refinement, no. actuator points = 54","5 levels of refinement, no. actuator points = 47",
#           "5 levels of refinement, no. actuator points = 59"]

# cases = ["Ex4","Ex5"]
# act_stations_cases = [19,19]
# dt_cases = [0.001,0.001]
# Titles = ["54 actuator points, no. levels of refinement = 4","54 actuator points, no. levels of refinement = 6"]

cases = ["Ex1","Ex1_dblade_1.0","Ex1_dblade_2.0","test3","test2"]
act_stations_cases = [54,54,54,94,94]
dt_cases = [0.001,0.0039,0.0078,0.0039,0.0078]

Titles = []
for i in np.arange(0,len(dt_cases)):
    Titles.append("54 actuator points, no. levels of refinement = 5: {0}s dt".format(dt_cases[i]))

#plotting options
plot_radial = False
plot_int = True
plot_spectra = False

ix = 0
for case in cases:

    dir = "../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/plots/".format(case)

    df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

    # for col in df.columns:
    #     print(col)

    time = df["Time_[s]"]
    time = np.array(time)

    Az = np.array(df["Azimuth_[deg]"])

    rotspeed = np.array(df["RotSpeed_[rpm]"])

    act_stations = act_stations_cases[ix]
    x = np.linspace(0,1,act_stations)


    def integrated_variable(Var,unit,time_start,time_end,Ylabel):

        txt = "{0}_{1}".format(Var,unit)

        Var_int = df[txt][:]

        tstart = np.searchsorted(time[:],time_start[ix])
        tend = np.searchsorted(time[:],time_end[ix])

        fig = plt.figure(figsize=(14,8))
        plt.plot(time[tstart:tend],Var_int[tstart:tend])
        plt.ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)
        plt.xlabel("time [s]",fontsize=16)
        plt.title(Titles[ix],fontsize=18)
        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Var))
        plt.close(fig)

        


    def radial_dist_variable(Var,unit, no_rots,YLabel):

        Var_list = []
        for i in np.arange(1,act_stations+1):
            if i < 10:
                txt = "AB1N00{0}{1}_{2}".format(i,Var,unit)
            elif i >= 10:
                txt = "AB1N0{0}{1}_{2}".format(i,Var,unit)

            Az_0 = Az[-1] - no_rots * 360
            tstart_ind = np.searchsorted(Az,Az_0)

            Var_dist = np.average(df[txt][tstart_ind:])
            Var_list.append(Var_dist)

        #plotting
        fig = plt.figure()
        plt.plot(x,Var_list,marker="o",markersize=6)
        plt.ylabel("{0} {1}".format(YLabel,unit),fontsize=16)
        plt.xlabel("Normalized blade radius [-]",fontsize=16)
        plt.title("Averaged over last {0} blade rotations\n".format(no_rots) + Titles[ix],fontsize=12)
        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Var))
        plt.close(fig)

    
    def temporal_spectra(Var,unit,time_start,time_end,Ylabel):
        
        txt = "{0}_{1}".format(Var,unit)

        tstart = np.searchsorted(time[:],time_start[ix])
        tend = np.searchsorted(time[:],time_end[ix])

        signal = np.array(df[txt][tstart:tend])


        m=0
        fs =1/dt_cases[ix]
        n = len(signal) 
        if n%2==0:
            nhalf = int(n/2+1)
        else:
            nhalf = int((n+1)/2)
        frq = np.arange(nhalf)*fs/n
        Y   = np.fft.fft(signal-m) #Y = np.fft.fft(y) 
        PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
        PSD[1:-1] = PSD[1:-1]*2



        fig = plt.figure(figsize=(14,8))
        plt.loglog(frq, PSD)
        plt.xlabel('Frequency (1/s)',fontsize=16)
        plt.ylabel('Power spectral density',fontsize=16)
        plt.title('{0} spectra, {1}'.format(Ylabel,Titles[ix]))
        plt.grid()
        plt.tight_layout()
        plt.savefig(dir+"{0}_spectra.png".format(Var))
        plt.close(fig)



    rad_variables = ["Vrel","Alpha", "Cl","Cd","Fn","Ft","Vx"]
    rad_YLabel = ["Local Relative Velocity", "Local Angle of Attack", "Local Coeffcient of Lift", "Local Coefficient of Drag",
                  "Local Aerofoil Normal Force", "Local Aerofoil Tangential Force", "Local Axial Velocity"]
    rad_units = ["[m/s]","[deg]","[-]","[-]","[N/m]","[N/m]","[m/s]"]
    number_rotor_rotations = 3

    if plot_radial == True:
        for i in np.arange(0,len(rad_variables),1):

            radial_dist_variable(rad_variables[i],rad_units[i], number_rotor_rotations,rad_YLabel[i])

    time_start = [10,10,10,10,10] #time in seconds to remove from start of data - insert 0 if plot all time
    time_end = [100,24,24,24,24]
    int_variables = ["RotSpeed","BldPitch1","Wind1VelX","RotTorq","RtAeroFxh","RtAeroFyh","RtAeroFzh","RtAeroMxh","RtAeroMyh","RtAeroMzh"]
    int_YLabel = ["Rotor speed","Blade pitch","Hub height Velocity", "Rotor Torque", "Rotor Force in X direction", "Rotor Force in Y direction", 
                  "Rotor Force in Z direction", "Rotor Moment in X direction", "Rotor Moment in Y direction", 
                  "Rotor Moment in Z direction"]
    int_units = ["[rpm]","[deg]","[m/s]","[kN-m]","[N]","[N]","[N]","[N-m]","[N-m]","[N-m]"]

    for i in np.arange(0,len(int_variables),1):

        if plot_int == True:
            integrated_variable(int_variables[i],int_units[i],time_start,time_end,int_YLabel[i])

        if plot_spectra == True:
            temporal_spectra(int_variables[i],int_units[i],time_start,time_end,int_YLabel[i])

    ix+=1
