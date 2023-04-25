#from pyFAST.input_output import FASTOutputFile
#import pyFAST.input_output as io
import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
import pyFAST.postpro as postpro
from scipy.fft import fft, fftfreq, fftshift
import pandas as pd
from scipy import interpolate
import math


cases = ["Ex1","Ex1.2"]
act_stations = 54
dt = 0.001

dir = "{0}/post_processing/plots/".format(cases[-1])


rad_variables = ["Vrel","Alpha", "Cl","Cd","Fn","Ft","Vx"]
rad_YLabel = ["Local Relative Velocity", "Local Angle of Attack", "Local Coeffcient of Lift", "Local Coefficient of Drag",
                "Local Aerofoil Normal Force", "Local Aerofoil Tangential Force", "Local Axial Velocity"]
rad_units = ["[m/s]","[deg]","[-]","[-]","[N/m]","[N/m]","[m/s]"]
number_rotor_rotations = 3
markers = ["o","D"]


int_variables = ["Wind1VelX","RotTorq","RtAeroFxh","RtAeroFyh","RtAeroFzh","RtAeroMxh","RtAeroMyh","RtAeroMzh"]
int_YLabel = ["Hub height Velocity", "Rotor Torque", "Rotor Force in X direction", "Rotor Force in Y direction", 
                "Rotor Force in Z direction", "Rotor Moment in X direction", "Rotor Moment in Y direction", 
                "Rotor Moment in Z direction"]
int_units = ["[m/s]","[kN-m]","[N]","[N]","[N]","[N-m]","[N-m]","[N-m]"]


#plotting options
plot_ints = False
plot_spectra = False
plot_radial = True


        

if plot_ints == True:
    #integrated plots
    for i in np.arange(0,len(int_variables),1):

        Var = int_variables[i]
        unit = int_units[i]
        Ylabel = int_YLabel[i]

        fig = plt.figure(figsize=(14,8))

        ix = 0 #case counter
        time_end = 0
        for case in cases:

            df = io.fast_output_file.FASTOutputFile("{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

            time = df["Time_[s]"]
            time = np.array(time) + time_end

            txt = "{0}_{1}".format(Var,unit)

            Var_int = df[txt][:]
            
            plt.plot(time[:],Var_int[:])

            ix+=1 #increase case counter

            time_end+=time[-1]


        plt.ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)
        plt.xlabel("time [s]",fontsize=16)
        plt.title("5 levels of refinement, dt = 0.001s",fontsize=18)
        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Var))
        plt.close(fig)




if plot_spectra == True:
    #spectral plots
    for i in np.arange(0,len(int_variables),1):

        Var = int_variables[i]
        unit = int_units[i]
        Ylabel = int_YLabel[i]

        fig = plt.figure(figsize=(14,8))

        ix = 0 #case counter
        for case in cases:

            df = io.fast_output_file.FASTOutputFile("{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

            time = df["Time_[s]"]
            time = np.array(time)
                
            txt = "{0}_{1}".format(Var,unit)

            tstart = np.searchsorted(time[:],10.0)

            signal = np.array(df[txt][tstart:])


            m=0
            fs =1/dt
            n = len(signal) 
            if n%2==0:
                nhalf = int(n/2+1)
            else:
                nhalf = int((n+1)/2)
            frq = np.arange(nhalf)*fs/n
            Y   = np.fft.fft(signal-m) #Y = np.fft.fft(y) 
            PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
            PSD[1:-1] = PSD[1:-1]*2


            plt.loglog(frq, PSD)

            ix+=1

        plt.xlabel('Frequency (1/s)',fontsize=16)
        plt.ylabel('Power spectral density',fontsize=16)
        plt.title('{0} spectra, 5 levels of refinement, dt = 0.001s'.format(Ylabel))
        plt.grid()
        plt.tight_layout()
        plt.savefig(dir+"{0}_spectra.png".format(Var))
        plt.close(fig)



if plot_radial == True:
    #radial plots
    for i in np.arange(0,len(rad_variables),1):

        Var = rad_variables[i]
        unit = rad_units[i]
        YLabel = rad_YLabel[i]
        no_rots = number_rotor_rotations

        fig = plt.figure()

        ix = 0 #case counter
        for case in cases:

            df = io.fast_output_file.FASTOutputFile("{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

            # for col in df.columns:
            #     print(col)

            time = df["Time_[s]"]
            time = np.array(time)
            
            Az = np.array(df["Azimuth_[deg]"])

            x = np.linspace(0,1,act_stations)

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

            plt.plot(x,Var_list,marker=markers[ix],markersize=4)

            ix+=1

        plt.ylabel("{0} {1}".format(YLabel,unit),fontsize=16)
        plt.xlabel("Normalized blade radius [-]",fontsize=16)
        plt.title("Averaged over last {0} blade rotations\n5 levels of refinement, dt = 0.001s".format(no_rots),fontsize=12)
        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Var))
        plt.close(fig)
