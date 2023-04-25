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


dir = "joint_plots/"


cases = ["Ex1","Ex2","Ex3"]
act_stations_cases = [54,47,59]
dt_cases = [0.001,0.001,0.001]

colors = ["red","blue","green"]
markers = ["o","D","s"]
trans = [1,0.5,0.25]

legends = []
for i in np.arange(0,len(act_stations_cases)):
    legends.append("{0}: {1} actuator points".format(cases[i],act_stations_cases[i]))


rad_variables = ["Vrel","Alpha", "Cl","Cd","Fn","Ft","Vx"]
rad_YLabel = ["Local Relative Velocity", "Local Angle of Attack", "Local Coeffcient of Lift", "Local Coefficient of Drag",
                "Local Aerofoil Normal Force", "Local Aerofoil Tangential Force", "Local Axial Velocity"]
rad_units = ["[m/s]","[deg]","[-]","[-]","[N/m]","[N/m]","[m/s]"]
number_rotor_rotations = 3



time_start = [10,10,10] #time in seconds to remove from start of data - insert 0 if plot all time
time_end = [100,100,100] #time in seconds to plot upto - insert False if plot all time
int_variables = ["Wind1VelX","RotTorq","RtAeroFxh","RtAeroFyh","RtAeroFzh","RtAeroMxh","RtAeroMyh","RtAeroMzh"]
int_YLabel = ["Hub height Velocity", "Rotor Torque", "Rotor Force in X direction", "Rotor Force in Y direction", 
                "Rotor Force in Z direction", "Rotor Moment in X direction", "Rotor Moment in Y direction", 
                "Rotor Moment in Z direction"]
int_units = ["[m/s]","[kN-m]","[N]","[N]","[N]","[N-m]","[N-m]","[N-m]"]


#plotting options
plot_ints = False
plot_spectra = False
plot_radial = True



def Root_mean_squared(data_set):

    
    RMSE = []
    for l in np.arange(1,len(cases)):

        MSE = np.square(np.subtract(data_set[0],data_set[l])).mean() 
 
        RMSE.append( math.sqrt(MSE)/np.average(data_set[0]) )

    return RMSE



def difference(data_set):

    norm_diff = []
    mean_diff = []
    perc_diff = []
    for l in np.arange(1,len(cases)):

        diff = np.subtract(data_set[l],data_set[0])

        perc_diff_i = np.true_divide(abs(diff),data_set[0])

        perc_diff_i = [x for x in perc_diff_i if math.isnan(x) == False]

        perc_diff.append( np.average(perc_diff_i) * 100 )
 
        norm_diff.append( diff/np.average(data_set[0]) )

        mean_diff.append( np.sum(abs(norm_diff[l-1]))/len(norm_diff[l-1]) )

    return norm_diff, mean_diff, perc_diff


def energy_contents_check(case,Var,e_fft,signal,dt):
    
    E = (1/dt)*np.sum(e_fft)

    q = np.sum(np.square(signal))

    E2 = q

    print(case, Var, E, E2, abs(E2/E))
        

if plot_ints == True:
    #integrated plots
    for i in np.arange(0,len(int_variables),1):

        legends = []
        for j in np.arange(0,len(act_stations_cases)):
            legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))

        Var = int_variables[i]
        unit = int_units[i]
        Ylabel = int_YLabel[i]

        data_set =  pd.DataFrame(data=None, columns=cases)

        fig = plt.figure(figsize=(14,8))

        ix = 0 #case counter
        for case in cases:

            df = io.fast_output_file.FASTOutputFile("{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

            time = df["Time_[s]"]
            time = np.array(time)

            txt = "{0}_{1}".format(Var,unit)

            Var_int = df[txt][:]

            tstart = np.searchsorted(time[:],time_start[ix])
            tend = np.searchsorted(time[:],time_end[ix])

            data_set[ix] = Var_int[tstart:tend]
            
            plt.plot(time[tstart:tend],Var_int[tstart:tend],color=colors[ix],alpha=trans[ix])

            ix+=1 #increase case counter


        RMS = Root_mean_squared(data_set)

        for k in np.arange(1,len(cases)):
            legends[k] = legends[k] + "\nRMSE = {}".format(round(RMS[k-1],6))
        plt.ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)
        plt.xlabel("time [s]",fontsize=16)
        plt.legend(legends)
        plt.title("5 levels of refinement, dt = 0.001s",fontsize=18)
        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Var))
        plt.close(fig)

        fig = plt.figure()
        plt.bar(cases[1:],RMS)
        plt.ylabel("Root Mean Squared Normalized")
        plt.xlabel("Experiment compared to")
        plt.title("{0} \nResults compared against Ex1".format(Ylabel))
        plt.savefig(dir+"{0}_int_RMSE.png".format(Var))
        plt.close(fig)



if plot_spectra == True:
    #spectral plots
    for i in np.arange(0,len(int_variables),1):

        legends = []
        for j in np.arange(0,len(act_stations_cases)):
            legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))

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


            energy_contents_check(case,Var,PSD,signal,dt_cases[ix])


            plt.loglog(frq, PSD,color=colors[ix],alpha=trans[ix])

            ix+=1

        plt.xlabel('Frequency (1/s)',fontsize=16)
        plt.ylabel('Power spectral density',fontsize=16)
        plt.legend(legends)
        plt.title('{0} spectra, 5 levels of refinement, dt = 0.001s'.format(Ylabel))
        plt.grid()
        plt.tight_layout()
        plt.savefig(dir+"{0}_spectra.png".format(Var))
        plt.close(fig)



if plot_radial == True:
    #radial plots
    for i in np.arange(0,len(rad_variables),1):

        legends = []
        for j in np.arange(0,len(act_stations_cases)):
            legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))

        Var = rad_variables[i]
        unit = rad_units[i]
        YLabel = rad_YLabel[i]
        no_rots = number_rotor_rotations

        data_set =  pd.DataFrame(data=None, columns=cases)

        fig = plt.figure()

        ix = 0 #case counter
        for case in cases:

            df = io.fast_output_file.FASTOutputFile("{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

            # for col in df.columns:
            #     print(col)

            time = df["Time_[s]"]
            time = np.array(time)
            
            Az = np.array(df["Azimuth_[deg]"])

            act_stations = act_stations_cases[ix]
            x = np.linspace(0,1,act_stations)
            x_max = np.linspace(0,1,act_stations_cases[0])

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

            data_set[ix] = interpolate.interp1d(x, Var_list,kind="linear")(x_max)            

            plt.plot(x,Var_list,color=colors[ix],marker=markers[ix],markersize=4,alpha=trans[ix])

            ix+=1


        RMSE = Root_mean_squared(data_set)

        norm_diff, mean_diff, percent_diff = difference(data_set)

        for k in np.arange(1,len(cases)):
            legends[k] = legends[k] + "\nRMS = {0} \nAverage Percentage difference = {1}%".format(round(RMSE[k-1],6),round(percent_diff[k-1],6))
        plt.ylabel("{0} {1}".format(YLabel,unit),fontsize=16)
        plt.xlabel("Normalized blade radius [-]",fontsize=16)
        plt.legend(legends)
        plt.title("Averaged over last {0} blade rotations\n5 levels of refinement, dt = 0.001s".format(no_rots),fontsize=12)
        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Var))
        plt.close(fig)


        fig, (ax1,ax2) = plt.subplots(1,2)
        ax1.bar(cases[1:],RMSE)
        ax1.set_ylabel("Root Mean Squared Normalized [-]",fontsize=16)

        ax2.bar(cases[1:],percent_diff)
        ax2.set_ylabel("Average Percentage difference [%]",fontsize=16)
        
        fig.supxlabel("Experiment compared to",fontsize=16)
        fig.suptitle("{0} \nResults compared against Ex1".format(YLabel),fontsize=12)
        fig.tight_layout()
        plt.savefig(dir+"{0}_rad_RMSE.png".format(Var))
        plt.close(fig)

        legends = []
        for j in np.arange(0,len(act_stations_cases)):
            legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))

        for k in np.arange(1,len(cases)):
            legends[k] = legends[k] + "\nTotal Normalized Mean difference = {}".format(round(mean_diff[k-1],6))

        fig = plt.figure()
        for norm_diff_i in norm_diff:
            plt.plot(x_max,norm_diff_i)
        plt.ylabel("Normalized difference [-]",fontsize=16)
        plt.xlabel("Normalized blade radius [-]",fontsize=16)
        plt.legend(legends[1:])
        plt.title("{0} \nResults compared against Ex1".format(YLabel),fontsize=12)
        plt.tight_layout()
        plt.savefig(dir+"{0}_mean_diff.png".format(Var))
        plt.close(fig)
