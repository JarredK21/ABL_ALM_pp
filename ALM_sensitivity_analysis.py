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


#study
time_step_study = False
delta_r_study = False
gravity_study = False
delta_x_study = False
dt_BSR_study = False
eps_c_study = False
fllc_concept_study = False
fllc_grid_study = False
fixed_vs_fllc_study = False
eps_c_comp_study = False
fllc_best_case = False
ALM_update_study = True


if fllc_best_case == True:
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/fllc_best_case/"
    cases = ["fllc_Ex4","fllc_Ex1","fllc_Ex2"]
    act_stations_cases = [300,74,300]
    dt_cases = [0.0039,0.0039,0.0039]
    dx_cases = [8.5,3.0,4.2]
    eps_c_mod = [0.25,0.5,0.25]
    eps_c_comp = [2.0,1.4,2.0]
    level_ref = [6,5,5]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]


if eps_c_comp_study == True:
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/eps_c_comp_study/"
    cases = ["fllc_Ex3", "fllc_Ex5"]
    act_stations_cases = [300,300]
    dt_cases = [0.0039,0.0039]
    dx_cases = [2.1,6.4]
    eps_dr = [0.85,0.85,0.85]
    eps_c_mod = [0.25, 0.25]
    eps_c_comp = [2.0, 6.0]
    level_ref = [4,4]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]

if fixed_vs_fllc_study == True:
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/fixed_vs_fllc_study/"
    cases = ["eps_6_lv4", "fllc_Ex5"]
    act_stations_cases = [300,300]
    dt_cases = [0.0039,0.0039]
    dx_cases = [6.4,6.4]
    eps_dr = [0.85,0.85,0.85]
    eps_c_mod = ["fixed", 0.25]
    eps_c_comp = [6.0, 6.0]
    level_ref = [4,4]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]

if fllc_grid_study == True:
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/fllc_grid_study/"
    cases = ["fllc_Ex3","fllc_Ex2","fllc_Ex4"]
    act_stations_cases = [300,300,300]
    dt_cases = [0.0039,0.0039,0.0039]
    dx_cases = [2.1,4.2,8.5]
    eps_dr = [0.85,0.85,0.85]
    eps_c_mod = [0.25,0.25,0.25]
    eps_c_comp = [2.0,2.0,2.0]
    level_ref = [4,5,6]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]

if fllc_concept_study == True:
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/eps_c_fllc_comp/"
    cases = ["eps_c_0.5","fllc_Ex4"]
    act_stations_cases = [74,300]
    dt_cases = [0.0039,0.0039]
    dx_cases = [3,8.5]
    eps_c = [0.5,0.25]

    colors = ["red","blue"]
    markers = ["o","D"]
    trans = [1,0.75]


if delta_r_study == True:
    #dir = "../../../jarred/ALM_sensitivity_analysis/joint_plots/dr_study/"
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/dr_study/"
    #cases = ["Ex2","Ex1","Ex3"]
    cases = ["eps_dr_0.75", "Ex1_dblade_1.0", "eps_dr_0.95"]
    act_stations_cases = [47,54,59]
    dt_cases = [0.0039,0.0039,0.0039]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]


elif time_step_study == True:
    #dir = "../../../jarred/ALM_sensitivity_analysis/joint_plots/dt_study/"
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/dt_study/"
    #cases = ["Ex1_dblade_2.0","Ex1_dblade_1.0","Ex1_dblade_0.5","Ex1"]
    cases = ["Ex1_dblade_2.0","Ex1_dblade_1.0","Ex1_dblade_0.5"]
    act_stations_cases = [54,54,54,54]
    #dt_cases = [0.0078, 0.0039, 0.00195, 0.001]
    dt_cases = [0.0078,0.0039, 0.00195]

    colors = ["red","blue","green","orange"]
    markers = ["o","D","s","v"]
    trans = [1,0.75,0.5,0.25]


elif gravity_study == True:
    dir = "../../gravity_study/plots/"
    cases = ["gravity_on", "gravity_off"]
    act_stations_cases = [54,54]
    dt_cases = [0.0039, 0.0039]
    colors = ["red","blue"]
    markers = ["o","D"]
    trans = [1,0.5]


elif delta_x_study == True:
    # dir = "../../../jarred/ALM_sensitivity_analysis/joint_plots/dx_study2/"
    # cases = ["Ex4","Ex1_dblade_0.5","Ex5"]
    # dx_cases = [1,2,4]
    # act_stations_cases = [54,54,54]
    # dt_cases = [0.0039,0.00195,0.001]
    # BSR = [0.5,0.5,0.5]

    # dir = "../../../jarred/ALM_sensitivity_analysis/joint_plots/dx_study/"
    # cases = ["Ex6","Ex1_dblade_0.5","Ex7"]
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/dx_study/"
    cases = ["eps_dx_1", "Ex1_dblade_1.0", "eps_dx_4"]
    dx_cases = [1,2,4]
    act_stations_cases = [54,54,54]
    dt_cases = [0.0039,0.0039,0.0039]
    BSR = [0.25,0.5,1.0]
    refinement_lev = [4,5,6]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]

elif dt_BSR_study == True:
    # dir = "../../../jarred/ALM_sensitivity_analysis/joint_plots/BSR_study/"
    # cases = ["Ex1_dblade_0.5","Ex4","Ex6"]
    # act_stations_cases = [54,54,54]
    # dx_cases = [2,1,1]
    # dt_cases = [0.00195,0.0039,0.00195]
    # BSR = [0.5,0.5,0.25]

    dir = "../../../jarred/ALM_sensitivity_analysis/joint_plots/BSR_study2/"
    cases = ["Ex1_dblade_0.5","Ex5","Ex7"]
    act_stations_cases = [54,54,54]
    dx_cases = [2,4,4]
    dt_cases = [0.00195,0.001,0.00195]
    BSR = [0.5,0.5,1.0]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]

elif eps_c_study == True:
    dir = "../../../jarred/ALM_sensitivity_analysis_nhalf/joint_plots/eps_c_study/"
    cases = ["eps_dx_4","eps_c_0.5"]
    eps_cases = ["Fixed 1m", "eps/c = 0.5"]
    act_stations_cases = [54,74]
    eps_dr = [0.85,0.85]
    dx_cases = [4,3]
    dt_cases = [0.0039,0.0039]
    BSR = [2.0,1.0]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]


elif ALM_update_study == True:
    dir = "../../ALM_update/plots/"
    cases = ["Ex1","Ex2"]
    eps_cases = ["Fixed 1m", "Fixed 1m"]
    act_stations_cases = [54,54]
    eps_dr = [0.85,0.85]
    dx_cases = [2,2]
    dt_cases = [0.0039,0.00195]
    BSR = [1.0,0.5]

    colors = ["red","blue","green"]
    markers = ["o","D","s"]
    trans = [1,0.5,0.25]


#variables
rad_variables = ["Vrel","Alpha", "Cl","Cd","Fn","Ft","Vx"]
rad_YLabel = ["Local Relative Velocity", "Local Angle of Attack", "Local Coeffcient of Lift", "Local Coefficient of Drag",
                "Local Aerofoil Normal Force", "Local Aerofoil Tangential Force", "Local Axial Velocity"]
rad_units = ["[m/s]","[deg]","[-]","[-]","[N/m]","[N/m]","[m/s]"]
number_rotor_rotations = 3

plot_radial_range = False
LF = 0.5; RT = 0.7
plot_time_end = False

time_start = np.ones(len(cases))*5
time_end = np.ones(len(cases))*24

int_variables = ["Wind1VelX","RotTorq","RtAeroFxh","RtAeroFyh","RtAeroFzh","RtAeroMxh","RtAeroMyh","RtAeroMzh"]
int_YLabel = ["Hub height Velocity", "Rotor Torque", "Rotor Force in X direction", "Rotor Force in Y direction", 
                "Rotor Force in Z direction", "Rotor Moment in X direction", "Rotor Moment in Y direction", 
                "Rotor Moment in Z direction"]
int_units = ["[m/s]","[kN-m]","[N]","[N]","[N]","[N-m]","[N-m]","[N-m]"]


elastic_variables = ["MLx","MLx","MLy","MLy","MLz","MLz","FLx","FLx","FLy","FLy","FLz","FLz"]
elastic_YLabel = ["Blade root edgewise moment", "Blade tip edgewise moment",
                  "Blade root flapwise moment","Blade tip flapwise moment",
                  "Blade root pitching moment","Blade tip pitching moment",
                  "Blade root flapwise shear force","Blade tip flapwise shear force",
                  "Blade root edgewise shear force","Blade tip edgewise shear force",
                  "Blade root axial force","Blade tip axial force"]
elastic_units = ["[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN-m]","[kN]","[kN]","[kN]","[kN]","[kN]","[kN]"]
blade_stations = [1,21,1,21,1,21,1,21,1,21,1,21]


aero_variables = ["Fn","Fn","Ft","Ft"]
aero_units = ["[N/m]","[N/m]","[N/m]","[N/m]"]
aero_YLabel = ["Blade root normal force [kN]", "Blade tip normal force [kN]", "Blade root tangential force [kN]", "Blade tip tangential force [kN]"]
aero_blade_stations = [1,54,1,54]




#plotting options
plot_ints = True
plot_spectra = True
plot_radial = True
plot_stats = False
plot_time_difference = True
plot_radial_difference = True

plot_elastic_ints = False
plot_elastic_spectra = False
plot_aero_ints = False
plot_aero_spectra = False

#statistics type
percent_change = False #compare against previous case
percent_difference = True #compare against best case


def stats(data_set):

    perc_diff = []
    rad_perc_diff = []
    avg_diff = []
    Difference = []
    scaled_difference = []
    for l in np.arange(1,len(cases)):
        
        if percent_change == True:
            diff = np.subtract(data_set[l+1],data_set[l])
        elif percent_difference == True:
            diff = np.subtract(data_set[0],data_set[l])

        Difference.append(diff)

        scaled_difference.append(Difference[l-1]/np.max(abs(data_set[l])))

        avg_diff.append(np.average(abs(diff)))

        perc_diff_i = np.true_divide(abs(diff),abs(data_set[l]))

        perc_diff_i[np.isnan(perc_diff_i)] = 0
        perc_diff_i[np.isinf(perc_diff_i)] = 0

        if plot_ints == True:
            for k in np.arange(0,len(perc_diff_i)):
                if np.min(data_set[l]) < 0:
                    if data_set[l][k] < (0.25*np.max(data_set[l])) and data_set[l][k] > (0.25*np.min(data_set[l])):
                        perc_diff_i[k] = 0

        rad_perc_diff.append(perc_diff_i)

        perc_diff.append( abs(np.average(perc_diff_i) * 100) )

    return perc_diff, Difference, rad_perc_diff,avg_diff,scaled_difference


def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5*fs
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y


def stats2(data_set):

    perc_diff = []
    rad_perc_diff = []
    avg_diff = []
    Difference = []
    scaled_difference = []
    for l in np.arange(1,len(cases)):

        if percent_change == True:
            diff = np.subtract(data_set[l+1],data_set[l])
        elif percent_difference == True:
            diff = np.subtract(data_set[0],data_set[l])

        Difference.append(diff)

        scaled_difference.append(Difference[l-1]/np.max(abs(data_set[l])))

        avg_diff.append(np.average(abs(diff)))

        perc_diff_i = np.true_divide(abs(diff),abs(data_set[l]))

        perc_diff_i[np.isnan(perc_diff_i)] = 0
        perc_diff_i[np.isinf(perc_diff_i)] = 0

        if delta_r_study == True or eps_c_study == True:
            perc_diff_mean = np.mean(perc_diff_i)
            for k in np.arange(0,len(perc_diff_i)):
                if perc_diff_i[k] >= perc_diff_mean:
                    perc_diff_i[k] = perc_diff_mean

        rad_perc_diff.append(perc_diff_i)

        perc_diff.append( abs(np.average(perc_diff_i) * 100) )

    return perc_diff, Difference, rad_perc_diff,avg_diff,scaled_difference


def energy_contents_check(case,Var,e_fft,signal,dt):
    
    E = (1/dt)*np.sum(e_fft)

    q = np.sum(np.square(signal))

    E2 = q

    print(case, Var, E, E2, abs(E2/E))


if plot_stats == True:
    #mean variance plots

    if gravity_study == True:
        dq = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
    else:
        #dq = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
        dq = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
        
    tmax = np.arange(time_start[-1],time_end[-1],dt_cases[-1])

    for i in np.arange(0,len(int_variables),1):
        
        legends = []
        if delta_r_study == True:
            for j in np.arange(0,len(act_stations_cases)):
                legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))
        elif time_step_study == True:
            for j in np.arange(0,len(dt_cases)):
                legends.append("{0}: {1}s dt".format(cases[j],dt_cases[j]))
        elif gravity_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}".format(cases[j]))
        elif delta_x_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, dt = {2}, BSR = {3}".format(cases[j],dx_cases[j],dt_cases[j],BSR[j]))
        elif fllc_concept_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("$\epsilon/c$ = {0}: $\epsilon/dx$ = {1}, dt = {2}, $\epsilon/dr$ = {3}".format(eps_c[j],dx_cases[j],dt_cases[j],eps_dr[j]))

        Var = int_variables[i]
        unit = int_units[i]
        Ylabel = int_YLabel[i]

        MEAN = []
        VARIANCE = []
        ix = 0 #case counter
        for case in cases:
            
            if gravity_study == True:
                df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
            else:
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
        

            time = df["Time_[s]"]
            time = np.array(time)

            txt = "{0}_{1}".format(Var,unit)

            tstart = np.searchsorted(time[:],time_start[ix])
            tend = np.searchsorted(time[:],time_end[ix])

            Var_int = df[txt][tstart:tend]
            
            x = time[tstart:tend]
            x[0] = time_start[ix]
            x[-1] = time_end[ix]

            MEAN.append(np.mean(Var_int))
            VARIANCE.append( np.average(np.square((Var_int - np.mean(Var_int)))) )

            ix+=1 #increase case counter

        fig, (ax1,ax2) = plt.subplots(2,1)
        if delta_r_study == True:
            ax1.plot(act_stations_cases,MEAN,marker="o")
            ax2.plot(act_stations_cases,VARIANCE,marker="o")
            fig.supxlabel("Number of actuator points",fontsize=16)
        elif time_step_study == True:
            ax1.plot(dt_cases,MEAN,marker="o")
            ax2.plot(dt_cases,VARIANCE,marker="o")   
            fig.supxlabel("Time step [s]",fontsize=16)    
        elif delta_x_study == True:
            ax1.plot(dx_cases,MEAN,marker="o")
            ax2.plot(dx_cases,VARIANCE,marker="o")   
            fig.supxlabel("$\epsilon/dx$ [-]",fontsize=16)
        elif fllc_concept_study == True:
            ax1.plot(cases,MEAN,marker="o")
            ax2.plot(cases,VARIANCE,marker="o")   
            fig.supxlabel("case",fontsize=16)                     
        
        ax1.set_ylabel("Mean [{0}]".format(unit))
        ax2.set_ylabel("Variance [{0}$^2$]".format(unit))
        
        fig.suptitle("{0}".format(Ylabel),fontsize=12)
        fig.tight_layout()
        plt.savefig(dir+"{0}_int_stats.png".format(Var))
        plt.close(fig)


if plot_ints == True:
    #integrated plots

    if gravity_study == True:
        dq = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
    else:
        #dq = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
        #dq = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
        dq = io.fast_output_file.FASTOutputFile("../../ALM_update/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
        
    tmax = np.arange(time_start[-1],time_end[-1],dt_cases[-1])

    for i in np.arange(0,len(int_variables),1):
        
        legends = []
        if delta_r_study == True:
            for j in np.arange(0,len(act_stations_cases)):
                legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))
        elif time_step_study == True or ALM_update_study == True:
            for j in np.arange(0,len(dt_cases)):
                legends.append("{0}: {1}s dt".format(cases[j],dt_cases[j]))
        elif gravity_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}".format(cases[j]))
        elif delta_x_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, dt = {2}, BSR = {3}".format(cases[j],dx_cases[j],dt_cases[j],BSR[j]))
        elif eps_c_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, $\epsilon/\Delta r$ = {2}, dt = {3}".format(eps_cases[j],dx_cases[j],eps_dr[j],dt_cases[j]))
        elif fllc_concept_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("$\epsilon/c$ = {0}: $\epsilon/dx$ = {1}".format(eps_c[j],dx_cases[j]))
        elif fllc_grid_study == True or fixed_vs_fllc_study == True or eps_c_comp_study == True or fllc_best_case == True:
            for j in np.arange(0,len(cases)):
                legends.append("$[\epsilon/c]_{mod}$ = " + "{}".format(eps_c_mod[j]) + ", $[\epsilon/c]_{comp}$ = " + "{}".format(eps_c_comp[j]) + ", $\epsilon/dx$ = " + "{}".format(dx_cases[j]) + ", levels of ref = " + "{}".format(level_ref[j]))


        Var = int_variables[i]
        unit = int_units[i]
        Ylabel = int_YLabel[i]

        data_set =  []

        fig = plt.figure(figsize=(14,8))

        ix = 0 #case counter
        for case in cases:
            
            if gravity_study == True:
                df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
            else:
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                df = io.fast_output_file.FASTOutputFile("../../ALM_update/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

            time = df["Time_[s]"]
            time = np.array(time)

            txt = "{0}_{1}".format(Var,unit)

            tstart = np.searchsorted(time[:],time_start[ix])
            tend = np.searchsorted(time[:],time_end[ix])

            Var_int = df[txt][tstart:tend]

            cutoff = 40 #Hz
            fs = 1/dt_cases[ix]
            order = 10

            Var_int = butter_lowpass_filter(Var_int, cutoff, fs, order)
            
            x = time[tstart:tend]
            x[0] = time_start[ix]
            x[-1] = time_end[ix]

            data_set.append(interpolate.interp1d(x, Var_int,kind="linear")(tmax))
            
            plt.plot(time[tstart:tend],Var_int,color=colors[ix],alpha=trans[ix])

            ix+=1 #increase case counter

        perc_diff, diff, rad_perc_diff,avg_diff,scaled_difference = stats(data_set)

        for k in np.arange(1,len(cases)):
            legends[k] = legends[k] + "\nAverage difference = {0} \nAverage percentage change = {1}%".format(round(avg_diff[k-1],6), round(perc_diff[k-1],6))
        
        plt.ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)
        plt.xlabel("time [s]",fontsize=16)
        plt.legend(legends)
        if delta_r_study == True:
            plt.title('{0}, 5 levels of refinement'.format(Ylabel))
        elif time_step_study == True or ALM_update_study == True:
            plt.title('{0}, 54 actuator points, 5 levels of refinement'.format(Ylabel))
        elif gravity_study == True:
            plt.title('{0}, 54 actuator points, 5 levels of refinement'.format(Ylabel))
        elif delta_x_study == True:
            plt.title('{0}, 54 actuator points'.format(Ylabel))
        elif eps_c_study == True:
            plt.title('{0}'.format(Ylabel))

        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Var))
        plt.close(fig)

        fig, (ax1,ax2) = plt.subplots(2,1)
        if delta_r_study == True:
            ax1.plot(act_stations_cases[1:],avg_diff,marker="o")
            ax2.plot(act_stations_cases[1:],perc_diff,marker="o")
            fig.supxlabel("Number of actuator points",fontsize=16)
        elif time_step_study == True:
            ax1.plot(dt_cases[1:],avg_diff,marker="o")
            ax2.plot(dt_cases[1:],perc_diff,marker="o")   
            fig.supxlabel("Time step [s]",fontsize=16)    
        elif delta_x_study == True:
            ax1.plot(dx_cases[1:],avg_diff,marker="o")
            ax2.plot(dx_cases[1:],perc_diff,marker="o")   
            fig.supxlabel("$\epsilon/dx$ [-]",fontsize=16)                     
        
        ax1.set_ylabel("Average difference {0}".format(unit))
        ax2.set_ylabel("Average Percentage \nchange [%]")
        
        fig.suptitle("{0}".format(Ylabel),fontsize=12)
        fig.tight_layout()
        plt.savefig(dir+"{0}_int_diff.png".format(Var))
        plt.close(fig)


        fig = plt.figure(figsize=(14,8))
        for j in np.arange(0,len(cases[1:])):
            plt.plot(tmax,rad_perc_diff[j]*100,color=colors[j+1])
        plt.xlabel("Time [s]",fontsize=16)
        plt.ylabel("Percentage change [%]",fontsize=16)

        if delta_r_study == True:
            plt.legend(act_stations_cases[1:])
            plt.title('{0}, 5 levels of refinement'.format(Ylabel))
        elif time_step_study == True or ALM_update_study == True:
            plt.legend(dt_cases[1:])
            plt.title('{0}, 54 actuator points, 5 levels of refinement'.format(Ylabel))
        elif delta_x_study == True:
            plt.legend(dx_cases[1:])
            plt.title('{0}, 54 actuator points'.format(Ylabel))
        elif dt_BSR_study == True:
            plt.legend(dt_cases[1:])
        elif eps_c_study == True:
            plt.title('{0}'.format(Ylabel))    
        elif fllc_concept_study == True:
            plt.title("Filtered lifting line correction: {0}".format(Ylabel))            

        plt.tight_layout()
        plt.savefig(dir+"time_series_perc_{0}.png".format(Var))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        for j in np.arange(0,len(cases[1:])):
            plt.plot(tmax,diff[j],color=colors[j+1])
        plt.xlabel("Time [s]",fontsize=16)
        plt.ylabel("Difference {0}".format(unit),fontsize=16)

        if delta_r_study == True:
            plt.legend(act_stations_cases[1:])
            plt.title('{0}, 5 levels of refinement'.format(Ylabel))
        elif time_step_study == True or ALM_update_study == True:
            plt.legend(dt_cases[1:])
            plt.title('{0}, 54 actuator points, 5 levels of refinement'.format(Ylabel))
        elif delta_x_study == True:
            plt.legend(dx_cases[1:])
            plt.title('{0}, 54 actuator points'.format(Ylabel))
        elif dt_BSR_study == True:
            plt.legend(dt_cases[1:])
        elif eps_c_study == True:
            plt.title('{0}'.format(Ylabel))    
        elif fllc_concept_study == True:
            plt.title("Filtered lifting line correction: {0}".format(Ylabel))            

        plt.tight_layout()
        plt.savefig(dir+"time_series_diff_{0}.png".format(Var))
        plt.close(fig)


if plot_time_difference == True:
    #integrated plots

    #dq = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
    dq = io.fast_output_file.FASTOutputFile("../../ALM_update/{0}/post_processing/NREL_5MW_Main.out".format(cases[0])).toDataFrame()
    
        
    tmax = np.arange(time_start[-1],time_end[-1],dt_cases[-1])

    for i in np.arange(0,len(int_variables),1):

        legends = []
        if delta_r_study == True:
            for j in np.arange(0,len(act_stations_cases)):
                legends.append("{0} actuator points".format(act_stations_cases[j]))
        elif time_step_study == True or ALM_update_study == True:
            for j in np.arange(0,len(dt_cases)):
                legends.append("{0}s dt".format(dt_cases[j]))
        elif gravity_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}".format(cases[j]))
        elif delta_x_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0} levels of refinement, $\epsilon/dx$ = {1}, dt = {2}, BSR = {3}".format(refinement_lev[j],dx_cases[j],dt_cases[j],BSR[j]))
        elif eps_c_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, $\epsilon/\Delta r$ = {2}, dt = {3}".format(eps_cases[j],dx_cases[j],eps_dr[j],dt_cases[j]))
        elif fllc_concept_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("$\epsilon/c$ = {0}: $\epsilon/dx$ = {1}".format(eps_c[j],dx_cases[j]))
        elif fllc_grid_study == True or fixed_vs_fllc_study == True or eps_c_comp_study == True or fllc_best_case == True:
            for j in np.arange(0,len(cases)):
                legends.append("$[\epsilon/c]_{mod}$ = " + "{}".format(eps_c_mod[j]) + ", $[\epsilon/c]_{comp}$ = " + "{}".format(eps_c_comp[j]) + ", $\epsilon/dx$ = " + "{}".format(dx_cases[j]) + ", levels of ref = " + "{}".format(level_ref[j]))


        Var = int_variables[i]
        unit = int_units[i]
        Ylabel = int_YLabel[i]

        data_set =  []

        fig,ax1 = plt.subplots(figsize=(14,8))
        ax2 = ax1.twinx()

        ix = 0 #case counter
        for case in cases:
            
            #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
            df = io.fast_output_file.FASTOutputFile("../../ALM_update/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()


            time = df["Time_[s]"]
            time = np.array(time)

            txt = "{0}_{1}".format(Var,unit)

            tstart = np.searchsorted(time[:],time_start[ix])
            tend = np.searchsorted(time[:],time_end[ix])

            Var_int = df[txt][tstart:tend]

            cutoff = 10 #Hz
            fs = 1/dt_cases[ix]
            order = 10

            Var_int = butter_lowpass_filter(Var_int, cutoff, fs, order)
            
            x = time[tstart:tend]
            x[0] = time_start[ix]
            x[-1] = time_end[ix]

            data_set.append(interpolate.interp1d(x, Var_int,kind="linear")(tmax))
            
            ax1.plot(time[tstart:tend],Var_int,color=colors[ix],alpha=trans[ix])


            ix+=1 #increase case counter

        ax1.legend(legends)

        perc_diff, diff, rad_perc_diff,avg_diff,scaled_difference = stats(data_set)

        for j in np.arange(0,len(cases)-1):
            ax2.plot(tmax,scaled_difference[j],color=colors[j+1],linestyle="--")

        ax1.set_ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)
        ax2.set_ylabel("Scaled Difference {0}".format(unit),fontsize=16)
        ax1.set_xlabel("Time [s]",fontsize=16)
        
        plt.tight_layout()
        plt.savefig(dir+"time_series_plus_difference_{0}.png".format(Var))
        plt.close(fig)


if plot_spectra == True:
    #spectral plots
    for i in np.arange(0,len(int_variables),1):
        legends = []
        if delta_r_study == True:
            for j in np.arange(0,len(act_stations_cases)):
                legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))
        elif time_step_study == True or ALM_update_study == True:
            for j in np.arange(0,len(dt_cases)):
                legends.append("{0}: {1}s dt".format(cases[j],dt_cases[j]))
        elif gravity_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}".format(cases[j]))
        elif delta_x_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, dt = {2}, BSR = {3}".format(cases[j],dx_cases[j],dt_cases[j],BSR[j]))
        elif eps_c_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, $\epsilon/\Delta r$ = {2}, dt = {3}".format(eps_cases[j],dx_cases[j],eps_dr[j],dt_cases[j]))
        elif fllc_concept_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("$\epsilon/c$ = {0}: $\epsilon/dx$ = {1}".format(eps_c[j],dx_cases[j]))
        elif fllc_grid_study == True or fixed_vs_fllc_study == True or eps_c_comp_study == True or fllc_best_case == True: 
            for j in np.arange(0,len(cases)):
                legends.append("$[\epsilon/c]_{mod}$ = " + "{}".format(eps_c_mod[j]) + ", $[\epsilon/c]_{comp}$ = " + "{}".format(eps_c_comp[j]) + ", $\epsilon/dx$ = " + "{}".format(dx_cases[j]) + ", levels of ref = " + "{}".format(level_ref[j]))


        Var = int_variables[i]
        unit = int_units[i]
        Ylabel = int_YLabel[i]

        fig = plt.figure(figsize=(14,8))
        AA = []
        ix = 0 #case counter
        for case in cases:

            if gravity_study == True:
                df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
            else:
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                df = io.fast_output_file.FASTOutputFile("../../ALM_update/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
        
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
            Y   = np.fft.fft(signal-m)
            PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
            PSD[1:-1] = PSD[1:-1]*2


            energy_contents_check(case,Var,PSD,signal,dt_cases[ix])


            plt.loglog(frq, PSD,color=colors[ix],alpha=trans[ix])

            ix+=1

        plt.xlabel('Frequency (1/s)',fontsize=16)
        plt.ylabel('Power spectral density',fontsize=16)
        plt.legend(legends)

        if delta_r_study == True:
            plt.title('{0}, 5 levels of refinement'.format(Ylabel))
        elif time_step_study == True or ALM_update_study == True:
            plt.title('{0}, 54 actuator points, 5 levels of refinement'.format(Ylabel))
        elif gravity_study == True:
            plt.title('{0}, 54 actuator points, 5 levels of refinement'.format(Ylabel))
        elif delta_x_study == True:
            plt.title('{0}, 54 actuator points'.format(Ylabel))
        elif eps_c_study == True:
            plt.title('{0}'.format(Ylabel))
        elif fllc_concept_study == True:
            plt.title("Filtered lifting line correction: {0}".format(Ylabel))

        plt.grid()
        plt.tight_layout()
        plt.savefig(dir+"{0}_spectra.png".format(Var))
        plt.close(fig)


if plot_radial == True:
    #radial plots
    for i in np.arange(0,len(rad_variables),1):

        legends = []
        if delta_r_study == True:
            for j in np.arange(0,len(act_stations_cases)):
                legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))
        elif time_step_study == True or ALM_update_study == True:
            for j in np.arange(0,len(dt_cases)):
                legends.append("{0}: {1}s dt".format(cases[j],dt_cases[j]))
        elif gravity_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}".format(cases[j]))
        elif delta_x_study == True or dt_BSR_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, dt = {2}, BSR = {3}".format(cases[j],dx_cases[j],dt_cases[j],BSR[j]))
        elif eps_c_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, $\epsilon/\Delta r$ = {2}, dt = {3}".format(eps_cases[j],dx_cases[j],eps_dr[j],dt_cases[j]))
        elif fllc_concept_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("$\epsilon/c$ = {0}: $\epsilon/dx$ = {1}".format(eps_c[j],dx_cases[j]))
        elif fllc_grid_study == True or fixed_vs_fllc_study == True or eps_c_comp_study == True or fllc_best_case == True:
            for j in np.arange(0,len(cases)):
                legends.append("$[\epsilon/c]_{mod}$ = " + "{}".format(eps_c_mod[j]) + ", $[\epsilon/c]_{comp}$ = " + "{}".format(eps_c_comp[j]) + ", $\epsilon/dx$ = " + "{}".format(dx_cases[j]) + ", levels of ref = " + "{}".format(level_ref[j]))


        Var = rad_variables[i]
        unit = rad_units[i]
        YLabel = rad_YLabel[i]
        no_rots = number_rotor_rotations

        data_set =  []


        fig = plt.figure()

        ix = 0 #case counter
        for case in cases:

            if gravity_study == True:
                df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
            else:
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                df = io.fast_output_file.FASTOutputFile("../../ALM_update/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
        

            time = df["Time_[s]"]
            time = np.array(time)
            
            Az = np.array(df["Azimuth_[deg]"])

            act_stations = act_stations_cases[ix]
            x = np.linspace(0,1,act_stations)
            #x_max = np.linspace(0,1,act_stations_cases[-1])
            x_min = np.linspace(0,1,act_stations_cases[0])

            Var_list = []
            for i in np.arange(1,act_stations+1):
                if i < 10:
                    txt = "AB1N00{0}{1}_{2}".format(i,Var,unit)
                elif i >= 10 and i < 100:
                    txt = "AB1N0{0}{1}_{2}".format(i,Var,unit)
                elif i >= 100:
                    txt = "AB1N{0}{1}_{2}".format(i,Var,unit)


                time_end_idx = np.searchsorted(time,time_end[ix])
                Az_0 = Az[time_end_idx] - no_rots * 360
                tstart_idx = np.searchsorted(Az,Az_0)

                if plot_time_end == True:
                    Var_dist = df[txt][time_end_idx]
                else:
                    Var_dist = np.average(df[txt][tstart_idx:time_end_idx])
                
                Var_list.append(Var_dist)

            #data_set.append(interpolate.interp1d(x, Var_list,kind="linear")(x_max))    
            data_set.append(interpolate.interp1d(x, Var_list,kind="linear")(x_min))

            plt.plot(x,Var_list,color=colors[ix],marker=markers[ix],markersize=4,alpha=trans[ix])

            if plot_radial_range == True:
                plt.xlim((LF,RT))
                plt.ylim((0,5))

            ix+=1

        perc_diff, diff, rad_perc_diff,avg_diff,scaled_difference = stats2(data_set)

        for k in np.arange(1,len(cases)):
            legends[k] = legends[k] + "\nAverage difference = {0} \nAverage Percentage change = {1}%".format(round(avg_diff[k-1],6),round(perc_diff[k-1],6))
    
        plt.ylabel("{0} {1}".format(YLabel,unit),fontsize=16)
        plt.xlabel("Normalized blade radius [-]",fontsize=16)
        plt.legend(legends)
        if plot_time_end == True:
            plt.title("Plotted at {0}".format(time_end[0]))
        else:
            plt.title("Averaged over last {0} blade rotations\n5 levels of refinement".format(no_rots),fontsize=12)
        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Var))
        plt.close(fig)


        fig, (ax1,ax2) = plt.subplots(2,1)
        if delta_r_study == True:
            ax1.plot(act_stations_cases[1:],avg_diff,marker="o")
            ax2.plot(act_stations_cases[1:],perc_diff,marker="o")
            fig.supxlabel("Number of actuator points",fontsize=16)
        elif time_step_study == True:
            ax1.plot(dt_cases[1:],avg_diff,marker="o")
            ax2.plot(dt_cases[1:],perc_diff,marker="o")  
            fig.supxlabel("Time step [s]",fontsize=16)
        elif delta_x_study == True:
            ax1.plot(dx_cases[1:],avg_diff,marker="o")
            ax2.plot(dx_cases[1:],perc_diff,marker="o")   
            fig.supxlabel("$\epsilon/dx$ [-]",fontsize=16)   

        ax1.set_ylabel("Average difference {0}".format(unit))
        ax2.set_ylabel("Average Percentage \nchange [%]")
        
        fig.suptitle("{0}".format(YLabel),fontsize=12)
        fig.tight_layout()
        plt.savefig(dir+"{0}_rad_diff.png".format(Var))
        plt.close(fig)


        fig = plt.figure()
        for j in np.arange(0,len(cases[1:])):
            plt.plot(x_min,rad_perc_diff[j]*100,color=colors[j+1])
        plt.xlabel("Normalized blade radius [-]",fontsize=16)
        plt.ylabel("Percentage change [%]",fontsize=16)

        if delta_r_study == True:
            plt.legend(act_stations_cases[1:])
        elif time_step_study == True or ALM_update_study == True:
            plt.legend(dt_cases[1:])
        elif delta_x_study == True:
            plt.legend(dx_cases[1:])
        elif dt_BSR_study == True:
            plt.legend(dt_cases[1:])


        if plot_time_end == True:
            plt.title("{0} Plotted at {1}".format(YLabel,time_end[0]))
        else:
            plt.title("{0} Averaged over last {1} blade rotations\n5 levels of refinement".format(YLabel,no_rots),fontsize=12)
        plt.tight_layout()
        plt.savefig(dir+"rad_perc_{0}.png".format(Var))
        plt.close(fig)


if plot_radial_difference == True:
    #radial plots
    for i in np.arange(0,len(rad_variables),1):

        legends = []
        if delta_r_study == True:
            for j in np.arange(0,len(act_stations_cases)):
                legends.append("{0} actuator points".format(act_stations_cases[j]))
        elif time_step_study == True or ALM_update_study == True:
            for j in np.arange(0,len(dt_cases)):
                legends.append("{0}s dt".format(dt_cases[j]))
        elif gravity_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}".format(cases[j]))
        elif delta_x_study == True or dt_BSR_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0} levels of refinement, $\epsilon/dx$ = {1}, dt = {2}, BSR = {3}".format(refinement_lev[j],dx_cases[j],dt_cases[j],BSR[j]))
        elif eps_c_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("{0}: $\epsilon/dx$ = {1}, $\epsilon/\Delta r$ = {2}, dt = {3}".format(eps_cases[j],dx_cases[j],eps_dr[j],dt_cases[j]))
        elif fllc_concept_study == True:
            for j in np.arange(0,len(cases)):
                legends.append("$\epsilon/c$ = {0}: $\epsilon/dx$ = {1}".format(eps_c[j],dx_cases[j]))
        elif fllc_grid_study == True or fixed_vs_fllc_study == True or eps_c_comp_study == True or fllc_best_case == True:
            for j in np.arange(0,len(cases)):
                legends.append("$[\epsilon/c]_{mod}$ = " + "{}".format(eps_c_mod[j]) + ", $[\epsilon/c]_{comp}$ = " + "{}".format(eps_c_comp[j]) + ", $\epsilon/dx$ = " + "{}".format(dx_cases[j]) + ", levels of ref = " + "{}".format(level_ref[j]))


        Var = rad_variables[i]
        unit = rad_units[i]
        YLabel = rad_YLabel[i]
        no_rots = number_rotor_rotations

        data_set =  []


        fig,ax1 = plt.subplots()
        ax2 = ax1.twinx()

        ix = 0 #case counter
        for case in cases:

            if gravity_study == True:
                df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
            else:
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
                df = io.fast_output_file.FASTOutputFile("../../ALM_update/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
        

            time = df["Time_[s]"]
            time = np.array(time)
            
            Az = np.array(df["Azimuth_[deg]"])

            act_stations = act_stations_cases[ix]
            x = np.linspace(0,1,act_stations)
            #x_max = np.linspace(0,1,act_stations_cases[-1])
            x_min = np.linspace(0,1,act_stations_cases[0])

            Var_list = []
            for i in np.arange(1,act_stations+1):
                if i < 10:
                    txt = "AB1N00{0}{1}_{2}".format(i,Var,unit)
                elif i >= 10 and i < 100:
                    txt = "AB1N0{0}{1}_{2}".format(i,Var,unit)
                elif i >= 100:
                    txt = "AB1N{0}{1}_{2}".format(i,Var,unit)


                time_end_idx = np.searchsorted(time,time_end[ix])
                Az_0 = Az[time_end_idx] - no_rots * 360
                tstart_idx = np.searchsorted(Az,Az_0)

                if plot_time_end == True:
                    Var_dist = df[txt][time_end_idx]
                else:
                    Var_dist = np.average(df[txt][tstart_idx:time_end_idx])
                
                Var_list.append(Var_dist)

            #data_set.append(interpolate.interp1d(x, Var_list,kind="linear")(x_max))    
            data_set.append(interpolate.interp1d(x, Var_list,kind="linear")(x_min))

            ax1.plot(x,Var_list,color=colors[ix],marker=markers[ix],markersize=4,alpha=trans[ix])

            ix+=1

        ax1.legend(legends)

        perc_diff, diff, rad_perc_diff,avg_diff,scaled_difference = stats2(data_set)

        for j in np.arange(0,len(cases)-1):
            ax2.plot(x_min,scaled_difference[j],color=colors[j+1],linestyle="--")

        ax1.set_ylabel("{0} {1}".format(YLabel,unit),fontsize=16)
        ax2.set_ylabel("Scaled Difference {0}".format(unit),fontsize=16)
        ax1.set_xlabel("Normalized blade radius [-]",fontsize=16)
        
        plt.tight_layout()
        plt.savefig(dir+"radial_plus_difference_{0}.png".format(Var))
        plt.close(fig)





if plot_elastic_ints == True:
    
    #integrated plots
    for i in np.arange(0,len(elastic_variables),1):

        # legends = []
        # for j in np.arange(0,len(act_stations_cases)):
        #     legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))
        # legends = []
        # for j in np.arange(0,len(dt_cases)):
        #     legends.append("{0}: {1}s dt".format(cases[j],dt_cases[j]))
        legends = []
        for j in np.arange(0,len(cases)):
            legends.append("{0}".format(cases[j]))

        Var = elastic_variables[i]
        unit = elastic_units[i]
        Ylabel = elastic_YLabel[i]
        blade_station = blade_stations[i]

        data_set =  []

        fig = plt.figure(figsize=(14,8))

        ix = 0 #case counter
        for case in cases:

            #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
            df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()


            time = df["Time_[s]"]
            time = np.array(time)

            if blade_station < 9:
                txt = "B1N00{0}{1}_{2}".format(blade_station,Var,unit)
            else:
                txt = "B1N0{0}{1}_{2}".format(blade_station,Var,unit)

            Var_int = df[txt][:]

            tstart = np.searchsorted(time[:],time_start[ix])
            tend = np.searchsorted(time[:],time_end[ix])

            data_set.append(np.mean(Var_int[tstart:tend]))
            
            plt.plot(time[tstart:tend],Var_int[tstart:tend],color=colors[ix],alpha=trans[ix])

            ix+=1 #increase case counter


        #mean_diff = mean_difference(data_set)

        for k in np.arange(0,len(cases)):
            legends[k] = legends[k] + "\nmean = {}".format(round(data_set[k],3))
        plt.ylabel("{0} {1}".format(Ylabel,unit),fontsize=16)
        plt.xlabel("time [s]",fontsize=16)
        plt.legend(legends)
        plt.axhline(data_set[0],color=colors[0],linestyle="dashed")
        plt.axhline(data_set[1],color=colors[1],linestyle="dotted")
        plt.title('{0}, 5 levels of refinement, 54 actuator points'.format(Ylabel))
        #plt.title("5 levels of refinement, 94 actuator points",fontsize=18)
        plt.tight_layout()
        plt.savefig(dir+"{0}.png".format(Ylabel))
        plt.close(fig)


if plot_elastic_spectra == True:
    #spectral plots
    for i in np.arange(0,len(elastic_variables),1):

        # legends = []
        # for j in np.arange(0,len(act_stations_cases)):
        #     legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))
        # legends = []
        # for j in np.arange(0,len(dt_cases)):
        #     legends.append("{0}: {1}s dt".format(cases[j],dt_cases[j]))
        legends = []
        for j in np.arange(0,len(cases)):
            legends.append("{0}".format(cases[j]))

        Var = elastic_variables[i]
        unit = elastic_units[i]
        Ylabel = elastic_YLabel[i]
        blade_station = blade_stations[i]

        fig = plt.figure(figsize=(14,8))

        ix = 0 #case counter
        for case in cases:

            #df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()
            df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

            time = df["Time_[s]"]
            time = np.array(time)
                
            if blade_station < 9:
                txt = "B1N00{0}{1}_{2}".format(blade_station,Var,unit)
            else:
                txt = "B1N0{0}{1}_{2}".format(blade_station,Var,unit)

            Var_int = df[txt][:]

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
        # plt.title('{0} spectra, 5 levels of refinement, dt = 0.001s'.format(Ylabel))
        plt.title('{0} spectra, 5 levels of refinement, 54 actuator points'.format(Ylabel))
        plt.grid()
        plt.tight_layout()
        plt.savefig(dir+"{0}_spectra.png".format(Ylabel))
        plt.close(fig)




elastic_variables = ["FLx","FLx","FLy","FLy"]
elastic_YLabel = ["Blade root flapwise shear force","Blade tip flapwise shear force",
                  "Blade root edgewise shear force","Blade tip edgewise shear force"]
elastic_units = ["[kN]","[kN]","[kN]","[kN]"]
elastic_blade_stations = [1,21,1,21]


if plot_aero_ints == True:
    
    #integrated plots
    for i in np.arange(0,len(aero_variables),1):

        aero_Var = aero_variables[i]
        elastic_var = elastic_variables[i]
        aero_unit = aero_units[i]
        elastic_unit = elastic_units[i]
        Ylabel_1 = aero_YLabel[i]
        Ylabel_2 = elastic_YLabel[i]
        blade_station_1 = aero_blade_stations[i]
        blade_station_2 = elastic_blade_stations[i]

        #fig,ax = plt.subplots(figsize=(14,8))
        fig = plt.figure(figsize=(14,8))


        case = "gravity_on" #check

        df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

        time = df["Time_[s]"]
        time = np.array(time)

        if blade_station_1 <= 9:
            txt_1 = "AB1N00{0}{1}_{2}".format(blade_station_1,aero_Var,aero_unit)
        elif blade_station_1 > 9:
            txt_1 = "AB1N0{0}{1}_{2}".format(blade_station_1,aero_Var,aero_unit)
        
        if blade_station_2 <= 9:
            txt_2 = "B1N00{0}{1}_{2}".format(blade_station_2,elastic_var,elastic_unit)
        elif blade_station_2 > 9:
            txt_2 = "B1N0{0}{1}_{2}".format(blade_station_2,elastic_var,elastic_unit)

        aero_Var_int = df[txt_1][:]*(1.18/1000)
        elastic_Var_int = df[txt_2][:]


        tstart = np.searchsorted(time[:],time_start[0])
        tend = np.searchsorted(time[:],time_end[0])
        
        # ax.plot(time[tstart:tend],aero_Var_int[tstart:tend],color=colors[0])
        # ax.set_ylabel("{0}".format(Ylabel_1),fontsize=16)
        # ax.axhline(y=np.mean(aero_Var_int),color=colors[0],linestyle="dashed")

        # ax2=ax.twinx()
        # ax2.plot(time[tstart:tend],elastic_Var_int[tstart:tend],color=colors[1])
        # ax2.set_ylabel("{0} {1}".format(Ylabel_2,elastic_unit),fontsize=16)
        # ax2.axhline(y=np.mean(elastic_Var_int),color=colors[1],linestyle="dotted")

        # ax.set_xlabel("time [s]",fontsize=16)

        plt.plot(time[tstart:tend],aero_Var_int[tstart:tend],color=colors[0])
        plt.plot(time[tstart:tend],elastic_Var_int[tstart:tend],color=colors[1])
        plt.legend(["Aerodyn","Elastodyn"])
        plt.axhline(y=np.mean(aero_Var_int),color=colors[0],linestyle="dashed")
        plt.axhline(y=np.mean(elastic_Var_int),color=colors[1],linestyle="dotted")
        plt.ylabel("{0} {1}".format(Ylabel_2,elastic_unit),fontsize=16)
        plt.xlabel("time [s]",fontsize=16)

        plt.title("Gravity on, comparing {0} aerodyn and {1} elastodyn".format(Ylabel_1,Ylabel_2))
        plt.tight_layout()
        plt.savefig(dir+"{0}_{1}_2.png".format(Ylabel_1,Ylabel_2))
        plt.close(fig)



if plot_aero_spectra == True:
    #spectral plots
    for i in np.arange(0,len(aero_variables),1):

        aero_Var = aero_variables[i]
        elastic_var = elastic_variables[i]
        aero_unit = aero_units[i]
        elastic_unit = elastic_units[i]
        Ylabel_1 = aero_YLabel[i]
        Ylabel_2 = elastic_YLabel[i]
        blade_station_1 = aero_blade_stations[i]
        blade_station_2 = elastic_blade_stations[i]

        fig,ax = plt.subplots(figsize=(14,8))

        case = "gravity_on"

        df = io.fast_output_file.FASTOutputFile("../../gravity_study/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

        time = df["Time_[s]"]
        time = np.array(time)

        if blade_station_1 <= 9:
            txt_1 = "AB1N00{0}{1}_{2}".format(blade_station_1,aero_Var,aero_unit)
        elif blade_station_1 > 9:
            txt_1 = "AB1N0{0}{1}_{2}".format(blade_station_1,aero_Var,aero_unit)
        
        if blade_station_2 <= 9:
            txt_2 = "B1N00{0}{1}_{2}".format(blade_station_2,elastic_var,elastic_unit)
        elif blade_station_2 > 9:
            txt_2 = "B1N0{0}{1}_{2}".format(blade_station_2,elastic_var,elastic_unit)

        tstart = np.searchsorted(time[:],time_start[0])
        tend = np.searchsorted(time[:],time_end[0])

        signal_1 = df[txt_1][tstart:tend]
        signal_2 = df[txt_2][tstart:tend]

        #fft aero signal
        m=0
        fs =1/dt_cases[0]
        n = len(signal_1) 
        if n%2==0:
            nhalf = int(n/2+1)
        else:
            nhalf = int((n+1)/2)
        frq = np.arange(nhalf)*fs/n
        Y   = np.fft.fft(signal_1-m)
        PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
        PSD[1:-1] = PSD[1:-1]*2

        plt.loglog(frq, PSD,color=colors[0],alpha=trans[0])
        plt.ylim(bottom=1e-10)

        #fft elastic signal
        m=0
        fs =1/dt_cases[0]
        n = len(signal_2) 
        if n%2==0:
            nhalf = int(n/2+1)
        else:
            nhalf = int((n+1)/2)
        frq = np.arange(nhalf)*fs/n
        Y   = np.fft.fft(signal_2-m)
        PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
        PSD[1:-1] = PSD[1:-1]*2

        plt.loglog(frq, PSD,color=colors[1],alpha=trans[1])
        plt.ylim(bottom=1e-10)



        plt.xlabel('Frequency (1/s)',fontsize=16)
        plt.ylabel('Power spectral density',fontsize=16)
        plt.legend(["AeroDyn","ElastoDyn"])
        plt.title("Gravity on, comparing spectra {0} aerodyn and {1} elastodyn".format(Ylabel_1,Ylabel_2))
        plt.grid()
        plt.tight_layout()
        plt.savefig(dir+"{0}_{1}_spectra.png".format(Ylabel_1,Ylabel_2))
        plt.close(fig)