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


#moment contributions

cases = ["eps_c_0.5", "eps_dr_0.75", "eps_dr_0.95", "eps_dx_1", "eps_dx_4", "Ex1_dblade_0.5", "Ex1_dblade_1.0",
         "Ex1_dblade_2.0", "fllc_Ex1"]
act_stations_cases = [74,47,59,54,54,54,54,54,74]



Var = "Ft"
Mom_Var = "Mt"
unit = "[N/m]"
YLabel = "Blade root local normal moment"
no_rots = 3
time_end = 24

ix = 0
for case in cases:

    df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis_nhalf/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()


    time = df["Time_[s]"]
    time = np.array(time)
    
    Az = np.array(df["Azimuth_[deg]"])

    act_stations = act_stations_cases[ix]
    x = np.linspace(0,1,act_stations)

    Var_list = []
    for i in np.arange(1,act_stations+1):
        if i < 10:
            txt = "AB1N00{0}{1}_{2}".format(i,Var,unit)
        elif i >= 10 and i < 100:
            txt = "AB1N0{0}{1}_{2}".format(i,Var,unit)
        elif i >= 100:
            txt = "AB1N{0}{1}_{2}".format(i,Var,unit)


        time_end_idx = np.searchsorted(time,time_end)
        Az_0 = Az[time_end_idx] - no_rots * 360
        tstart_idx = np.searchsorted(Az,Az_0)

        Var_dist = np.average(df[txt][tstart_idx:time_end_idx])
        
        Var_list.append(Var_dist)


    if case == "fllc_Ex1":
        new_Var_list = Var_list[:-2]
        x_list = [-3,-2]
        for j in x_list:
            y_pri = (new_Var_list[-2] - new_Var_list[-1])/(x[j] - x[j-1])

            new_point = new_Var_list[-1] + y_pri*(x[j] - x[j+1])

            if new_point < 80:
                new_Var_list.append(80.0)
            else:
                new_Var_list.append(new_point)
    else:
        new_Var_list = Var_list[:-1]

        y_pri = (Var_list[-2] - Var_list[-3])/(x[-2] - x[-3])

        new_point = Var_list[-2] + y_pri*(x[-1] - x[-2])

        new_Var_list.append(new_point)

    # if case == "fllc_Ex1":
    #     new_Var_list = Var_list[:-2]
    #     new_x = np.linspace(0,1,act_stations-2)
    # else:
    #     new_Var_list = Var_list[:-1]
    #     new_x = np.linspace(0,1,act_stations-1)


    # Mn_old = []
    # Mn_old_i = 0
    # dr = x[1] - x[0]
    # R = np.linspace(0,61.5,len(x))
    # for i in np.arange(0,len(x)):
    #     if i == 0 or i == len(x):
    #         Mn_old_i += Var_list[i]*(dr/2)*R[i]
    #         Mn_old.append(Mn_old_i)
    #     else:
    #         Mn_old_i += Var_list[i]*dr*R[i]
    #         Mn_old.append(Mn_old_i)
    
    # Mn_new = []
    # Mn_new_i = 0
    # new_dr = new_x[1] - new_x[0]  
    # new_R = np.linspace(0,61.5,len(new_x))
    # for i in np.arange(0,len(new_x)):
    #     if i == 0 or i == len(new_x):
    #         Mn_new_i += Var_list[i]*(new_dr/2)*new_R[i]
    #         Mn_new.append(Mn_new_i)
    #     else:
    #         Mn_new_i += Var_list[i]*new_dr*new_R[i]
    #         Mn_new.append(Mn_new_i)


    Mn_old = []
    Mn_old_i = 0
    Mn_new = []
    Mn_new_i = 0
    dr = x[1] - x[0]
    R = np.linspace(0,61.5,len(x))
    for i in np.arange(0,len(x)):
        if i == 0 or i == len(x):
            Mn_old_i += Var_list[i]*(dr/2)*R[i]
            Mn_new_i += new_Var_list[i]*(dr/2)*R[i]
            Mn_old.append(Mn_old_i)
            Mn_new.append(Mn_new_i)
        else:
            Mn_old_i += Var_list[i]*dr*R[i]
            Mn_new_i += new_Var_list[i]*dr*R[i]  
            Mn_old.append(Mn_old_i)
            Mn_new.append(Mn_new_i)

    percent_diff = np.round((Mn_old[-1]-Mn_new[-1])/Mn_new[-1]*100,3)
    
    fig = plt.figure()
    plt.plot(x,Mn_old,marker="o",color="b",markersize=4)
    plt.plot(x,Mn_new,marker="s",color="r",markersize=4)
    plt.xlabel("span [-]")
    plt.ylabel("Cumulative Tangential moment contribution to root moment [N-m]")
    plt.title("Averaged over last {0} blade rotations".format(no_rots),fontsize=12)
    plt.legend(["Percentage difference = {0}%".format(percent_diff),"-"])
    plt.tight_layout()
    plt.savefig("../../ALM_sensitivity_analysis_nhalf/joint_plots/moment_contributions_3/{0}_{1}.png".format(Mom_Var,case))
    plt.close(fig)

    fig = plt.figure()
    plt.plot(x,Var_list,marker="o",color="b",markersize=4)
    plt.plot(x,new_Var_list,marker="s",color="r",markersize=4)
    plt.xlabel("span [-]")
    plt.ylabel("Local Tangential force [N/m]")
    plt.legend(["Original", "Extrapolated last point"])
    #plt.legend(["Original", "removed spike"])
    plt.title("Averaged over last {0} blade rotations".format(no_rots),fontsize=12)
    plt.tight_layout()
    plt.savefig("../../ALM_sensitivity_analysis_nhalf/joint_plots/moment_contributions_3/{0}_{1}.png".format(Var,case))
    plt.close(fig)


    ix+=1








