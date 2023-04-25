# coding: utf-8
import netCDF4 as nc
from scipy.fft import fft2, fftfreq, fftshift
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
from netCDF4 import Dataset
import glob, os
from math import ceil, floor
import scipy as sp
import math

def d_dt(time, var):
    d_var_dt = []
    for i in np.arange(0,len(var)-1,1):
        d_var_dt.append((var[i+1]-var[i])/(time[i+1]-time[i]))
    return d_var_dt 


def plottingdvar_dt(dir):


    a = Dataset("./Ex1/post_processing/abl_statistics65000.nc")
    b = Dataset("./test9/post_processing/abl_statistics75000.nc")

    time_a = a.variables["time"]
    time_b = b.variables["time"]
    zi_a = a.variables["zi"]
    zi_b = b.variables["zi"]
    ustar_a = a.variables["ustar"]
    ustar_b = b.variables["ustar"]
    wstar_a = a.variables["wstar"]
    wstar_b = b.variables["wstar"]
    Tsurf_a = a.variables["Tsurf"]
    Tsurf_b = b.variables["Tsurf"]

    vars_a = [zi_a,ustar_a,wstar_a,Tsurf_a,zi_a,zi_a]
    vars_b = [zi_b,ustar_b,wstar_b,Tsurf_b,zi_b,zi_b]
    labels = ["$dz_i/dt$ [m/s]", "$du_*/dt [m/s^2]$", "$dw_*/dt [m/s^2]$", "$dT_0/dt [K m/s^2]$",
                "$dz_i/dt 1/w_*$ [-]","$dz_i/dt 1/u_*$ [-]"]
    filenames = ["dzi_dt", "dustar_dt", "dwstar_dt", "dT0_dt", "dzi_dt_wstar","dzi_dt_ustar"]


    ic = 0
    for var_a, var_b in zip(vars_a,vars_b):
        d_var_dt_a = d_dt(time_a,var_a)
        d_var_dt_b = d_dt(time_b,var_b)

        if filenames[ic] == "dzi_dt_wstar":
            d_var_dt_a = np.divide(d_var_dt_a,wstar_a[:-1])
            d_var_dt_b = np.divide(d_var_dt_b,wstar_b[:-1])
        elif filenames[ic] == "dzi_dt_ustar":
            d_var_dt_a = np.divide(d_var_dt_a,ustar_a[:-1])
            d_var_dt_b = np.divide(d_var_dt_b,ustar_b[:-1])

        fig = plt.figure()

        def plot_std(time, d_var_dt,colors, case):
            #calculate bin edges using 3Tau
            bins = [0] #bin edges index's
            Tau = np.divide(case.variables["zi"],case.variables["wstar"])

            i = 0 #bin edge counter
            time_stamp = case.variables["time"][0]
            while time_stamp < case.variables["time"][-1]:
                Tau_i = Tau[bins[i]]
                time_stamp += 3*Tau_i #increase time by 3Tau
                bins.append( np.searchsorted(case.variables['time'][:],time_stamp) ) #append next bin edge

                i+=1 #increase bin edge counter


            mean = []
            standard_dev = []
            bin_center = []
            for i_bin in np.arange(0,np.size(bins)-1,1):
                left = bins[i_bin]
                right = bins[i_bin + 1]

                numbers = d_var_dt[left:right]
                bin_center.append( np.mean( time[left:right] ) )
                mean.append( np.mean(numbers) )
                variance = 0
                for number in numbers:
                    variance += ( number - np.mean(numbers) )**2

                variance = variance/len(numbers)

                standard_dev.append(np.sqrt(variance))


            under_line = np.array(mean) - np.array(standard_dev)
            over_line = np.array(mean) + np.array(standard_dev)

            #plt.plot(bin_center, mean, linewidth = 2, linestyle = "solid", color=colors)#mean curve
            plt.axhline(mean,linestyle="solid",color=colors)
            plt.axhline(0.01, linestyle="--")
            plt.axhline(-0.01, linestyle="--")
            #plt.fill_between(bin_center,under_line,over_line, linestyle = "solid", color=colors, alpha=0.1)#std curve
            
        plot_std(time_a,d_var_dt_a,colors="blue",case=a)
        plot_std(time_b,d_var_dt_b,colors="red",case=b)
        #plt.xlim([30000,40000])
        #plt.ylim([-0.05, 0.05])
        
        plt.ylabel(labels[ic])
        plt.xlabel("Time [s]")
        plt.legend(["Ex1 ABL ALM","-",
                   "Ex1 ABL ALM restart","-"])
    

        path = dir + "{0}.png".format(filenames[ic])

        #separate plots
        plt.savefig(path)
        plt.close(fig)

        ic += 1


def plottingTimeVars(cases, Titles, TimeVars, Timelabels, TimeFilenames, i, dir, linestyles, colors):
    cc = 0 #case counter

    fig = plt.figure()

    for case in cases:

        if case == "Ex1":
            a = Dataset("./{0}/post_processing/abl_statistics65000.nc".format(case))
        elif case == "test9":
            a = Dataset("./{0}/post_processing/abl_statistics75000.nc".format(case))

        if i == 2:
            zi_ustar = np.divide(a.variables["zi"],a.variables["ustar"])
            plt.plot(a.variables['time'][:], zi_ustar, linestyle=linestyles[cc], color=colors[cc])
            plt.ylabel(Timelabels[i])
        elif i == 5:
            zi_wstar = np.divide(a.variables["zi"], a.variables["wstar"])
            plt.plot(a.variables['time'][:],zi_wstar,linestyle = linestyles[cc], color=colors[cc])
            plt.ylabel(Timelabels[i])
        elif i == 6:
            zi_L = -(np.divide(a.variables["zi"],a.variables["L"]))
            plt.plot(a.variables["time"][:],zi_L,linestyle = linestyles[cc], color=colors[cc])
            plt.ylabel(Timelabels[i])
        else:
            plt.plot(a.variables['time'][:], a.variables["{}".format(TimeVars[i])][:]
                     , linestyle=linestyles[cc], color=colors[cc])
            plt.ylabel("{0}".format(Timelabels[i]))

        cc+=1


    path = dir + "{0}.png".format(TimeFilenames[i])

    #plt.xlim([30000,40000])
    plt.xlabel("Time [s]")
    plt.legend(Titles)
    plt.savefig(path)
    plt.close(fig)


def Ave_props(case):

    if case == "AR_0.66_120K":
        a = Dataset("./{0}/post_processing/abl_statistics00000.nc".format(case))
    elif case == "AR_0.66_120K_restart":
        a = Dataset("./{0}/post_processing/abl_statistics60000.nc".format(case))

    tstart = np.searchsorted(a.variables['time'][:],30000.0)
    tend = np.searchsorted(a.variables["time"][:],40000.0)

    zi = a.variables["zi"][tstart:tend]
    zi_ave = np.average(zi)
    ustar = a.variables["ustar"][tstart:tend]
    ustar_ave = np.average(ustar)
    tau_u = zi_ave/ustar_ave
    wstar = a.variables["wstar"][tstart:tend]
    wstar_ave = np.average(wstar)
    tau_w = zi_ave/wstar_ave
    L = np.average(a.variables["L"][tstart:tend])

    dzi_dt = np.average( d_dt(time=a.variables["time"][tstart:tend],var=zi) )
    dustar_dt = np.average( d_dt(time=a.variables["time"][tstart:tend], var=ustar) )
    dwstar_dt = np.average( d_dt(a.variables["time"][tstart:tend], wstar) )

    d_zi_ustar = dzi_dt/ustar_ave
    d_zi_wstar = dzi_dt/wstar_ave

    

    print("{0} average properties \n $z_i$ = {1}m \n $u_*$ = {2}m/s \n $\Tau_u$ = {3}mins \n $w_*$ = {4}m/s \n $\Tau_w$ = {5}mins \n -L = {6}m \n -$z_i/L$ = {7} \n $dz_i/dt$ = {8}m/s \n $du_*/dt$ = {9}m \n $dw_*$ = {10}m \n $dz_i/dt 1/u_*$ = {11} \n $dz_i/dt 1/w_*$ = {12}".format(case, zi_ave, ustar_ave, tau_u/60, wstar_ave, tau_w/60,-L,-zi_ave/L,dzi_dt,dustar_dt,dwstar_dt,d_zi_ustar,d_zi_wstar))


def Plotting(dir, cases, labels, j, filenames, Vars,Titles, linestyles, colors):
    cc = 0
    fig = plt.figure()

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Var = Vars[j]

        if case == "Ex1":
            kk = -1
        elif case == "test9":
            kk = 0
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        plt.plot(mean_profiles["{}".format(Var)][kk][:], z, linestyle=linestyles[cc], color=colors[cc])    

        cc+=1


    plt.xlabel("{}".format(labels[j]))
    plt.ylabel('Height from surface [m]')
    plt.title('Time = {0} [s]'.format(Time))
    plt.legend(Titles)
        
    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


dir = "./test9/post_processing/plots/"

cases = ["Ex1","test9"]
Titles = ["Ex1","Ex1 ABL ALM restart"]
markerstyle = ["o"]
linestyles = ["solid","dashed"]
colors = ["blue","red"]


#plotting height variables
Vars = ["u","theta"]
labels = ["velocity x [m/s]","Potential temperature [K]"]
filenames = ["vel_x","theta"]

for j in np.arange(0,len(Vars),1):
    Plotting(dir, cases, labels, j, filenames, Vars,Titles, linestyles, colors)



#plotting time variables
TimeVars = ["zi", "ustar", "zi_ustar","wstar","Tsurf", "zi_wstar","zi_L"]
Timelabels = ["$z_i$ [m]", "$u_*$ [m/s]", "$z_i/{u_*}$ [s]","$w_*$ [m/s]","$T_0$ [K]", "$z_i/{w_*}$ [s]", "$-z_i/L$ [-]"]
TimeFilenames = ["z_i_time", "u_star_time", "zi_ustar_time","w_star_time","T0_time", "zi_wstar_time", "zi_L_time"]

#plottingdvar_dt(dir)


# for case in cases:
#     Ave_props(case)


# for i in np.arange(0, len(TimeVars)):
#     plottingTimeVars(cases, Titles, TimeVars, Timelabels, TimeFilenames, i, dir, linestyles, colors)