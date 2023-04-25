from math import ceil, floor
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os, glob


def TI(case):

    a = Dataset("./{0}/post_processing/abl_statistics00000.nc".format(case))

    mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

    tstart = np.searchsorted(a.variables['time'][:],15000.0)

    u_var = np.average(mean_profiles["u'u'_r"][tstart:][:],axis=0)
    v_var = np.average(mean_profiles["v'v'_r"][tstart:][:],axis=0)
    w_var = np.average(mean_profiles["w'w'_r"][tstart:][:],axis=0)

    U_var = np.average(mean_profiles["u"][tstart:][:],axis=0)
    V_var = np.average(mean_profiles["v"][tstart:][:],axis=0)
    W_var = np.average(mean_profiles["w"][tstart:][:],axis=0)

    u_pri = np.sqrt((1/3)*(u_var + v_var + w_var))
    U_bar = np.sqrt((U_var**2 + V_var**2 + W_var**2))

    I = np.round((u_pri/U_bar),decimals=2)

    return I
    


def turbine_height_vars(dir, labels,cases, Vars, j, k, rows, columns, Titles, filenames, WT_colors, WT_heights, WT_rotor_D):
    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))
    fig.supylabel('Height [m]')

    for case in cases:

        a = Dataset("./{0}/post_processing/abl_statistics60000.nc".format(case))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Var = Vars[j]

        kk = k[cc]
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        #phi_m
        u = mean_profiles.variables["u"][kk]
        v = mean_profiles.variables["v"][kk]
        hvelmag = np.sqrt(u**2 + v**2)

        del_z = mean_profiles["h"][1] - mean_profiles["h"][0]

        #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
        d_dz = []
        for i in np.arange(0,len(hvelmag)-4,1):
            if i == 0:
                d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
            else:
                d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

            d_dz.append(d_dz_i)

        kappa = 0.41

        #Phi_m
        u_star = a.variables["ustar"][kk]
        z_star = (z[0:-4]*kappa)/u_star
        PHI_m = np.multiply(z_star,d_dz)


        #Phi_h
        Q0 = a.variables["Q"][kk]
        T_star = Q0/a.variables["ustar"][kk]
        z_star = (z[0:-4]*kappa)/T_star
        PHI_h = np.multiply(z_star,d_dz)


        if Var == "phi_m":
            plot_var = PHI_m
        elif Var == "phi_h":
            plot_var = PHI_h
        else:
            plot_var = mean_profiles["{}".format(Var)][kk][:]


        #plot on same plot showing wind turbine heights and variable values at height
        ax = plt.subplot(rows,columns,(cc+1))

        if len(cases) < 4:
            plt.subplots_adjust(wspace=1.2, 
                                hspace=1.2)
        else:
            plt.subplots_adjust(wspace=0.6, 
                                hspace=0.6)
        

        if Var == "phi_m" or Var == "phi_h":
            plt.plot(plot_var, z[0:-4])
            ax.set_xlim([0,5.0])
        else:
            plt.plot(plot_var, z)
        
        
        for ic in np.arange(0,len(WT_heights),1):
            plt.axhline(y=WT_heights[ic], color=WT_colors[ic], linestyle='-')

            if Var == "u":
                hub_height_ind = np.searchsorted(z,WT_heights[ic])
                hub_height_var = np.round(plot_var[hub_height_ind],decimals=2)
                upper_height_ind = np.searchsorted(z,WT_heights[ic]+WT_rotor_D[ic]/2)
                upper_height_var = np.round(plot_var[upper_height_ind],decimals=2)
                lower_height_ind = np.searchsorted(z,WT_heights[ic]-WT_rotor_D[ic]/2)
                lower_height_var = np.round(plot_var[lower_height_ind],2)
                alpha = np.round(np.log((upper_height_var/lower_height_var))/
                                 np.log(((WT_heights[ic]+WT_rotor_D[ic]/2)/(WT_heights[ic]-WT_rotor_D[ic]/2))),decimals=2)
                
                # I = TI(case)
                # turb_int = I[hub_height_ind]

                if ic == 0:
                    plt.text(7.0, WT_heights[ic]-24,"velocity x = {0} [m/s]  \nShear exp = {1}".format(hub_height_var,alpha),fontsize=8)
                else:
                    plt.text(7.0, WT_heights[ic]+5,"velocity x = {0} [m/s]  \nShear exp = {1}".format(hub_height_var,alpha),fontsize=8)
            
            
        ax.set_ylim([0,WT_heights[-1]+100])  
        ax.set_title('{0},  Time = {1} [s]'.format(Titles[cc], Time), fontsize=6)   

        cc+=1
        
    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)



def TI_z(cases,Titles,rows,columns, WT_heights, WT_colors):
    cc = 0
    fig = plt.figure()
    fig.supylabel("Turbulence Intensity [%]")
    fig.supxlabel('Height [m]')
    for case in cases:

        a = Dataset("./{0}/post_processing/abl_statistics00000.nc".format(case))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        z = mean_profiles["h"][:]

        I = TI(case)
        turb_int = I*100

        #plotting
        ax = plt.subplot(rows,columns,(cc+1))

        if len(cases) < 4:
            plt.subplots_adjust(wspace=1.2, 
                                hspace=1.2)
        else:
            plt.subplots_adjust(wspace=0.6, 
                                hspace=0.6)
            
        plt.plot(z,turb_int)

        ax.set_title('{0}'.format(Titles[cc]), fontsize=6)   

        for ic in np.arange(0,len(WT_heights),1):
            plt.axvline(x=WT_heights[ic], color=WT_colors[ic], linestyle='-')

        ax.set_xlim([0,WT_heights[-1]+100])

        cc+=1
        
    path = dir + "turbulence_intensity.png"
    plt.savefig(path)
    plt.close(fig)



dir = './AR_0.66_120K_restart/post_processing/plots/'


cases = ["AR_0.66_120K_restart"]
WT_heights = [85, 90, 150]
WT_rotor_D = [77, 126, 240]
WT_colors = ["b", "r", "g"]
Titles = ["AR = 0.66 strong capping inv \nNz = 128 $\Delta \Theta$ = 120K/km \nQ0 = 0.15Km/s Ug = 15m/s"]

Vars = ["u", "phi_m", "phi_h"]
labels = ["Velocity x [m/s]", "$\Phi_m$ [-]", "$\Phi_h$ [-]"]
filenames = ["WT_velocity_x", "WT_phi_m", "WT_phi_h"]


a = Dataset("./{0}/post_processing/abl_statistics60000.nc".format(cases[0]))

times = np.round(np.linspace(0, a.variables["time"][-1],num=6),-3)
Ind = []
for time in times:
    Ind_row = []
    for case in cases:
    
        a = Dataset("./{0}/post_processing/abl_statistics60000.nc".format(case))

        Ind_row.append(np.searchsorted(a.variables['time'][:],time))

    Ind.append(Ind_row)


if len(cases) < 4:
    rows = 1
    columns = len(cases)
else:
    rows = 2
    columns = ceil(len(cases)/2)

#TI_z(cases,Titles,rows,columns, WT_heights, WT_colors)

for j in np.arange(0,len(Vars),1):
    for k in Ind:

        turbine_height_vars(dir, labels,cases, Vars, j, k, rows, columns, Titles, filenames, WT_colors, WT_heights, WT_rotor_D)