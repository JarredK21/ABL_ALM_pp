from math import ceil, floor
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os, glob



def Main(dir, cases, Titles,rows,columns, scaling_zi, Ind, plot_on_same_plot, plot_side_by_side, linestyles, colors):


    Vars = ["u", "u", "u", "u", "theta", "theta", "theta","u'u'_r", "w'w'_r", "w'theta'_r", "u'w'_r", "u", "u", "u", "u", "u", "TKE",
            "u'u'_r", "w'w'_r", "u'w'_r"]
    labels = ["velocity x [m/s]",'$du/{dz}$ [1/s]','$\phi_m$ [-]', '$\phi_m$ [-]', "Potential temperature [K]",
                '$d theta /{dz}$ [K/m]', '$\phi_h$ [-]',"<u'u'> [$m^2/s^2$]", "<w'w'> [$m^2/s^2$]","<w'theta'> [mK/s]", 
                "<u'w'> [$m^2/s^2$]","TR+, TS+ -normalized stress [-]", "TR/TS [-]", "v_les [$m^2/s$]", "$l_{v les}$ [m]", 
                '$du/{dz}$ [1/s]', "Turbulent kinetic energy [$m^2/s^2$]", "<u'u'> [$m^2/s^2$]", "<w'w'> [$m^2/s^2$]",
                "<u'w'> [$m^2/s^2$]"]
    filenames = ["vel_x", "du_dz", "Phi_m", "Phi_m_250", "theta", "dtheta_dz", "Phi_h", "u'u'_r", "w'w'_r", 
                    "w'theta'_r", "u'w'_r", "TR_TS_z", "TRdivTS_z", "v_les", "l_v_les", "du_dz_250", "TKE", 
                    "u'u'_r_0.2", "w'w'_r_0.2","u'w'_r_0.2"]

    for j in np.arange(0,len(Vars),1):
        for k in Ind:

                if j == 1 or j == 15:
                    dU_dz(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors)
                elif j == 5:
                    dtheta_dz(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors)
                elif j == 2 or j == 3:
                    Phi_m(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors)
                elif j == 6 and a.variables["wstar"][-1] != 0:
                    Phi_h(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors)
                elif j == 11 or j == 12 or j == 13 or j == 14:
                    HAZ(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, linestyles, colors)
                elif j == 16:
                    TKE(dir, cases, labels, j, filenames,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors)
                else:
                    Plotting(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors)


def Main2(dir, cases, Titles,rows,columns, scaling_zi, Ind, plot_on_same_plot, plot_side_by_side, linestyles, colors):
    
    Vars = ["u'u'_r", "w'w'_r", "u'w'_r", "u'u'_r", "w'w'_r", "u'w'_r", "u'u'_r", "w'w'_r", "u'w'_r", "u'u'_r", 
            "w'w'_r", "u'w'_r", "u'u'_r", "w'w'_r", "v'v'_r", "v'v'_r", "v'v'_r", "v'v'_r"]
    labels = ["<u'u'> [$m^2/s^2$]", "<w'w'> [$m^2/s^2$]", "<u'w'> [$m^2/s^2$]", "<u'u'> [$m^2/s^2$]", 
                "<w'w'> [$m^2/s^2$]", "<u'w'> [$m^2/s^2$]", "<u'u'> [$m^2/s^2$]", "<w'w'> [$m^2/s^2$]", "<u'w'> [$m^2/s^2$]",
                "<u'u'> [$m^2/s^2$]", "<w'w'> [$m^2/s^2$]", "<u'w'> [$m^2/s^2$]","<u'u'> [$m^2/s^2$]", "<w'w'> [$m^2/s^2$]",
                "<v'v'> [$m^2/s^2$]", "<v'v'> [$m^2/s^2$]", "<v'v'> [$m^2/s^2$]", "<v'v'> [$m^2/s^2$]"]
    filenames = ["u'u'_r_0.05", "w'w'_r_0.05","u'w'_r_0.05", "u'u'_r_0.2", "w'w'_r_0.2","u'w'_r_0.2",
                     "u'u'_r_0.5", "w'w'_r_0.5","u'w'_r_0.5", "u'u'_r_0.8", "w'w'_r_0.8","u'w'_r_0.8", "u'u'_r", "w'w'_r",
                     "v'v'_r_0.05", "v'v'_r_0.2", "v'v'_r_0.5", "v'v'_r_0.8"]

    for j in np.arange(0,len(Vars),1):
        #only final time step

        Plotting2(dir, cases, labels, j, filenames, Vars, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors)


def dU_dz(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors):

    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))

    if scaling_zi == True:
        fig.supylabel("$z/z_{i}$ non-dimensionalised height [-]")
    else:
        fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        kk = k[cc] #case specific time index

        u = mean_profiles.variables["u"][kk]
        v = mean_profiles.variables["v"][kk]
        hvelmag = np.sqrt(u**2 + v**2)
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        Zi,u_star,w_star = Ave_props(case)

        del_z = z[1] - z[0]

        #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
        dU_dz = []
        for i in np.arange(0,len(hvelmag)-4,1):
            if i == 0:
                dU_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
            else:
                dU_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

            dU_dz.append(dU_dz_i)

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                hspace=0.6)

            ax.set_title('{0}, Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)


        if scaling_zi == True:
            plt.plot(dU_dz, z[0:-4]/Zi, linestyle = linestyles[cc], color = colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,1.5])
            elif plot_on_same_plot == True:
                plt.ylim([0,1.5])
        else:
            plt.plot(dU_dz, z[0:-4], linestyle = linestyles[cc], color = colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,1.5])
            elif plot_on_same_plot == True:
                plt.ylim([0,1.5])
        

        if j == 15 and scaling_zi == True:
            if plot_side_by_side == True:
                ax.set_ylim([0,0.5])
            elif plot_on_same_plot == True:
                plt.ylim([0,0.5])
        elif j == 15:
            if plot_side_by_side == True:
                ax.set_ylim([0,250])
            elif plot_on_same_plot == True:
                plt.ylim([0,250])

        cc+=1
    
    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)
    

    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def dtheta_dz(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors):

    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))

    if scaling_zi == True:
        fig.supylabel("$z/z_{i}$ non-dimensionalised height [-]")
    else:
        fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        kk = k[cc] #case specific time index

        theta = mean_profiles.variables["theta"][kk]
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        Zi,u_star,w_star = Ave_props(case)

        del_z = z[1] - z[0]

        #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
        dtheta_dz = []
        for i in np.arange(0,len(theta)-4,1):
            if i == 0:
                dtheta_dz_i = ((-(25/12)*theta[i]+4*theta[i+1]-3*theta[i+2]+(4/3)*theta[i+3]-(1/4)*theta[i+4])/del_z)
            else:
                dtheta_dz_i = ((theta[i+1] - theta[i-1])/(2*del_z))

            dtheta_dz.append(dtheta_dz_i)

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                hspace=0.6)

            ax.set_title('{0}, Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)


        if scaling_zi == True:
            plt.plot(dtheta_dz, z[0:-4]/Zi, linestyle = linestyles[cc], color = colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,1.5])
            elif plot_on_same_plot == True:
                plt.ylim([0,1.5])
        else:
            plt.plot(dtheta_dz, z[0:-4], linestyle = linestyles[cc], color = colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,1.5])
            elif plot_on_same_plot == True:
                plt.ylim([0,1.5])
        

        if j == 15 and scaling_zi == True:
            if plot_side_by_side == True:
                ax.set_ylim([0,0.5])
            elif plot_on_same_plot == True:
                plt.ylim([0,0.5])
        elif j == 15:
            if plot_side_by_side == True:
                ax.set_ylim([0,250])
            elif plot_on_same_plot == True:
                plt.ylim([0,250])

        cc+=1
    
    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)
    

    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def dvar_dz2(dir, cases, labels, j, filenames, Vars, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors):

    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))

    if scaling_zi == True:
        fig.supylabel("$z/z_{i}$ non-dimensionalised height [-]")
    else:
        fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Var = Vars[j]

        kk = -1 #final time step

        u = mean_profiles["{}".format(Var)][kk]
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        Zi,u_star,w_star = Ave_props(case)

        del_z = mean_profiles["h"][1] - mean_profiles["h"][0]

        #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
        d_dz = []
        for i in np.arange(0,len(u)-4,1):
            if i == 0:
                d_dz_i = ((-(25/12)*u[i]+4*u[i+1]-3*u[i+2]+(4/3)*u[i+3]-(1/4)*u[i+4])/del_z)
            else:
                d_dz_i = ((u[i+1] - u[i-1])/(2*del_z))

            d_dz.append(d_dz_i)

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                hspace=0.6)

            ax.set_title('{0}, Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)


        if scaling_zi == True:
            plt.plot(d_dz, z[0:-4]/Zi, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,0.2])
            elif plot_on_same_plot == True:
                plt.ylim([0,0.2])
        else:
            plt.plot(d_dz, z[0:-4], linestyle=linestyles[cc], color = colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,0.2])
            elif plot_on_same_plot == True:
                plt.ylim([0,0.2])

        cc+=1
    
    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)
    

    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def TKE(dir, cases, labels, j, filenames,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors):
    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))

    if scaling_zi == True:
        fig.supylabel("$z/z_{i}$ non-dimensionalised height [-]")
    else:
        fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        kk = k[cc] #case specific time index
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        Zi,u_star,w_star = Ave_props(case)

        TKE = 0.5*(mean_profiles["u'u'_r"][kk]+mean_profiles["v'v'_r"][kk]+mean_profiles["w'w'_r"][kk])

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                    hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                    hspace=0.6)
        
            ax.set_title('{0}, Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)

        if scaling_zi == True:
            plt.plot(TKE[:], z/Zi, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,1.5])
            elif plot_on_same_plot == True:
                plt.ylim([0,1.5])
        else:
            plt.plot(TKE[:], z[:], linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,700])
            elif plot_on_same_plot == True:
                plt.ylim([0,700])


        cc+=1

    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)
    
    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def Phi_m(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors):

    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))

    if scaling_zi == True:
        fig.supylabel("$z/z_{i}$ non-dimensionalised height [-]")
    else:
        fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Var = Vars[j]

        kk = k[cc] #case specific time index

        u = mean_profiles.variables["u"][kk]
        v = mean_profiles.variables["v"][kk]
        hvelmag = np.sqrt(u**2 + v**2)
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        Zi,u_star_ave,w_star_ave = Ave_props(case)

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
        u_star = a.variables["ustar"][kk]
        #Phi_m
        z_star = (z[0:-4]*kappa)/u_star
        PHI_m = np.multiply(z_star,d_dz)

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                    hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                    hspace=0.6)

            ax.set_title('{0}, Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)

        if scaling_zi == True:
            plt.plot(PHI_m, z[0:-4],Zi, linestyle=linestyles[cc], color=colors[cc])
        else:
            plt.plot(PHI_m, z[0:-4], linestyle=linestyles[cc], color=colors[cc])


            if j == 3 and scaling_zi == True:
                if plot_side_by_side == True:
                    ax.set_ylim([0,0.4])
                elif plot_on_same_plot == True:
                    plt.ylim([0,0.4])
            elif j == 3:
                if plot_side_by_side == True:
                    ax.set_ylim([0,250])
                elif plot_on_same_plot == True:
                    plt.ylim([0,250])
        
        #ax.set_xlim([0,2.5])

        cc+=1

    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)

    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def Phi_h(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors):

    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))

    
    if scaling_zi == True:
        fig.supylabel("$z/z_{i}$ non-dimensionalised height [-]")
    else:
        fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Var = Vars[j]

        kk = k[cc]

        u = mean_profiles["u"][kk]
        v = mean_profiles["v"][kk]
        hvelmag = np.sqrt(u**2 + v**2)
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        Zi,u_star_ave,w_star_ave = Ave_props(case)

        del_z = mean_profiles["h"][1] - mean_profiles["h"][0]

        #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
        d_dz = []
        for i in np.arange(0,len(u)-4,1):
            if i == 0:
                d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
            else:
                d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

            d_dz.append(d_dz_i)

        kappa = 0.41
        Q0 = a.variables["Q"][kk]
        T_star = Q0/a.variables["ustar"][kk]

        #Phi_h
        z_star = (z[0:-4]*kappa)/T_star
        PHI_h = np.multiply(z_star,d_dz)

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                    hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                    hspace=0.6)

            ax.set_title('{0}, Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)

        if scaling_zi == True:
            plt.plot(PHI_h, z[0:-4]/Zi, linestyle=linestyles[cc], color=colors[cc])
        else:
            plt.plot(PHI_h, z[0:-4], linestyle=linestyles[cc], color=colors[cc])

        cc+=1

    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)

    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def HAZ(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, linestyles, colors):

    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))
    fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        kk = k[cc] #Case specific time index
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        Zi,u_star_ave,w_star_ave = Ave_props(case)

        Var = Vars[j]

        u = mean_profiles.variables["u"][kk]
        v = mean_profiles.variables["v"][kk]
        hvelmag = np.sqrt(u**2 + v**2)

        del_z = mean_profiles["h"][1] - mean_profiles["h"][0]

        #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
        d_dz = []
        for i in np.arange(0,len(u)-4,1):
            if i == 0:
                d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
            else:
                d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

            d_dz.append(d_dz_i)

        rho = 1
        TR = -mean_profiles.variables["u'w'_r"][kk][:]*rho #resolved stress (z)
        TS = -mean_profiles.variables["u'w'_sfs"][kk][:]*rho #sfs stress (z)

        T0 = rho*a.variables["ustar"][kk]**2 #Total stress
        TR_plus = np.true_divide(TR, T0) #normalised resolved stress
        TS_plus = np.true_divide(TS, T0) #normalised sfs stress

        v_les = np.divide(TS[0:-4],(d_dz*rho)) #LES false viscosity
        v_LES = v_les[0] #LES false viscosity at first grid level
        
        l_vles = v_les/a.variables["ustar"][kk]
        l_vLES = v_LES/a.variables["ustar"][kk] #LES false length scale

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                    hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                    hspace=0.6)

            ax.set_title('{0},  Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)


        if j == 11:
            plt.plot(TR_plus, z/Zi, linestyle=linestyles[cc], color=colors[cc])
            plt.plot(TS_plus, z/Zi, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.legend(["TR+", "TS+"])
                ax.set_ylim([0,0.4])
            elif plot_on_same_plot == True:
                plt.legend(["TR+", "TS+"])
                plt.ylim([0,0.4])
        elif j == 12:
            plt.plot(np.true_divide(TR, TS), z/Zi, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,0.4])
            elif plot_on_same_plot == True:
                plt.ylim([0,0.4])
        elif j == 13:
            plt.plot(v_les, z[0:-4]/Zi, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,0.4])
            elif plot_on_same_plot == True:
                plt.ylim([0,0.4])
        elif j == 14:
            plt.plot(l_vles, z[0:-4],Zi, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,0.4])
            elif plot_on_same_plot == True:
                plt.ylim([0,0.4])

        cc+=1

    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)

    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def Plotting(dir, cases, labels, j, filenames, Vars,k, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors):
    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))

    if scaling_zi == True:
        fig.supylabel("$z/z_{i}$ non-dimensionalised height [-]")
    else:
        fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Var = Vars[j]

        kk = k[cc]
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        Zi,u_star_ave,w_star_ave = Ave_props(case)

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                    hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                    hspace=0.6)
            
            ax.set_title('{0},  Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)

        if scaling_zi == True:
            plt.plot(mean_profiles["{}".format(Var)][kk][:], z/Zi, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                if j == 17 or j == 18 or j == 19:
                    ax.set_ylim([0, 0.2])
                else:
                    ax.set_ylim([0,1.5])
            elif plot_on_same_plot == True:
                if j == 17 or j == 18 or j == 19:
                    plt.ylim([0, 0.2])
                else:
                    plt.ylim([0,1.5])
        else:
            plt.plot(mean_profiles["{}".format(Var)][kk][:], z, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,700])
            elif plot_on_same_plot == True:
                plt.ylim([0,700])        

        cc+=1

    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)
        
    #path = "./GB_MB/{0}_{1}.png".format(filenames[j],Time) #Ganesh vs main branch
    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def plottingTimeVars(cases, Titles, TimeVars, Timelabels, TimeFilenames, i, dir, linestyles, colors):
    cc = 0 #case counter

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        fig = plt.figure() #separate plots

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

        plt.xlabel("Time [s]")
    

        path = dir + "{0}.png".format(TimeFilenames[i])

        #separate plots
        plt.title('{0}'.format(Titles[cc]))
        plt.savefig(path)
        plt.close(fig)

        cc+=1


def Plotting2(dir, cases, labels, j, filenames, Vars, rows, columns, Titles, scaling_zi, plot_on_same_plot, plot_side_by_side, linestyles, colors):
    cc = 0
    fig = plt.figure()
    fig.supxlabel('{0}'.format(labels[j]))

    if scaling_zi == True:
        fig.supylabel("$z/z_{i}$ non-dimensionalised height [-]")
    else:
        fig.supylabel('Height [m]')

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Var = Vars[j]

        kk = -1 #final time step
        
        Time = floor(a.variables["time"][kk])

        z = mean_profiles["h"][:]

        #Zi,u_star_ave,w_star_ave = Ave_props(case)

        Zi = a.variables["zi"][-1]

        if plot_side_by_side == True:
            ax = plt.subplot(rows,columns,(cc+1))

            if len(cases) < 4:
                plt.subplots_adjust(wspace=1.2, 
                                    hspace=1.2)
            else:
                plt.subplots_adjust(wspace=0.6, 
                                    hspace=0.6)
            
            ax.set_title('{0},  Time = {1} [s]'.format(Titles[cc], Time), fontsize=10)

        if scaling_zi == True:
            plt.plot(mean_profiles["{}".format(Var)][kk][:], z/Zi, linestyle=linestyles[cc], color=colors[cc])

            if plot_side_by_side == True:
                if j == 0 or j == 1 or j == 2 or j == 14:
                    ax.set_ylim([0,0.05])
                elif j == 3 or j == 4 or j == 5 or j == 15:
                    ax.set_ylim([0,0.2])
                elif j == 6 or j == 7 or j == 8 or j == 16:
                    ax.set_ylim([0.4,0.6])
                elif j == 9 or j == 10 or j == 11 or j == 17:
                    ax.set_ylim([0.7,0.9])
            elif plot_on_same_plot == True:
                if j == 0 or j == 1 or j == 2 or j == 14:
                    plt.ylim([0,0.05])
                elif j == 3 or j == 4 or j == 5 or j == 15:
                    plt.ylim([0,0.2])
                elif j == 6 or j == 7 or j == 8 or j == 16:
                    plt.ylim([0.4,0.6])
                elif j == 9 or j == 10 or j == 11 or j == 17:
                    plt.ylim([0.7,0.9])
        else:
            plt.plot(mean_profiles["{}".format(Var)][kk][:], z, linestyle=linestyles[cc], color=colors[cc])
            if plot_side_by_side == True:
                ax.set_ylim([0,700])
            elif plot_on_same_plot == True:
                plt.ylim([0,700])        

        cc+=1

    if plot_on_same_plot == True:
        plt.title('Time = {0} [s]'.format(Time))
        plt.legend(Titles)
        
    path = dir + "{0}_{1}.png".format(filenames[j],Time)
    plt.savefig(path)
    plt.close(fig)


def Ave_props(case):

    stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
    a = Dataset("./{}".format(stats[0]))

    Zi = np.average(a.variables["zi"][:])
    u_star = np.average(a.variables["ustar"][:])
    tau_u = Zi/u_star
    w_star = np.average(a.variables["wstar"][:])
    tau_w = Zi/w_star
    L = np.average(a.variables["L"][:])

    print(Zi, u_star, tau_u/60, w_star, tau_w/60,-L,-Zi/L)

    return Zi, u_star, w_star


def spec_props(case):

    stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
    a = Dataset("./{}".format(stats[0]))

    t_spec = np.searchsorted(a.variables["time"],32300)
    t_end = np.searchsorted(a.variables["time"],33500)

    Zi = np.average(a.variables["zi"][t_spec:t_end])
    u_star = np.average(a.variables["ustar"][t_spec:t_end])
    tau_u = Zi/u_star
    w_star = np.average(a.variables["wstar"][t_spec:t_end])
    tau_w = Zi/w_star
    L = np.average(a.variables["L"][t_spec:t_end])

    print(Zi, u_star, tau_u/60, w_star, tau_w/60,-L,-Zi/L)


def AvePhi_m(dir, cases, Titles, markerstyle, scaling_zi, linestyles, colors):

    fig = plt.figure()
    cc = 0
    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        u = np.average(mean_profiles.variables["u"][:][:],axis=0)
        v = np.average(mean_profiles.variables["v"][:][:],axis=0)
        hvelmag = np.sqrt(u**2 + v**2)

        z = mean_profiles["h"][:]

        Zi,u_star_ave,w_star_ave = Ave_props(case)

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
        z_star = (z[0:-4]*kappa)/u_star_ave
        PHI_m = np.multiply(z_star,d_dz)
    
        if scaling_zi == True:
            plt.plot(PHI_m, z[0:-4]/Zi, linestyle=linestyles[cc], color=colors[cc], label="{}".format(Titles[cc]))
            #plt.plot(PHI_m, z[0:-4]/Zi)
            plt.ylabel("$z/z_{i}$ non-dimensionalised height [-]")
            plt.ylim(0,0.2)
        else:
            plt.plot(PHI_m, z,linestyle=linestyles[cc], color=colors[cc], label="{}".format(Titles[cc]))
            plt.ylabel("Height [m]")
            plt.ylim([0,100])

        cc+=1

    x = [1,1]; y = [0,0.4]
    plt.plot(x,y, 'k--')
    plt.xlim([0,2])
    plt.grid()
    plt.legend(Titles)
    plt.title("$phi_m$")
    plt.xlabel("$phi_m$(z)")
    plt.savefig(dir + "Phi_m.png")
    plt.close(fig)


def AvePhi_h(dir, cases, Titles, markerstyle, scaling_zi, linestyles, colors):

    fig = plt.figure()
    cc = 0
    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profile

        u = np.average(mean_profiles.variables["u"][:][:],axis=0)
        v = np.average(mean_profiles.variables["v"][:][:],axis=0)
        hvelmag = np.sqrt(u**2 + v**2)

        z = mean_profiles["h"][:]

        Zi,u_star_ave,w_star_ave = Ave_props(case)

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
        Q0 = a.variables["Q"][-1]
        T_star = Q0/u_star_ave

        #Phi_h
        z_star = (z[0:-4]*kappa)/T_star
        PHI_h = np.multiply(z_star,d_dz)
    
        if scaling_zi == True:
            plt.plot(PHI_h, z[0:-4]/Zi, linestyle=linestyles[cc], color=colors[cc], label="{}".format(Titles[cc]))
            #plt.plot(PHI_m, z[0:-4]/Zi)
            plt.ylabel("$z/z_{i}$ non-dimensionalised height [-]")
            plt.ylim(0,0.2)
        else:
            plt.plot(PHI_h, z[0:-4],linestyle=linestyles[cc], color=colors[cc], label="{}".format(Titles[cc]))
            plt.ylabel("Height [m]")
            plt.ylim([0,100])

        cc+=1

    x = [1,1]; y = [0,0.4]
    plt.plot(x,y, 'k--')
    plt.xlim([0,5])
    plt.grid()
    plt.legend(Titles, loc='upper right')
    plt.title("$phi_h$")
    plt.xlabel("$phi_h$(z)")
    plt.savefig(dir + "Phi_h.png")
    plt.close(fig)


def R_Re_LES(dir, cases, markerstyle, colors):

    cc = 0
    fig = plt.figure()

    for case in cases:

        stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
        a = Dataset("./{}".format(stats[0]))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        u = np.average(mean_profiles.variables["u"][:][:],axis=0)
        v = np.average(mean_profiles.variables["v"][:][:],axis=0)
        hvelmag = np.sqrt(u**2 + v**2)

        cosine_theta = u/hvelmag


        u_w_r = np.average(mean_profiles.variables["u'w'_r"][:][:],axis=0)
        v_w_r = np.average(mean_profiles.variables["v'w'_r"][:][:],axis=0)
        hvelmag_w_r = np.sqrt(u_w_r**2 + v_w_r**2)
        u_w_sfs = np.average(mean_profiles.variables["u'w'_sfs"][:][:],axis=0)
        v_w_sfs = np.average(mean_profiles.variables["v'w'_sfs"][:][:],axis=0)
        hvelmag_w_sfs = np.sqrt(u_w_sfs**2 + v_w_sfs**2)

        Zi,u_star_ave,w_star_ave = Ave_props(case)

        del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
        
        #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
        d_dz = []
        for i in np.arange(0,len(hvelmag)-4,1):
            if i == 0:
                d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
            else:
                d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

            d_dz.append(d_dz_i)

        rho = 1
        # TR = -u_w_r*rho #resolved stress (z)
        # TS = -u_w_sfs*rho #sfs stress (z)

        TR = hvelmag_w_r * rho

        TS = hvelmag_w_sfs * rho

        R = TR[0]/TS[0]

        v_les = np.divide(TS[0:-4],(d_dz* rho)) #LES false viscosity
        v_LES = v_les[0] #LES false viscosity at first grid level
        
        l_vLES = v_LES/u_star_ave #LES false length scale

        Re_LES = Zi/l_vLES

        plt.plot(Re_LES, R, color=colors[cc], marker=markerstyle[cc],markersize=10)

        cc+=1

    plt.axhline(y=1, color='k', linestyle='--', linewidth=2)
    plt.axvline(x=350, color='k', linestyle='--', linewidth=2)
    plt.grid()
    plt.legend(Titles)
    plt.xlabel("$Re_{LES}$ - False viscous Reynolds number [-]")
    plt.ylabel("$R$ - Ratio Resolved stress to SFS stress [-]")
    plt.title("HAZ")
    plt.savefig(dir + "HAZ.png")
    plt.close(fig)


def plottingdvar_dt(cases, Titles, dir, linestyles, colors):

    variables = ["zi","ustar","wstar","Tsurf","zi","zi"]
    labels = ["$dz_i/dt$ [m/s]", "$du_*/dt [m/s^2]$", "$dw_*/dt [m/s^2]$", "$dT_0/dt [K m/s^2]$",
                "$dz_i/dt 1/w_*$ [-]","$dz_i/dt 1/u_*$ [-]"]
    filenames = ["dzi_dt", "dustar_dt", "dwstar_dt", "dT0_dt", "dzi_dt_wstar","dzi_dt_ustar"]


    ic = 0
    for variable in variables:
        cc = 0
        for case in cases:

            stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
            a = Dataset("./{}".format(stats[0]))
            var = a.variables[variable]
            time = a.variables["time"][:]
            d_var_dt = []
            for i in np.arange(0,len(var)-1,1):
                d_var_dt.append((var[i+1]-var[i])/(time[i+1]-time[i]))

            if filenames[ic] == "dzi_dt_wstar":
                d_var_dt = np.divide(d_var_dt,a.variables["wstar"][:-1])
            elif filenames[ic] == "dzi_dt_ustar":
                d_var_dt = np.divide(d_var_dt,a.variables["ustar"][:-1])

            fig = plt.figure()
            plt.rcParams.update({'font.size': 12})

            def plot_std(time, d_var_dt,colors):
                #calculate bin edges using 3Tau
                bins = [0] #bin edges index's
                Tau = np.divide(a.variables["zi"],a.variables["wstar"])

                i = 0 #bin edge counter
                time_stamp = a.variables["time"][0]
                while time_stamp < a.variables["time"][-1]:
                    Tau_i = Tau[bins[i]]
                    time_stamp += 3*Tau_i #increase time by 3Tau
                    bins.append( np.searchsorted(a.variables['time'][:],time_stamp) ) #append next bin edge

                    i+=1 #increase bin edge counter


                #Declare the array containing the series you want to plot. 
                n_bins = 20
                n_steps           = int(len(time[:-1])/n_bins) #number of rolling steps for the mean/std.

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

                plt.plot(bin_center, mean, linewidth = 2, linestyle = "solid", color=colors)#mean curve
                plt.axhline(0.01, linestyle="--")
                plt.axhline(-0.01, linestyle="--")
                plt.fill_between(bin_center,under_line,over_line, linestyle = "solid", color=colors, alpha=0.1)#std curve
                
            plot_std(time,d_var_dt,colors="blue")
            
            plt.ylabel(labels[ic])
            plt.xlabel("Time [s]")        

            path = dir + "{0}.png".format(filenames[ic])

            #separate plots
            #plt.title('{0}'.format(Titles[cc]))
            plt.tight_layout()
            plt.savefig(path)
            plt.close(fig)

            cc += 1

        ic += 1


dir = "../../ABL_precursor/post_processing/plots/"

cases = ["../../ABL_precursor"]
Titles = ["ABL precursor"]
markerstyle = ["o","s"]
linestyles = ["solid","solid"]
colors = ["k"]


TimeVars = ["zi", "ustar", "zi_ustar","wstar","Tsurf", "zi_wstar","zi_L"]
Timelabels = ["$z_i$ [m]", "$u_*$ [m/s]", "$z_i/{u_*}$ [s]","$w_*$ [m/s]","$T_0$ [K]", "$z_i/{w_*}$ [s]", "$-z_i/L$ [-]"]
TimeFilenames = ["z_i_time", "u_star_time", "zi_ustar_time","w_star_time","T0_time", "zi_wstar_time", "zi_L_time"]


stats = glob.glob("{0}/post_processing/abl_statistics*".format(cases[0]))
a = Dataset("./{}".format(stats[0]))


times = np.linspace(a.variables["time"][0], a.variables["time"][-1],num=3)
Ind = []
for time in times:
    Ind_row = []
    for case in cases:
    
        stats = glob.glob("{0}/post_processing/abl_statistics*".format(cases[0]))
        a = Dataset("./{}".format(stats[0]))

        Ind_row.append(np.searchsorted(a.variables['time'][:],time))

    Ind.append(Ind_row)

scaling_zi = True

if len(cases) < 4:
    rows = 1
    columns = len(cases)
else:
    rows = 2
    columns = ceil(len(cases)/2)

#choosing format for plot
plot_side_by_side = False
plot_on_same_plot = True

spec_props(case=cases[0])
# Main(dir, cases, Titles,rows,columns, scaling_zi, Ind, plot_on_same_plot, plot_side_by_side, linestyles, colors)

# AvePhi_m(dir, cases, Titles, markerstyle, scaling_zi, linestyles, colors)

# AvePhi_h(dir, cases, Titles, markerstyle, scaling_zi, linestyles, colors)

# R_Re_LES(dir, cases,markerstyle, colors)

# for i in np.arange(0, len(TimeVars)):
#    plottingTimeVars(cases, Titles, TimeVars, Timelabels, TimeFilenames, i, dir, linestyles, colors)

#Main2(dir, cases, Titles,rows,columns, scaling_zi, Ind, plot_on_same_plot, plot_side_by_side, linestyles, colors)

#Cosine_theta_z(dir, cases, scaling_zi, markerstyle, colors, linestyles)
#theta_z(dir, cases, scaling_zi, markerstyle, colors, linestyles)

#plottingdvar_dt(cases, Titles, dir, linestyles, colors)