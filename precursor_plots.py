from math import ceil, floor
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os, glob


#Mean statistics
plot_corliolis_twist = False
plot_velmag = False
plot_ww_r = False
plot_w = False
plot_pot_temp = False
plot_horz_vel = False
plot_hub_height_horz_vel = False

#HAZ analysis plots
plot_weno_z_Re_LES = False
plot_weno_z_phi_m = False
plot_weno_z_ppm_nolim_Re_LES = False
plot_weno_z_ppm_nolim_phi_m = False
plot_reduced_oneEq_Re_LES = False
plot_reduced_oneEq_phi_m = False
plot_uu_r_spectra = True
plot_ww_r_spectra = True


dir = "../../ABL_precursor/post_processing/plots/"

case = "../../ABL_precursor"

a = Dataset("./{0}/post_processing/abl_statistics60000.nc".format(case))

mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

z = mean_profiles["h"][:]

t_start = np.searchsorted(a.variables["time"],32300)
t_end = np.searchsorted(a.variables["time"],33500)

def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist

def TI(case):

    a = Dataset("{0}/post_processing/abl_statistics60000.nc".format(case))

    mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

    tstart = np.searchsorted(a.variables["time"],32500)
    t_end = np.searchsorted(a.variables["time"],33700)

    u_var = np.average(mean_profiles["u'u'_r"][tstart:t_end][:],axis=0)
    v_var = np.average(mean_profiles["v'v'_r"][tstart:t_end][:],axis=0)
    w_var = np.average(mean_profiles["w'w'_r"][tstart:t_end][:],axis=0)

    U_var = np.average(mean_profiles["u"][tstart:t_end][:],axis=0)
    V_var = np.average(mean_profiles["v"][tstart:t_end][:],axis=0)
    W_var = np.average(mean_profiles["w"][tstart:t_end][:],axis=0)

    u_pri = np.sqrt((1/3)*(u_var + v_var + w_var))
    U_bar = np.sqrt((U_var**2 + V_var**2 + W_var**2))

    I = np.round((u_pri/U_bar),decimals=2)

    return I

if plot_corliolis_twist == True:
    fig = plt.figure()

    hub_height_ind = np.searchsorted(z,90)
    u = np.array(mean_profiles.variables["u"])
    v = np.array(mean_profiles.variables["v"])

    u_2 = np.average(u[t_start:t_end],axis=0)
    v_2 = np.average(v[t_start:t_end],axis=0)

    u = u[t_start:t_end,hub_height_ind]
    v = v[t_start:t_end,hub_height_ind]


    twist = coriolis_twist(u,v)


    #coriolis twist plots
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})
    twist_2 = coriolis_twist(u=u_2,v=v_2)
    plt.plot(twist_2*(180/np.pi),z,linewidth=2)
    plt.xlabel("$\Theta$ - twist induced by coriolis [deg]")
    plt.ylabel("Distance from the surface [m]")
    plt.grid()
    plt.tight_layout()       
    path = dir + "coriolis_twist_avg.png"
    plt.savefig(path)
    plt.close(fig)

    z_zi = z/1007
    z_zi_loc = [0.1,0.8,1.1,1.2]
    text = ["$0.1z_i$","$0.8z_i$","$1.1z_i$","$1.2z_i$"]
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})
    for loc in z_zi_loc:
        z_zi_idx = np.searchsorted(z_zi,loc)
        u_loc = u_2[z_zi_idx]
        v_loc = v_2[z_zi_idx]
        plt.plot(u_loc,v_loc,"o",markersize=10)
    plt.xlabel("U [m/s]")
    plt.ylabel("V [m/s]")
    plt.legend(text)

    for loc in z_zi_loc:
        z_zi_idx = np.searchsorted(z_zi,loc)
        u_loc = u_2[z_zi_idx]
        v_loc = v_2[z_zi_idx]
        plt.arrow(0,0,u_loc,v_loc,length_includes_head=True,head_width=0.25,color="k")

    plt.grid()
    plt.tight_layout()       
    path = dir + "hodograph_avg.png"
    plt.savefig(path)
    plt.close(fig)


if plot_velmag == True:
    hvelmag = np.add( np.multiply(u,np.cos(twist[hub_height_ind])) , np.multiply(v,np.sin(twist[hub_height_ind])) )

    plt.plot(a.variables["time"][t_start:t_end],hvelmag)


u = np.average(mean_profiles.variables["u"][t_start:t_end][:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end][:],axis=0)

twist = coriolis_twist(u=u,v=v)

hvelmag = []
for i in np.arange(0,len(u),1):

    hvel = u[i]*np.cos(twist[i]) + v[i]*np.sin(twist[i])
    hvelmag.append(hvel)


w = np.average(mean_profiles.variables["w"][t_start:t_end][:],axis=0)

theta = np.average(mean_profiles.variables["theta"][t_start:t_end][:],axis=0)

u_w_r = np.average(mean_profiles.variables["u'w'_r"][t_start:t_end][:],axis=0)

w_w_r = np.average(mean_profiles.variables["w'w'_r"][t_start:t_end][:],axis=0)


if plot_ww_r == True:
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})

    plt.plot(w_w_r,z,"b-")

    plt.xlabel("Ensemble averaged vertical velocity variance [m/s]")
    plt.ylabel("Distance from surface [m]") 
    plt.grid()
    plt.tight_layout()
            
    path = dir + "w_w_r_avg.png"
    plt.savefig(path)
    plt.close(fig)

if plot_w == True:
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})

    plt.plot(w,z,"b-")

    plt.xlabel("Ensemble averaged vertical velocity [m/s]")
    plt.ylabel("Distance from surface [m]") 
    plt.grid()
    plt.tight_layout()
            
    path = dir + "w_avg.png"
    plt.savefig(path)
    plt.close(fig)

if plot_pot_temp == True:
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})

    plt.plot(theta,z,"b-")
    plt.xlabel("Ensemble averaged Potential temperature [K]")
    plt.ylabel("Distance from surface [m]") 
    plt.grid()
    plt.tight_layout()
            
    path = dir + "theta_avg.png"
    plt.savefig(path)
    plt.close(fig)


if plot_horz_vel == True:
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})

    plt.plot(hvelmag,z,"b-")
    plt.grid()
    plt.xlabel("Ensemble averaged Horizontal velocity [m/s]")
    plt.ylabel("Distance from surface [m]")

    plt.axhline(y=90, color="k", linestyle='--') 
    plt.tight_layout()
    path = dir + "Horz_vel_avg.png"
    plt.savefig(path)
    plt.close(fig)

if plot_hub_height_horz_vel == True:
    fig = plt.figure()
    plt.rcParams.update({'font.size': 12})

    plt.plot(hvelmag,z,"b-")
    plt.grid()
    plt.xlabel("Ensemble averaged Horizontal velocity [m/s]")
    plt.ylabel("Distance from surface [m]")

    plt.axhline(y=90, color="k", linestyle='--')


    hub_height_ind = np.searchsorted(z,90)
    hub_height_var = np.round(hvelmag[hub_height_ind],decimals=2)
    upper_height_ind = np.searchsorted(z,90+63)
    upper_height_var = hvelmag[upper_height_ind]
    upper_height = z[upper_height_ind]
    lower_height_ind = np.searchsorted(z,90-63)
    lower_height_var = hvelmag[lower_height_ind]
    lower_height = z[lower_height_ind]
    alpha = np.round(np.log((upper_height_var/lower_height_var))/
                        np.log((upper_height/lower_height)),decimals=2)

    I = TI(case)
    I = I[hub_height_ind]*100

    plt.text(8, 90+10,"velocity x = {0} [m/s]  \nShear exp = {1}  \nTurbulence intensity = {2}%".format(hub_height_var,alpha,I),fontsize=12)
                
                
    plt.ylim([0,90+100])  

    plt.tight_layout()
    path = dir + "Horz_vel_avg_hub_height.png"
    plt.savefig(path)
    plt.close(fig)



def R_Re_LES(dir, cases, Titles,filename,markers,colors):

    fig = plt.figure()
    R = []
    Re_LES = []
    cc=0
    for case in cases:

        a = Dataset("../../ABL/HAZ_analysis/{0}/abl_statistics00000.nc".format(case))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Tau_u = 6000
        time_start = a.variables["time"][-1] - Tau_u

        tstart = np.searchsorted(a.variables['time'][:],time_start)

        u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
        v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)

        u_w_r = np.average(mean_profiles.variables["u'w'_r"][tstart:][:],axis=0)
        v_w_r = np.average(mean_profiles.variables["v'w'_r"][tstart:][:],axis=0)

        u_w_sfs = np.average(mean_profiles.variables["u'w'_sfs"][tstart:][:],axis=0)
        v_w_sfs = np.average(mean_profiles.variables["v'w'_sfs"][tstart:][:],axis=0)

        twist = coriolis_twist(u=u,v=v)

        hvelmag = []
        hvelmag_w_r = []
        hvelmag_w_sfs = []
        for i in np.arange(0,len(u),1):

            hvel = u[i]*np.cos(twist[i]) + v[i]*np.sin(twist[i])
            hvel_w_r = u_w_r[i]*np.cos(twist[i]) + v_w_r[i]*np.sin(twist[i])
            hvel_w_sfs = u_w_sfs[i]*np.cos(twist[i]) + v_w_sfs[i]*np.sin(twist[i])
            hvelmag.append(hvel)
            hvelmag_w_r.append(hvel_w_r)
            hvelmag_w_sfs.append(hvel_w_sfs)


        Zi = np.average(a.variables["zi"][tstart:])
        u_star_ave = np.average(a.variables["ustar"][tstart:])

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

        TR = np.array(hvelmag_w_r) * -rho

        TS = np.array(hvelmag_w_sfs) * -rho

        R.append(TR[0]/TS[0])

        v_les = np.divide(TS[0:-4],(d_dz* rho)) #LES false viscosity
        v_LES = v_les[0] #LES false viscosity at first grid level
        
        l_vLES = v_LES/u_star_ave #LES false length scale

        Re_LES.append(Zi/l_vLES)

        plt.plot(Re_LES[cc], R[cc] ,marker=markers[cc],color=colors[cc],markersize=10)

        cc+=1

    if filename == "weno_z":
        plt.arrow(Re_LES[0],R[0],(Re_LES[-1]-Re_LES[0]),(R[-1]-R[0]),color="k")
        plt.arrow(Re_LES[0],R[0],130,0.8,color="k")

    plt.ylim(bottom=0.1,top=1.0)
    plt.grid()
    plt.legend(Titles)
    plt.xlabel("$Re_{LES}$ - False viscous Reynolds number [-]",fontsize=12)
    plt.ylabel("$R$ - Ratio Resolved stress to SFS stress [-]",fontsize=12)
    plt.tight_layout()
    plt.savefig(dir + "{0}_HAZ.png".format(filename))
    plt.close(fig)



def AvePhi_m(dir, cases, Titles, filename,colors,linestyles):

    fig = plt.figure()
    cc=0
    for case in cases:

        a = Dataset("../../ABL/HAZ_analysis/{0}/abl_statistics00000.nc".format(case))

        mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

        Tau_u = 6000
        time_start = a.variables["time"][-1] - Tau_u

        tstart = np.searchsorted(a.variables['time'][:],time_start)

        u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
        v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)

        twist = coriolis_twist(u=u,v=v)

        hvelmag = []
        for i in np.arange(0,len(u),1):

            hvel = u[i]*np.cos(twist[i]) + v[i]*np.sin(twist[i])
            hvelmag.append(hvel)


        Zi = np.average(a.variables["zi"][tstart:])
        u_star_ave = np.average(a.variables["ustar"][tstart:])

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
        z = mean_profiles["h"]
        #Phi_m
        z_star = (z[0:-4]*kappa)/u_star_ave
        PHI_m = np.multiply(z_star,d_dz)
    
        plt.plot(PHI_m, z[0:-4]/Zi,color=colors[cc],linestyle=linestyles[cc])

        cc+=1
 
    x = [1,1]; y = [0,0.4]
    plt.plot(x,y, 'k--')
    plt.xlim([0,2])
    plt.ylim(0,0.2)
    plt.grid()
    plt.legend(Titles)
    plt.ylabel("$z/z_{i}$ non-dimensionalised height [-]",fontsize=12)
    plt.xlabel("$\Phi_m$(z)",fontsize=12)
    plt.tight_layout()
    plt.savefig(dir + "{0}_phi_m.png".format(filename))
    plt.close(fig)    
    



if plot_weno_z_Re_LES == True:
    dir = "../../ABL/HAZ_analysis/plots/"
    cases = ["weno_z/AR_1.0","weno_z/AR_0.8","weno_z/AR_0.66","weno_z/AR_0.6"]
    Titles = ["weno_z AR = 1.0", "weno_z AR_0.8","weno_z AR_0.66","weno_z AR_0.6"]
    filename = "weno_z"
    markers =["o","o","o","o"]
    colors = ["blue","orange","green","red"]
    R_Re_LES(dir, cases, Titles,filename,markers,colors)


if plot_weno_z_phi_m == True:
    dir = "../../ABL/HAZ_analysis/plots/"
    cases = ["weno_z/AR_1.0","weno_z/AR_0.8","weno_z/AR_0.66","weno_z/AR_0.6"]
    Titles = ["weno_z AR = 1.0", "weno_z AR_0.8","weno_z AR_0.6","weno_z AR_0.6"]
    filename = "weno_z"
    colors = ["blue","orange","green","red"]
    linestyles = ["solid","solid","solid","solid"]
    AvePhi_m(dir, cases, Titles, filename,colors,linestyles)


if plot_weno_z_ppm_nolim_Re_LES == True:
    dir = "../../ABL/HAZ_analysis/plots/"
    cases = ["weno_z/AR_1.0","weno_z/AR_0.6","ppm_nolim/AR_1.0","ppm_nolim/AR_0.6"]
    Titles = ["weno_z AR = 1.0", "weno_z AR_0.6","ppm_nolim AR_1.0","ppm_nolim AR_0.6"]
    filename = "ppm_nolim_weno_z"
    markers =["o","x","o","x"]
    colors = ["blue","blue","red","red"]
    R_Re_LES(dir, cases, Titles,filename,markers,colors)

if plot_weno_z_ppm_nolim_phi_m == True:
    dir = "../../ABL/HAZ_analysis/plots/"
    cases = ["weno_z/AR_1.0","weno_z/AR_0.6","ppm_nolim/AR_1.0","ppm_nolim/AR_0.6"]
    Titles = ["weno_z AR = 1.0", "weno_z AR_0.6","ppm_nolim AR_1.0","ppm_nolim AR_0.6"]
    filename = "ppm_nolim_weno_z"
    colors = ["blue","blue","red","red"]
    linestyles = ["dashed","solid","dashed","solid"]
    AvePhi_m(dir, cases, Titles, filename,colors,linestyles)

if plot_reduced_oneEq_Re_LES == True:
    dir = "../../ABL/HAZ_analysis/plots/"
    cases = ["weno_z/AR_0.66","weno_z/AR_0.66_reduced_model_const"]
    Titles = ["weno_z AR = 0.66 $C_k = 0.1$", "weno_z AR_0.66 $C_k = 0.07$"]
    filename = "reduced_model_const"
    markers =["o","x"]
    colors = ["blue","red"]
    R_Re_LES(dir, cases, Titles,filename,markers,colors)


if plot_reduced_oneEq_phi_m == True:
    dir = "../../ABL/HAZ_analysis/plots/"
    cases = ["weno_z/AR_0.66","weno_z/AR_0.66_reduced_model_const"]
    Titles = ["weno_z AR = 0.66 $C_k = 0.1$", "weno_z AR_0.66 $C_k = 0.07$"]
    filename = "reduced_model_const"
    colors = ["blue","red"]
    linestyles = ["solid","solid"]
    AvePhi_m(dir, cases, Titles, filename,colors,linestyles)

if plot_uu_r_spectra == True:
    dir = "../../ABL/HAZ_analysis/plots/"
    cases = ["weno_z/AR_1.0","weno_z/AR_0.6","ppm_nolim/AR_1.0"]
    Titles = ["weno_z AR = 1.0  <u'u'> = ", "weno_z AR = 0.6 <u'u'> = ",
              "ppm_nolim AR = 1.0 <u'u'> = "]
    linestyles = ["dashed","solid","dashed","solid"]
    colors = ["blue","blue","red","red"]
    
    fig = plt.figure()
    cc=0
    for case in cases:
        filepath = "../../ABL/HAZ_analysis/{0}/spectral_data_uu.csv".format(case)
        df = pd.read_csv(filepath)
        freq1d = df['freq']
        e_1d = df["10.0"]

        E = str(round(np.sum(e_1d),4))
        Titles[cc] = Titles[cc] + E

        plt.loglog(freq1d, e_1d,linestyle=linestyles[cc],color=colors[cc])

        cc+=1

    plt.ylim([1e-06, 1])
    plt.xlabel('k - Wave number [1/m]',fontsize=12)
    plt.ylabel("$E_{uu}$ - Power spectral density [$m^2/s^2$]",fontsize=12)
    plt.title("$z/z_i$ = 0.025",fontsize=12)
    plt.grid()
    plt.legend(Titles)
    plt.tight_layout()
    plt.savefig(dir + "uu_spectra.png")
    plt.close(fig)


if plot_ww_r_spectra == True:
    dir = "../../ABL/HAZ_analysis/plots/"
    cases = ["weno_z/AR_1.0","weno_z/AR_0.6","ppm_nolim/AR_1.0"]
    Titles = ["weno_z AR = 1.0  <u'u'> = ", "weno_z AR = 0.6 <u'u'> = ",
              "ppm_nolim AR = 1.0 <u'u'> = "]
    linestyles = ["dashed","solid","dashed","solid"]
    colors = ["blue","blue","red","red"]
    
    fig = plt.figure()
    cc=0
    for case in cases:
        filepath = "../../ABL/HAZ_analysis/{0}/spectral_data_ww.csv".format(case)
        df = pd.read_csv(filepath)
        freq1d = df['freq']
        e_1d = df["40.0"]

        E = str(round(np.sum(e_1d),4))
        Titles[cc] = Titles[cc] + E

        plt.loglog(freq1d, e_1d,linestyle=linestyles[cc],color=colors[cc])

        cc+=1

    plt.ylim([1e-06, 1])
    plt.xlabel('k - Wave number [1/m]',fontsize=12)
    plt.ylabel("$E_{uu}$ - Power spectral density [$m^2/s^2$]",fontsize=12)
    plt.title("$z/z_i$ = 0.076",fontsize=12)
    plt.grid()
    plt.legend(Titles)
    plt.tight_layout()
    plt.savefig(dir + "ww_spectra.png")
    plt.close(fig)