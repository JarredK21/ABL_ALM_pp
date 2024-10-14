from math import ceil, floor
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy import interpolate
import csv
import matplotlib.patches as patches


def dz_calc(u,z):
    d_dz = []
    for i in np.arange(0,len(u)-1,1):
        d_dz.append((u[i+1]-u[i])/(z[i+1]-z[i]))

    return np.array(d_dz)


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def dt_calc(u,dt):
    #compute time derivative using first order forward difference
    d_dt = []
    for i in np.arange(0,len(u)-1,1):
        d_dt.append( (u[i+1]-u[i])/dt )

    return d_dt




#Figure 3-2 evolution boundary layer
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/weno_z/AR_1.0/"
df = Dataset(in_dir+"abl_statistics00000.nc")
Mean_profiles = df.groups["mean_profiles"]
time = np.array(df.variables["time"])
z = np.array(Mean_profiles.variables["h"])
plt.rcParams['font.size'] = 16
times = [0.0,1000,3000,9000,15000,20000]
for it in times:
    Tstart_idx = np.searchsorted(time,it)
    theta = np.array(Mean_profiles.variables["theta"][Tstart_idx])
    dtheta_dz = dz_calc(theta,z)
    w_theta = np.array(Mean_profiles.variables["w'theta'_r"][Tstart_idx])
    w_w_r = np.array(Mean_profiles.variables["w'w'_r"][Tstart_idx])
    fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,10),sharey=True)
    ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax3.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    ax1.plot(dtheta_dz,z[:-1])
    ax1.set_xlabel("$d\\theta/dz$\n[K/m]")
    ax1.grid()
    ax1.set_ylim([0,700])
    ax2.plot(w_theta,z)
    ax2.set_xlabel("$\langle w' \\theta' \\rangle$\n[Km/s]")
    ax2.grid()
    ax2.set_ylim([0,700])
    ax3.plot(w_w_r,z)
    ax3.set_xlabel("$\langle w'w' \\rangle$\n$[m^2/s^2]$")
    ax3.grid()
    ax3.set_ylim([0,700])
    fig.supylabel("Height from surface [m]")
    fig.suptitle("Time={}s".format(it))
    plt.tight_layout()
    plt.savefig("../../Thesis/Figures/evo_{}.png".format(it))
    plt.close(fig)


#Figure 3-3 Tau u time
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/weno_z/AR_1.0/"
df = Dataset(in_dir+"abl_statistics00000.nc")
time = np.array(df.variables["time"])
zi = np.array(df.variables["zi"])
u_star = np.array(df.variables["ustar"])
tau_u = zi/u_star
plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(14,8))
plt.plot(time,tau_u)
plt.xlabel("Time [s]")
plt.ylabel("$\\tau_u$ [s]")
plt.grid()
plt.tight_layout()
plt.savefig("../../Thesis/Figures/Tau_u.png")
plt.close(fig)





#Figure 3-4 Haz analysis weno_z
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

cases = ["weno_z new BC/AR_1.0/","weno_z/AR_0.8/","weno_z new BC/AR_0.66/","weno_z new BC/AR_0.6/"]
colors = ["b","y","g","r"]
labels = ["weno_z AR = 1.0", "weno_z AR = 0.8", "weno_z AR = 0.66", "weno_z AR = 0.6"]

plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(8.5,6.4))
ix = 0
for case in cases:
    df = Dataset(in_dir+case+"abl_statistics00000.nc")

    mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

    tstart = np.searchsorted(df.variables['time'][:],15000.0)

    zi = np.average(np.array(df.variables["zi"][tstart:]))
    ustar = np.average(np.array(df.variables["ustar"][tstart:]))

    u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
    v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
    twist = coriolis_twist(u,v)
    hvelmag = []
    for i in np.arange(0,len(twist)):
        hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
    hvelmag = np.array(hvelmag)

    u_w_r = np.average(mean_profiles.variables["u'w'_r"][tstart:][:],axis=0)
    v_w_r = np.average(mean_profiles.variables["v'w'_r"][tstart:][:],axis=0)
    hvelmag_w_r = []
    for i in np.arange(0,len(twist)):
        hvelmag_w_r.append(u_w_r[i] * np.cos(twist[i]) + v_w_r[i] * np.sin(twist[i]))
    hvelmag_w_r = np.array(hvelmag_w_r)

    u_w_sfs = np.average(mean_profiles.variables["u'w'_sfs"][tstart:][:],axis=0)
    v_w_sfs = np.average(mean_profiles.variables["v'w'_sfs"][tstart:][:],axis=0)
    hvelmag_w_sfs = []
    for i in np.arange(0,len(twist)):
        hvelmag_w_sfs.append(u_w_sfs[i] * np.cos(twist[i]) + v_w_sfs[i] * np.sin(twist[i]))
    hvelmag_w_sfs = np.array(hvelmag_w_sfs)

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

    TR = -hvelmag_w_r * rho

    TS = -hvelmag_w_sfs * rho

    R = TR[0]/TS[0]

    v_les = np.divide(TS[0:-4],(d_dz* rho)) #LES false viscosity
    v_LES = v_les[0] #LES false viscosity at first grid level
    
    l_vLES = v_LES/ustar #LES false length scale

    Re_LES = zi/l_vLES


    plt.plot(Re_LES, R, color=colors[ix], marker="o",markersize=10)

    ix+=1


plt.xlabel("False viscous Reynolds number [-]")
plt.ylabel("$\mathfrak{R}$ - Ratio Resolved stress to SFS stress [-]")
plt.grid()
plt.legend(labels)
plt.ylim([0,1.0])
# plt.xlim([300,600])
plt.tight_layout()
plt.savefig("../../Thesis/Figures/R_ReLES_weno_z.png")
plt.close(fig)

#phi_m
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

cases = ["weno_z new BC/AR_1.0/","weno_z/AR_0.8/","weno_z new BC/AR_0.66/","weno_z new BC/AR_0.6/"]
colors = ["b","y","g","r"]
labels = ["weno_z AR = 1.0", "weno_z AR = 0.8", "weno_z AR = 0.66", "weno_z AR = 0.6"]

plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(8.5,6.4))
ix = 0
for case in cases:
    df = Dataset(in_dir+case+"abl_statistics00000.nc")

    mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

    z = np.array(mean_profiles.variables["h"])

    tstart = np.searchsorted(df.variables['time'][:],15000.0)

    zi = np.average(np.array(df.variables["zi"][tstart:]))
    z_zi = z/zi
    ustar = np.average(np.array(df.variables["ustar"][tstart:]))
    kappa = 0.41

    u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
    v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
    twist = coriolis_twist(u,v)
    hvelmag = []
    for i in np.arange(0,len(twist)):
        hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
    hvelmag = np.array(hvelmag)

    del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
    #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
    d_dz = []
    for i in np.arange(0,len(hvelmag)-4,1):
        if i == 0:
            d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
        else:
            d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

        d_dz.append(d_dz_i)
    
    d_dz = np.array(d_dz)

    phi_m = ((kappa*z[:-4])/ustar)*d_dz

    z_idx = np.searchsorted(z_zi,0.22)

    plt.plot(phi_m[:z_idx],z_zi[:z_idx],color=colors[ix],linestyle="-")
    ix+=1

plt.xlabel("$\Phi_m(z)$")
plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
plt.grid()
plt.legend(labels)
plt.axvline(x=1.0,linestyle="--",color="k")
plt.ylim([0,0.2])
plt.xlim(left=0.0)
plt.tight_layout()
plt.savefig("../../Thesis/Figures/phi_m_weno_z.png")
plt.close(fig)



#figure 3-5 HAZ weno_z vs ppm_nolim
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

cases = ["weno_z new BC/AR_1.0/","weno_z new BC/AR_0.6/","ppm_no_lim/AR_1.0/","ppm_no_lim/AR_0.6/"]
colors = ["b","b","r","r"]
markers = ["o","x","o","x"]
linestyles = ["--","-","--","-"]
labels = ["weno_z AR = 1.0","weno_z AR = 0.6","ppm_nolim AR = 1.0","ppm_nolim AR = 0.6"]

plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(8.5,6.4))
ix = 0
for case in cases:
    df = Dataset(in_dir+case+"abl_statistics00000.nc")

    mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

    tstart = np.searchsorted(df.variables['time'][:],15000.0)

    zi = np.average(np.array(df.variables["zi"][tstart:]))
    ustar = np.average(np.array(df.variables["ustar"][tstart:]))

    u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
    v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
    twist = coriolis_twist(u,v)
    hvelmag = []
    for i in np.arange(0,len(twist)):
        hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
    hvelmag = np.array(hvelmag)

    u_w_r = np.average(mean_profiles.variables["u'w'_r"][tstart:][:],axis=0)
    v_w_r = np.average(mean_profiles.variables["v'w'_r"][tstart:][:],axis=0)
    hvelmag_w_r = []
    for i in np.arange(0,len(twist)):
        hvelmag_w_r.append(u_w_r[i] * np.cos(twist[i]) + v_w_r[i] * np.sin(twist[i]))
    hvelmag_w_r = np.array(hvelmag_w_r)

    u_w_sfs = np.average(mean_profiles.variables["u'w'_sfs"][tstart:][:],axis=0)
    v_w_sfs = np.average(mean_profiles.variables["v'w'_sfs"][tstart:][:],axis=0)
    hvelmag_w_sfs = []
    for i in np.arange(0,len(twist)):
        hvelmag_w_sfs.append(u_w_sfs[i] * np.cos(twist[i]) + v_w_sfs[i] * np.sin(twist[i]))
    hvelmag_w_sfs = np.array(hvelmag_w_sfs)

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

    TR = -hvelmag_w_r * rho

    TS = -hvelmag_w_sfs * rho

    R = TR[0]/TS[0]

    v_les = np.divide(TS[0:-4],(d_dz* rho)) #LES false viscosity
    v_LES = v_les[0] #LES false viscosity at first grid level
    
    l_vLES = v_LES/ustar #LES false length scale

    Re_LES = zi/l_vLES


    plt.plot(Re_LES, R, color=colors[ix], marker=markers[ix],markersize=10)

    ix+=1


plt.xlabel("False viscous Reynolds number [-]")
plt.ylabel("$\mathfrak{R}$ - Ratio Resolved stress to SFS stress [-]")
plt.grid()
plt.legend(labels)
plt.ylim([0,1.0])
# plt.xlim([300,600])
plt.tight_layout()
plt.savefig("../../Thesis/Figures/R_Re_weno_z_ppm_nolim.png")
plt.close(fig)

#phi_m
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

cases = ["weno_z new BC/AR_1.0/","weno_z new BC/AR_0.6/","ppm_no_lim/AR_1.0/","ppm_no_lim/AR_0.6/"]
colors = ["b","b","r","r"]
markers = ["o","x","o","x"]
linestyles = ["--","-","--","-"]
labels = ["weno_z AR = 1.0","weno_z AR = 0.6","ppm_nolim AR = 1.0","ppm_nolim AR = 0.6"]

plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(8.5,6.4))
ix = 0
for case in cases:
    df = Dataset(in_dir+case+"abl_statistics00000.nc")

    mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

    z = np.array(mean_profiles.variables["h"])

    tstart = np.searchsorted(df.variables['time'][:],15000.0)

    zi = np.average(np.array(df.variables["zi"][tstart:]))
    z_zi = z/zi
    ustar = np.average(np.array(df.variables["ustar"][tstart:]))
    kappa = 0.41

    u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
    v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
    twist = coriolis_twist(u,v)
    hvelmag = []
    for i in np.arange(0,len(twist)):
        hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
    hvelmag = np.array(hvelmag)

    del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
    #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
    d_dz = []
    for i in np.arange(0,len(hvelmag)-4,1):
        if i == 0:
            d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
        else:
            d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

        d_dz.append(d_dz_i)
    
    d_dz = np.array(d_dz)

    phi_m = ((kappa*z[:-4])/ustar)*d_dz

    z_idx = np.searchsorted(z_zi,0.22)

    plt.plot(phi_m[:z_idx],z_zi[:z_idx],color=colors[ix],linestyle=linestyles[ix])
    ix+=1

plt.xlabel("$\Phi_m(z)$")
plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
plt.grid()
plt.legend(labels)
plt.axvline(x=1.0,linestyle="--",color="k")
plt.ylim([0,0.2])
plt.xlim(left=0.0)
plt.tight_layout()
plt.savefig("../../Thesis/Figures/phi_m_weno_z_ppm_nolim.png")
plt.close(fig)


#find spectral data
# #fig 3-6 u'u' spectra
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# cases = ["weno_z/AR_1.0/","weno_z/AR_0.6/","ppm_no_lim/AR_0.6/"]
# colors = ["b","b","r"]
# linestyles = ["--","-","--"]
# labels = ["weno_z AR = 1.0", "weno_z AR = 0.6", "ppm_nolim AR = 1.0"]

# plt.rcParams['font.size'] = 16
# fig = plt.figure()
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")
#     time = np.array(df.variables["time"])
#     tstart = np.searchsorted(df.variables['time'][:],15000.0)
#     zi = np.average(df.variables["zi"][tstart:])
#     height = 15
#     col = "7.5"
#     df = pd.read_csv(in_dir+case+"spectral_data.csv")
#     freq1d = df['freq']
#     e_1d = df[col]

#     plt.loglog(freq1d, e_1d, linestyle=linestyles[ix], color=colors[ix])


#     ix+=1

# plt.xlabel("Wave number [1/m]")
# plt.ylabel("$E_{uu}(k)$ - Power spectral density [$m^4/s^4$]")
# plt.title("$z_z_i = 0.025$")
# plt.grid()
# plt.ylim(bottom = 1e-06)
# plt.xlim(right = 1e-01)
# plt.legend(labels)
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Euu.png")
# plt.close(fig)


# #fig 3-6 w'w' spectra
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# cases = ["weno_z/AR_1.0/","weno_z/AR_0.6/","ppm_no_lim/AR_0.6/"]
# colors = ["b","b","r"]
# linestyles = ["--","-","--"]
# labels = ["weno_z AR = 1.0", "weno_z AR = 0.6", "ppm_nolim AR = 1.0"]

# plt.rcParams['font.size'] = 16
# fig = plt.figure()
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")
#     time = np.array(df.variables["time"])
#     tstart = np.searchsorted(df.variables['time'][:],15000.0)
#     zi = np.average(df.variables["zi"][tstart:])
#     height = 40
#     col = "47.5"
#     df = pd.read_csv(in_dir+case+"spectral_data.csv")
#     freq1d = df['freq']
#     e_1d = df[col]

#     plt.loglog(freq1d, e_1d, linestyle=linestyles[ix], color=colors[ix])


#     ix+=1

# plt.xlabel("Wave number [1/m]")
# plt.ylabel("$E_{ww}(k)$ - Power spectral density [$m^4/s^4$]")
# plt.title("$z_z_i = 0.08$")
# plt.grid()
# plt.ylim(bottom = 1e-06)
# plt.xlim(right = 1e-01)
# plt.legend(labels)
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Eww.png")
# plt.close(fig)



#figure 3-6 reduced model constant HAZ analysis
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

cases = ["weno_z new BC/AR_0.66/","weno_z new BC/AR_0.66_reduced_model_constant/"]
colors = ["b","r"]
markers = ["o","x"]
linestyles = ["-","-"]
labels = ["weno_z AR = 0.66 $C_k=0.1$","weno_z AR = 0.66 $C_k=0.07$"]

plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(8.5,6.4))
ix = 0
for case in cases:
    df = Dataset(in_dir+case+"abl_statistics00000.nc")

    mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

    tstart = np.searchsorted(df.variables['time'][:],15000.0)

    zi = np.average(np.array(df.variables["zi"][tstart:]))
    ustar = np.average(np.array(df.variables["ustar"][tstart:]))

    u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
    v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
    twist = coriolis_twist(u,v)
    hvelmag = []
    for i in np.arange(0,len(twist)):
        hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
    hvelmag = np.array(hvelmag)

    u_w_r = np.average(mean_profiles.variables["u'w'_r"][tstart:][:],axis=0)
    v_w_r = np.average(mean_profiles.variables["v'w'_r"][tstart:][:],axis=0)
    hvelmag_w_r = []
    for i in np.arange(0,len(twist)):
        hvelmag_w_r.append(u_w_r[i] * np.cos(twist[i]) + v_w_r[i] * np.sin(twist[i]))
    hvelmag_w_r = np.array(hvelmag_w_r)

    u_w_sfs = np.average(mean_profiles.variables["u'w'_sfs"][tstart:][:],axis=0)
    v_w_sfs = np.average(mean_profiles.variables["v'w'_sfs"][tstart:][:],axis=0)
    hvelmag_w_sfs = []
    for i in np.arange(0,len(twist)):
        hvelmag_w_sfs.append(u_w_sfs[i] * np.cos(twist[i]) + v_w_sfs[i] * np.sin(twist[i]))
    hvelmag_w_sfs = np.array(hvelmag_w_sfs)

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

    TR = -hvelmag_w_r * rho

    TS = -hvelmag_w_sfs * rho

    R = TR[0]/TS[0]

    v_les = np.divide(TS[0:-4],(d_dz* rho)) #LES false viscosity
    v_LES = v_les[0] #LES false viscosity at first grid level
    
    l_vLES = v_LES/ustar #LES false length scale

    Re_LES = zi/l_vLES


    plt.plot(Re_LES, R, color=colors[ix], marker=markers[ix],markersize=10)

    ix+=1


plt.xlabel("False viscous Reynolds number [-]")
plt.ylabel("$\mathfrak{R}$ - Ratio Resolved stress to SFS stress [-]")
plt.grid()
plt.legend(labels)
plt.ylim([0,1.0])
# plt.xlim([300,600])
plt.tight_layout()
plt.savefig("../../Thesis/Figures/R_Re_weno_z_reduced_model_const.png")
plt.close(fig)

#phi_m
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

cases = ["weno_z new BC/AR_0.66/","weno_z new BC/AR_0.66_reduced_model_constant/"]
colors = ["b","r"]
markers = ["o","x"]
linestyles = ["-","-"]
labels = ["weno_z AR = 0.66 $C_k=0.1$","weno_z AR = 0.66 $C_k=0.07$"]

plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(8.5,6.4))
ix = 0
for case in cases:
    df = Dataset(in_dir+case+"abl_statistics00000.nc")

    mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

    z = np.array(mean_profiles.variables["h"])

    tstart = np.searchsorted(df.variables['time'][:],15000.0)

    zi = np.average(np.array(df.variables["zi"][tstart:]))
    z_zi = z/zi
    ustar = np.average(np.array(df.variables["ustar"][tstart:]))
    kappa = 0.41

    u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
    v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
    twist = coriolis_twist(u,v)
    hvelmag = []
    for i in np.arange(0,len(twist)):
        hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
    hvelmag = np.array(hvelmag)

    del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
    #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
    d_dz = []
    for i in np.arange(0,len(hvelmag)-4,1):
        if i == 0:
            d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
        else:
            d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

        d_dz.append(d_dz_i)
    
    d_dz = np.array(d_dz)

    phi_m = ((kappa*z[:-4])/ustar)*d_dz

    z_idx = np.searchsorted(z_zi,0.22)

    plt.plot(phi_m[:z_idx],z_zi[:z_idx],color=colors[ix],linestyle=linestyles[ix])
    ix+=1

plt.xlabel("$\Phi_m(z)$")
plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
plt.grid()
plt.legend(labels)
plt.axvline(x=1.0,linestyle="--",color="k")
plt.ylim([0,0.2])
plt.xlim(left=0.0)
plt.tight_layout()
plt.savefig("../../Thesis/Figures/phi_m_weno_z_reduced_model_const.png")
plt.close(fig)


#fig 3-7 u'u'
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

cases = ["weno_z/AR_1.0/","weno_z/AR_0.6/","ppm_no_lim/AR_0.6/"]
colors = ["b","b","r"]
linestyles = ["--","-","--"]
labels = ["weno_z AR = 1.0", "weno_z AR = 0.6", "ppm_nolim AR = 1.0"]

plt.rcParams['font.size'] = 16
fig = plt.figure()
ix = 0
for case in cases:
    df = Dataset(in_dir+case+"abl_statistics00000.nc")
    Mean_profiles = df.groups["mean_profiles"]
    print(Mean_profiles)
    time = np.array(df.variables["time"])
    Tstart_idx = np.searchsorted(time,np.max(time)-5000)
    z = np.array(Mean_profiles.variables["h"])
    zi = np.average(np.array(df["zi"][Tstart_idx:]))
    z_zi = z/zi
    uu = np.average(np.array(Mean_profiles.variables["u'u'_r"][Tstart_idx:]),axis=0)
    plt.plot(uu,z_zi,color=colors[ix],linestyle=linestyles[ix])
    ix+=1
plt.xlabel("$\langle u'u' \\rangle$ $[m^2/s^2]$")
plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
plt.ylim([0,0.2])
plt.legend(labels,loc="upper left")
plt.grid()
plt.axhline(y=0.025,linestyle="-",color="k")
plt.tight_layout()
plt.savefig("../../Thesis/Figures/uu.png")
plt.close(fig)

#w'w'
fig = plt.figure()
ix = 0
for case in cases:
    df = Dataset(in_dir+case+"abl_statistics00000.nc")
    Mean_profiles = df.groups["mean_profiles"]
    time = np.array(df.variables["time"])
    Tstart_idx = np.searchsorted(time,np.max(time)-5000)
    z = np.array(Mean_profiles.variables["h"])
    zi = np.average(np.array(df["zi"][Tstart_idx:]))
    z_zi = z/zi
    ww = np.average(np.array(Mean_profiles.variables["w'w'_r"][Tstart_idx:]),axis=0)
    plt.plot(ww,z_zi,color=colors[ix],linestyle=linestyles[ix])
    ix+=1
plt.xlabel("$\langle w'w' \\rangle$ $[m^2/s^2]$")
plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
plt.legend(labels,loc="upper left")
plt.ylim([0,0.2])
plt.grid()
plt.axhline(y=0.076,linestyle="-",color="k")
plt.tight_layout()
plt.savefig("../../Thesis/Figures/ww.png")
plt.close(fig)


#fig 3-9
#dzi/dt 1/u*
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/60000/"
df = Dataset(in_dir+"abl_statistics60000.nc")
Mean_profiles = df.groups["mean_profiles"]

Time = np.array(df.variables["time"])
dt = Time[1]-Time[0]
zi = np.array(df.variables["zi"])
u_star = np.array(df.variables["ustar"])
w_star = np.array(df.variables["wstar"])
dzi_dt = dt_calc(zi,dt)
dzi_dt_u_star = np.true_divide(dzi_dt,u_star[:-1])
dzi_dt_w_star = np.true_divide(dzi_dt,w_star[:-1])

#moving statistics
ts_dzi_dt_u_star = pd.Series(dzi_dt_u_star, index=Time[:-1])
ts_dzi_dt_w_star = pd.Series(dzi_dt_w_star, index=Time[:-1])


tau_u = np.true_divide(zi,u_star)
tau_w = np.true_divide(zi,w_star)

#Average global statistics
glob_u_star = np.average(u_star)
glob_w_star = np.average(w_star)
glob_tau_u = np.average(tau_u)
glob_tau_w = np.average(tau_w)


plt.figure(figsize=(14,8))
plt.plot(Time[:-1],dzi_dt_u_star)
window_idx = int((glob_tau_u)/dt)
ts_dzi_dt_u_star.rolling(window=window_idx).mean().plot(style='k--')
ts_dzi_dt_u_star.rolling(window=window_idx).std().plot(style='r--')
plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("$dz_i/dt 1/u_{star}$ [-]",fontsize=16)
plt.legend(["$dz_i/dt 1/u_{star}$","Mean","Std","0.01","-0.01"])
plt.tight_layout()
plt.savefig("../../Thesis/Figures/dzi_dt_1_ustar.png")
plt.close(fig)

#dzi/dt 1/w*
plt.figure(figsize=(14,8))
plt.plot(Time[:-1],dzi_dt_w_star)
window_idx = int((glob_tau_w)/dt)
ts_dzi_dt_w_star.rolling(window=window_idx).mean().plot(style='k--')
ts_dzi_dt_w_star.rolling(window=window_idx).std().plot(style='r--')
plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("$dz_i/dt 1/w_{star}$ [-]",fontsize=16)
plt.legend(["$dz_i/dt 1/w_{star}$","Mean","Std","0.01","-0.01"])
plt.tight_layout()
plt.savefig("../../Thesis/Figures/dzi_dt_1_wstar.png")
plt.close(fig)



#Figure 3-10
#horizontal velocity
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/60000/"
df = Dataset(in_dir+"abl_statistics60000.nc")
Time = np.array(df.variables["time"])
dt = Time[1]-Time[0]

zi = np.array(df.variables["zi"])
L = np.array(df.variables["L"])
zi_L = -np.true_divide(zi,L)

Mean_profiles = df.groups["mean_profiles"]
z = np.array(Mean_profiles.variables["h"])

u = np.array(Mean_profiles.variables["u"])
v = np.array(Mean_profiles.variables["v"])

twist = coriolis_twist(u,v)
hvelmag = []
for i in np.arange(0,len(twist)):
    hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
hvelmag = np.array(hvelmag)

#hub height
z_hub = 90
z_hub_idx = np.searchsorted(z,z_hub)
hvelmag_hub = hvelmag[:,z_hub_idx]

in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
df = Dataset(in_dir+"abl_statistics70000.nc")
Time_2 = np.array(df.variables["time"])
dt = Time[1]-Time[0]

zi = np.array(df.variables["zi"])
L = np.array(df.variables["L"])
zi_L_2 = -np.true_divide(zi,L)

Mean_profiles = df.groups["mean_profiles"]
z = np.array(Mean_profiles.variables["h"])

u = np.array(Mean_profiles.variables["u"])
v = np.array(Mean_profiles.variables["v"])

twist = coriolis_twist(u,v)
hvelmag = []
for i in np.arange(0,len(twist)):
    hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
hvelmag = np.array(hvelmag)

#hub height
z_hub = 90
z_hub_idx = np.searchsorted(z,z_hub)
hvelmag_hub_2 = hvelmag[:,z_hub_idx]

Time = np.concatenate((Time,Time_2))
hvelmag_hub = np.concatenate((hvelmag_hub,hvelmag_hub_2))
zi_L = np.concatenate((zi_L,zi_L_2))


plt.figure(figsize=(14,8))
plt.plot(Time,hvelmag_hub)
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("horizontal velocity [m/s]",fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig("../../Thesis/Figures/Time_Uhub.png")
plt.close(fig)


#-zi/L
plt.figure(figsize=(14,8))
plt.plot(Time,zi_L)
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("$-z_i/L$ [m]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig("../../Thesis/Figures/Time_zi_L.png")
plt.close(fig)



#Figure 3-11
#horizontal velocity
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
df = Dataset(in_dir+"abl_statistics70000.nc")
Time = np.array(df.variables["time"])

tstart = 38000
tstart_idx = np.searchsorted(Time,tstart)

zi = np.array(df.variables["zi"])
glob_zi = np.average(zi[tstart_idx:])

Mean_profiles = df.groups["mean_profiles"]
z = np.array(Mean_profiles.variables["h"])

z_zi = z/glob_zi

u = np.average(Mean_profiles.variables["u"][tstart_idx:],axis=0)
v = np.average(Mean_profiles.variables["v"][tstart_idx:],axis=0)

twist = coriolis_twist(u,v)
hvelmag = []
for i in np.arange(0,len(twist)):
    hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
hvelmag = np.array(hvelmag)


theta = np.average(Mean_profiles.variables["theta"][tstart_idx:],axis=0)

u_w_r = np.average(Mean_profiles.variables["u'w'_r"][tstart_idx:],axis=0)

#mean profiles
#U
plt.figure(figsize=(14,8))
plt.plot(hvelmag,z_zi)
plt.xlabel("Streamwise velocity [m/s]",fontsize=16)
plt.ylabel("$z/z_i$ [-]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig("../../Thesis/Figures/U.png")
plt.close(fig)

#mean profiles 0.2zi
#U
plt.figure(figsize=(14,8))
plt.plot(hvelmag,z_zi)
plt.xlabel("Streamwise velocity [m/s]",fontsize=16)
plt.ylabel("$z/z_i$ [-]",fontsize=16)
plt.ylim([0,0.2])
plt.tight_layout()
plt.grid()
plt.savefig("../../Thesis/Figures/U_rotor.png")
plt.close(fig)

#Coriolis twist
plt.figure(figsize=(14,8))
plt.plot(np.degrees(twist),z_zi)
plt.xlabel("Flow angle [deg]",fontsize=16)
plt.ylabel("$z/z_i$ [-]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig("../../Thesis/Figures/flow_angle.png")
plt.close(fig)


#u'w'_r
plt.figure(figsize=(14,8))
plt.plot(u_w_r,z_zi)
plt.xlabel("$\langle u'w' \\rangle ^r$ $[m^2/s^2]$",fontsize=16)
plt.ylabel("$z/z_i$ [-]",fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig("../../Thesis/Figures/u_w_r.png")
plt.close(fig)

#Theta
plt.figure(figsize=(14,8))
plt.plot(theta,z_zi)
plt.xlabel("Potential temperature [K]",fontsize=16)
plt.ylabel("$z/z_i$ [-]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig("../../Thesis/Figures/Pot_temp.png")
plt.close(fig)

fu = interpolate.interp1d(z,u)
fv = interpolate.interp1d(z,v)
heights = np.array([0.1,0.4,0.8,1.0,1.1,1.2])
plt.figure(figsize=(14,8))
for height in heights:
    height_m = height*glob_zi
    u_h = fu(height_m)
    v_h = fv(height_m)
    plt.arrow(0,0,u_h,v_h,length_includes_head=True,color="#1f77b4",head_length=0.05,head_width=0.05)
    plt.text(u_h,v_h,"${}z_i$".format(height))

plt.xlabel("$U$ - average velocity [m/s]",fontsize=16)
plt.ylabel("$V$ - average velocity [m/s]",fontsize=16)
xtemp = [-1,-2]; ytemp= [-1,-2]
plt.xlim(left=0)
plt.ylim(bottom=0)
plt.grid()
plt.tight_layout()
plt.savefig("../../Thesis/Figures/hodograph.png")
plt.close(fig)

#3-12 isocontour xy plane u'
in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
df = Dataset(in_dir+"sampling70000.nc")
print(df)
