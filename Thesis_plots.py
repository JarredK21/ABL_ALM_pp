from math import ceil, floor
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy import interpolate
import csv
import matplotlib.patches as patches
import pyFAST.input_output as io
from multiprocessing import Pool


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


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


def energy_contents_check(Var,e_fft,signal,dt):

    E = (1/dt)*np.sum(e_fft)

    q = np.sum(np.square(signal))

    E2 = q

    print(Var, E, E2, abs(E2/E))    


def temporal_spectra(signal,dt,Var):

    fs =1/dt
    n = len(signal) 
    if n%2==0:
        nhalf = int(n/2+1)
    else:
        nhalf = int((n+1)/2)
    frq = np.arange(nhalf)*fs/n
    Y   = np.fft.fft(signal)
    PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
    PSD[1:-1] = PSD[1:-1]*2


    energy_contents_check(Var,PSD,signal,dt)

    return frq, PSD


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def tranform_fixed_frame(y,z,Theta):

    Y = y*np.cos(Theta) - z*np.sin(Theta)
    Z = y*np.sin(Theta) + z*np.cos(Theta)

    return Y,Z


def hard_filter(signal,cutoff,dt,filter_type):

    N = len(signal)
    spectrum = np.fft.fft(signal)
    F = np.fft.fftfreq(N,dt)
    #F = (1/(dt*N)) * np.arange(N)
    if filter_type=="lowpass":
        spectrum_filter = spectrum*(np.abs(F)<cutoff)
    elif filter_type=="highpass":
        spectrum_filter = spectrum*(np.abs(F)>cutoff)
    elif filter_type=="bandpass":
        spectrum_filter = spectrum*(np.abs(F)>cutoff[0])
        spectrum_filter = spectrum_filter*(np.abs(F)<cutoff[1])
        

    spectrum_filter = np.fft.ifft(spectrum_filter)

    return np.real(spectrum_filter)

def stats(y):

    print(round(np.mean(y),2))
    print(round(np.std(y),2))


def actuator_asymmetry_calc(it):

    xo = np.array(WT.variables["xyz"][it,1:301,0])
    yo = np.array(WT.variables["xyz"][it,1:301,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB1 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB1 = np.array(WT.variables["xyz"][it,1:301,2]) - Rotor_coordinates[2]

    xo = np.array(WT.variables["xyz"][it,301:601,0])
    yo = np.array(WT.variables["xyz"][it,301:601,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB2 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB2 = np.array(WT.variables["xyz"][it,301:601,2]) - Rotor_coordinates[2]


    xo = np.array(WT.variables["xyz"][it,601:901,0])
    yo = np.array(WT.variables["xyz"][it,601:901,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB3 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB3 = np.array(WT.variables["xyz"][it,601:901,2]) - Rotor_coordinates[2]


    IyB1 = np.sum(hvelB1[it]*zB1)*dr
    IzB1 = np.sum(hvelB1[it]*yB1)*dr

    IyB2 = np.sum(hvelB2[it]*zB2)*dr
    IzB2 = np.sum(hvelB2[it]*yB2)*dr


    IyB3 = np.sum(hvelB3[it]*zB3)*dr
    IzB3 = np.sum(hvelB3[it]*yB3)*dr

    return IyB1+IyB2+IyB3, IzB1+IzB2+IzB3





# #Figure 3-14 mean CDF 2d x-y plane 90m u, w velocities
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
# df = pd.read_csv(in_dir+"PDF_data_uu.csv")
# PDF_uu_mean = np.array(df["mean"])
# X_uu = np.array(df["X"])

# fig = plt.figure()
# plt.plot(X_uu,PDF_uu_mean,"-k")
# plt.xlabel("Fluctuating streamwise velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.axvline(x=-0.61,linestyle="--",color="b",label="low speed streaks")
# plt.axvline(x=0.76,linestyle="--",color="r",label="high speed regions")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_u_velocity_PDF.png")
# plt.close(fig)


# CDF_i = 0
# CDF = []
# dx = X_uu[1]-X_uu[0]
# for f in PDF_uu_mean:
#     CDF_i+=f*dx
#     CDF.append(CDF_i)

# fig = plt.figure()
# plt.plot(X_uu,CDF,"-k")
# plt.xlabel("Fluctuating streamwise velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_u_velocity_CDF.png")
# plt.close(fig)


# df = pd.read_csv(in_dir+"PDF_data_ww.csv")
# PDF_ww_mean = np.array(df["mean"])
# X_ww = np.array(df["X"])

# fig = plt.figure()
# plt.plot(X_ww,PDF_ww_mean,"-k")
# plt.xlabel("Fluctuating vertical velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.axvline(x=-0.48,linestyle="--",color="b",label="downdrafts")
# plt.axvline(x=0.42,linestyle="--",color="r",label="updrafts")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_w_velocity_PDF.png")
# plt.close(fig)

# CDF_i = 0
# CDF = []
# dx = X_ww[1]-X_ww[0]
# for f in PDF_ww_mean:
#     CDF_i+=f*dx
#     CDF.append(CDF_i)

# fig = plt.figure()
# plt.plot(X_ww,CDF,"-k")
# plt.xlabel("Fluctuating vertical velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_w_velocity_CDF.png")
# plt.close(fig)

# #Figure 3-2 evolution boundary layer
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/weno_z/AR_1.0/"
# df = Dataset(in_dir+"abl_statistics00000.nc")
# Mean_profiles = df.groups["mean_profiles"]
# time = np.array(df.variables["time"])
# z = np.array(Mean_profiles.variables["h"])
# plt.rcParams['font.size'] = 16
# times = [0.0,1000,3000,9000,15000,20000]
# for it in times:
#     Tstart_idx = np.searchsorted(time,it)
#     theta = np.array(Mean_profiles.variables["theta"][Tstart_idx])
#     dtheta_dz = dz_calc(theta,z)
#     w_theta = np.array(Mean_profiles.variables["w'theta'_r"][Tstart_idx])
#     w_w_r = np.array(Mean_profiles.variables["w'w'_r"][Tstart_idx])
#     fig,(ax1,ax2,ax3) = plt.subplots(1,3,figsize=(10,10),sharey=True)
#     ax1.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#     ax2.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#     ax3.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
#     ax1.plot(dtheta_dz,z[:-1])
#     ax1.set_xlabel("$d\\theta/dz$\n[K/m]")
#     ax1.grid()
#     ax1.set_ylim([0,700])
#     ax2.plot(w_theta,z)
#     ax2.set_xlabel("$\langle w' \\theta' \\rangle$\n[Km/s]")
#     ax2.grid()
#     ax2.set_ylim([0,700])
#     ax3.plot(w_w_r,z)
#     ax3.set_xlabel("$\langle w'w' \\rangle$\n$[m^2/s^2]$")
#     ax3.grid()
#     ax3.set_ylim([0,700])
#     fig.supylabel("Height from surface [m]")
#     fig.suptitle("Time={}s".format(it))
#     plt.tight_layout()
#     plt.savefig("../../Thesis/Figures/evo_{}.png".format(it))
#     plt.close(fig)


# #Figure 3-3 Tau u time
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/weno_z/AR_1.0/"
# df = Dataset(in_dir+"abl_statistics00000.nc")
# time = np.array(df.variables["time"])
# zi = np.array(df.variables["zi"])
# u_star = np.array(df.variables["ustar"])
# tau_u = zi/u_star
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(time,tau_u)
# plt.xlabel("Time [s]")
# plt.ylabel("$\\tau_u$ [s]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Tau_u.png")
# plt.close(fig)





# #Figure 3-4 Haz analysis weno_z
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# cases = ["weno_z new BC/AR_1.0/","weno_z/AR_0.8/","weno_z new BC/AR_0.66/","weno_z new BC/AR_0.6/"]
# colors = ["b","y","g","r"]
# labels = ["weno_z AR = 1.0", "weno_z AR = 0.8", "weno_z AR = 0.66", "weno_z AR = 0.6"]

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(8.5,6.4))
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")

#     mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

#     tstart = np.searchsorted(df.variables['time'][:],15000.0)

#     zi = np.average(np.array(df.variables["zi"][tstart:]))
#     ustar = np.average(np.array(df.variables["ustar"][tstart:]))

#     u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
#     v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
#     twist = coriolis_twist(u,v)
#     hvelmag = []
#     for i in np.arange(0,len(twist)):
#         hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
#     hvelmag = np.array(hvelmag)

#     u_w_r = np.average(mean_profiles.variables["u'w'_r"][tstart:][:],axis=0)
#     v_w_r = np.average(mean_profiles.variables["v'w'_r"][tstart:][:],axis=0)
#     hvelmag_w_r = []
#     for i in np.arange(0,len(twist)):
#         hvelmag_w_r.append(u_w_r[i] * np.cos(twist[i]) + v_w_r[i] * np.sin(twist[i]))
#     hvelmag_w_r = np.array(hvelmag_w_r)

#     u_w_sfs = np.average(mean_profiles.variables["u'w'_sfs"][tstart:][:],axis=0)
#     v_w_sfs = np.average(mean_profiles.variables["v'w'_sfs"][tstart:][:],axis=0)
#     hvelmag_w_sfs = []
#     for i in np.arange(0,len(twist)):
#         hvelmag_w_sfs.append(u_w_sfs[i] * np.cos(twist[i]) + v_w_sfs[i] * np.sin(twist[i]))
#     hvelmag_w_sfs = np.array(hvelmag_w_sfs)

#     del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
#     #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
#     d_dz = []
#     for i in np.arange(0,len(hvelmag)-4,1):
#         if i == 0:
#             d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
#         else:
#             d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

#         d_dz.append(d_dz_i)

#     rho = 1
#     # TR = -u_w_r*rho #resolved stress (z)
#     # TS = -u_w_sfs*rho #sfs stress (z)

#     TR = -hvelmag_w_r * rho

#     TS = -hvelmag_w_sfs * rho

#     R = TR[0]/TS[0]

#     v_les = np.divide(TS[0:-4],(d_dz* rho)) #LES false viscosity
#     v_LES = v_les[0] #LES false viscosity at first grid level
    
#     l_vLES = v_LES/ustar #LES false length scale

#     Re_LES = zi/l_vLES


#     plt.plot(Re_LES, R, color=colors[ix], marker="o",markersize=10)

#     ix+=1


# plt.xlabel("False viscous Reynolds number [-]")
# plt.ylabel("$\mathfrak{R}$ - Ratio Resolved stress to SFS stress [-]")
# plt.grid()
# plt.legend(labels)
# plt.ylim([0,1.0])
# # plt.xlim([300,600])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/R_ReLES_weno_z.png")
# plt.close(fig)

# #phi_m
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# cases = ["weno_z new BC/AR_1.0/","weno_z/AR_0.8/","weno_z new BC/AR_0.66/","weno_z new BC/AR_0.6/"]
# colors = ["b","y","g","r"]
# labels = ["weno_z AR = 1.0", "weno_z AR = 0.8", "weno_z AR = 0.66", "weno_z AR = 0.6"]

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(8.5,6.4))
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")

#     mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

#     z = np.array(mean_profiles.variables["h"])

#     tstart = np.searchsorted(df.variables['time'][:],15000.0)

#     zi = np.average(np.array(df.variables["zi"][tstart:]))
#     z_zi = z/zi
#     ustar = np.average(np.array(df.variables["ustar"][tstart:]))
#     kappa = 0.41

#     u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
#     v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
#     twist = coriolis_twist(u,v)
#     hvelmag = []
#     for i in np.arange(0,len(twist)):
#         hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
#     hvelmag = np.array(hvelmag)

#     del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
#     #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
#     d_dz = []
#     for i in np.arange(0,len(hvelmag)-4,1):
#         if i == 0:
#             d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
#         else:
#             d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

#         d_dz.append(d_dz_i)
    
#     d_dz = np.array(d_dz)

#     phi_m = ((kappa*z[:-4])/ustar)*d_dz

#     z_idx = np.searchsorted(z_zi,0.22)

#     plt.plot(phi_m[:z_idx],z_zi[:z_idx],color=colors[ix],linestyle="-")
#     ix+=1

# plt.xlabel("$\Phi_m(z)$")
# plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
# plt.grid()
# plt.legend(labels)
# plt.axvline(x=1.0,linestyle="--",color="k")
# plt.ylim([0,0.2])
# plt.xlim(left=0.0)
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/phi_m_weno_z.png")
# plt.close(fig)



# #figure 3-5 HAZ weno_z vs ppm_nolim
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# cases = ["weno_z new BC/AR_1.0/","weno_z new BC/AR_0.6/","ppm_no_lim/AR_1.0/","ppm_no_lim/AR_0.6/"]
# colors = ["b","b","r","r"]
# markers = ["o","x","o","x"]
# linestyles = ["--","-","--","-"]
# labels = ["weno_z AR = 1.0","weno_z AR = 0.6","ppm_nolim AR = 1.0","ppm_nolim AR = 0.6"]

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(8.5,6.4))
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")

#     mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

#     tstart = np.searchsorted(df.variables['time'][:],15000.0)

#     zi = np.average(np.array(df.variables["zi"][tstart:]))
#     ustar = np.average(np.array(df.variables["ustar"][tstart:]))

#     u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
#     v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
#     twist = coriolis_twist(u,v)
#     hvelmag = []
#     for i in np.arange(0,len(twist)):
#         hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
#     hvelmag = np.array(hvelmag)

#     u_w_r = np.average(mean_profiles.variables["u'w'_r"][tstart:][:],axis=0)
#     v_w_r = np.average(mean_profiles.variables["v'w'_r"][tstart:][:],axis=0)
#     hvelmag_w_r = []
#     for i in np.arange(0,len(twist)):
#         hvelmag_w_r.append(u_w_r[i] * np.cos(twist[i]) + v_w_r[i] * np.sin(twist[i]))
#     hvelmag_w_r = np.array(hvelmag_w_r)

#     u_w_sfs = np.average(mean_profiles.variables["u'w'_sfs"][tstart:][:],axis=0)
#     v_w_sfs = np.average(mean_profiles.variables["v'w'_sfs"][tstart:][:],axis=0)
#     hvelmag_w_sfs = []
#     for i in np.arange(0,len(twist)):
#         hvelmag_w_sfs.append(u_w_sfs[i] * np.cos(twist[i]) + v_w_sfs[i] * np.sin(twist[i]))
#     hvelmag_w_sfs = np.array(hvelmag_w_sfs)

#     del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
#     #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
#     d_dz = []
#     for i in np.arange(0,len(hvelmag)-4,1):
#         if i == 0:
#             d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
#         else:
#             d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

#         d_dz.append(d_dz_i)

#     rho = 1
#     # TR = -u_w_r*rho #resolved stress (z)
#     # TS = -u_w_sfs*rho #sfs stress (z)

#     TR = -hvelmag_w_r * rho

#     TS = -hvelmag_w_sfs * rho

#     R = TR[0]/TS[0]

#     v_les = np.divide(TS[0:-4],(d_dz* rho)) #LES false viscosity
#     v_LES = v_les[0] #LES false viscosity at first grid level
    
#     l_vLES = v_LES/ustar #LES false length scale

#     Re_LES = zi/l_vLES


#     plt.plot(Re_LES, R, color=colors[ix], marker=markers[ix],markersize=10)

#     ix+=1


# plt.xlabel("False viscous Reynolds number [-]")
# plt.ylabel("$\mathfrak{R}$ - Ratio Resolved stress to SFS stress [-]")
# plt.grid()
# plt.legend(labels)
# plt.ylim([0,1.0])
# # plt.xlim([300,600])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/R_Re_weno_z_ppm_nolim.png")
# plt.close(fig)

# #phi_m
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# cases = ["weno_z new BC/AR_1.0/","weno_z new BC/AR_0.6/","ppm_no_lim/AR_1.0/","ppm_no_lim/AR_0.6/"]
# colors = ["b","b","r","r"]
# markers = ["o","x","o","x"]
# linestyles = ["--","-","--","-"]
# labels = ["weno_z AR = 1.0","weno_z AR = 0.6","ppm_nolim AR = 1.0","ppm_nolim AR = 0.6"]

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(8.5,6.4))
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")

#     mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

#     z = np.array(mean_profiles.variables["h"])

#     tstart = np.searchsorted(df.variables['time'][:],15000.0)

#     zi = np.average(np.array(df.variables["zi"][tstart:]))
#     z_zi = z/zi
#     ustar = np.average(np.array(df.variables["ustar"][tstart:]))
#     kappa = 0.41

#     u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
#     v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
#     twist = coriolis_twist(u,v)
#     hvelmag = []
#     for i in np.arange(0,len(twist)):
#         hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
#     hvelmag = np.array(hvelmag)

#     del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
#     #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
#     d_dz = []
#     for i in np.arange(0,len(hvelmag)-4,1):
#         if i == 0:
#             d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
#         else:
#             d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

#         d_dz.append(d_dz_i)
    
#     d_dz = np.array(d_dz)

#     phi_m = ((kappa*z[:-4])/ustar)*d_dz

#     z_idx = np.searchsorted(z_zi,0.22)

#     plt.plot(phi_m[:z_idx],z_zi[:z_idx],color=colors[ix],linestyle=linestyles[ix])
#     ix+=1

# plt.xlabel("$\Phi_m(z)$")
# plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
# plt.grid()
# plt.legend(labels)
# plt.axvline(x=1.0,linestyle="--",color="k")
# plt.ylim([0,0.2])
# plt.xlim(left=0.0)
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/phi_m_weno_z_ppm_nolim.png")
# plt.close(fig)


# #find spectral data
# # #fig 3-6 u'u' spectra
# # in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# # cases = ["weno_z/AR_1.0/","weno_z/AR_0.6/","ppm_no_lim/AR_0.6/"]
# # colors = ["b","b","r"]
# # linestyles = ["--","-","--"]
# # labels = ["weno_z AR = 1.0", "weno_z AR = 0.6", "ppm_nolim AR = 1.0"]

# # plt.rcParams['font.size'] = 16
# # fig = plt.figure()
# # ix = 0
# # for case in cases:
# #     df = Dataset(in_dir+case+"abl_statistics00000.nc")
# #     time = np.array(df.variables["time"])
# #     tstart = np.searchsorted(df.variables['time'][:],15000.0)
# #     zi = np.average(df.variables["zi"][tstart:])
# #     height = 15
# #     col = "7.5"
# #     df = pd.read_csv(in_dir+case+"spectral_data.csv")
# #     freq1d = df['freq']
# #     e_1d = df[col]

# #     plt.loglog(freq1d, e_1d, linestyle=linestyles[ix], color=colors[ix])


# #     ix+=1

# # plt.xlabel("Wave number [1/m]")
# # plt.ylabel("$E_{uu}(k)$ - Power spectral density [$m^4/s^4$]")
# # plt.title("$z_z_i = 0.025$")
# # plt.grid()
# # plt.ylim(bottom = 1e-06)
# # plt.xlim(right = 1e-01)
# # plt.legend(labels)
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/Euu.png")
# # plt.close(fig)


# # #fig 3-6 w'w' spectra
# # in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# # cases = ["weno_z/AR_1.0/","weno_z/AR_0.6/","ppm_no_lim/AR_0.6/"]
# # colors = ["b","b","r"]
# # linestyles = ["--","-","--"]
# # labels = ["weno_z AR = 1.0", "weno_z AR = 0.6", "ppm_nolim AR = 1.0"]

# # plt.rcParams['font.size'] = 16
# # fig = plt.figure()
# # ix = 0
# # for case in cases:
# #     df = Dataset(in_dir+case+"abl_statistics00000.nc")
# #     time = np.array(df.variables["time"])
# #     tstart = np.searchsorted(df.variables['time'][:],15000.0)
# #     zi = np.average(df.variables["zi"][tstart:])
# #     height = 40
# #     col = "47.5"
# #     df = pd.read_csv(in_dir+case+"spectral_data.csv")
# #     freq1d = df['freq']
# #     e_1d = df[col]

# #     plt.loglog(freq1d, e_1d, linestyle=linestyles[ix], color=colors[ix])


# #     ix+=1

# # plt.xlabel("Wave number [1/m]")
# # plt.ylabel("$E_{ww}(k)$ - Power spectral density [$m^4/s^4$]")
# # plt.title("$z_z_i = 0.08$")
# # plt.grid()
# # plt.ylim(bottom = 1e-06)
# # plt.xlim(right = 1e-01)
# # plt.legend(labels)
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/Eww.png")
# # plt.close(fig)



# #figure 3-6 reduced model constant HAZ analysis
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# cases = ["weno_z new BC/AR_0.66/","weno_z new BC/AR_0.66_reduced_model_constant/"]
# colors = ["b","r"]
# markers = ["o","x"]
# linestyles = ["-","-"]
# labels = ["weno_z AR = 0.66 $C_k=0.1$","weno_z AR = 0.66 $C_k=0.07$"]

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(8.5,6.4))
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")

#     mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

#     tstart = np.searchsorted(df.variables['time'][:],15000.0)

#     zi = np.average(np.array(df.variables["zi"][tstart:]))
#     ustar = np.average(np.array(df.variables["ustar"][tstart:]))

#     u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
#     v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
#     twist = coriolis_twist(u,v)
#     hvelmag = []
#     for i in np.arange(0,len(twist)):
#         hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
#     hvelmag = np.array(hvelmag)

#     u_w_r = np.average(mean_profiles.variables["u'w'_r"][tstart:][:],axis=0)
#     v_w_r = np.average(mean_profiles.variables["v'w'_r"][tstart:][:],axis=0)
#     hvelmag_w_r = []
#     for i in np.arange(0,len(twist)):
#         hvelmag_w_r.append(u_w_r[i] * np.cos(twist[i]) + v_w_r[i] * np.sin(twist[i]))
#     hvelmag_w_r = np.array(hvelmag_w_r)

#     u_w_sfs = np.average(mean_profiles.variables["u'w'_sfs"][tstart:][:],axis=0)
#     v_w_sfs = np.average(mean_profiles.variables["v'w'_sfs"][tstart:][:],axis=0)
#     hvelmag_w_sfs = []
#     for i in np.arange(0,len(twist)):
#         hvelmag_w_sfs.append(u_w_sfs[i] * np.cos(twist[i]) + v_w_sfs[i] * np.sin(twist[i]))
#     hvelmag_w_sfs = np.array(hvelmag_w_sfs)

#     del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
#     #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
#     d_dz = []
#     for i in np.arange(0,len(hvelmag)-4,1):
#         if i == 0:
#             d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
#         else:
#             d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

#         d_dz.append(d_dz_i)

#     rho = 1
#     # TR = -u_w_r*rho #resolved stress (z)
#     # TS = -u_w_sfs*rho #sfs stress (z)

#     TR = -hvelmag_w_r * rho

#     TS = -hvelmag_w_sfs * rho

#     R = TR[0]/TS[0]

#     v_les = np.divide(TS[0:-4],(d_dz* rho)) #LES false viscosity
#     v_LES = v_les[0] #LES false viscosity at first grid level
    
#     l_vLES = v_LES/ustar #LES false length scale

#     Re_LES = zi/l_vLES


#     plt.plot(Re_LES, R, color=colors[ix], marker=markers[ix],markersize=10)

#     ix+=1


# plt.xlabel("False viscous Reynolds number [-]")
# plt.ylabel("$\mathfrak{R}$ - Ratio Resolved stress to SFS stress [-]")
# plt.grid()
# plt.legend(labels)
# plt.ylim([0,1.0])
# # plt.xlim([300,600])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/R_Re_weno_z_reduced_model_const.png")
# plt.close(fig)

# #phi_m
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/HAZ Analysis/"

# cases = ["weno_z new BC/AR_0.66/","weno_z new BC/AR_0.66_reduced_model_constant/"]
# colors = ["b","r"]
# markers = ["o","x"]
# linestyles = ["-","-"]
# labels = ["weno_z AR = 0.66 $C_k=0.1$","weno_z AR = 0.66 $C_k=0.07$"]

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(8.5,6.4))
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")

#     mean_profiles = df.groups["mean_profiles"] #create variable to hold mean profiles

#     z = np.array(mean_profiles.variables["h"])

#     tstart = np.searchsorted(df.variables['time'][:],15000.0)

#     zi = np.average(np.array(df.variables["zi"][tstart:]))
#     z_zi = z/zi
#     ustar = np.average(np.array(df.variables["ustar"][tstart:]))
#     kappa = 0.41

#     u = np.average(mean_profiles.variables["u"][tstart:][:],axis=0)
#     v = np.average(mean_profiles.variables["v"][tstart:][:],axis=0)
#     twist = coriolis_twist(u,v)
#     hvelmag = []
#     for i in np.arange(0,len(twist)):
#         hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
#     hvelmag = np.array(hvelmag)

#     del_z = mean_profiles["h"][1] - mean_profiles["h"][0]
    
#     #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
#     d_dz = []
#     for i in np.arange(0,len(hvelmag)-4,1):
#         if i == 0:
#             d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/del_z)
#         else:
#             d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*del_z))

#         d_dz.append(d_dz_i)
    
#     d_dz = np.array(d_dz)

#     phi_m = ((kappa*z[:-4])/ustar)*d_dz

#     z_idx = np.searchsorted(z_zi,0.22)

#     plt.plot(phi_m[:z_idx],z_zi[:z_idx],color=colors[ix],linestyle=linestyles[ix])
#     ix+=1

# plt.xlabel("$\Phi_m(z)$")
# plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
# plt.grid()
# plt.legend(labels)
# plt.axvline(x=1.0,linestyle="--",color="k")
# plt.ylim([0,0.2])
# plt.xlim(left=0.0)
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/phi_m_weno_z_reduced_model_const.png")
# plt.close(fig)


# #fig 3-7 u'u'
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
#     Mean_profiles = df.groups["mean_profiles"]
#     print(Mean_profiles)
#     time = np.array(df.variables["time"])
#     Tstart_idx = np.searchsorted(time,np.max(time)-5000)
#     z = np.array(Mean_profiles.variables["h"])
#     zi = np.average(np.array(df["zi"][Tstart_idx:]))
#     z_zi = z/zi
#     uu = np.average(np.array(Mean_profiles.variables["u'u'_r"][Tstart_idx:]),axis=0)
#     plt.plot(uu,z_zi,color=colors[ix],linestyle=linestyles[ix])
#     ix+=1
# plt.xlabel("$\langle u'u' \\rangle$ $[m^2/s^2]$")
# plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
# plt.ylim([0,0.2])
# plt.legend(labels,loc="upper left")
# plt.grid()
# plt.axhline(y=0.025,linestyle="-",color="k")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/uu.png")
# plt.close(fig)

# #w'w'
# fig = plt.figure()
# ix = 0
# for case in cases:
#     df = Dataset(in_dir+case+"abl_statistics00000.nc")
#     Mean_profiles = df.groups["mean_profiles"]
#     time = np.array(df.variables["time"])
#     Tstart_idx = np.searchsorted(time,np.max(time)-5000)
#     z = np.array(Mean_profiles.variables["h"])
#     zi = np.average(np.array(df["zi"][Tstart_idx:]))
#     z_zi = z/zi
#     ww = np.average(np.array(Mean_profiles.variables["w'w'_r"][Tstart_idx:]),axis=0)
#     plt.plot(ww,z_zi,color=colors[ix],linestyle=linestyles[ix])
#     ix+=1
# plt.xlabel("$\langle w'w' \\rangle$ $[m^2/s^2]$")
# plt.ylabel("$z/z_i$ non-dimensional height from surface [-]")
# plt.legend(labels,loc="upper left")
# plt.ylim([0,0.2])
# plt.grid()
# plt.axhline(y=0.076,linestyle="-",color="k")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/ww.png")
# plt.close(fig)


# #fig 3-9
# #dzi/dt 1/u*
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/60000/"
# df = Dataset(in_dir+"abl_statistics60000.nc")
# Mean_profiles = df.groups["mean_profiles"]

# Time = np.array(df.variables["time"])
# dt = Time[1]-Time[0]
# zi = np.array(df.variables["zi"])
# u_star = np.array(df.variables["ustar"])
# w_star = np.array(df.variables["wstar"])
# dzi_dt = dt_calc(zi,dt)
# dzi_dt_u_star = np.true_divide(dzi_dt,u_star[:-1])
# dzi_dt_w_star = np.true_divide(dzi_dt,w_star[:-1])

# #moving statistics
# ts_dzi_dt_u_star = pd.Series(dzi_dt_u_star, index=Time[:-1])
# ts_dzi_dt_w_star = pd.Series(dzi_dt_w_star, index=Time[:-1])


# tau_u = np.true_divide(zi,u_star)
# tau_w = np.true_divide(zi,w_star)

# #Average global statistics
# glob_u_star = np.average(u_star)
# glob_w_star = np.average(w_star)
# glob_tau_u = np.average(tau_u)
# glob_tau_w = np.average(tau_w)


# plt.figure(figsize=(14,8))
# plt.plot(Time[:-1],dzi_dt_u_star)
# window_idx = int((glob_tau_u)/dt)
# ts_dzi_dt_u_star.rolling(window=window_idx).mean().plot(style='k--')
# ts_dzi_dt_u_star.rolling(window=window_idx).std().plot(style='r--')
# plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
# plt.xlabel("Time [s]",fontsize=16)
# plt.ylabel("$dz_i/dt 1/u_{star}$ [-]",fontsize=16)
# plt.legend(["$dz_i/dt 1/u_{star}$","Mean","Std","0.01","-0.01"])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dzi_dt_1_ustar.png")
# plt.close(fig)

# #dzi/dt 1/w*
# plt.figure(figsize=(14,8))
# plt.plot(Time[:-1],dzi_dt_w_star)
# window_idx = int((glob_tau_w)/dt)
# ts_dzi_dt_w_star.rolling(window=window_idx).mean().plot(style='k--')
# ts_dzi_dt_w_star.rolling(window=window_idx).std().plot(style='r--')
# plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
# plt.xlabel("Time [s]",fontsize=16)
# plt.ylabel("$dz_i/dt 1/w_{star}$ [-]",fontsize=16)
# plt.legend(["$dz_i/dt 1/w_{star}$","Mean","Std","0.01","-0.01"])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dzi_dt_1_wstar.png")
# plt.close(fig)



# #Figure 3-10
# #horizontal velocity
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/60000/"
# df = Dataset(in_dir+"abl_statistics60000.nc")
# Time = np.array(df.variables["time"])
# dt = Time[1]-Time[0]

# zi = np.array(df.variables["zi"])
# L = np.array(df.variables["L"])
# zi_L = -np.true_divide(zi,L)

# Mean_profiles = df.groups["mean_profiles"]
# z = np.array(Mean_profiles.variables["h"])

# u = np.array(Mean_profiles.variables["u"])
# v = np.array(Mean_profiles.variables["v"])

# twist = coriolis_twist(u,v)
# hvelmag = []
# for i in np.arange(0,len(twist)):
#     hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
# hvelmag = np.array(hvelmag)

# #hub height
# z_hub = 90
# z_hub_idx = np.searchsorted(z,z_hub)
# hvelmag_hub = hvelmag[:,z_hub_idx]

# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
# df = Dataset(in_dir+"abl_statistics70000.nc")
# Time_2 = np.array(df.variables["time"])
# dt = Time[1]-Time[0]

# zi = np.array(df.variables["zi"])
# L = np.array(df.variables["L"])
# zi_L_2 = -np.true_divide(zi,L)

# Mean_profiles = df.groups["mean_profiles"]
# z = np.array(Mean_profiles.variables["h"])

# u = np.array(Mean_profiles.variables["u"])
# v = np.array(Mean_profiles.variables["v"])

# twist = coriolis_twist(u,v)
# hvelmag = []
# for i in np.arange(0,len(twist)):
#     hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
# hvelmag = np.array(hvelmag)

# #hub height
# z_hub = 90
# z_hub_idx = np.searchsorted(z,z_hub)
# hvelmag_hub_2 = hvelmag[:,z_hub_idx]

# Time = np.concatenate((Time,Time_2))
# hvelmag_hub = np.concatenate((hvelmag_hub,hvelmag_hub_2))
# zi_L = np.concatenate((zi_L,zi_L_2))


# plt.figure(figsize=(14,8))
# plt.plot(Time,hvelmag_hub)
# plt.xlabel("Time [s]",fontsize=16)
# plt.ylabel("horizontal velocity [m/s]",fontsize=16)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Time_Uhub.png")
# plt.close(fig)


# #-zi/L
# plt.figure(figsize=(14,8))
# plt.plot(Time,zi_L)
# plt.xlabel("Time [s]",fontsize=16)
# plt.ylabel("$-z_i/L$ [m]",fontsize=16)
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Time_zi_L.png")
# plt.close(fig)



# #Figure 3-11
# #horizontal velocity
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
# df = Dataset(in_dir+"abl_statistics70000.nc")
# Time = np.array(df.variables["time"])

# tstart = 38000
# tstart_idx = np.searchsorted(Time,tstart)

# zi = np.array(df.variables["zi"])
# glob_zi = np.average(zi[tstart_idx:])

# Mean_profiles = df.groups["mean_profiles"]
# z = np.array(Mean_profiles.variables["h"])

# z_zi = z/glob_zi

# u = np.average(Mean_profiles.variables["u"][tstart_idx:],axis=0)
# v = np.average(Mean_profiles.variables["v"][tstart_idx:],axis=0)

# twist = coriolis_twist(u,v)
# hvelmag = []
# for i in np.arange(0,len(twist)):
#     hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
# hvelmag = np.array(hvelmag)


# theta = np.average(Mean_profiles.variables["theta"][tstart_idx:],axis=0)

# u_w_r = np.average(Mean_profiles.variables["u'w'_r"][tstart_idx:],axis=0)

# #mean profiles
# #U
# plt.figure(figsize=(14,8))
# plt.plot(hvelmag,z_zi)
# plt.xlabel("Streamwise velocity [m/s]",fontsize=16)
# plt.ylabel("$z/z_i$ [-]",fontsize=16)
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/U.png")
# plt.close(fig)

# #mean profiles 0.2zi
# #U
# plt.figure(figsize=(14,8))
# plt.plot(hvelmag,z_zi)
# plt.xlabel("Streamwise velocity [m/s]",fontsize=16)
# plt.ylabel("$z/z_i$ [-]",fontsize=16)
# plt.ylim([0,0.2])
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/U_rotor.png")
# plt.close(fig)

# #Coriolis twist
# plt.figure(figsize=(14,8))
# plt.plot(np.degrees(twist),z_zi)
# plt.xlabel("Flow angle [deg]",fontsize=16)
# plt.ylabel("$z/z_i$ [-]",fontsize=16)
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/flow_angle.png")
# plt.close(fig)


# #u'w'_r
# plt.figure(figsize=(14,8))
# plt.plot(u_w_r,z_zi)
# plt.xlabel("$\langle u'w' \\rangle ^r$ $[m^2/s^2]$",fontsize=16)
# plt.ylabel("$z/z_i$ [-]",fontsize=16)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/u_w_r.png")
# plt.close(fig)

# #Theta
# plt.figure(figsize=(14,8))
# plt.plot(theta,z_zi)
# plt.xlabel("Potential temperature [K]",fontsize=16)
# plt.ylabel("$z/z_i$ [-]",fontsize=16)
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Pot_temp.png")
# plt.close(fig)

# fu = interpolate.interp1d(z,u)
# fv = interpolate.interp1d(z,v)
# heights = np.array([0.1,0.4,0.8,1.0,1.1,1.2])
# plt.figure(figsize=(14,8))
# for height in heights:
#     height_m = height*glob_zi
#     u_h = fu(height_m)
#     v_h = fv(height_m)
#     plt.arrow(0,0,u_h,v_h,length_includes_head=True,color="#1f77b4",head_length=0.05,head_width=0.05)
#     plt.text(u_h,v_h,"${}z_i$".format(height))

# plt.xlabel("$U$ - average velocity [m/s]",fontsize=16)
# plt.ylabel("$V$ - average velocity [m/s]",fontsize=16)
# xtemp = [-1,-2]; ytemp= [-1,-2]
# plt.xlim(left=0)
# plt.ylim(bottom=0)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/hodograph.png")
# plt.close(fig)

# #3-12 isocontour xy plane u'
# # in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
# # df = Dataset(in_dir+"sampling70000.nc")
# # print(df)


# #Figure 3-13 mean Spectra 2d x-y plane 90m u, w velocities
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
# df = pd.read_csv(in_dir+"spectral_data_uu.csv")
# Mean_uu = np.array(df["mean"])
# frq_uu = np.array(df["freqs"])

# df = pd.read_csv(in_dir+"spectral_data_ww.csv")
# Mean_ww = np.array(df["mean"])
# frq_ww = np.array(df["freqs"])
# k = (frq_uu**(-5/3))/1e05

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.loglog(frq_uu,Mean_uu,"-r",label="$\langle u' u' \\rangle _{T=1200s}$")
# plt.loglog(frq_ww,Mean_ww,"-b",label="$\langle w' w' \\rangle _{T=1200s}$")
# plt.loglog(frq_uu,k,"--k",label="$k^{-5/3}$")
# plt.axvline(x=3e-03,linestyle="--",color="k",label="maximum peak in $\langle w' w' \\rangle _{T=1200s}$")
# plt.xlabel("k - wavenumber [1/m]")
# plt.ylabel("(k) - power spectral density [$m^2/s^2$]")
# plt.title("Distance from surface: 90m")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_velocity_spectra.png")
# plt.close(fig)

# #Figure 3-14 mean CDF 2d x-y plane 90m u, w velocities
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
# df = pd.read_csv(in_dir+"PDF_data_uu.csv")
# PDF_uu_mean = np.array(df["mean"])
# X_uu = np.array(df["X"])

# fig = plt.figure()
# plt.plot(X_uu,PDF_uu_mean,"-k")
# plt.xlabel("Fluctuating streamwise velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.axvline(y=-0.61,linestyle="--",color="b",label="low speed streaks")
# plt.axvline(y=0.76,linestyle="--",color="r",label="high speed regions")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_u_velocity_PDF.png")
# plt.close(fig)


# CDF_i = 0
# CDF = []
# dx = X_uu[1]-X_uu[0]
# for f in PDF_uu_mean:
#     CDF_i+=f*dx
#     CDF.append(CDF_i)

# fig = plt.figure()
# plt.plot(X_uu,CDF,"-k")
# plt.xlabel("Fluctuating streamwise velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_u_velocity_CDF.png")
# plt.close(fig)


# df = pd.read_csv(in_dir+"PDF_data_ww.csv")
# PDF_ww_mean = np.array(df["mean"])
# X_ww = np.array(df["X"])

# fig = plt.figure()
# plt.plot(X_ww,PDF_ww_mean,"-k")
# plt.xlabel("Fluctuating vertical velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.axvline(y=-0.48,linestyle="--",color="b",label="downdrafts")
# plt.axvline(y=0.42,linestyle="--",color="r",label="updrafts")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_w_velocity_PDF.png")
# plt.close(fig)

# CDF_i = 0
# CDF = []
# dx = X_ww[1]-X_ww[0]
# for f in PDF_ww_mean:
#     CDF_i+=f*dx
#     CDF.append(CDF_i)

# fig = plt.figure()
# plt.plot(X_ww,CDF,"-k")
# plt.xlabel("Fluctuating vertical velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/2d_w_velocity_CDF.png")
# plt.close(fig)



# #Eddy passage time plots
# offsets = [22.5,85,142.5]

# filter_cutoff = [1/(1*3e-03),1/(1.5*3e-03),1/(2*3e-03),1/(2.5*3e-03),1/(3*3e-03)]

# colors = ["r","b","g"]

# mean_high_D = np.array([[568,440,360,306,266],[593,380,289,235,203],[546,351,266,215,187]])
# mean_low_D = np.array([[719,496,396,339,306],[738,495,375,308,263],[725,474,349,278,238]])

# std_high_D = np.array([[719,603,550,509,473],[821,633,530,463,421],[777,590,519,442,403]])
# std_low_D = np.array([[961,781,691,622,583],[1020,799,671,602,544],[992,775,659,573,518]])

# mean_high_Tau = np.array([[45.1,34.7,28.3,24.0,20.9],[42.5,27.2,20.7,16.8,14.6],[38.1,24.5,18.6,15.0,13.1]])
# mean_low_Tau = np.array([[70.9,49.3,39.7,34.2,31.0],[63.6,43.0,32.6,26.8,23.0],[60.4,39.6,29.2,23.3,19.9]])

# std_high_Tau = np.array([[56.1,46.6,42.2,39.0,36.2],[58.1,44.6,37.3,32.6,29.6],[53.7,40.6,35.7,30.4,27.7]])
# std_low_Tau = np.array([[96.7,79.7,71.0,64.5,60.8],[89.5,70.7,59.8,53.8,48.8],[83.8,65.9,56.3,49.1,44.4]])

# mean_high_ux = np.array([[12.39,12.44,12.44,12.44,12.42],[13.77,13.73,13.73,13.72,13.71],[14.16,14.14,14.13,14.13,14.12]])
# mean_low_ux = np.array([[10.37,10.33,10.29,10.26,10.25],[11.83,11.82,11.82,11.82,11.83],[12.27,12.29,12.3,12.32,12.34]])

# std_high_ux = np.array([[0.28,0.31,0.33,0.33,0.33],[0.23,0.22,0.22,0.21,0.21],[0.2,0.19,0.18,0.18,0.18]])
# std_low_ux = np.array([[0.27,0.31,0.34,0.37,0.39],[0.28,0.3,0.31,0.32,0.33],[0.28,0.29,0.29,0.29,0.28]])


# #Figure 3-14a mean Eddy length
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5

#     plt.plot(filter_cutoff,mean_high_D[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,mean_low_D[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]")
# plt.ylabel("Mean Eddy length [m]")
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Mean_Eddy_length_unedit.png")
# plt.close()

# #figure 3-14b std eddy length
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5


#     plt.plot(filter_cutoff,std_high_D[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,std_low_D[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]")
# plt.ylabel("Standard deviation Eddy length [m]")
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Std_Eddy_length_unedit.png")
# plt.close()


# #Figure 3-15a mean Eddy velocity
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5

#     plt.plot(filter_cutoff,mean_high_ux[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,mean_low_ux[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]")
# plt.ylabel("Mean Eddy velocity [m/s]")
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Mean_Eddy_velocity_unedit.png")
# plt.close()

# #Figure 3-15b std eddy velocity
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5


#     plt.plot(filter_cutoff,std_high_ux[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,std_low_ux[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]")
# plt.ylabel("Standard deviation Eddy velocity [m/s]")
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Std_Eddy_velocity_unedit.png")
# plt.close()


# #Figure 3-16a mean Eddy passage time
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5

#     plt.plot(filter_cutoff,mean_high_Tau[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,mean_low_Tau[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]")
# plt.ylabel("Mean Eddy Passage time [s]")
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Mean_Eddy_passage_time_unedit.png")
# plt.close()

# #Figure 3-16b std Eddy passage time
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5


#     plt.plot(filter_cutoff,std_high_Tau[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,std_low_Tau[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]")
# plt.ylabel("Standard deviation Eddy Passage time [s]")
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Std_Eddy_passage_time_unedit.png")
# plt.close()

# #Figure 3-17 summary eddy passage time 167m filter
# x = ["High speed\n30m","High speed\n92.5m","High speed\n150m","Low speed\n30m","Low speed\n92.5m","Low speed\n150m"]


# T_mean = [28.3,20.7,18.6,39.7,32.6,29.2]
# T_std = [[0,0,0,0,0,0],[42.2,37.3,35.7,71.0,59.8,56.3]]

# fig = plt.figure(figsize=(14,8))
# plt.bar(x,T_mean)
# plt.errorbar(x,T_mean,yerr=T_std,fmt="o", color="k",capsize=10)
# plt.ylabel("Eddy passage time [s]")
# plt.grid()
# plt.title("Filter cutoff 167m")
# plt.savefig("../../Thesis/Figures/eddy_passage_time_summary_bar_unedit.png")
# plt.close()


# #Figure 4-1a
# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/ALM sensitivity Analysis/"
# cases = ["dblade_2.0","dblade_1.0","dblade_0.5"]
# colors = ["red","blue","green"]
# labels = ["1a","1b","1c"]
# variables = ["RtAeroMxh_[N-m]","RtAeroMyh_[N-m]"]
# ylabels = ["PSD - Aerodynamic Hub moment around x axis $M_{H,x}$ [kN-m]","PSD - Aerodynamic Hub moment around y axis $M_{\widehat{H},y}$ [kN-m]"]
# iv = 0
# for var in variables:
#     ix = 0
#     fig = plt.figure(figsize=(14,8))
#     for case in cases:
#         df = io.fast_output_file.FASTOutputFile(in_dir+case+"/NREL_5MW_Main.out").toDataFrame()
#         Time = df["Time_[s]"]
#         dt = Time[1] - Time[0]
#         Time_start_idx = np.searchsorted(Time,Time[0]+5)
#         Time = np.array(Time[Time_start_idx:])

#         y = np.array(df[var][Time_start_idx:])

#         frq,PSD = temporal_spectra(y,dt,var)

#         plt.loglog(frq,PSD,color=colors[ix],label=labels[ix])
        
#         ix+=1

#     plt.xlabel("Frequency [Hz]")
#     plt.ylabel(ylabels[iv])
#     plt.legend()
#     plt.grid()
#     plt.tight_layout()
#     plt.savefig("../../Thesis/Figures/ALM_sensitivity_BSR_spectra_{}.png".format(var))
#     plt.close(fig)

#     iv+=1


# #Figure 4-4
# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/ALM sensitivity Analysis/"
# cases = ["eps_dr_0.75","dblade_1.0","eps_dr_0.95"]
# colors = ["red","blue","green"]
# labels = ["1a","1b","1c"]
# variables = ["AoA_[deg]"]
# act_stations_cases = [47,54,59]

# ix = 0
# data = []
# for case in cases:
#     df = io.fast_output_file.FASTOutputFile(in_dir+case+"/NREL_5MW_Main.out").toDataFrame()

#     act_stations = act_stations_cases[ix]
#     x = np.linspace(0,1,act_stations)
#     x_min = np.linspace(0,1,act_stations_cases[0])

#     Var_list = []
#     for i in np.arange(1,act_stations+1):
#         if i < 10:
#             txt = "AB1N00{0}Alpha_[deg]".format(i)
#         elif i >= 10 and i < 100:
#             txt = "AB1N0{0}Alpha_[deg]".format(i)
#         elif i >= 100:
#             txt = "AB1N{0}Alpha_[deg]".format(i)


#         tstart_idx = len((df["Time_[s]"])) - int((3 * 5)/0.0039)

#         Var_list.append(np.average(df[txt][tstart_idx:]))

   
#     data.append(interpolate.interp1d(x, Var_list,kind="linear")(x_min))

# data = np.array(data)


#Section 5

# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroFxh = np.array(OpenFAST_vars.variables["RtAeroFxh"][Time_start_idx:])
# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)
# RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000

# RtAeroMxa = np.array(OpenFAST_vars.variables["RtAeroMxh"][Time_start_idx:])/1000
# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
# RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

# FR = np.sqrt(np.add(np.square(RtAeroFys),np.square(RtAeroFzs)))
# FTheta = np.degrees(np.arctan2(RtAeroFzs,RtAeroFys))
# FTheta = theta_360(FTheta)

# MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))
# MTheta = np.degrees(np.arctan2(RtAeroMys,-RtAeroMzs))
# MTheta = theta_360(MTheta)

# #rotor averaged variables
# Time_sampling = np.array(df_OF.variables["Time_sampling"])
# dt_sample = Time_sampling[1] - Time_sampling[0]
# Time_start_sample_idx = np.searchsorted(Time_sampling,Time_sampling[0]+200)
# Time_sampling = Time_sampling[Time_start_sample_idx:]

# Rotor_avg_vars = df_OF.groups["Rotor_Avg_Variables"]
# Rotor_avg_vars = Rotor_avg_vars.groups["63.0"]

# Ux = np.array(Rotor_avg_vars.variables["Ux"][Time_start_sample_idx:])
# IA = np.array(Rotor_avg_vars.variables["IA"][Time_start_sample_idx:])
# Iy = np.array(Rotor_avg_vars.variables["Iy"][Time_start_sample_idx:])
# Iz = -np.array(Rotor_avg_vars.variables["Iz"][Time_start_sample_idx:])
# I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
# ITheta = np.degrees(np.arctan2(Iy,-Iz))
# ITheta = theta_360(ITheta)

# # I_norm = I/np.mean(I); MR_norm = MR/np.mean(MR)
# # fig = plt.figure(figsize=(14,8))
# # frq,PSD = temporal_spectra(MR_norm,dt,"MR")
# # plt.loglog(frq,PSD,"-r",label="$\widetilde{M}_{H,\perp}$ [-]")
# # frq,PSD = temporal_spectra(I_norm,dt_sample,"I")
# # plt.loglog(frq,PSD,"-b",label="$I$ [-]")
# # plt.title("Magnitude of vector normalized on mean")
# # plt.grid()
# # plt.legend()
# # plt.xlabel("Frequency [Hz]")
# # plt.ylabel("PSD")
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/spectra_I_MR.png")
# # plt.close(fig)

# # cc = round(correlation_coef(IA,I),2)
# # fig,ax = plt.subplots(figsize=(14,8))
# # ax.plot(Time_sampling,IA,"-r")
# # ax.set_ylabel("Asymmetry parameter [$m^4/s$]")
# # ax.yaxis.label.set_color('red')
# # ax2=ax.twinx()
# # ax2.plot(Time_sampling,I,"-b")
# # ax2.set_ylabel("Asymmetry vector magnitude [$m^4/s$]")
# # ax2.yaxis.label.set_color('blue')
# # ax2.grid()
# # fig.supxlabel("Time [s]")
# # fig.suptitle("Correlation coefficient = {}".format(cc))
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/cc_IA_A.png")
# # plt.close(fig)



# Ux_LPF = hard_filter(Ux,0.3,dt_sample,"lowpass")
# IA_LPF = hard_filter(IA,0.3,dt_sample,"lowpass")
# I_LPF = hard_filter(I,0.3,dt_sample,"lowpass")
# Iy_LPF = hard_filter(Iy,0.3,dt_sample,"lowpass")
# Iz_LPF = hard_filter(Iz,0.3,dt_sample,"lowpass")

# ITheta_LPF = hard_filter(ITheta,0.3,dt_sample,"lowpass")

# f = interpolate.interp1d(Time_OF,MR)
# MR_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,MTheta)
# MTheta_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,RtAeroFxh)
# RtAeroFxh_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,RtAeroMxa)
# RtAeroMxh_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,RtAeroMys)
# RtAeroMys_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,RtAeroMzs)
# RtAeroMzs_interp = f(Time_sampling)

# MR_LPF = hard_filter(MR_interp,0.3,dt_sample,"lowpass")
# MTheta_LPF = hard_filter(MTheta_interp,0.3,dt_sample,"lowpass")
# RtAeroFxh_LPF = hard_filter(RtAeroFxh_interp,0.3,dt_sample,"lowpass")
# RtAeroMxh_LPF  = hard_filter(RtAeroMxh_interp,0.3,dt_sample,"lowpass")
# RtAeroMys_LPF  = hard_filter(RtAeroMys_interp,0.3,dt_sample,"lowpass")
# RtAeroMzs_LPF  = hard_filter(RtAeroMzs_interp,0.3,dt_sample,"lowpass")



# Time_shift_idx = np.searchsorted(Time_OF,Time_OF[0]+4.6)


# MR_shift = MR[Time_shift_idx:]
# RtAeroFxh_shift = RtAeroFxh[Time_shift_idx:]
# RtAeroMxa_shift = RtAeroMxa[Time_shift_idx:]


# Time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+4.6)
# Time_sampling_shift = Time_sampling[:-Time_shift_idx]
# MR_LPF_shift = MR_LPF[Time_shift_idx:]
# MTheta_LPF_shift = MTheta_LPF[Time_shift_idx:]
# RtAeroFxh_LPF_shift = RtAeroFxh_LPF[Time_shift_idx:]
# RtAeroMxa_LPF_shift = RtAeroMxh_LPF[Time_shift_idx:]
# RtAeroMys_LPF_shift = RtAeroMys_LPF[Time_shift_idx:]
# RtAeroMzs_LPF_shift = RtAeroMzs_LPF[Time_shift_idx:]

# Ux_LPF_shift = Ux_LPF[:-Time_shift_idx]
# IA_LPF_shift = IA_LPF[:-Time_shift_idx]
# I_LPF_shift = I_LPF[:-Time_shift_idx]
# Iy_LPF_shift = Iy_LPF[:-Time_shift_idx]
# Iz_LPF_shift = Iz_LPF[:-Time_shift_idx]
# ITheta_LPF_shift = ITheta_LPF[:-Time_shift_idx]


# print(correlation_coef(ITheta_LPF_shift,MTheta_LPF_shift))
# print(correlation_coef(I_LPF_shift,MR_LPF_shift))
# print(correlation_coef(Iy_LPF_shift,RtAeroMys_LPF_shift))
# print(correlation_coef(Iz_LPF_shift,RtAeroMzs_LPF_shift))

# print(correlation_coef(IA_LPF_shift,MR_LPF_shift))
# print(correlation_coef(MR,RtAeroMxa))
# print(correlation_coef(RtAeroMxa_LPF_shift,Ux_LPF_shift))
# print(correlation_coef(RtAeroFxh_LPF_shift,Ux_LPF_shift))
# print(correlation_coef(RtAeroFxh,RtAeroMxa))


# Time_OF_shift = Time_OF[:-Time_shift_idx]
# fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(16,24),sharex=True)
# ax1.plot(Time_sampling_shift,IA_LPF_shift,"-r")
# ax1.set_title("Asymmetry Parameter [$m^4/s$]")
# ax1.grid()
# ax2.plot(Time_OF_shift,MR_shift,"-b")
# ax2.plot(Time_sampling_shift,MR_LPF_shift,"-r")
# ax2.set_title("out-of-plane bending moment [kN-m]")
# ax2.grid()
# ax3.plot(Time_OF_shift,RtAeroMxa_shift,"-b")
# ax3.plot(Time_sampling_shift,RtAeroMxa_LPF_shift,"-r")
# ax3.set_title("Torque [kN-m]")
# ax3.grid()
# ax4.plot(Time_sampling_shift,Ux_LPF_shift,"-r")
# ax4.set_title("Rotor averaged wind speed [$m/s$]")
# ax4.grid()
# ax5.plot(Time_OF_shift,RtAeroFxh_shift,"-b")
# ax5.plot(Time_sampling_shift,RtAeroFxh_LPF_shift,"-r")
# ax5.set_title("Thrust [kN]")
# ax5.grid()
# fig.supxlabel("Time [s]")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Time_correlations.png")
# plt.close(fig)


# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# WR = 1079
# nu = np.linspace(0,1.0,11)
# Mean = []
# Var = []
# for n in nu:
#     FBz_hat = RtAeroMys/L2 + (RtAeroFzs - n*WR)*((L1+L2)/L2)
#     Mean_FBR_hat = np.sqrt(np.add(np.square(np.mean(FBy)),np.square(np.mean(FBz_hat))))
#     Mean.append(Mean_FBR_hat)
#     Var_FBR_hat = np.var(FBy) + np.var(FBz)
#     Var.append(Var_FBR_hat)

# fig = plt.figure(figsize=(14,8))
# plt.plot(nu,Mean,"-o")
# plt.xlabel("Percentage of weight $\\nu$")
# plt.ylabel("Magnitude of mean of main bearing force vector [kN]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Mean_FBR.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# plt.plot(nu,Var,"-o")
# plt.xlabel("Percentage of weight $\\nu$")
# plt.ylabel("Variance main bearing force vector [kN]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Var_FBR.png")
# plt.close(fig)


# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
# Theta_FBR = np.degrees(np.arctan2(FBz,FBy))
# Theta_FBR = theta_360(Theta_FBR)

# # df = io.fast_output_file.FASTOutputFile("../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/Steady_Rigid_blades_shear_0.085/NREL_5MW_Main.out").toDataFrame()
# # Time_OF_SS = np.array(df["Time_[s]"])

# # Azimuth_SS = np.radians(np.array(df["Azimuth_[deg]"]))

# # RtAeroMyh_SS = np.array(df["RtAeroMyh_[N-m]"]); RtAeroMzh_SS = np.array(df["RtAeroMzh_[N-m]"])
# # RtAeroMys_SS, RtAeroMzs_SS = tranform_fixed_frame(RtAeroMyh_SS,RtAeroMzh_SS,Azimuth_SS)
# # RtAeroMys_SS = np.array(RtAeroMys_SS)/1000; RtAeroMzs_SS = np.array(RtAeroMzs_SS)/1000
# # MR_SS = np.sqrt(np.add(np.square(RtAeroMys_SS),np.square(RtAeroMzs_SS)))
# # MTheta_SS = np.degrees(np.arctan2(RtAeroMys_SS,-RtAeroMzs_SS))
# # MTheta_SS = theta_360(MTheta_SS)


# # fig = plt.figure(figsize=(8,8))
# # ax = fig.add_subplot(projection='polar')
# # label="{}-{}s".format(Time_OF[2050],Time_OF[2500])
# # ax.plot(np.radians(MTheta[2050:2500]),MR[2050:2500]/np.max(MR),"-r",label="MCBL ABL\n"+label)
# # label="{}-{}s".format(Time_OF[4550],Time_OF[5000])
# # ax.plot(np.radians(MTheta[4550:5000]),MR[4550:5000]/np.max(MR),"-b",label="MCBL ABL\n"+label)
# # label="{}-{}s".format(Time_OF[49450],Time_OF[50000])
# # ax.plot(np.radians(MTheta[49450:50000]),MR[49450:50000]/np.max(MR),"-g",label="MCBL ABL\n"+label)
# # label="{}-{}s".format(Time_OF_SS[80],Time_OF_SS[255])
# # ax.plot(np.radians(MTheta_SS[80:255]),(8*MR_SS[80:255])/np.max(MR),"-k",label="Steady shear\n"+label)
# # ax.legend()
# # ax.set_title("Normalized out-of-plane bending\nmoment trajectory [-]")
# # plt.savefig("../../Thesis/Figures/MR_trajectory.png")
# # plt.close(fig)



# # fig = plt.figure(figsize=(8,8))
# # ax = fig.add_subplot(projection='polar')
# # ax.plot(np.radians(Theta_FBR[:2500]),FBR[:2500]/np.max(FBR),"-r")
# # ax.plot(np.radians(FTheta[:2500]),FR[:2500]/np.max(FR),"-g")
# # ax.plot(np.radians(MTheta[:2500]),MR[:2500]/np.max(MR),"-b")

# # ax.plot(np.radians(Theta_FBR[2500]),FBR[2500]/np.max(FBR),"or",label="$\\tilde{F}_{B\perp}$")
# # ax.arrow(0, 0, np.radians(Theta_FBR[2500]), FBR[2500]/np.max(FBR), length_includes_head=True,color="r")
# # ax.plot(np.radians(FTheta[2500]),FR[2500]/np.max(FR),"og",label="$\\tilde{F}_{H\perp}$")
# # ax.arrow(0, 0, np.radians(FTheta[2500]), FR[2500]/np.max(FR), length_includes_head=True,color="g")
# # ax.plot(np.radians(MTheta[2500]),MR[2500]/np.max(MR),"ob",label="$\\tilde{M}_{H\perp,mod}$")
# # ax.arrow(0, 0, np.radians(MTheta[2500]), MR[2500]/np.max(MR), length_includes_head=True,color="b")
# # ax.legend()
# # ax.set_ylim(0,0.7)
# # ax.set_title("Normalized vector trajectories\n25s period")
# # plt.savefig("../../Thesis/Figures/FR_MR_FBR_trajectory.png")
# # plt.close(fig)



# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/65000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]

# Azimuth = np.radians(np.array(df_OF.variables["Azimuth"][Time_start_idx:]))

# RtAeroFxh = np.array(df_OF.variables["RtAeroFxh"][Time_start_idx:])

# RtAeroMxa = np.array(df_OF.variables["RtAeroMxh"][Time_start_idx:])/1000
# RtAeroMyh = np.array(df_OF.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(df_OF.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
# RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

# MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

# #rotor averaged variables
# Time_sampling = np.array(df_OF.variables["time_sampling"])
# dt_sample = Time_sampling[1] - Time_sampling[0]
# Time_start_sample_idx = np.searchsorted(Time_sampling,Time_sampling[0]+200)
# Time_sampling = Time_sampling[Time_start_sample_idx:]

# Rotor_avg_vars = df_OF.groups["63.0"]

# Ux = np.array(Rotor_avg_vars.variables["Ux"][Time_start_sample_idx:])
# IA = np.array(Rotor_avg_vars.variables["IA"][Time_start_sample_idx:])

# Ux_LPF = hard_filter(Ux,0.3,dt_sample,"lowpass")
# IA_LPF = hard_filter(IA,0.3,dt_sample,"lowpass")

# f = interpolate.interp1d(Time_OF,MR)
# MR_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,RtAeroFxh)
# RtAeroFxh_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,RtAeroMxa)
# RtAeroMxh_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,RtAeroMys)
# RtAeroMys_interp = f(Time_sampling)

# f = interpolate.interp1d(Time_OF,RtAeroMzs)
# RtAeroMzs_interp = f(Time_sampling)

# MR_LPF = hard_filter(MR_interp,0.3,dt_sample,"lowpass")
# RtAeroFxh_LPF = hard_filter(RtAeroFxh_interp,0.3,dt_sample,"lowpass")
# RtAeroMxh_LPF  = hard_filter(RtAeroMxh_interp,0.3,dt_sample,"lowpass")
# RtAeroMys_LPF  = hard_filter(RtAeroMys_interp,0.3,dt_sample,"lowpass")
# RtAeroMzs_LPF  = hard_filter(RtAeroMzs_interp,0.3,dt_sample,"lowpass")



# Time_shift_idx = np.searchsorted(Time_OF,Time_OF[0]+4.6)
# Time_OF_shift = Time_OF[:-Time_shift_idx]

# MR_shift = MR[Time_shift_idx:]
# RtAeroFxh_shift = RtAeroFxh[Time_shift_idx:]
# RtAeroMxa_shift = RtAeroMxa[Time_shift_idx:]


# Time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+4.6)
# Time_sampling_shift = Time_sampling[:-Time_shift_idx]
# MR_LPF_shift = MR_LPF[Time_shift_idx:]
# RtAeroFxh_LPF_shift = RtAeroFxh_LPF[Time_shift_idx:]
# RtAeroMxa_LPF_shift = RtAeroMxh_LPF[Time_shift_idx:]
# RtAeroMys_LPF_shift = RtAeroMys_LPF[Time_shift_idx:]
# RtAeroMzs_LPF_shift = RtAeroMzs_LPF[Time_shift_idx:]

# Ux_LPF_shift = Ux_LPF[:-Time_shift_idx]
# IA_LPF_shift = IA_LPF[:-Time_shift_idx]


# print(correlation_coef(IA_LPF_shift,MR_LPF_shift))
# print(correlation_coef(MR,RtAeroMxa))
# print(correlation_coef(RtAeroMxa_LPF_shift,Ux_LPF_shift))
# print(correlation_coef(RtAeroFxh_LPF_shift,Ux_LPF_shift))
# print(correlation_coef(RtAeroFxh,RtAeroMxa))



# fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(16,24),sharex=True)
# ax1.plot(Time_sampling_shift,IA_LPF_shift,"-r")
# ax1.set_title("Asymmetry Parameter [$m^4/s$]")
# ax1.grid()
# ax2.plot(Time_OF_shift,MR_shift,"-b")
# ax2.plot(Time_sampling_shift,MR_LPF_shift,"-r")
# ax2.set_title("out-of-plane bending moment [kN-m]")
# ax2.grid()
# ax3.plot(Time_OF_shift,RtAeroMxa_shift,"-b")
# ax3.plot(Time_sampling_shift,RtAeroMxa_LPF_shift,"-r")
# ax3.set_title("Torque [kN-m]")
# ax3.grid()
# ax4.plot(Time_sampling_shift,Ux_LPF_shift,"-r")
# ax4.set_title("Rotor averaged wind speed [$m/s$]")
# ax4.grid()
# ax5.plot(Time_OF_shift,RtAeroFxh_shift,"-b")
# ax5.plot(Time_sampling_shift,RtAeroFxh_LPF_shift,"-r")
# ax5.set_title("Thrust [kN]")
# ax5.grid()
# fig.supxlabel("Time [s]")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Time_correlations_65000.png")
# plt.close(fig)



# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# #rotor averaged variables
# Time_sampling = np.array(df_OF.variables["Time_sampling"])
# Time_start_sample_idx = np.searchsorted(Time_sampling,Time_sampling[0]+200)
# Time_sampling = Time_sampling[Time_start_sample_idx:]

# Rotor_avg_vars = df_OF.groups["Rotor_Avg_Variables"]
# Rotor_avg_vars = Rotor_avg_vars.groups["63.0"]

# Ux_63 = np.array(Rotor_avg_vars.variables["Ux"][Time_start_sample_idx:])
# Iy = np.array(Rotor_avg_vars.variables["Iy"][Time_start_sample_idx:])
# Iz = -np.array(Rotor_avg_vars.variables["Iz"][Time_start_sample_idx:])
# I_63 = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

# Rotor_avg_vars = df_OF.groups["Rotor_Avg_Variables"]
# Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]

# Ux_5 = np.array(Rotor_avg_vars.variables["Ux"][Time_start_sample_idx:])
# Iy = np.array(Rotor_avg_vars.variables["Iy"][Time_start_sample_idx:])
# Iz = -np.array(Rotor_avg_vars.variables["Iz"][Time_start_sample_idx:])
# I_5 = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

# Time_shifts = np.linspace(4,7,10)

# cc_I = []; cc_ux = []
# for Time_shift in Time_shifts:

#     Time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+Time_shift)
#     Ux_5_shift = Ux_5[Time_shift_idx:]
#     Ux_63_shift = Ux_63[:-Time_shift_idx]

#     cc_ux.append(correlation_coef(Ux_5_shift,Ux_63_shift))

#     I_5_shift = I_5[Time_shift_idx:]
#     I_63_shift = I_63[:-Time_shift_idx]

#     cc_I.append(correlation_coef(I_5_shift,I_63_shift))

# max_I = np.argmax(cc_I); max_Ux = np.argmax(cc_ux)
# T = (Time_shifts[max_Ux]+Time_shifts[max_I])/2
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_shifts,cc_I,"-b",label="$|I|$")
# plt.plot(Time_shifts,cc_ux,"-r",label="$\langle u_{x'} \\rangle _{A}$") 
# plt.axvline(x=4.6,linestyle="--",color="k")
# plt.xlabel("Time shift [s]")
# plt.ylabel("correlation coefficient [-]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Time_shift_cc.png")
# plt.close(fig)


# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/Steady_Rigid_blades_shear_0.098/"

# df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

# z_SS = [7.5, 22.5, 37.5, 52.5, 82.5, 97.5, 112.5, 127.5, 157.5]
# Wind1 = np.array(df["Wind1VelX_[m/s]"][0]); Wind2 = np.array(df["Wind2VelX_[m/s]"][0]); Wind3 = np.array(df["Wind3VelX_[m/s]"][0])
# Wind4 = np.array(df["Wind4VelX_[m/s]"][0]); Wind5 = np.array(df["Wind5VelX_[m/s]"][0]); Wind6 = np.array(df["Wind6VelX_[m/s]"][0])
# Wind7 = np.array(df["Wind7VelX_[m/s]"][0]); Wind8 = np.array(df["Wind8VelX_[m/s]"][0]); Wind9 = np.array(df["Wind9VelX_[m/s]"][0])
# Wind_profile = [Wind1,Wind2,Wind3,Wind4,Wind5,Wind6,Wind7,Wind8,Wind9]

# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"

# df_pre = Dataset(in_dir+"abl_statistics70000.nc")

# Time_pre = np.array(df_pre.variables["time"])
# Time_start_idx = np.searchsorted(Time_pre,38200)

# mean_profiles = df_pre.groups["mean_profiles"]


# z = np.array(mean_profiles.variables["h"])

# u = np.average(mean_profiles.variables["u"][Time_start_idx:],axis=0)
# v = np.average(mean_profiles.variables["v"][Time_start_idx:],axis=0)

# twist = coriolis_twist(u,v)
# hvelmag = []
# for i in np.arange(0,len(twist)):
#     hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
# hvelmag = np.array(hvelmag)

# z_end = np.searchsorted(z,z_SS[-1])

# fig = plt.figure()
# plt.plot(Wind_profile,z_SS,"-or",label="Steady shear inflow")
# plt.plot(hvelmag[:z_end+1],z[:z_end+1],"-*b",label="Precursor MCBL")
# plt.xlabel("$\langle u_{x'} \\rangle _{T=38200-39200s}$ [m/s]")
# plt.ylabel("Height from surface [m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Wind_profile_precursor_steady_shear.png")
# plt.close(fig)



# Time_SS = np.array(df["Time_[s]"])
# dt_SS = Time_SS[1]-Time_SS[0]
# Time_start_idx = np.searchsorted(Time_SS,Time_SS[0]+200)
# Time_end_idx = np.searchsorted(Time_SS,Time_SS[0]+300)
# Time_SS = Time_SS[Time_start_idx:Time_end_idx]
# Azimuth = np.radians(np.array(df["Azimuth_[deg]"][Time_start_idx:Time_end_idx]))
# RtAeroMyh_SS = np.array(df["RtAeroMyh_[N-m]"][Time_start_idx:Time_end_idx])/1000
# RtAeroMzh_SS = np.array(df["RtAeroMzh_[N-m]"][Time_start_idx:Time_end_idx])/1000

# RtAeroMys_SS, RtAeroMzs_SS = tranform_fixed_frame(RtAeroMyh_SS,RtAeroMzh_SS,Azimuth)

# MR_SS = np.sqrt(np.add(np.square(RtAeroMys_SS),np.square(RtAeroMzs_SS)))

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_SS,MR_SS)
# plt.xlabel("Time [s]")
# plt.ylabel("out-of-plane bending moment [kN-m]")
# plt.title("Steady shear ($\\alpha=0.098$) inflow")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Steady_shear_MR.png")
# plt.close(fig)

# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"
# df = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df.variables["Time_OF"])
# dt_OF = Time_OF[1]-Time_OF[0]
# Time_start_idx = np.searchsorted(Time_OF,Time_OF[0]+200)
# Time_OF = Time_OF[Time_start_idx:]

# Openfast_vars = df.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(Openfast_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroMyh = np.array(Openfast_vars.variables["RtAeroMyh"][Time_start_idx:])/1000
# RtAeroMzh = np.array(Openfast_vars.variables["RtAeroMzh"][Time_start_idx:])/1000

# RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

# MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(MR,dt_OF,"MR")
# plt.loglog(frq,PSD,"-r",label="LES")
# frq,PSD = temporal_spectra(MR_SS,dt_SS,"MR_SS")
# plt.loglog(frq,PSD,"-b",label="Steady shear inflow")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - out-of-plane bending moment [$kNm^2$]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Spectra_LES_steady_shear_MR.png")
# plt.close(fig)



# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
# #OOPBM
# OOPBM = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

# LPF_1_OOPBM = np.array(hard_filter(OOPBM,0.3,dt,"lowpass"))
# BPF_OOPBM = np.array(hard_filter(OOPBM,[0.3,0.9],dt,"bandpass"))
# HPF_OOPBM = np.array(hard_filter(OOPBM,[1.5,40],dt,"bandpass"))


# #BPF calc
# dBPF_OOPBM = np.array(dt_calc(BPF_OOPBM,dt))
# zero_crossings_index_BPF_OOPBM = np.where(np.diff(np.sign(dBPF_OOPBM)))[0]
# Env_BPF_OOPBM = []
# Env_Times = []
# for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM),2):
#     idx = zero_crossings_index_BPF_OOPBM[i]
#     Env_BPF_OOPBM.append(BPF_OOPBM[idx]); Env_Times.append(Time_OF[idx])

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,BPF_OOPBM,"-r",label="BPF $M_{H,\perp}$")
# plt.plot(Env_Times,Env_BPF_OOPBM,"-b",label="Env BPF $M_{H,\perp}$")
# plt.xlabel("Time [s]")
# plt.ylabel("Aerodynamic out-of-plane bending moment magnitude [kN-m]")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Env_BPF_OOPBM.png")
# plt.close(fig)


# fig = plt.figure(figsize=(14,8))
# idx1 = np.searchsorted(Time_OF,200); idx2 = np.searchsorted(Time_OF,300)
# plt.plot(Time_OF[idx1:idx2],BPF_OOPBM[idx1:idx2],"-r",label="BPF $M_{H,\perp}$")
# idx1 = np.searchsorted(Env_Times,200); idx2 = np.searchsorted(Env_Times,300)
# plt.plot(Env_Times[idx1:idx2],Env_BPF_OOPBM[idx1:idx2],"--b",label="Env BPF $M_{H,\perp}$")
# plt.xlabel("Time [s]")
# plt.ylabel("Aerodynamic out-of-plane bending moment magnitude [kN-m]")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Env_BPF_OOPBM_200_300.png")
# plt.close(fig)

# # f = interpolate.interp1d(Env_Times,Env_BPF_OOPBM)
# # Env_Times = np.arange(Env_Times[0],Env_Times[-1],0.39)
# # Env_BPF_OOPBM = f(Env_Times)
# # dt_BPF = Env_Times[1] - Env_Times[0]
# # Env_BPF_OOPBM = hard_filter(Env_BPF_OOPBM,0.3,dt_BPF,"lowpass")

# Env_BPF_OOPBM = np.array(Env_BPF_OOPBM); Env_Times = np.array(Env_Times)

# f_BPF = interpolate.interp1d(Env_Times,Env_BPF_OOPBM)
# f_LPF = interpolate.interp1d(Time_OF,LPF_1_OOPBM)


# #HPF calc
# cc_BPF = []
# cc_LPF = []
# abs_HPF_OOPBM = abs(HPF_OOPBM)
# windows = [3,4,5,6,7,8,9,10,11,12]
# colors=["g","r","b"]
# offset=0
# ix=0
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,abs_HPF_OOPBM,"-k",label="Absolute HPF $M_{H,\perp}$")
# for window in windows:
#     window_idx = np.searchsorted(Time_OF,Time_OF[0]+window)
#     if (window_idx % 2) != 0:
#         window_idx+=1
#     Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]
#     avg_HPF_OOPBM = []
#     for i in np.arange(0,len(Time_OF)-window_idx):
#         avg_HPF_OOPBM.append(np.average(abs_HPF_OOPBM[i:i+window_idx]))
    
#     idx_min = np.searchsorted(Times_avg_HPF,np.min(Env_Times)); idx_max = np.searchsorted(Times_avg_HPF,np.max(Env_Times))
#     Env_BPF_OOPBM_interp = f_BPF(Times_avg_HPF)
#     cc_BPF.append(round(correlation_coef(Env_BPF_OOPBM_interp,avg_HPF_OOPBM[idx_min:idx_max]),3))

#     LPF_1_OOPBM_interp = f_LPF(Times_avg_HPF)
#     cc_LPF.append(round(correlation_coef(LPF_1_OOPBM_interp,avg_HPF_OOPBM[idx_min:idx_max]),3))

#     if window == 3 or window == 6 or window == 9:
#         label="Effectively filtered HPF $M_{H,\perp}$"+"\nWindow = {}s, offset: {}kN-m".format(window,offset)
#         plt.plot(Times_avg_HPF,np.add(avg_HPF_OOPBM,offset),color=colors[ix],label=label)
#         offset+=200000

#         ix+=1

# plt.xlabel("Time [s]")
# plt.ylabel("Aerodynamic out-of-plane bending moment magnitude [kN-m]")
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Effectively_filtered_HPF_OOPBM.png")
# plt.close(fig)



# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(windows,cc_BPF,"-r",label="Env BPF $M_{H,\perp}$ cc eff(|$M_{H,\perp,HPF}$|)")
# plt.plot(windows,cc_LPF,"-b",label="LPF $M_{H,\perp}$ cc eff(|$M_{H,\perp,HPF}$|)")
# idx = np.argmax(cc_BPF)
# plt.axvline(x=windows[idx],linestyle="--",color="r")
# idx = np.argmax(cc_LPF)
# plt.axvline(x=windows[idx],linestyle="--",color="b")
# plt.xlabel("Window size [s]")
# plt.ylabel("Correlation coefficient [-]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_avg_MH_HPF.png")
# plt.close(fig)


# # dt_HPF = Times_avg_HPF[1] - Times_avg_HPF[0]
# # Times_avg_HPF = np.array(Times_avg_HPF); avg_HPF_OOPBM = np.array(hard_filter(avg_HPF_OOPBM,0.3,dt_HPF,"lowpass"))

# window_idx = np.searchsorted(Time_OF,Time_OF[0]+9)
# if (window_idx % 2) != 0:
#     window_idx+=1
# Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]
# avg_HPF_OOPBM = []
# for i in np.arange(0,len(Time_OF)-window_idx):
#     avg_HPF_OOPBM.append(np.average(abs_HPF_OOPBM[i:i+window_idx]))

# Times_avg_HPF = np.array(Times_avg_HPF); avg_HPF_OOPBM = np.array(avg_HPF_OOPBM)

# perc_overlap_LPF_HPF = []
# perc_overlap_HPF_LPF = []
# perc_overlap_BPF_HPF = []
# perc_overlap_HPF_BPF = []
# perc_overlap_LPF_BPF = []
# perc_overlap_BPF_LPF = []
# thresholds = np.linspace(0,1,5)
# ix = 0
# for threshold in thresholds:
#     print(threshold)


#     idx_min = np.searchsorted(Time_OF,np.min(Times_avg_HPF)); idx_max = np.searchsorted(Time_OF,np.max(Times_avg_HPF))
#     xco_array_LPF = []
#     xco = []
#     for it in np.arange(idx_min,idx_max,dtype=int):
#         if len(xco) == 0 and LPF_1_OOPBM[it] >= np.mean(LPF_1_OOPBM)+threshold*np.std(LPF_1_OOPBM):
#             xco.append(Time_OF[it])
        
#         if len(xco) == 1 and LPF_1_OOPBM[it] < np.mean(LPF_1_OOPBM)+threshold*np.std(LPF_1_OOPBM):
#             xco.append(Time_OF[it])
#             xco_array_LPF.append(xco)
#             xco = []
#         print(it)


#     xco = []
#     xco_array_BPF = []
#     for it in np.arange(0,len(Env_Times)):
#         if len(xco) == 0 and Env_BPF_OOPBM[it] >= np.mean(Env_BPF_OOPBM)+threshold*np.std(Env_BPF_OOPBM):
#             xco.append(Env_Times[it])
        
#         if len(xco) == 1 and Env_BPF_OOPBM[it] < np.mean(Env_BPF_OOPBM)+threshold*np.std(Env_BPF_OOPBM):
#             xco.append(Env_Times[it])
#             xco_array_BPF.append(xco)
#             xco = []
#         print(it)


#     xco_array_HPF = []
#     xco = []
#     for it in np.arange(0,len(Times_avg_HPF)):
#         if len(xco) == 0 and avg_HPF_OOPBM[it] >= np.mean(avg_HPF_OOPBM)+threshold*np.std(avg_HPF_OOPBM):
#             xco.append(Times_avg_HPF[it])
        
#         if len(xco) == 1 and avg_HPF_OOPBM[it] < np.mean(avg_HPF_OOPBM)+threshold*np.std(avg_HPF_OOPBM):
#             xco.append(Times_avg_HPF[it])
#             xco_array_HPF.append(xco)
#             xco = []
#         print(it)

#     T_overlap_LPF_HPF = 0
#     T_LPF = 0
#     for xco_LPF in xco_array_LPF:
#         T_LPF+=(xco_LPF[1]-xco_LPF[0])
#         for xco_HPF in xco_array_HPF:
#             if xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
#                 T_overlap_LPF_HPF+=(xco_LPF[1] - xco_LPF[0])
#             elif xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1]:
#                 T_overlap_LPF_HPF+=(xco_HPF[1] - xco_LPF[0])
#             elif xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
#                 T_overlap_LPF_HPF+=(xco_LPF[1] - xco_HPF[0])
#             elif xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1] and xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
#                 T_overlap_LPF_HPF+=(xco_HPF[1] - xco_HPF[0])
    

#     perc_overlap_LPF_HPF.append(round((T_overlap_LPF_HPF/T_LPF)*100,1))

#     T_overlap_HPF_LPF = 0
#     T_HPF = 0
#     for xco_HPF in xco_array_HPF:
#         T_HPF+=(xco_HPF[1]-xco_HPF[0])
#         for xco_LPF in xco_array_LPF:
#             if xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1] and xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
#                 T_overlap_HPF_LPF+=(xco_HPF[1] - xco_HPF[0])
#             elif xco_LPF[0] <= xco_HPF[0] <= xco_LPF[1]:
#                 T_overlap_HPF_LPF+=(xco_LPF[1] - xco_HPF[0])
#             elif xco_LPF[0] <= xco_HPF[1] <= xco_LPF[1]:
#                 T_overlap_HPF_LPF+=(xco_HPF[1] - xco_LPF[0])
#             elif xco_HPF[0] <= xco_LPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_LPF[1] <= xco_HPF[1]:
#                 T_overlap_HPF_LPF+=(xco_LPF[1] - xco_LPF[0])

#     perc_overlap_HPF_LPF.append(round((T_overlap_HPF_LPF/T_HPF)*100,1))



#     T_overlap_BPF_HPF = 0
#     T_BPF = 0
#     for xco_BPF in xco_array_BPF:
#         T_BPF+=(xco_BPF[1]-xco_BPF[0])
#         for xco_HPF in xco_array_HPF:
#             if xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
#                 T_overlap_BPF_HPF+=(xco_BPF[1] - xco_BPF[0])
#             elif xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1]:
#                 T_overlap_BPF_HPF+=(xco_HPF[1] - xco_BPF[0])
#             elif xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
#                 T_overlap_BPF_HPF+=(xco_BPF[1] - xco_HPF[0])
#             elif xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1] and xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
#                 T_overlap_BPF_HPF+=(xco_HPF[1] - xco_HPF[0])

#     perc_overlap_BPF_HPF.append(round((T_overlap_BPF_HPF/T_BPF)*100,1))

#     T_overlap_HPF_BPF = 0
#     T_HPF = 0
#     for xco_HPF in xco_array_HPF:
#         T_HPF+=(xco_HPF[1]-xco_HPF[0])
#         for xco_BPF in xco_array_BPF:
#             if xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1] and xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
#                 T_overlap_HPF_BPF+=(xco_HPF[1] - xco_HPF[0])
#             elif xco_BPF[0] <= xco_HPF[0] <= xco_BPF[1]:
#                 T_overlap_HPF_BPF+=(xco_BPF[1] - xco_HPF[0])
#             elif xco_BPF[0] <= xco_HPF[1] <= xco_BPF[1]:
#                 T_overlap_HPF_BPF+=(xco_HPF[1] - xco_BPF[0])
#             elif xco_HPF[0] <= xco_BPF[0] <= xco_HPF[1] and xco_HPF[0] <= xco_BPF[1] <= xco_HPF[1]:
#                 T_overlap_HPF_BPF+=(xco_BPF[1] - xco_BPF[0])

#     perc_overlap_HPF_BPF.append(round((T_overlap_HPF_BPF/T_HPF)*100,1))


#     T_overlap_LPF_BPF = 0
#     T_LPF = 0
#     for xco_LPF in xco_array_LPF:
#         T_LPF+=(xco_LPF[1]-xco_LPF[0])
#         for xco_BPF in xco_array_BPF:
#             if xco_BPF[0] <= xco_LPF[0] <= xco_BPF[1] and xco_BPF[0] <= xco_LPF[1] <= xco_BPF[1]:
#                 T_overlap_LPF_BPF+=(xco_LPF[1] - xco_LPF[0])
#             elif xco_BPF[0] <= xco_LPF[0] <= xco_BPF[1]:
#                 T_overlap_LPF_BPF+=(xco_BPF[1] - xco_LPF[0])
#             elif xco_BPF[0] <= xco_LPF[1] <= xco_BPF[1]:
#                 T_overlap_LPF_BPF+=(xco_LPF[1] - xco_BPF[0])
#             elif xco_LPF[0] <= xco_BPF[0] <= xco_LPF[1] and xco_LPF[0] <= xco_BPF[1] <= xco_LPF[1]:
#                 T_overlap_LPF_BPF+=(xco_BPF[1] - xco_BPF[0])

#     perc_overlap_LPF_BPF.append(round((T_overlap_LPF_BPF/T_LPF)*100,1))


#     T_overlap_BPF_LPF = 0
#     T_BPF = 0
#     for xco_BPF in xco_array_BPF:
#         T_BPF+=(xco_BPF[1]-xco_BPF[0])
#         for xco_LPF in xco_array_LPF:
#             if xco_LPF[0] <= xco_BPF[0] <= xco_LPF[1] and xco_LPF[0] <= xco_BPF[1] <= xco_LPF[1]:
#                 T_overlap_BPF_LPF+=(xco_BPF[1] - xco_BPF[0])
#             elif xco_LPF[0] <= xco_BPF[0] <= xco_LPF[1]:
#                 T_overlap_BPF_LPF+=(xco_LPF[1] - xco_BPF[0])
#             elif xco_LPF[0] <= xco_BPF[1] <= xco_LPF[1]:
#                 T_overlap_BPF_LPF+=(xco_BPF[1] - xco_LPF[0])
#             elif xco_BPF[0] <= xco_LPF[0] <= xco_BPF[1] and xco_BPF[0] <= xco_LPF[1] <= xco_BPF[1]:
#                 T_overlap_BPF_LPF+=(xco_LPF[1] - xco_LPF[0])

#     perc_overlap_BPF_LPF.append(round((T_overlap_BPF_LPF/T_BPF)*100,1))



#     if threshold == 1.0:
#         fig,ax = plt.subplots(figsize=(14,8))
#         idx_min = np.searchsorted(Time_OF,np.min(Times_avg_HPF)); idx_max = np.searchsorted(Time_OF,np.max(Times_avg_HPF))
#         ax.plot(Time_OF[idx_min:idx_max],LPF_1_OOPBM[idx_min:idx_max],"-g")
#         ax.set_ylabel("LPF OOPBM magnitude [kN-m]")
#         ax.grid()
#         ax.axhline(y=np.mean(LPF_1_OOPBM)+threshold*np.std(LPF_1_OOPBM),linestyle="--",color="g")

#         for xco in xco_array_LPF:
#             square = patches.Rectangle((xco[0],np.min(LPF_1_OOPBM)), (xco[1]-xco[0]), (np.max(LPF_1_OOPBM)-np.min(LPF_1_OOPBM)), fill=True,color="g",alpha=0.1)
#             ax.add_patch(square)


#         ax2=ax.twinx()
#         ax2.plot(Times_avg_HPF,avg_HPF_OOPBM,"-b")
#         ax2.axhline(y=np.mean(avg_HPF_OOPBM)+threshold*np.std(avg_HPF_OOPBM),linestyle="--",color="b")

#         for xco in xco_array_HPF:
#             square = patches.Rectangle((xco[0],np.min(avg_HPF_OOPBM)), (xco[1]-xco[0]), (np.max(avg_HPF_OOPBM)-np.min(avg_HPF_OOPBM)), fill=True,color="b",alpha=0.1)
#             ax2.add_patch(square)

#         ax2.set_ylabel("Effectively filtered (9s) HPF OOPBM magnitude [kN-m]")
#         fig.supxlabel("Time [s]")
#         plt.tight_layout()
#         plt.savefig("../../Thesis/Figures/LPF_HPF_OOPBM_bursting_periods.png")
#         plt.close(fig)

#     ix+=1



# plt.figure(figsize=(14,8))
# plt.plot(thresholds,perc_overlap_LPF_HPF,"-o",label="LPF overlaps HPF")
# plt.plot(thresholds,perc_overlap_HPF_LPF,"-o",label="HPF overlaps LPF")
# plt.plot(thresholds,perc_overlap_BPF_HPF,"-o",label="BPF overlaps HPF")
# plt.plot(thresholds,perc_overlap_HPF_BPF,"-o",label="HPF overlaps BPF")
# plt.plot(thresholds,perc_overlap_LPF_BPF,"-o",label="LPF overlaps BPF")
# plt.plot(thresholds,perc_overlap_BPF_LPF,"-o",label="BPF overlaps LPF")
# plt.xlabel("Threshold $T$: $mean(x)+T*std(x)$ [kN]")
# plt.ylabel("Percentage overlap [%]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Threshold_percentage_overlap.png")
# plt.close(fig)


in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["Time_OF"])
dt = Time_OF[1] - Time_OF[0]

Time_start = 200
Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_steps = np.arange(Time_start_idx,len(Time_OF)-1)
Time_OF = Time_OF[Time_start_idx:]

OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

OOPBM = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))
MTheta = np.degrees(np.arctan2(RtAeroMys,-RtAeroMzs))
MTheta = theta_360(MTheta)



Time_sampling = np.array(df_OF.variables["Time_sampling"])
dt_sampling = Time_sampling[1] - Time_sampling[0]
Time_start = 200
Time_sampling_start_idx = np.searchsorted(Time_sampling,Time_start)

Time_sampling = Time_sampling[Time_sampling_start_idx:]

Rotor_avg_vars = df_OF.groups["Rotor_Avg_Variables"]
Rotor_avg_vars_63 = Rotor_avg_vars.groups["5.5"]

IA = np.array(Rotor_avg_vars_63.variables["IA"][Time_sampling_start_idx:])
LPF_IA = hard_filter(IA,0.3,dt_sampling,"lowpass")


df_WT = Dataset(in_dir+"WTG01b.nc")

WT = df_WT.groups["WTG01"]


Rotor_coordinates = [np.float64(WT.variables["xyz"][0,0,0]),np.float64(WT.variables["xyz"][0,0,1]),np.float64(WT.variables["xyz"][0,0,2])]


df = Dataset(in_dir+"WTG01a.nc")
uvelB1 = np.array(df.variables["uvel"][:,1:301])
vvelB1 = np.array(df.variables["vvel"][:,1:301])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
hvelB1 = np.add(np.cos(np.radians(29.29))*uvelB1, np.sin(np.radians(29.29))*vvelB1)
uvelB2 = np.array(df.variables["uvel"][:,301:601])
vvelB2 = np.array(df.variables["vvel"][:,301:601])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
hvelB2 = np.add(np.cos(np.radians(29.29))*uvelB2, np.sin(np.radians(29.29))*vvelB2)
uvelB3 = np.array(df.variables["uvel"][:,601:901])
vvelB3 = np.array(df.variables["vvel"][:,601:901])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
hvelB3 = np.add(np.cos(np.radians(29.29))*uvelB3, np.sin(np.radians(29.29))*vvelB3)


R = np.linspace(0,63,300)
dr = R[1] - R[0]

Iy = []
Iz = []
ix=0
with Pool() as pool:
    for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,Time_steps):
        Iy.append(Iy_it); Iz.append(Iz_it)
        print(ix)
        ix+=1
Iy = np.array(Iy); Iz = -np.array(Iz)
I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
ITheta = np.degrees(np.arctan2(Iy,-Iz))
ITheta = theta_360(ITheta)

LPF_I = hard_filter(I,0.3,dt,"lowpass")
BPF_I = hard_filter(I,[0.3,0.9],dt,"bandpass")
HPF_I = hard_filter(I,[1.5,40],dt,"bandpass")

LPF_OOPBM = hard_filter(OOPBM,0.3,dt,"lowpass")
BPF_OOPBM = hard_filter(OOPBM,[0.3,0.9],dt,"bandpass")
HPF_OOPBM = hard_filter(OOPBM,[1.5,40],dt,"bandpass")

print(correlation_coef(I,OOPBM[:-1]))
print(correlation_coef(LPF_I,LPF_OOPBM[:-1]))
print(correlation_coef(BPF_I,BPF_OOPBM[:-1]))
print(correlation_coef(HPF_I,HPF_OOPBM[:-1]))

print(correlation_coef(ITheta,MTheta[:-1]))

plt.rcParams['font.size'] = 16
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,OOPBM,"-r")
ax.set_ylabel("Aerodynamic out-of-plane bending moment magnitude [kN-m]")
ax.yaxis.label.set_color('red') 
ax2=ax.twinx()
ax2.plot(Time_OF[:-1],I,"-b")
ax2.set_ylabel("Blade asymmetry [$m^3/s$]")
ax2.yaxis.label.set_color('blue') 
ax2.grid()
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig("../../Thesis/Figures/cc_IB_OOPBM.png")
plt.close(fig)

OOPBM_norm = OOPBM/np.mean(OOPBM)
IB_norm = I/np.mean(I)

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(OOPBM_norm,dt,"OOPBM")
plt.loglog(frq,PSD,"-r",label="Normalised $M_{H,\perp}$")
frq,PSD = temporal_spectra(IB_norm,dt,"IB")
plt.loglog(frq,PSD,"-b",label="Normalised $I_B$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig("../../Thesis/Figures/spectra_IB_OOPBM.png")
plt.close(fig)


f = interpolate.interp1d(Time_OF[:-1],LPF_I)
LPF_I_interp = f(Time_sampling)

time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+4.6)

Time_sampling_shift = Time_sampling[:-time_shift_idx]

#LPF_I_interp_shift = LPF_I_interp[time_shift_idx:]

#LPF_IA_shift = LPF_IA[:-time_shift_idx]

cc = round(correlation_coef(LPF_I_interp,LPF_IA),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling,LPF_IA,"-r")
ax.set_ylabel("LPF (0.3Hz) Asymmetry parameter [$m^4/s$]")
ax.yaxis.label.set_color('red') 
ax2=ax.twinx()
ax2.plot(Time_sampling,LPF_I_interp,"-b")
ax2.set_ylabel("LPF (0.3Hz) Blade asymmetry [$m^3/s$]")
ax2.yaxis.label.set_color('blue') 
ax2.grid()
fig.supxlabel("Time [s]")
fig.suptitle("Correlation coefficient = {}".format(cc))
plt.tight_layout()
plt.savefig("../../Thesis/Figures/cc_LPF_IB_I.png")
plt.close(fig)