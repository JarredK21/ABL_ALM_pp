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
from scipy.signal import butter,filtfilt


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

    E = np.sum(e_fft)

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
    PSD = (abs(Y[range(nhalf)])**2) /n # PSD
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


def low_pass_filter(signal, cutoff,dt):

    fs = 1/dt     # sample rate, Hz
    nyq = 0.5 * fs  # Nyquist Frequency
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


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


def actuator_asymmetry_calc_75(it):

    xo = np.array(WT.variables["xyz"][it,225,0])
    yo = np.array(WT.variables["xyz"][it,225,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB1 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB1 = np.array(WT.variables["xyz"][it,225,2]) - Rotor_coordinates[2]

    xo = np.array(WT.variables["xyz"][it,526,0])
    yo = np.array(WT.variables["xyz"][it,526,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB2 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB2 = np.array(WT.variables["xyz"][it,526,2]) - Rotor_coordinates[2]


    xo = np.array(WT.variables["xyz"][it,827,0])
    yo = np.array(WT.variables["xyz"][it,827,1])
    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    yB3 = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
    zB3 = np.array(WT.variables["xyz"][it,827,2]) - Rotor_coordinates[2]


    IyB1 = np.sum(hvelB1[it,225]*zB1)*dr
    IzB1 = np.sum(hvelB1[it,225]*yB1)*dr

    IyB2 = np.sum(hvelB2[it,225]*zB2)*dr
    IzB2 = np.sum(hvelB2[it,225]*yB2)*dr


    IyB3 = np.sum(hvelB3[it,225]*zB3)*dr
    IzB3 = np.sum(hvelB3[it,225]*yB3)*dr

    return IyB1+IyB2+IyB3, IzB1+IzB2+IzB3


def probability_dist(y,N):
    std = np.std(y)
    if N=="default":
        N=20
    bin_width = std/N
    x = np.arange(np.min(y),np.max(y)+bin_width,bin_width)
    dx = x[1]-x[0]
    P = []
    X = []
    for i in np.arange(0,len(x)-1):
        p = 0
        for yi in y:
            if x[i] <= yi <= x[i+1]:
                p+=1
        P.append(p/(dx*len(y)))
        X.append((x[i+1]+x[i])/2)

    print(np.sum(P)*dx)

    return P,X


def moments(y,X,P):
    mu = np.mean(y)
    std = np.std(y)

    mode = X[np.argmax(P)]
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    print("mode = {}, mean = {}, std = {}, skew = {}, flat = {}".format(round(mode,4),round(mu,4),round(std,4),round(skewness,4),round(kurotsis,4)))

    return round(mode,2), round(mu,2), round(std,2), round(skewness,2), round(kurotsis,2)



def isInside(x, y):
     
    if ((x - 2560) * (x - 2560) +
        (y - 90) * (y - 90) < 63 * 63):
        return True
    else:
        return False
    

def Area_calc(it):

    H = Heights[it]
    A = 0
    h = []
    for j in np.arange(0,len(ys)):
        #is coordinate inside rotor disk
        cc = isInside(ys[j],H[j])

        if cc == True:

            z = np.min( np.roots([1,-180,(90**2-63**2+(ys[j]-2560)**2)]) )
            h.append(H[j]-z) #height from coordinate zs to coordinate z on rotor disk

        #is coordinate above rotor disk so it is still covering it
        elif ys[j] > 2497 and ys[j] < 2623 and H[j] > 153:
            z = np.roots([1,-180,(90**2-63**2+(ys[j]-2560)**2)])
            h.append(z[0]-z[1]) #height

        if len(h) > 1 and isInside(ys[j+1],H[j+1]) == False:

            #integrate over sub area covering rotor disk
            for i in np.arange(0,len(h)-1):
                A+=((h[i+1] + h[i])/2)*dy

            h = [] #reset h array


    return A

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

# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
# df = Dataset(in_dir+"abl_statistics70000.nc")
# Mean_profiles = df.groups["mean_profiles"]
# print(Mean_profiles)
# time = np.array(df.variables["time"])
# Time_start_idx = np.searchsorted(time,38000)
# z = np.array(Mean_profiles.variables["h"])
# hub_idx = np.searchsorted(z,90)

# u = np.average(Mean_profiles.variables["u"][Time_start_idx:][:],axis=0)
# v = np.average(Mean_profiles.variables["v"][Time_start_idx:][:],axis=0)
# twist = coriolis_twist(u,v)
# f = interpolate.interp1d(z,twist); twist_90 = f(90); print(twist_90)


# u_w_r = np.average(Mean_profiles.variables["u'w'_r"][Time_start_idx:][:],axis=0)
# u_w_sfs = np.average(Mean_profiles.variables["u'w'_sfs"][Time_start_idx:][:],axis=0)

# u_v_r = np.average(Mean_profiles.variables["u'v'_r"][Time_start_idx:][:],axis=0)
# u_v_sfs = np.average(Mean_profiles.variables["u'v'_sfs"][Time_start_idx:][:],axis=0)

# v_w_r = np.average(Mean_profiles.variables["v'w'_r"][Time_start_idx:][:],axis=0)
# v_w_sfs = np.average(Mean_profiles.variables["v'w'_sfs"][Time_start_idx:][:],axis=0)


# fu = interpolate.interp1d(z,u_w_r); fv = interpolate.interp1d(z,v_w_r)
# print("u'w'_r(90) = ",fu(90)*np.cos(twist_90)+fv(90)*np.sin(twist_90))
# fu_sfs = interpolate.interp1d(z,u_w_sfs); fv_sfs = interpolate.interp1d(z,v_w_sfs)
# print("u'w'_t(90) = ",fu(90)*np.cos(twist_90)+fv(90)*np.sin(twist_90) + fu_sfs(90)*np.cos(twist_90)+fv_sfs(90)*np.sin(twist_90))

# fu = interpolate.interp1d(z,u_w_r); fv = interpolate.interp1d(z,v_w_r)
# print("v'w'_r(90) = ",-fu(90)*np.sin(twist_90)+fv(90)*np.cos(twist_90))
# fu_sfs = interpolate.interp1d(z,u_w_sfs); fv_sfs = interpolate.interp1d(z,v_w_sfs)
# print("v'w'_t(90) = ",-fu(90)*np.sin(twist_90)+fv(90)*np.cos(twist_90) - fu_sfs(90)*np.sin(twist_90)+fv_sfs(90)*np.cos(twist_90))

# f = interpolate.interp1d(z,u_v_r)
# print("u'v'_r(90) = ",f(90))
# f_sfs = interpolate.interp1d(z,u_v_sfs)
# print("u'v'_t(90) = ",f(90)+f_sfs(90))

# dz = z[1] - z[0]
# theta = np.average(Mean_profiles.variables["theta"][Time_start_idx:][:],axis=0)
# f = interpolate.interp1d(z,theta)
# print(f(90))
# dtheta_dz = dz_calc(theta,z)
# f = interpolate.interp1d(z[:-1],dtheta_dz)
# print(f(90))



# hvelmag = []
# for i in np.arange(0,len(twist)):
#     hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
# hvelmag = np.array(hvelmag)



# #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
# d_dz = []
# for i in np.arange(0,len(hvelmag)-4,1):
#     if i == 0:
#         d_dz_i = ((-(25/12)*hvelmag[i]+4*hvelmag[i+1]-3*hvelmag[i+2]+(4/3)*hvelmag[i+3]-(1/4)*hvelmag[i+4])/dz)
#     else:
#         d_dz_i = ((hvelmag[i+1] - hvelmag[i-1])/(2*dz))

#     d_dz.append(d_dz_i)

# f = interpolate.interp1d(z[:-4],d_dz)
# print(f(90))

# a = 1

# # #Figure 3-2 evolution boundary layer
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
#     ax1.set_xlabel("$d\\theta/dz$\n[K/m]",fontsize=24)
#     ax1.grid()
#     ax1.set_ylim([0,700])
#     ax2.plot(w_theta,z)
#     ax2.set_xlabel("$\langle w' \\theta' \\rangle$\n[Km/s]",fontsize=24)
#     ax2.grid()
#     ax2.set_ylim([0,700])
#     ax3.plot(w_w_r,z)
#     ax3.set_xlabel("$\langle w'w' \\rangle$\n$[m^2/s^2]$",fontsize=24)
#     ax3.grid()
#     ax3.set_ylim([0,700])
#     fig.supylabel("Height from surface [m]",fontsize=24)
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
# print(np.max(time/tau_u))
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(time,tau_u)
# plt.xlabel("Time [s]",fontsize=24)
# plt.ylabel("Eddy turnover time associated with the\ncharacteristic velocity of the surface layer\n$\\tau_u=z_i/u_*$ [s]",fontsize=24)
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


# plt.xlabel("$Re_{LES}$ [-]")
# plt.ylabel("$\mathfrak{R}$ [-]")
# plt.grid()
# plt.legend(labels)
# plt.axvline(x=400,linestyle="--",color="k")
# plt.axhline(y=0.9,linestyle="--",color="k")
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
# plt.ylabel("$z/z_i$ [-]")
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


# plt.xlabel("$Re_{LES}$ [-]")
# plt.ylabel("$\mathfrak{R}$ [-]")
# plt.grid()
# plt.legend(labels)
# plt.axvline(x=400,linestyle="--",color="k")
# plt.axhline(y=0.9,linestyle="--",color="k")
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
# plt.ylabel("$z/z_i$ [-]")
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


# plt.xlabel("$Re_{LES}$ [-]")
# plt.ylabel("$\mathfrak{R}$ [-]")
# plt.grid()
# plt.legend(labels)
# plt.axvline(x=400,linestyle="--",color="k")
# plt.axhline(y=0.9,linestyle="--",color="k")
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
# plt.ylabel("$z/z_i$ [-]")
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
# plt.rcParams['font.size'] = 16
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


# fig = plt.figure(figsize=(14,8))
# plt.plot(Time[:-1],dzi_dt_u_star)
# window_idx = int((glob_tau_u)/dt)
# ts_dzi_dt_u_star.rolling(window=window_idx).mean().plot(style='k--')
# ts_dzi_dt_u_star.rolling(window=window_idx).std().plot(style='r--')
# plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("$dz_i/dt 1/u_*$ [-]",fontsize=22)
# plt.legend(["$dz_i/dt 1/u_*$","Average","Standard deviation","0.01","-0.01"])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dzi_dt_1_ustar.png")
# plt.close(fig)

# #dzi/dt 1/w*
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time[:-1],dzi_dt_w_star)
# window_idx = int((glob_tau_w)/dt)
# ts_dzi_dt_w_star.rolling(window=window_idx).mean().plot(style='k--')
# ts_dzi_dt_w_star.rolling(window=window_idx).std().plot(style='r--')
# plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("$dz_i/dt 1/w_*$ [-]",fontsize=22)
# plt.legend(["$dz_i/dt 1/w_*$","Average","Standard deviation","0.01","-0.01"])
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
# u_star = np.array(df.variables["ustar"])
# w_star = np.array(df.variables["wstar"])
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

in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
df = Dataset(in_dir+"abl_statistics70000.nc")
Time_2 = np.array(df.variables["time"])
#dt = Time[1]-Time[0]

# zi_2 = np.array(df.variables["zi"])
# u_star_2 = np.array(df.variables["ustar"])
# w_star_2 = np.array(df.variables["wstar"])
# L_2 = np.array(df.variables["L"])
# zi_L_2 = -np.true_divide(zi_2,L_2)

# Time = np.concatenate((Time,Time_2))
# zi = np.concatenate((zi,zi_2))
# u_star = np.concatenate((u_star,u_star_2))
# w_star = np.concatenate((w_star,w_star_2))

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

# #fig 3-9
# #dzi/dt 1/u*
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time[:-1],dzi_dt_u_star)
# window_idx = int(3*(glob_tau_u)/dt)
# ts_dzi_dt_u_star.rolling(window=window_idx).mean().plot(style='k--')
# ts_dzi_dt_u_star.rolling(window=window_idx).std().plot(style='r--')
# plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("$dz_i/dt 1/u_*$ [-]",fontsize=22)
# plt.legend(["$dz_i/dt 1/u_*$","Average","Standard deviation","0.01","-0.01"])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dzi_dt_1_ustar.png")
# plt.close(fig)

# #dzi/dt 1/w*
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time[:-1],dzi_dt_w_star)
# window_idx = int(3*(glob_tau_w)/dt)
# ts_dzi_dt_w_star.rolling(window=window_idx).mean().plot(style='k--')
# ts_dzi_dt_w_star.rolling(window=window_idx).std().plot(style='r--')
# plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("$dz_i/dt 1/w_*$ [-]",fontsize=22)
# plt.legend(["$dz_i/dt 1/w_*$","Average","Standard deviation","0.01","-0.01"])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dzi_dt_1_wstar.png")
# plt.close(fig)


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

tstart_idx = np.searchsorted(Time_2,38000); tend_idx = np.searchsorted(Time_2,39200)
uhub = np.average(hvelmag_hub_2[tstart_idx:tend_idx])
ustd = np.std(hvelmag_hub_2[tstart_idx:tend_idx])

TI = (ustd/uhub)*100

print(uhub)
print(TI)

plt.plot(Time_2,hvelmag_hub_2)
plt.show()

a =1

#hvelmag_hub = np.concatenate((hvelmag_hub,hvelmag_hub_2))
# zi_L = np.concatenate((zi_L,zi_L_2))


# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,hvelmag_hub)
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("Streamwise velocity $u_{x'}$ [m/s]",fontsize=22)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Time_Uhub.png")
# plt.close(fig)


# #-zi/L
# plt.figure(figsize=(14,8))
# plt.plot(Time,zi_L)
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("Capping inversion height $-z_i/L$ [m]",fontsize=22)
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
# plt.figure()
# plt.plot(hvelmag,z_zi)
# plt.xlabel("Streamwise velocity $u_{x'}$ [m/s]",fontsize=22)
# plt.ylabel("$z/z_i$ [-]",fontsize=22)
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/U.png")
# plt.close(fig)

# #mean profiles 0.2zi
# #U
# plt.figure()
# plt.plot(hvelmag,z_zi)
# plt.xlabel("Streamwise velocity $u_{x'}$ [m/s]",fontsize=22)
# plt.ylabel("$z/z_i$ [-]",fontsize=22)
# plt.ylim([0,0.2])
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/U_rotor.png")
# plt.close(fig)

# #Coriolis twist
# plt.figure()
# plt.plot(np.degrees(twist),z_zi)
# plt.xlabel("Flow angle [$\circ$]",fontsize=22)
# plt.ylabel("$z/z_i$ [-]",fontsize=22)
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/flow_angle.png")
# plt.close(fig)


# #u'w'_r
# plt.figure()
# plt.plot(u_w_r,z_zi)
# plt.xlabel("$\langle u'w' \\rangle ^r$ $[m^2/s^2]$",fontsize=22)
# plt.ylabel("$z/z_i$ [-]",fontsize=22)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/u_w_r.png")
# plt.close(fig)

# #Theta
# plt.figure()
# plt.plot(theta,z_zi)
# plt.xlabel("Potential temperature [K]",fontsize=22)
# plt.ylabel("$z/z_i$ [-]",fontsize=22)
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Pot_temp.png")
# plt.close(fig)

# fu = interpolate.interp1d(z,u)
# fv = interpolate.interp1d(z,v)
# heights = np.array([0.1,0.4,0.8,1.0,1.1,1.2])
# plt.figure()
# for height in heights:
#     height_m = height*glob_zi
#     u_h = fu(height_m)
#     v_h = fv(height_m)
#     plt.arrow(0,0,u_h,v_h,length_includes_head=True,color="#1f77b4",head_length=0.2,head_width=0.2)
#     if height == 0.1:
#         plt.text(u_h-0.2,v_h,"${}z_i$".format(height))
#     elif height == 0.4:
#         plt.text(u_h+0.2,v_h-0.2,"${}z_i$".format(height))
#     else:
#         plt.text(u_h,v_h,"${}z_i$".format(height))

# plt.xlabel("$U$ - average velocity [m/s]",fontsize=22)
# plt.ylabel("$V$ - average velocity [m/s]",fontsize=22)
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


#Figure 3-13 mean Spectra 2d x-y plane 90m u, w velocities
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
# plt.rcParams['font.size'] = 16
# fig = plt.figure()
# plt.plot(X_uu,PDF_uu_mean,"-k")
# plt.xlabel("Fluctuating streamwise velocity [m/s]")
# plt.ylabel("Probability [-]")
# plt.axvline(x=-0.61,linestyle="--",color="b",label="low speed streaks")
# plt.axvline(x=0.76,linestyle="--",color="r",label="high speed regions")
# plt.grid()
# plt.legend(loc="upper left")
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
# plt.legend(loc="upper left")
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
# plt.xlabel("Filter width [m]",fontsize=20)
# plt.ylabel("Average Eddy length [m]",fontsize=20)
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Mean_Eddy_length.png")
# plt.close()

# #figure 3-14b std eddy length
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5


#     plt.plot(filter_cutoff,std_high_D[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,std_low_D[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]",fontsize=20)
# plt.ylabel("Standard deviation Eddy length [m]",fontsize=20)
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Std_Eddy_length.png")
# plt.close()


# #Figure 3-15a mean Eddy velocity
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5

#     plt.plot(filter_cutoff,mean_high_ux[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,mean_low_ux[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]",fontsize=20)
# plt.ylabel("Average Eddy velocity [m/s]",fontsize=20)
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Mean_Eddy_velocity.png")
# plt.close()

# #Figure 3-15b std eddy velocity
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5


#     plt.plot(filter_cutoff,std_high_ux[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,std_low_ux[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]",fontsize=20)
# plt.ylabel("Standard deviation Eddy velocity [m/s]",fontsize=20)
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Std_Eddy_velocity.png")
# plt.close()


# #Figure 3-16a mean Eddy passage time
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5

#     plt.plot(filter_cutoff,mean_high_Tau[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,mean_low_Tau[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]",fontsize=20)
# plt.ylabel("Average Eddy Passage time [s]",fontsize=20)
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Mean_Eddy_passage_time.png")
# plt.close()

# #Figure 3-16b std Eddy passage time
# fig = plt.figure(figsize=(14,8))
# for i in np.arange(0,len(offsets)):

#     height = offsets[i]+7.5


#     plt.plot(filter_cutoff,std_high_Tau[i],"-o",color=colors[i],label="high speed: {}m".format(height))
#     plt.plot(filter_cutoff,std_low_Tau[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
# plt.xlabel("Filter width [m]",fontsize=20)
# plt.ylabel("Standard deviation Eddy Passage time [s]",fontsize=20)
# plt.legend()
# plt.tight_layout()
# plt.grid()
# plt.savefig("../../Thesis/Figures/Std_Eddy_passage_time.png")
# plt.close()

#Figure 3-17 summary eddy passage time 167m filter
# plt.rcParams['font.size'] = 16
# x = ["High speed\n30m","High speed\n92.5m","High speed\n150m","Low speed\n30m","Low speed\n92.5m","Low speed\n150m"]


# T_mean = [28.3,20.7,18.6,39.7,32.6,29.2]
# T_std = [[0,0,0,0,0,0],[42.2-28.3,37.3-20.7,35.7-18.6,71.0-39.7,59.8-32.6,56.3-29.2]]

# fig = plt.figure(figsize=(14,8))
# plt.bar(x,T_mean)
# plt.errorbar(x,T_mean,yerr=T_std,fmt="o", color="k",capsize=10)
# plt.ylabel("Eddy passage time [s]")
# plt.grid()
# plt.title("Filter width 167m")
# plt.savefig("../../Thesis/Figures/eddy_passage_time_summary_bar_unedit.png")
# plt.close()


# BlFract = np.array([0.0000000E+00, 3.2500000E-03, 1.9510000E-02, 3.5770000E-02, 5.2030000E-02, 6.8290000E-02, 8.4550000E-02, 1.0081000E-01, 1.1707000E-01, 1.3335000E-01, 1.4959000E-01,
#     1.6585000E-01, 1.8211000E-01, 1.9837000E-01, 2.1465000E-01, 2.3089000E-01, 2.4715000E-01, 2.6341000E-01, 2.9595000E-01, 3.2846000E-01, 3.6098000E-01, 3.9350000E-01, 
#     4.2602000E-01, 4.5855000E-01, 4.9106000E-01, 5.2358000E-01, 5.5610000E-01, 5.8862000E-01, 6.2115000E-01, 6.5366000E-01, 6.8618000E-01, 7.1870000E-01, 7.5122000E-01,
#     7.8376000E-01, 8.1626000E-01, 8.4878000E-01, 8.8130000E-01, 8.9756000E-01, 9.1382000E-01, 9.3008000E-01, 9.3821000E-01, 9.4636000E-01, 9.5447000E-01, 9.6260000E-01,
#     9.7073000E-01, 9.7886000E-01, 9.8699000E-01, 9.9512000E-01, 1.0000000E+00])

# twist = np.array([1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01, 1.3308000E+01,
#     1.3308000E+01, 1.3181000E+01, 1.2848000E+01, 1.2192000E+01, 1.1561000E+01, 1.1072000E+01, 1.0792000E+01, 1.0232000E+01, 9.6720000E+00, 9.1100000E+00, 8.5340000E+00,
#     7.9320000E+00, 7.3210000E+00, 6.7110000E+00, 6.1220000E+00, 5.5460000E+00, 4.9710000E+00, 4.4010000E+00, 3.8340000E+00, 3.3320000E+00, 2.8900000E+00, 2.5030000E+00,
#     2.1160000E+00, 1.7300000E+00, 1.3420000E+00, 9.5400000E-01, 7.6000000E-01, 5.7400000E-01, 4.0400000E-01, 3.1900000E-01, 2.5300000E-01, 2.1600000E-01, 1.7800000E-01,
#     1.4000000E-01, 1.0100000E-01, 6.2000000E-02, 2.3000000E-02, 0.0000000E+00])

# BMassDen = [6.7893500E+02, 6.7893500E+02, 7.7336300E+02, 7.4055000E+02, 7.4004200E+02, 5.9249600E+02, 4.5027500E+02, 4.2405400E+02, 4.0063800E+02, 3.8206200E+02, 3.9965500E+02,
#             4.2632100E+02, 4.1682000E+02, 4.0618600E+02, 3.8142000E+02, 3.5282200E+02, 3.4947700E+02, 3.4653800E+02, 3.3933300E+02, 3.3000400E+02, 3.2199000E+02, 3.1382000E+02,
#             2.9473400E+02, 2.8712000E+02, 2.6334300E+02, 2.5320700E+02, 2.4166600E+02, 2.2063800E+02, 2.0029300E+02, 1.7940400E+02, 1.6509400E+02, 1.5441100E+02, 1.3893500E+02, 
#             1.2955500E+02, 1.0726400E+02, 9.8776000E+01, 9.0248000E+01, 8.3001000E+01, 7.2906000E+01, 6.8772000E+01, 6.6264000E+01, 5.9340000E+01, 5.5914000E+01, 5.2484000E+01,
#             4.9114000E+01, 4.5818000E+01, 4.1669000E+01, 1.1453000E+01, 1.0319000E+01]

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(BlFract,twist)
# plt.xlabel("Non-dimensionalised Blade Span from the Root [-]",fontsize=20)
# plt.ylabel("Structural twist [$\circ$]",fontsize=20)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/NREL_5MW_twist_angle.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# plt.plot(BlFract,BMassDen)
# plt.xlabel("Non-dimensionalised Blade Span from the Root [-]",fontsize=20)
# plt.ylabel("Blade mass density [kg/m]",fontsize=20)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/NREL_5MW_blade_mass_density.png")
# plt.close()


# #Figure 4-1a
# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/ALM sensitivity Analysis/"
# cases = ["dblade_2.0","dblade_1.0","dblade_0.5"]
# colors = ["red","blue","green"]
# labels = ["1a","1b","1c"]
# variables = ["RtAeroMxh_[N-m]","RtAeroMyh_[N-m]"]
# ylabels = ["PSD - Aerodynamic Hub moment around x axis $\widetilde{M}_{H,x}$ [kN-m]","PSD - Aerodynamic Hub moment around y axis $\widetilde{M}_{\widehat{H},y}$ [kN-m]"]
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


# # Section 5

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

# RtAeroFxh = np.array(OpenFAST_vars.variables["RtAeroFxh"][Time_start_idx:])/1000
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

# # # L1 = 1.912; L2 = 2.09
# # # fig = plt.figure(figsize=(14,8))
# # # frq,PSD = temporal_spectra(MR/L2,dt,"MR/L2")
# # # plt.loglog(frq,PSD,"-b",label="$\widetilde{M}_{H,\perp,mod}/L_2$")
# # # frq,PSD = temporal_spectra(FR*((L1+L2)/L2),dt,"FRL/L2")
# # # plt.loglog(frq,PSD,"-r",label="$\widetilde{F}_{H,\perp}(L/L_2)$")
# # # plt.xlabel("Frequency [Hz]",fontsize=20)
# # # plt.ylabel("PSD - Out-of-plane hub force and moment\ncontributions to main bearing force [$kN^2$]",fontsize=20)
# # # plt.legend()
# # # plt.grid()
# # # plt.tight_layout()
# # # plt.savefig("../../Thesis/Figures/FR_MR_spectra.png")
# # # plt.close(fig)

# # # I_norm = I/np.mean(I); MR_norm = MR/np.mean(MR)
# # # fig = plt.figure(figsize=(14,8))
# # # frq,PSD = temporal_spectra(MR_norm,dt,"MR")
# # # plt.loglog(frq,PSD,"-r",label="$\widetilde{M}_{H,\perp}$ [-]")
# # # frq,PSD = temporal_spectra(I_norm,dt_sample,"I")
# # # plt.loglog(frq,PSD,"-b",label="$I$ [-]")
# # # plt.title("Magnitude of vector normalized on mean")
# # # plt.grid()
# # # plt.legend()
# # # plt.xlabel("Frequency [Hz]")
# # # plt.ylabel("PSD")
# # # plt.tight_layout()
# # # plt.savefig("../../Thesis/Figures/spectra_I_MR.png")
# # # plt.close(fig)

# # # # # cc = round(correlation_coef(IA,I),2)
# # # # # fig,ax = plt.subplots(figsize=(14,8))
# # # # # ax.plot(Time_sampling,IA,"-r")
# # # # # ax.set_ylabel("Asymmetry parameter [$m^4/s$]")
# # # # # ax.yaxis.label.set_color('red')
# # # # # ax2=ax.twinx()
# # # # # ax2.plot(Time_sampling,I,"-b")
# # # # # ax2.set_ylabel("Asymmetry vector magnitude [$m^4/s$]")
# # # # # ax2.yaxis.label.set_color('blue')
# # # # # ax2.grid()
# # # # # fig.supxlabel("Time [s]")
# # # # # fig.suptitle("Correlation coefficient = {}".format(cc))
# # # # # plt.tight_layout()
# # # # # plt.savefig("../../Thesis/Figures/cc_IA_A.png")
# # # # # plt.close(fig)



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


# cc1 = correlation_coef(ITheta_LPF_shift,MTheta_LPF_shift)
# cc2 = correlation_coef(I_LPF_shift,MR_LPF_shift)
# cc3 = correlation_coef(Iy_LPF_shift,RtAeroMys_LPF_shift)
# cc4 = correlation_coef(Iz_LPF_shift,RtAeroMzs_LPF_shift)

# xlabel = ["$|\mathbf{I}|$ cc $|\widetilde{\mathbf{M}}_{H,\perp,mod}|$", "$\\theta (I)$ cc $\\theta (\widetilde{\mathbf{M}}_{H,\perp,mod})$",
#           "$I_y$ cc $\widetilde{M}_{H,y}$", "$I_z$ cc $\widetilde{M}_{H,z}$"]
# colors = ["r","b","g","c"]

# fig = plt.figure(figsize=(14,8))
# plt.bar(xlabel,[cc2,cc1,cc3,cc4],color=colors)
# plt.ylabel("Correlation coefficient [-]")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_I_MR.png")
# plt.close(fig)



# # # print(correlation_coef(IA_LPF_shift,MR_LPF_shift))
# # # print(correlation_coef(MR,RtAeroMxa))
# # # print(correlation_coef(RtAeroMxa_LPF_shift,Ux_LPF_shift))
# # # print(correlation_coef(RtAeroFxh_LPF_shift,Ux_LPF_shift))
# # # print(correlation_coef(RtAeroFxh,RtAeroMxa))


# # # Time_OF_shift = Time_OF[:-Time_shift_idx]
# # # fig,(ax1,ax2,ax3,ax4,ax5) = plt.subplots(5,1,figsize=(16,24),sharex=True)
# # # ax1.plot(Time_sampling_shift,IA_LPF_shift,"-r")
# # # ax1.set_title("Asymmetry Parameter [$m^4/s$]")
# # # ax1.grid()
# # # ax2.plot(Time_OF_shift,MR_shift,"-b")
# # # ax2.plot(Time_sampling_shift,MR_LPF_shift,"-r")
# # # ax2.set_title("out-of-plane bending moment [kN-m]")
# # # ax2.grid()
# # # ax3.plot(Time_OF_shift,RtAeroMxa_shift,"-b")
# # # ax3.plot(Time_sampling_shift,RtAeroMxa_LPF_shift,"-r")
# # # ax3.set_title("Torque [kN-m]")
# # # ax3.grid()
# # # ax4.plot(Time_sampling_shift,Ux_LPF_shift,"-r")
# # # ax4.set_title("Rotor averaged wind speed [$m/s$]")
# # # ax4.grid()
# # # ax5.plot(Time_OF_shift,RtAeroFxh_shift,"-b")
# # # ax5.plot(Time_sampling_shift,RtAeroFxh_LPF_shift,"-r")
# # # ax5.set_title("Thrust [kN]")
# # # ax5.grid()
# # # fig.supxlabel("Time [s]")
# # # plt.tight_layout()
# # # plt.savefig("../../Thesis/Figures/Time_correlations.png")
# # # plt.close(fig)


# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FR = np.sqrt(np.add(np.square(RtAeroFys*((L1+L2)/L2)),np.square(RtAeroFzs*((L1+L2)/L2))))
# FTheta = np.degrees(np.arctan2(RtAeroFzs*((L1+L2)/L2),RtAeroFys*((L1+L2)/L2)))
# FTheta = theta_360(FTheta)

# FR2 = np.sqrt(np.add(np.square(RtAeroFys*((L1+L2)/L2)*30),np.square(RtAeroFzs*((L1+L2)/L2)*30)))
# FTheta2 = np.degrees(np.arctan2(RtAeroFzs*((L1+L2)/L2)*30,RtAeroFys*((L1+L2)/L2)*30))
# FTheta2 = theta_360(FTheta2)

# MR = np.sqrt(np.add(np.square(RtAeroMys/L2),np.square(RtAeroMzs/L2)))
# MTheta = np.degrees(np.arctan2(RtAeroMys/L2,-RtAeroMzs/L2))
# MTheta = theta_360(MTheta)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# # WR = 1079
# # nu = np.linspace(0,1.0,11)
# # Mean = []
# # Var = []
# # for n in nu:
# #     FBz_hat = RtAeroMys/L2 + (RtAeroFzs - n*WR)*((L1+L2)/L2)
# #     Mean_FBR_hat = np.sqrt(np.add(np.square(np.mean(FBy)),np.square(np.mean(FBz_hat))))
# #     Mean.append(Mean_FBR_hat)
# #     Var_FBR_hat = np.var(FBy) + np.var(FBz)
# #     Var.append(Var_FBR_hat)

# # fig = plt.figure(figsize=(14,8))
# # plt.plot(nu,Mean,"-o")
# # plt.xlabel("Percentage of weight $\\nu$",fontsize=22)
# # plt.ylabel("Magnitude of average \nmain bearing force vector [kN]",fontsize=22)
# # plt.grid()
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/Mean_FBR.png")
# # plt.close(fig)

# # fig = plt.figure(figsize=(14,8))
# # plt.plot(nu,Var,"-o")
# # plt.xlabel("Percentage of weight $\\nu$",fontsize=22)
# # plt.ylabel("Variance of main bearing force vector [kN]",fontsize=22)
# # plt.grid()
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/Var_FBR.png")
# # plt.close(fig)


# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
# Theta_FBR = np.degrees(np.arctan2(FBz,FBy))
# Theta_FBR = theta_360(Theta_FBR)

# # df = io.fast_output_file.FASTOutputFile("../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/Steady_Rigid_blades_shear_0.098/NREL_5MW_Main.out").toDataFrame()
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
# # label="{}-{}s".format(round(Time_OF[2050],2),round(Time_OF[2500],2))
# # ax.plot(np.radians(MTheta[2050:2500]),MR[2050:2500]/np.max(MR),"-r",label="ABL-turbine\n"+label)
# # label="{}-{}s".format(round(Time_OF[4550],2),round(Time_OF[5000],2))
# # ax.plot(np.radians(MTheta[4550:5000]),MR[4550:5000]/np.max(MR),"-b",label="ABL-turbine\n"+label)
# # label="{}-{}s".format(round(Time_OF[49450],2),round(Time_OF[50000],2))
# # ax.plot(np.radians(MTheta[49450:50000]),MR[49450:50000]/np.max(MR),"-g",label="ABL-turbine\n"+label)
# # label="{}-{}s".format(round(Time_OF_SS[80],2),round(Time_OF_SS[505],2))
# # ax.plot(np.radians(MTheta_SS[80:505]),(8*MR_SS[80:505])/np.max(MR),"-k",label="Steady shear, $\\times 8$\n"+label)
# # ax.plot(np.radians(MTheta_SS[80:505]),(MR_SS[80:505])/np.max(MR),"-k",label="Steady shear\n"+label)
# # ax.legend()
# # ax.set_title("Normalized out-of-plane bending\nmoment trajectory [-]")
# # plt.savefig("../../Thesis/Figures/MR_trajectory.png")
# # plt.close(fig)



# # #Role of thrust
# # in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

# # df = Dataset(in_dir+"Dataset.nc")

# # Time = np.array(df.variables["Time_OF"])

# # Time_start_idx = np.searchsorted(Time,200)
# # Time = Time[Time_start_idx:]

# # OF_vars = df.groups["OpenFAST_Variables"]

# # LSShftFxa = np.array(OF_vars.variables["LSShftFxa"][Time_start_idx:])
# # LSShftFys = np.array(OF_vars.variables["LSShftFys"][Time_start_idx:])
# # LSShftFzs = np.array(OF_vars.variables["LSShftFzs"][Time_start_idx:])
# # LSSTipMys = np.array(OF_vars.variables["LSSTipMys"][Time_start_idx:])
# # LSSTipMzs = np.array(OF_vars.variables["LSSTipMzs"][Time_start_idx:])

# # L1 = 1.912; L2 = 2.09

# # FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
# # FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

# # FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)
# # FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# # FB = np.sqrt(np.square(LSShftFxa)+np.square(FBy)+np.square(FBz))

# # cc = round(correlation_coef(FB,FBR),2)
# # cc2 = round(correlation_coef(FB,LSShftFxa),2)
# # fig = plt.figure(figsize=(14,8))
# # plt.plot(Time,FB,"-k",label="$F_B$")
# # plt.plot(Time,FBR,"-b",label="$F_{B,\perp}$")
# # plt.plot(Time,LSShftFxa,"-r",label="$F_{B,x}$")
# # plt.xlabel("Time [s]")
# # plt.ylabel("Main Bearing forces [kN]")
# # plt.title("Correlation coefficient ($F_B$, $F_{B,\perp}$) = "+"{}".format(cc)+"\nCorrelation coefficient ($F_B$, $F_{B,x}$) = "+"{}".format(cc2))
# # plt.grid()
# # plt.legend()
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/cc_Bearing_force_components.png")
# # plt.close(fig)



# # Fa_Fr = np.true_divide(LSShftFxa,FBR)


# # E = [0.22,0.3]

# # for e in E:

# #     alpha = np.arctan(e/1.5)

# #     Pr = []
# #     XFr = []
# #     YFa = []
# #     e_it = []
# #     XFr_2 = []
# #     YFa_2 = []
# #     for it in np.arange(0,len(Time)):

# #         if Fa_Fr[it] <= e:
# #             e_it.append(0)
# #             X = 1; Y = 0.45*(1/np.tan(alpha))
# #         elif Fa_Fr[it] > e:
# #             e_it.append(1)
# #             X = 0.67; Y = 0.67*(1/np.tan(alpha))
# #             XFr_2.append(X*FBR[it]); YFa_2.append(Y*LSShftFxa[it])

# #         XFr.append(X*FBR[it]); YFa.append(Y*LSShftFxa[it])
# #         Pr.append(X*FBR[it]+Y*LSShftFxa[it])
    
# #     cc = round(correlation_coef(Pr,FBR),2)
# #     cc2 = round(correlation_coef(Pr,LSShftFxa),2)

# #     fig = plt.figure(figsize=(14,8))
# #     plt.plot(Time,XFr,"-b",label="$XF_{B,\perp}$")
# #     plt.plot(Time,YFa,"-r",label="$YF_{B,x}$")
# #     plt.xlabel("Time [s]")
# #     plt.ylabel("Modified Bearing force components [kN]")
# #     plt.title("Correlation coefficient ($P_r$, $F_{B,\perp}$) = "+"{}".format(cc)+"\nCorrelation coefficient ($P_r$, $F_{B,x}$) = "+"{}".format(cc2))
# #     plt.legend()
# #     plt.grid()
# #     plt.tight_layout()
# #     plt.savefig("../../Thesis/Figures/cc_XFr_Y_Fa_{}.png".format(e))
# #     plt.close(fig)




# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(projection='polar')
# ax.plot(np.radians(Theta_FBR[:2500]),FBR[:2500],"-r")
# ax.plot(np.radians(FTheta[:2500]),FR[:2500],"-g")
# ax.plot(np.radians(FTheta2[:2500]),FR2[:2500],"--g")
# ax.plot(np.radians(MTheta[:2500]),MR[:2500],"-b")

# ax.plot(np.radians(Theta_FBR[2500]),FBR[2500],"or",label="$\\tilde{F}_{B\perp}$")
# ax.arrow(0, 0, np.radians(Theta_FBR[2500]), FBR[2500], length_includes_head=True,color="r")
# ax.plot(np.radians(FTheta[2500]),FR[2500],"og",label="$\\tilde{F}_{H\perp}(L/L_2)$")
# ax.arrow(0, 0, np.radians(FTheta[2500]), FR[2500], length_includes_head=True,color="g")
# ax.plot(np.radians(FTheta2[2500]),FR2[2500],"--g",label="$\\tilde{F}_{H\perp}(L/L_2)X30$")
# ax.arrow(0, 0, np.radians(FTheta2[2500]), FR2[2500], length_includes_head=True,color="g")
# ax.plot(np.radians(MTheta[2500]),MR[2500],"ob",label="$\\tilde{M}_{H\perp,mod}/L_2$")
# ax.arrow(0, 0, np.radians(MTheta[2500]), MR[2500], length_includes_head=True,color="b")
# ax.legend()
# ax.set_title("Vector trajectories [kN]\n25s period")
# plt.savefig("../../Thesis/Figures/FR_MR_FBR_trajectory.png")
# plt.close(fig)



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

# # z_SS = [7.5, 22.5, 37.5, 52.5, 82.5, 97.5, 112.5, 127.5, 157.5]
# # Wind1 = np.array(df["Wind1VelX_[m/s]"][0]); Wind2 = np.array(df["Wind2VelX_[m/s]"][0]); Wind3 = np.array(df["Wind3VelX_[m/s]"][0])
# # Wind4 = np.array(df["Wind4VelX_[m/s]"][0]); Wind5 = np.array(df["Wind5VelX_[m/s]"][0]); Wind6 = np.array(df["Wind6VelX_[m/s]"][0])
# # Wind7 = np.array(df["Wind7VelX_[m/s]"][0]); Wind8 = np.array(df["Wind8VelX_[m/s]"][0]); Wind9 = np.array(df["Wind9VelX_[m/s]"][0])
# # Wind_profile = [Wind1,Wind2,Wind3,Wind4,Wind5,Wind6,Wind7,Wind8,Wind9]

# # in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"

# # df_pre = Dataset(in_dir+"abl_statistics70000.nc")

# # Time_pre = np.array(df_pre.variables["time"])
# # Time_start_idx = np.searchsorted(Time_pre,38200)

# # mean_profiles = df_pre.groups["mean_profiles"]


# # z = np.array(mean_profiles.variables["h"])

# # u = np.average(mean_profiles.variables["u"][Time_start_idx:],axis=0)
# # v = np.average(mean_profiles.variables["v"][Time_start_idx:],axis=0)

# # twist = coriolis_twist(u,v)
# # hvelmag = []
# # for i in np.arange(0,len(twist)):
# #     hvelmag.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
# # hvelmag = np.array(hvelmag)

# # z_end = np.searchsorted(z,z_SS[-1])

# # fig = plt.figure()
# # plt.plot(Wind_profile,z_SS,"-or",label="Steady shear inflow")
# # plt.plot(hvelmag[:z_end+1],z[:z_end+1],"-*b",label="Precursor MCBL")
# # plt.xlabel("$\langle u_{x'} \\rangle _{T=38200-39200s}$ [m/s]")
# # plt.ylabel("Height from surface [m]")
# # plt.legend()
# # plt.grid()
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/Wind_profile_precursor_steady_shear.png")
# # plt.close(fig)



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

# # fig = plt.figure(figsize=(14,8))
# # plt.plot(Time_SS,MR_SS)
# # plt.xlabel("Time [s]")
# # plt.ylabel("out-of-plane bending moment [kN-m]")
# # plt.title("Steady shear ($\\alpha=0.098$) inflow")
# # plt.grid()
# # plt.tight_layout()
# # plt.savefig("../../Thesis/Figures/Steady_shear_MR.png")
# # plt.close(fig)

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



in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["Time_OF"])
dt = Time_OF[1] - Time_OF[0]

Time_start = 200
Time_start_idx = np.searchsorted(Time_OF,Time_start)

Time_OF = Time_OF[Time_start_idx:]

OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])/1000
RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])/1000

RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
#OOPBM
OOPBM = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

LPF_1_OOPBM = np.array(hard_filter(OOPBM,0.3,dt,"lowpass"))
BPF_OOPBM = np.array(hard_filter(OOPBM,[0.3,0.9],dt,"bandpass"))
HPF_OOPBM = np.array(hard_filter(OOPBM,[1.5,40],dt,"bandpass"))

# plt.rcParams['font.size'] = 16
# times = np.arange(200,1300,100)
# for i in np.arange(0,len(times)-1):
#     idx1 = np.searchsorted(Time_OF,times[i])
#     idx2 = np.searchsorted(Time_OF,times[i+1])
#     fig = plt.figure(figsize=(14,8))
#     plt.plot(Time_OF[idx1:idx2],OOPBM[idx1:idx2],"-k",label="Total $M_{H,\perp}$")
#     plt.plot(Time_OF[idx1:idx2],LPF_1_OOPBM[idx1:idx2],"-g",label="LPF 0.3Hz $M_{H,\perp}$")
#     plt.plot(Time_OF[idx1:idx2],BPF_OOPBM[idx1:idx2],"-r",label="BPF 0.3-0.9Hz $M_{H,\perp}$")
#     plt.plot(Time_OF[idx1:idx2],HPF_OOPBM[idx1:idx2]-1000,"-b",label="HPF 1.5-40Hz $M_{H,\perp}$\noffset: -1000kN-m")
#     plt.grid()
#     plt.xlabel("Time [s]",fontsize=22)
#     plt.ylabel("Magnitude aerodynamic OOPBM vector [kN-m]",fontsize=22)
#     plt.legend(fontsize=20)
#     plt.tight_layout()
#     plt.savefig("../../Thesis/Figures/{}_{}.png".format(times[i],times[i+1]))
#     plt.close(fig)


#BPF calc
dBPF_OOPBM = np.array(dt_calc(BPF_OOPBM,dt))
zero_crossings_index_BPF_OOPBM = np.where(np.diff(np.sign(dBPF_OOPBM)))[0]
Env_BPF_OOPBM = []
Env_Times = []
for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM),2):
    idx = zero_crossings_index_BPF_OOPBM[i]
    Env_BPF_OOPBM.append(BPF_OOPBM[idx]); Env_Times.append(Time_OF[idx])

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,BPF_OOPBM,"-r",label="BPF $M_{H,\perp}$")
# plt.plot(Env_Times,Env_BPF_OOPBM,"-b",label="Env BPF $M_{H,\perp}$")
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("Magnitude aerodynamic OOPBM [kN-m]",fontsize=22)
# plt.grid()
# plt.legend(fontsize=20)
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Env_BPF_OOPBM.png")
# plt.close(fig)


# fig = plt.figure(figsize=(14,8))
# idx1 = np.searchsorted(Time_OF,200); idx2 = np.searchsorted(Time_OF,300)
# plt.plot(Time_OF[idx1:idx2],BPF_OOPBM[idx1:idx2],"-r",label="BPF $M_{H,\perp}$")
# idx1 = np.searchsorted(Env_Times,200); idx2 = np.searchsorted(Env_Times,300)
# plt.plot(Env_Times[idx1:idx2],Env_BPF_OOPBM[idx1:idx2],"--b",label="Env BPF $M_{H,\perp}$")
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("Magnitude aerodynamic OOPBM [kN-m]",fontsize=22)
# plt.grid()
# plt.legend(fontsize=20)
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Env_BPF_OOPBM_200_300.png")
# plt.close(fig)

f = interpolate.interp1d(Env_Times,Env_BPF_OOPBM)
Env_Times = np.arange(Env_Times[0],Env_Times[-1],0.39)
Env_BPF_OOPBM = f(Env_Times)
dt_BPF = Env_Times[1] - Env_Times[0]
Env_BPF_OOPBM = hard_filter(Env_BPF_OOPBM,0.3,dt_BPF,"lowpass")

Env_BPF_OOPBM = np.array(Env_BPF_OOPBM); Env_Times = np.array(Env_Times)

f_BPF = interpolate.interp1d(Env_Times,Env_BPF_OOPBM)
f_LPF = interpolate.interp1d(Time_OF,LPF_1_OOPBM)

plt.rcParams['font.size'] = 16
#HPF calc
cc_BPF = []
cc_LPF = []
abs_HPF_OOPBM = abs(HPF_OOPBM)
windows = [3,4,5,6,7,8,9,10,11,12]
colors=["g","r","b"]
offset=0
ix=0
fig,ax = plt.subplots(figsize=(14,8))
ax2=ax.twinx()
# plt.plot(Time_OF,abs_HPF_OOPBM,"-k",label="Absolute HPF $M_{H,\perp}$")
ax.plot(Time_OF,LPF_1_OOPBM,"-g",label="LPF $\widetilde{M}_{H,\perp}$")
ax.plot(Env_Times,Env_BPF_OOPBM,"-r",label="Env $\widetilde{M}_{H,\perp,BPF}$")
for window in windows:
    window_idx = np.searchsorted(Time_OF,Time_OF[0]+window)
    if (window_idx % 2) != 0:
        window_idx+=1
    Times_avg_HPF = Time_OF[int(window_idx/2):-int(window_idx/2)]
    avg_HPF_OOPBM = []
    for i in np.arange(0,len(Time_OF)-window_idx):
        avg_HPF_OOPBM.append(np.average(abs_HPF_OOPBM[i:i+window_idx]))
    
    idx_min = np.searchsorted(Times_avg_HPF,np.min(Env_Times)); idx_max = np.searchsorted(Times_avg_HPF,np.max(Env_Times))
    Env_BPF_OOPBM_interp = f_BPF(Times_avg_HPF)
    cc_BPF.append(round(correlation_coef(Env_BPF_OOPBM_interp,avg_HPF_OOPBM[idx_min:idx_max]),3))

    LPF_1_OOPBM_interp = f_LPF(Times_avg_HPF)
    cc_LPF.append(round(correlation_coef(LPF_1_OOPBM_interp,avg_HPF_OOPBM[idx_min:idx_max]),3))

    # if window == 3 or window == 6 or window == 9:
    #     label="Effectively filtered HPF $M_{H,\perp}$"+"\nWindow = {}s, offset: {}kN-m".format(window,offset)
    #     plt.plot(Times_avg_HPF,np.add(avg_HPF_OOPBM,offset),color=colors[ix],label=label)
    #     offset+=200

    #     ix+=1
    if window == 9:
        ax2.set_ylabel("Effectively filtered HPF $\widetilde{M}_{H,\perp}$"+"\nWindow = {}s [kN-m]".format(window),fontsize=22)
        ax2.yaxis.label.set_color('blue')
        ax2.plot(Times_avg_HPF,avg_HPF_OOPBM,"-b")


fig.supxlabel("Time [s]",fontsize=22)
ax.set_ylabel("Magnitude aerodynamic OOPBM [kN-m]",fontsize=22)
ax.grid()
ax.legend(fontsize=18)
plt.tight_layout()
plt.savefig("../../Thesis/Figures/LPF_BPF_HPF_OOPBM.png")
plt.close(fig)



# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(windows,cc_BPF,"-r",label="Env BPF $M_{H,\perp}$ cc eff(|$M_{H,\perp,HPF}$|)")
# plt.plot(windows,cc_LPF,"-b",label="LPF $M_{H,\perp}$ cc eff(|$M_{H,\perp,HPF}$|)")
# idx = np.argmax(cc_BPF)
# plt.axvline(x=windows[idx],linestyle="--",color="r")
# idx = np.argmax(cc_LPF)
# plt.axvline(x=windows[idx],linestyle="--",color="b")
# plt.xlabel("Window size [s]",fontsize=22)
# plt.ylabel("Correlation coefficient [-]",fontsize=22)
# plt.legend(fontsize=20)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_avg_MH_HPF.png")
# plt.close(fig)

# abs_HPF_OOPBM = abs(HPF_OOPBM)
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


# plt.rcParams['font.size'] = 16
# plt.figure(figsize=(14,8))
# plt.plot(thresholds,perc_overlap_LPF_HPF,"-or",label="LPF overlaps HPF")
# plt.plot(thresholds,perc_overlap_HPF_LPF,"-ob",label="HPF overlaps LPF")
# plt.plot(thresholds,perc_overlap_BPF_HPF,"-og",label="BPF overlaps HPF")
# plt.plot(thresholds,perc_overlap_HPF_BPF,"-oc",label="HPF overlaps BPF")
# plt.plot(thresholds,perc_overlap_LPF_BPF,"-om",label="LPF overlaps BPF")
# plt.plot(thresholds,perc_overlap_BPF_LPF,"-oy",label="BPF overlaps LPF")
# plt.xlabel("Fraction $t$ of standard deviation $\sigma(x)$ above mean $\\bar{x}$\nThreshold: $T = \\bar{x}+t \sigma(x)$ [kN-m]",fontsize=22)
# plt.ylabel("Percentage overlap [%]",fontsize=22)
# plt.legend(fontsize=18)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Threshold_percentage_overlap.png")
# plt.close(fig)


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)
# Time_steps = np.arange(Time_start_idx,len(Time_OF)-1)
# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

# OOPBM = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))
# MTheta = np.degrees(np.arctan2(RtAeroMys,-RtAeroMzs))
# MTheta = theta_360(MTheta)



# Time_sampling = np.array(df_OF.variables["Time_sampling"])
# dt_sampling = Time_sampling[1] - Time_sampling[0]
# Time_start = 200
# Time_sampling_start_idx = np.searchsorted(Time_sampling,Time_start)

# Time_sampling = Time_sampling[Time_sampling_start_idx:]

# # Rotor_avg_vars = df_OF.groups["Rotor_Avg_Variables"]
# # Rotor_avg_vars_63 = Rotor_avg_vars.groups["5.5"]

# # IA = np.array(Rotor_avg_vars_63.variables["IA"][Time_sampling_start_idx:])
# # LPF_IA = hard_filter(IA,0.3,dt_sampling,"lowpass")


# df_WT = Dataset(in_dir+"WTG01b.nc")

# WT = df_WT.groups["WTG01"]


# Rotor_coordinates = [np.float64(WT.variables["xyz"][0,0,0]),np.float64(WT.variables["xyz"][0,0,1]),np.float64(WT.variables["xyz"][0,0,2])]


# df = Dataset(in_dir+"WTG01a.nc")
# uvelB1 = np.array(df.variables["uvel"][:,1:301])
# vvelB1 = np.array(df.variables["vvel"][:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1 = np.add(np.cos(np.radians(29.29))*uvelB1, np.sin(np.radians(29.29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][:,301:601])
# vvelB2 = np.array(df.variables["vvel"][:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2 = np.add(np.cos(np.radians(29.29))*uvelB2, np.sin(np.radians(29.29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][:,601:901])
# vvelB3 = np.array(df.variables["vvel"][:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3 = np.add(np.cos(np.radians(29.29))*uvelB3, np.sin(np.radians(29.29))*vvelB3)


# R = np.linspace(0,63,300)
# dr = R[1] - R[0]

# Iy = []
# Iz = []
# ix=0
# with Pool() as pool:
#     for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,Time_steps):
#         Iy.append(Iy_it); Iz.append(Iz_it)
#         print(ix)
#         ix+=1
# Iy = np.array(Iy); Iz = -np.array(Iz)
# I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
# ITheta = np.degrees(np.arctan2(Iy,-Iz))
# ITheta = theta_360(ITheta)

# LPF_I = hard_filter(I,0.3,dt,"lowpass")
# BPF_I = hard_filter(I,[0.3,0.9],dt,"bandpass")
# HPF_I = hard_filter(I,[1.5,40],dt,"bandpass")

# LPF_OOPBM = hard_filter(OOPBM,0.3,dt,"lowpass")
# BPF_OOPBM = hard_filter(OOPBM,[0.3,0.9],dt,"bandpass")
# HPF_OOPBM = hard_filter(OOPBM,[1.5,40],dt,"bandpass")

# cc1 = correlation_coef(I,OOPBM[:-1])
# cc2 = correlation_coef(ITheta,MTheta[:-1])
# cc3 = correlation_coef(Iy,RtAeroMys[:-1])
# cc4 = correlation_coef(Iz,RtAeroMzs[:-1])
# cc5 = correlation_coef(LPF_I,LPF_OOPBM[:-1])
# cc6 = correlation_coef(BPF_I,BPF_OOPBM[:-1])
# cc7 = correlation_coef(HPF_I,HPF_OOPBM[:-1])

# xlabel = ["$|\mathbf{I}_B|$ cc $|\widetilde{\mathbf{M}}_{H,\perp,mod}|$", "$\\theta (I_B)$ cc $\\theta (\widetilde{\mathbf{M}}_{H,\perp,mod})$",
#           "$I_{B,y}$ cc $\widetilde{M}_{H,y}$", "$I_{B,z}$ cc $\widetilde{M}_{H,z}$", "$|\mathbf{I}_B|_{LPF}$ cc $|\widetilde{\mathbf{M}}_{H,\perp,mod}|_{LPF}$",
#           "$|\mathbf{I}_B|_{BPF}$ cc $|\widetilde{\mathbf{M}}_{H,\perp,mod}|_{BPF}$", "$|\mathbf{I}_B|_{HPF}$ cc $|\widetilde{\mathbf{M}}_{H,\perp,mod}|_{HPF}$"]
# colors = ["r","b","g","c","m","y","k"]

# fig = plt.figure(figsize=(22,8))
# plt.bar(xlabel,[cc1,cc2,cc3,cc4,cc5,cc6,cc7],color=colors)
# plt.ylabel("Correlation coefficient [-]")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_IB_MR.png")
# plt.close(fig)

# plt.rcParams['font.size'] = 16
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,OOPBM,"-r")
# ax.set_ylabel("Aerodynamic out-of-plane bending moment magnitude [kN-m]")
# ax.yaxis.label.set_color('red') 
# ax2=ax.twinx()
# ax2.plot(Time_OF[:-1],I,"-b")
# ax2.set_ylabel("Blade asymmetry [$m^3/s$]")
# ax2.yaxis.label.set_color('blue') 
# ax2.grid()
# fig.supxlabel("Time [s]")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_IB_OOPBM.png")
# plt.close(fig)

# OOPBM_norm = OOPBM/np.mean(OOPBM)
# IB_norm = I/np.mean(I)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(OOPBM_norm,dt,"OOPBM")
# plt.loglog(frq,PSD,"-r",label="Normalised $M_{H,\perp}$")
# frq,PSD = temporal_spectra(IB_norm,dt,"IB")
# plt.loglog(frq,PSD,"-b",label="Normalised $I_B$")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/spectra_IB_OOPBM.png")
# plt.close(fig)


# f = interpolate.interp1d(Time_OF[:-1],LPF_I)
# LPF_I_interp = f(Time_sampling)

# time_shift_idx = np.searchsorted(Time_sampling,Time_sampling[0]+4.6)

# Time_sampling_shift = Time_sampling[:-time_shift_idx]

# #LPF_I_interp_shift = LPF_I_interp[time_shift_idx:]

# #LPF_IA_shift = LPF_IA[:-time_shift_idx]

# cc = round(correlation_coef(LPF_I_interp,LPF_IA),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_sampling,LPF_IA,"-r")
# ax.set_ylabel("LPF (0.3Hz) Asymmetry parameter [$m^4/s$]")
# ax.yaxis.label.set_color('red') 
# ax2=ax.twinx()
# ax2.plot(Time_sampling,LPF_I_interp,"-b")
# ax2.set_ylabel("LPF (0.3Hz) Blade asymmetry [$m^3/s$]")
# ax2.yaxis.label.set_color('blue') 
# ax2.grid()
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficient = {}".format(cc))
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_LPF_IB_I.png")
# plt.close(fig)


# Iy_75 = []
# Iz_75 = []
# ix=0
# with Pool() as pool:
#     for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc_75,Time_steps):
#         Iy_75.append(Iy_it); Iz_75.append(Iz_it)
#         print(ix)
#         ix+=1
# Iy_75 = np.array(Iy_75); Iz_75 = -np.array(Iz_75)
# I_75 = np.sqrt(np.add(np.square(Iy_75),np.square(Iz_75)))

# cc = round(correlation_coef(I,I_75),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF[:-1],I,"-b")
# ax.set_ylabel("Blade asymmetry [$m^3/s$]")
# ax.yaxis.label.set_color('blue')
# ax2=ax.twinx()
# ax2.plot(Time_OF[:-1],I_75,"-r")
# ax2.set_ylabel("Blade asymmetry $I_{B,75\%}$ [$m^3/s$]")
# ax2.yaxis.label.set_color('red')
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficient = {}".format(cc))
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_IB_IB_75.png")
# plt.close(fig)


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)
# Time_steps = np.arange(Time_start_idx,len(Time_OF)-1)
# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)


# dHPF_FBR = dt_calc(HPF_OOPBM,dt)

# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]



# df = Dataset(in_dir+"WTG01a.nc")

# t = np.array(df.variables["time"][:])
# Tstart_idx = np.searchsorted(t,t[0]+200)
# uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
# vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1 = np.add(np.cos(np.radians(29.29))*uvelB1, np.sin(np.radians(29.29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
# vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2 = np.add(np.cos(np.radians(29.29))*uvelB2, np.sin(np.radians(29.29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
# vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3 = np.add(np.cos(np.radians(29.29))*uvelB3, np.sin(np.radians(29.29))*vvelB3)


# #HPF FBR calc
# dF_mag_HPF = []
# dUxB1 = []
# dUxB2 = []
# dUxB3 = []
# Time_mag_HPF = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#     it_1 = zero_crossings_index_HPF_FBR[i]
#     it_2 = zero_crossings_index_HPF_FBR[i+1]

#     Time_mag_HPF.append(Time_OF[it_1])

#     dF_mag_HPF.append(abs(HPF_OOPBM[it_2] - HPF_OOPBM[it_1]))
#     dUxB1.append(hvelB1[it_2,225]-hvelB1[it_1,225])
#     dUxB2.append(hvelB2[it_2,225]-hvelB2[it_1,225])
#     dUxB3.append(hvelB3[it_2,225]-hvelB3[it_1,225])



# max_dUx = []
# for i in np.arange(0,len(dUxB1)):
#     max_dUx.append(np.max([abs(dUxB1[i]),abs(dUxB2[i]),abs(dUxB3[i])]))






# cc1 = round(correlation_coef(max_dUx,dF_mag_HPF),2)
# cc2 = round(correlation_coef(HPF_I,HPF_OOPBM[:-1]),2)
# fig,ax1 = plt.subplots(figsize=(14,8),sharex=True)
# ax1.plot(Time_mag_HPF,dF_mag_HPF,"-or",markersize=3)
# ax1.set_ylabel("$|d \widetilde{M}_{H,\perp,HPF}|_{HPF}$ [kN-m]")
# ax2=ax1.twinx()
# ax2.plot(Time_mag_HPF,max_dUx,"-ob",markersize=3)
# ax2.set_ylabel("$max[|du_{x',i}|]$ B1,B2,B3 [m/s]")
# ax1.grid()
# fig.supxlabel("Time [s]")
# fig.suptitle("correlation coefficient $d \widetilde{M}_{H,\perp,HPF}$, $max|du_{x'}|$ ="+"{}".format(cc1)+"\ncorrelation coefficient $\widetilde{M}_{H,\perp,HPF}$, $I_{B,HPF}$ = "+"{}".format(cc2))
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_dux_dMR_HPF.png")
# plt.close(fig)


# Time_mag_HPF_interp = np.linspace(Time_mag_HPF[0],Time_mag_HPF[-1],len(Time_OF))
# f = interpolate.interp1d(Time_mag_HPF,dF_mag_HPF)
# dF_mag_HPF_interp = f(Time_mag_HPF_interp)
# f = interpolate.interp1d(Time_mag_HPF,max_dUx)
# max_dUx_interp = f(Time_mag_HPF_interp)

# idx = np.searchsorted(Time_mag_HPF_interp,Time_mag_HPF_interp[0]+10)
# cc = []
# for it in np.arange(0,len(Time_mag_HPF_interp)-idx):
#     cc.append(correlation_coef(dF_mag_HPF_interp[it:it+idx],max_dUx_interp[it:it+idx]))

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)-1],cc,"-k")
# plt.ylabel("Local correlation T=10s")
# plt.title("correlation ($d \widetilde{M}_{H,\perp,HPF}$, $max|du_{x'}|$)")
# plt.grid()
# plt.xlabel("Time [s]")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/local_correlation_dux_dMR_HPF.png")
# plt.close(fig)


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"
# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_sampling = np.array(df_OF.variables["Time_sampling"])
# dt_sample = Time_sampling[1] - Time_sampling[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_sampling,Time_sampling[0]+Time_start)

# Time_sampling = Time_sampling[Time_start_idx:]


# Rotor_gradients = df_OF.groups["Rotor_Gradients"]

# drUx = np.array(Rotor_gradients.variables["drUx"][Time_start_idx:])

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
# BPF_OOPBM = np.array(hard_filter(OOPBM,[0.3,0.9],dt,"bandpass"))

# #BPF calc
# dBPF_OOPBM = np.array(dt_calc(BPF_OOPBM,dt))
# zero_crossings_index_BPF_OOPBM = np.where(np.diff(np.sign(dBPF_OOPBM)))[0]
# Env_BPF_OOPBM = []
# Env_Times = []
# for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM),2):
#     idx = zero_crossings_index_BPF_OOPBM[i]
#     Env_BPF_OOPBM.append(BPF_OOPBM[idx]); Env_Times.append(Time_OF[idx])



# Time_interp = np.arange(Env_Times[0],Env_Times[-1],0.39)
# dt = Time_interp[1]-Time_interp[0]
# f = interpolate.interp1d(Env_Times,Env_BPF_OOPBM)

# Env_BPF_OOPBM_interp = f(Time_interp)

# f = interpolate.interp1d(Time_sampling,drUx)
# dUx_interp = f(Time_interp)

# Env_BPF_OOPBM_interp = hard_filter(Env_BPF_OOPBM_interp,0.3,dt,"lowpass")
# dUx_interp = hard_filter(dUx_interp,0.3,dt,"lowpass")

# Time_shift_idx = np.searchsorted(Time_interp,Time_interp[0]+4.6)
# Time_interp_shift = Time_interp[:-Time_shift_idx]
# Env_BPF_OOPBM_interp_shift = Env_BPF_OOPBM_interp[Time_shift_idx:]
# dUx_interp_shift = dUx_interp[:-Time_shift_idx]

# plt.rcParams['font.size'] = 16
# cc = round(correlation_coef(dUx_interp_shift,Env_BPF_OOPBM_interp_shift),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_interp_shift,Env_BPF_OOPBM_interp_shift,"-r")
# ax.set_ylabel("Envelope $\widetilde{M}_{H,\perp,BPF}$")
# ax.yaxis.label.set_color('red')
# ax2=ax.twinx()
# ax2.plot(Time_interp_shift,dUx_interp_shift,"-b")
# ax2.set_ylabel("$du_{x'}/dr$")
# ax2.yaxis.label.set_color('blue')
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficient = {}".format(cc))
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_drux_Env_MR_HPF.png")
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

# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])/1000
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])/1000

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])/1000
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])/1000

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)


# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# #FBR
# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# WR = 1079*((L1+L2)/L2)

# LPF_FBR = hard_filter(FBR,0.3,dt,"lowpass")
# BPF_FBR = hard_filter(FBR,[0.3,0.9],dt,"bandpass")
# HPF_FBR = hard_filter(FBR,[1.5,40],dt,"bandpass")

# dLPF_FBR = dt_calc(LPF_FBR,dt)
# zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]


# #jump analysis
# # frq,PSD = temporal_spectra(LPF_FBR,dt,"LPF")
# # fig = plt.figure()
# # plt.loglog(frq,PSD)
# # plt.ylabel("LPF")

# # frq,PSD = temporal_spectra(BPF_FBR,dt,"BPF")
# # fig = plt.figure()
# # plt.loglog(frq,PSD)
# # plt.ylabel("BPF")

# # frq,PSD = temporal_spectra(HPF_FBR,dt,"HPF")
# # fig = plt.figure()
# # plt.loglog(frq,PSD)
# # plt.ylabel("HPF")

# dFBR = dt_calc(FBR,dt)
# zero_crossings_index_FBR = np.where(np.diff(np.sign(dFBR)))[0]

# dLPF_FBR = dt_calc(LPF_FBR,dt)
# zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]

# dBPF_FBR = dt_calc(BPF_FBR,dt)
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

# dHPF_FBR = dt_calc(HPF_FBR,dt)
# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]


# #FBR calc
# dF_mag = []
# dt_mag = []
# Time_mag = []
# FBR_mag = []
# for i in np.arange(0,len(zero_crossings_index_FBR)-1):

#     it_1 = zero_crossings_index_FBR[i]
#     it_2 = zero_crossings_index_FBR[i+1]

#     dt = Time_OF[it_2]-Time_OF[it_1]

#     Time_mag.append(Time_OF[it_1])
#     FBR_mag.append(FBR[it_1])

#     dF_mag.append(abs(FBR[it_2] - FBR[it_1])/WR)
#     dt_mag.append(dt)

# print("Total")
# PFBR,XFBR = probability_dist(dF_mag,5)
# Pdt_FBR,Xdt_FBR = probability_dist(dt_mag,5)

# moments(dF_mag,XFBR,PFBR)
# moments(dt_mag,Xdt_FBR,Pdt_FBR)

# threshold = np.mean(dF_mag)+2*np.std(dF_mag)
# dt_threshold = []
# dF_threshold = []

# for i in np.arange(0,len(dF_mag)):

#     if dF_mag[i] >= threshold:
#         dF_threshold.append(dF_mag[i])
#         dt_threshold.append(dt_mag[i])

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.scatter(dt_threshold,dF_threshold)
# plt.xlabel("$\Delta t$ [s]")
# plt.ylabel("$|\Delta \widetilde{F}_{B,\perp}|/W_R(L/L_2)$ [-]")
# plt.title("{} $2 \sigma$ jumps in 1000s".format(len(dF_threshold)))
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dF_dt_threshold.png")
# plt.close(fig)


# #LPF FBR calc
# dF_LPF = []
# dt_LPF = []
# Time_mag_LPF = []
# FBR_LPF_mag = []
# for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-1):

#     it_1 = zero_crossings_index_LPF_FBR[i]
#     it_2 = zero_crossings_index_LPF_FBR[i+1]

#     dt = Time_OF[it_2]-Time_OF[it_1]

#     Time_mag_LPF.append(Time_OF[it_1])
#     FBR_LPF_mag.append(LPF_FBR[it_1])

#     dF_LPF.append(abs(LPF_FBR[it_2] - LPF_FBR[it_1])/WR)
#     dt_LPF.append(dt)

# #print(np.max(dt_LPF),np.min(dt_LPF))

# # fig = plt.figure()
# # plt.plot(Time_OF,LPF_FBR)
# # plt.scatter(Time_mag_LPF,FBR_LPF_mag)
# # plt.show()

# print("LPF")
# PF_LPF,XF_LPF = probability_dist(dF_LPF,5)
# Pdt_LPF,Xdt_LPF = probability_dist(dt_LPF,5)

# moments(dF_LPF,XF_LPF,PF_LPF)
# moments(dt_LPF,Xdt_LPF,Pdt_LPF)

# #BPF FBR calc
# dF_BPF = []
# dt_BPF = []
# Time_mag_BPF = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

#     it_1 = zero_crossings_index_BPF_FBR[i]
#     it_2 = zero_crossings_index_BPF_FBR[i+1]

#     Time_mag_BPF.append(Time_OF[it_1])

#     dF_BPF.append(abs(BPF_FBR[it_2] - BPF_FBR[it_1])/WR)
#     dt_BPF.append(Time_OF[it_2]-Time_OF[it_1])

# #print(np.max(dt_BPF),np.min(dt_BPF))
# print("BPF")
# PF_BPF,XF_BPF = probability_dist(dF_BPF,5)
# Pdt_BPF,Xdt_BPF = probability_dist(dt_BPF,5)

# moments(dF_BPF,XF_BPF,PF_BPF)
# moments(dt_BPF,Xdt_BPF,Pdt_BPF)

# #HPF FBR calc
# dF_HPF = []
# dt_HPF = []
# Time_mag_HPF = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#     it_1 = zero_crossings_index_HPF_FBR[i]
#     it_2 = zero_crossings_index_HPF_FBR[i+1]

#     Time_mag_HPF.append(Time_OF[it_1])

#     dF_HPF.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1])/WR)
#     dt_HPF.append(Time_OF[it_2]-Time_OF[it_1])

# #print(np.max(dt_HPF),np.min(dt_HPF))
# print("HPF")
# PF_HPF,XF_HPF = probability_dist(dF_HPF,5)
# Pdt_HPF,Xdt_HPF = probability_dist(dt_HPF,5)

# moments(dF_HPF,XF_HPF,PF_HPF)
# moments(dt_HPF,Xdt_HPF,Pdt_HPF)



# plt.rcParams['font.size'] = 16
# fig,ax = plt.subplots(figsize=(14,5))
# ax.scatter(Time_mag,dF_mag,s=5)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$|\Delta \widetilde{F}_{B,\perp}|/W_R(L/L_2)$ [-]")
# ax2=ax.twiny()
# ax2.plot(PFBR,XFBR,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dF_total.png")
# plt.close(fig)


# fig,ax = plt.subplots(figsize=(14,5))
# ax.scatter(Time_mag,dt_mag,s=5)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$\Delta t$ [s]")
# ax2=ax.twiny()
# ax2.plot(Pdt_FBR,Xdt_FBR,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dt_total.png")
# plt.close(fig)



# plt.rcParams['font.size'] = 16
# fig,ax = plt.subplots(figsize=(14,5))
# ax.scatter(Time_mag_LPF,dF_LPF,s=5)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$|\Delta \widetilde{F}_{B,\perp,LPF}|/W_R(L/L_2)$ [-]")
# ax2=ax.twiny()
# ax2.plot(PF_LPF,XF_LPF,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dF_LPF.png")
# plt.close(fig)


# fig,ax = plt.subplots(figsize=(14,5))
# ax.scatter(Time_mag_LPF,dt_LPF,s=5)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$\Delta t_{LPF}$ [s]")
# ax2=ax.twiny()
# ax2.plot(Pdt_LPF,Xdt_LPF,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dt_LPF.png")
# plt.close(fig)

# fig,ax = plt.subplots(figsize=(14,5))
# ax.scatter(Time_mag_BPF,dF_BPF,s=2)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$|\Delta \widetilde{F}_{B,\perp,BPF}|/W_R(L/L_2)$ [-]")
# ax2=ax.twiny()
# ax2.plot(PF_BPF,XF_BPF,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dF_BPF.png")
# plt.close(fig)

# fig,ax = plt.subplots(figsize=(14,5))
# ax.scatter(Time_mag_BPF,dt_BPF,s=2)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$\Delta t_{BPF}$ [s]")
# ax2=ax.twiny()
# ax2.plot(Pdt_BPF,Xdt_BPF,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dt_BPF.png")
# plt.close(fig)


# fig,ax = plt.subplots(figsize=(14,5))
# ax.scatter(Time_mag_HPF,dF_HPF,s=1)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$|\Delta \widetilde{F}_{B,\perp,HPF}|/W_R(L/L_2)$ [-]")
# ax2=ax.twiny()
# ax2.plot(PF_HPF,XF_HPF,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dF_HPF.png")
# plt.close(fig)

# fig,ax = plt.subplots(figsize=(14,5))
# ax.scatter(Time_mag_HPF,dt_HPF,s=1)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$\Delta t_{HPF}$ [s]")
# ax2=ax.twiny()
# ax2.plot(Pdt_HPF,Xdt_HPF,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dt_HPF.png")
# plt.close(fig)


# dt = Time_OF[1] - Time_OF[0]

# LPF_2_FBR = hard_filter(FBR,0.9,dt,"lowpass")


# dLPF_2_FBR = dt_calc(LPF_2_FBR,dt)
# zero_crossings_index_LPF_2_FBR = np.where(np.diff(np.sign(dLPF_2_FBR)))[0]

# #LPF+BPF FBR calc
# LPF_BPF_mag = []
# FBR_mag = []
# dF_LPF_BPF = []
# dt_LPF_BPF = []
# Time_mag_LPF_BPF = []
# dFBR_mag = []
# for i in np.arange(0,len(zero_crossings_index_LPF_2_FBR)-1):

#     it_1 = zero_crossings_index_LPF_2_FBR[i]
#     it_2 = zero_crossings_index_LPF_2_FBR[i+1]

#     Time_mag_LPF_BPF.append(Time_OF[it_1])
#     FBR_mag.append(FBR[it_1])
#     LPF_BPF_mag.append(LPF_2_FBR[it_1])

#     dF_LPF_BPF.append(abs(LPF_2_FBR[it_2] - LPF_2_FBR[it_1]))
#     dFBR_mag.append(abs(FBR[it_2]-FBR[it_1]))


# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,FBR,"-k",label="Total")
# plt.scatter(Time_mag_LPF_BPF,FBR_mag,color="y",label="peaks in $F_{B,\perp,LPF+BPF}$")
# plt.plot(Time_OF,LPF_2_FBR,"-r",label="LPF+BPF (LPF 0.9Hz)")
# plt.scatter(Time_mag_LPF_BPF,LPF_BPF_mag,color="b",label="peaks in $F_{B,\perp,LPF+BPF}$")
# plt.xlabel("Time [s]")
# plt.ylabel("Main bearing radial force magnitude [kN]")
# plt.legend()
# plt.grid()
# plt.xlim([200,220])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/FBR_LPF_BPF.png")
# plt.close(fig)

# plt.rcParams['font.size'] = 16
# diff_FB = np.subtract(dFBR_mag,dF_LPF_BPF)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.scatter(Time_mag_LPF_BPF,diff_FB,s=2)
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$(|\Delta F_{B,\perp}|-|\Delta F_{B,\perp,LPF+BPF}|)/|\Delta F_{B,\perp,LPF+BPF}|$ [-]")
# ax2=ax.twiny()
# P,X = probability_dist(diff_FB,5)
# moments(diff_FB,X,P)
# ax2.plot(P,X,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dF_total_dF_LPF_BPF.png")
# plt.close(fig)



# #LPF+BPF FBR calc
# LPF_BPF_mag = []
# FBR_mag = []
# dF_LPF_BPF = []
# dt_LPF_BPF = []
# Time_mag_LPF_BPF = []
# dFBR_mag = []
# for i in np.arange(0,len(zero_crossings_index_FBR)-1):

#     it_1 = zero_crossings_index_FBR[i]
#     it_2 = zero_crossings_index_FBR[i+1]

#     Time_mag_LPF_BPF.append(Time_OF[it_1])
#     FBR_mag.append(FBR[it_1])
#     LPF_BPF_mag.append(LPF_2_FBR[it_1])

#     dF_LPF_BPF.append(abs(LPF_2_FBR[it_2] - LPF_2_FBR[it_1]))
#     dFBR_mag.append(abs(FBR[it_2]-FBR[it_1]))


# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,FBR,"-k",label="Total")
# plt.scatter(Time_mag_LPF_BPF,FBR_mag,color="y",label="peaks in $F_{B,\perp,LPF+BPF}$")
# plt.plot(Time_OF,LPF_2_FBR,"-r",label="LPF+BPF (LPF 0.9Hz)")
# plt.scatter(Time_mag_LPF_BPF,LPF_BPF_mag,color="b",label="peaks in $F_{B,\perp,LPF+BPF}$")
# plt.xlabel("Time [s]")
# plt.ylabel("Main bearing radial force magnitude [kN]")
# plt.legend()
# plt.grid()
# plt.xlim([200,220])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/FBR_LPF_BPF.png")
# plt.close(fig)

# diff_FB = np.subtract(dFBR_mag,dF_LPF_BPF)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.scatter(Time_mag_LPF_BPF,diff_FB,s=2,label="$X_{max}=-7.79kN$, $\mu_1=62.25kN$, $\mu_2=153.7kN$, $\mu_3=3.26$, $\mu_4=17.37$")
# ax.legend()
# ax.set_xlabel("Time [s]")
# ax.set_ylabel("$|\Delta F_{B,\perp}|-|\Delta F_{B,\perp,LPF+BPF}|$ [kN]")
# ax2=ax.twiny()
# P,X = probability_dist(diff_FB,5)
# moments(diff_FB,X,P)
# ax2.plot(P,X,"-k")
# ax.set_title("Probability [-]")
# ax.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/dF_total_dF_LPF_BPF.png")
# plt.close(fig)


# #Section 7

# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_E_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroFxh = np.array(OpenFAST_vars.variables["RtAeroFxh"][Time_start_idx:])/1000
# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)
# RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
# RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

# L1 = 1.912; L2 = 2.09
# FR = np.sqrt(np.add(np.square(RtAeroFys),np.square(RtAeroFzs)))*((L1+L2)/L2)
# FTheta = np.degrees(np.arctan2(RtAeroFzs,RtAeroFys))
# FTheta = theta_360(FTheta)

# MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))*(1/L2)
# MTheta = np.degrees(np.arctan2(RtAeroMys,-RtAeroMzs))
# MTheta = theta_360(MTheta)


# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
# FBTheta = np.degrees(np.arctan2(FBz,FBy))
# FBTheta = theta_360(FBTheta)

# print("FHperp")
# print(np.mean(FR)); print(np.std(FR)); print(correlation_coef(FR,FBR))
# print("MHperp")
# print(np.mean(MR)); print(np.std(MR)); print(correlation_coef(MR,FBR))
# print("FHperp Theta")
# print(np.mean(FTheta)); print(np.std(FTheta)); print(correlation_coef(FTheta,FBTheta))
# print("MHperp Theta")
# print(np.mean(MTheta)); print(np.std(MTheta)); print(correlation_coef(MTheta,FBTheta))
# print("FBy")
# print(np.mean(FBy)); print(np.std(FBy)); print(correlation_coef(FBy,FBz))
# print("FBz")
# print(np.mean(FBz)); print(np.std(FBz)); print(correlation_coef(FBz,FBy))
# print("-1/L2 MHz")
# print(np.mean(-FBMy)); print(np.std(-FBMy)); print(correlation_coef(-FBMy,FBy))
# print("L/L2 FHy")
# print(np.mean(-FBFy)); print(np.std(-FBFy)); print(correlation_coef(-FBFy,FBy))
# print("1/L2 MHy")
# print(np.mean(-FBMz)); print(np.std(-FBMz)); print(correlation_coef(-FBMz,FBz))
# print("L/L2 FHz")
# print(np.mean(-FBFz)); print(np.std(-FBFz)); print(correlation_coef(-FBFz,FBz))



# def coordinate_rotation(it):

#     xo = np.array(WT_E.variables["xyz"][it,1:301,0])
#     yo = np.array(WT_E.variables["xyz"][it,1:301,1])
#     zs_E = np.array(WT_E.variables["xyz"][it,1:301,2])


#     x_trans = xo - Rotor_coordinates[0]
#     y_trans = yo - Rotor_coordinates[1]

#     phi = np.radians(-29.29)
#     xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
#     ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))

#     xs_E = xs + Rotor_coordinates[0]
#     ys_E = ys + Rotor_coordinates[1]

#     xo = np.array(WT_R.variables["xyz"][it,1:301,0])
#     yo = np.array(WT_R.variables["xyz"][it,1:301,1])
#     zs_R = np.array(WT_R.variables["xyz"][it,1:301,2])

#     x_trans = xo - Rotor_coordinates[0]
#     y_trans = yo - Rotor_coordinates[1]

#     phi = np.radians(-29.29)
#     xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
#     ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))

#     xs_R = xs + Rotor_coordinates[0]
#     ys_R = ys + Rotor_coordinates[1]

#     return xs_E,ys_E,zs_E, xs_R,ys_R,zs_R


# def tranform_blade_fixed_frame(Y_pri,Z_pri,it):

#     Y = ((Y_pri-Rotor_coordinates[1])*np.cos(Azimuth[it]) - (Z_pri-Rotor_coordinates[2])*np.sin(Azimuth[it])) + Rotor_coordinates[1]
#     Z = ((Y_pri-Rotor_coordinates[1])*np.sin(Azimuth[it]) + (Z_pri-Rotor_coordinates[2])*np.cos(Azimuth[it])) + Rotor_coordinates[2]

#     return Y,Z



# def rotating_frame_coordinates(it):

#     xco_E,yco_E,zco_E, xco_R,yco_R,zco_R = coordinate_rotation(it)

#     yE_fixed,zE_fixed = tranform_blade_fixed_frame(yco_E,zco_E,it)
#     yR_fixed,zR_fixed = tranform_blade_fixed_frame(yco_R,zco_R,it)

#     return xco_E,yE_fixed,zE_fixed, xco_R, yR_fixed, zR_fixed





# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_E_CRPM/76000/"

# df = Dataset(in_dir+"Dataset.nc")
# OF_vars = df.groups["OpenFAST_Variables"]
# Azimuth = np.array(OF_vars.variables["Azimuth"])

# Azimuth = 360 - Azimuth[1:]
# Azimuth = np.radians(Azimuth)

# RtAeroFxh = np.array(OF_vars["RtAeroFxh"])

# df_E = Dataset(in_dir+"WTG01b.nc")

# WT_E = df_E.groups["WTG01"]

# Time = np.array(WT_E.variables["time"])
# dt = Time[1] - Time[0]

# Tstart_idx = np.searchsorted(Time,Time[0]+200)
# Time_steps = np.arange(Tstart_idx,len(Time))


# Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"
# df_R = Dataset(in_dir+"WTG01b.nc")

# WT_R = df_R.groups["WTG01"]


# ix = 0
# xE = []; yE = []; zE = []
# xR = []; yR = []; zR = []



# with Pool() as pool:
#     for xEit,yEit,zEit,xRit,yRit,zRit in pool.imap(rotating_frame_coordinates,Time_steps):
#         xE.append(xEit); yE.append(yEit); zE.append(zEit)
#         xR.append(xRit); yR.append(yRit); zR.append(zRit)
#         #print(np.shape(x))
#         print(ix)
#         ix+=1

# xD = np.subtract(xE,xR); yD = np.subtract(yE,yR); zD = np.subtract(zE,zR)

# #7-4
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(xD[:,75],dt,"xH")
# plt.loglog(frq,PSD,"-g",label="15.75m")
# frq,PSD = temporal_spectra(xD[:,225],dt,"xH")
# plt.loglog(frq,PSD,"-r",label="47.25m")
# frq,PSD = temporal_spectra(xD[:,-1],dt,"xH")
# plt.loglog(frq,PSD,"-b",label="63m")
# plt.legend()
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Displacement $x_{\widehat{H}}$ relative to rigid blade [$m^2$]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Spectra_displacement_x.png")
# plt.close(fig)


# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(yD[:,75],dt,"yH")
# plt.loglog(frq,PSD,"-g",label="15.75m")
# frq,PSD = temporal_spectra(yD[:,225],dt,"yH")
# plt.loglog(frq,PSD,"-r",label="47.25m")
# frq,PSD = temporal_spectra(yD[:,-1],dt,"yH")
# plt.loglog(frq,PSD,"-b",label="63m")
# plt.legend()
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Displacement $y_{\widehat{H}}$ relative to rigid blade [$m^2$]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Spectra_displacement_y.png")
# plt.close(fig)


# xE = np.mean(xE,axis=0); yE = np.mean(yE,axis=0); zE = np.mean(zE,axis=0)
# xR = np.mean(xR,axis=0); yR = np.mean(yR,axis=0); zR = np.mean(zR,axis=0)


# ix = 0
# xE = []; yE = []; zE = []
# xR = []; yR = []; zR = []
# with Pool() as pool:
#     for xEit,yEit,zEit,xRit,yRit,zRit in pool.imap(rotating_frame_coordinates,Time_steps):
#         xE.append(xEit[-1]); yE.append(yEit[-1]); zE.append(zEit[-1])
#         xR.append(xRit[-1]); yR.append(yRit[-1]); zR.append(zRit[-1])
#         #print(np.shape(x))
#         print(ix)
#         ix+=1

# xD = np.subtract(xE,xR); yD = np.subtract(yE,yR); zD = np.subtract(zE,zR)

# print(np.mean(xD)); print(np.std(xD)); print(np.min(xD)); print(np.max(xD))
# print(np.mean(yD)); print(np.std(yD)); print(np.min(yD)); print(np.max(yD))
# print(np.mean(zD)); print(np.std(zD)); print(np.min(zD)); print(np.max(zD))

# LPF_xD = hard_filter(xD,0.3,dt,"lowpass"); LPF_RtAeroFxh = hard_filter(RtAeroFxh,0.3,dt,"lowpass")
# print(correlation_coef(LPF_xD,LPF_RtAeroFxh[Tstart_idx:-1]))

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time[Tstart_idx:],xD)
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("Tip displacement $x_{\widehat{H}}$\n relative to rigid blade [m]",fontsize=22)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Tip_displacement_xD.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time[Tstart_idx:],yD)
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("Tip displacement $y_{\widehat{H}}$\nrelative to rigid blade [m]",fontsize=22)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Tip_displacement_yD.png")
# plt.close(fig)


# # BMassDen = [6.7893500E+02, 6.7893500E+02, 7.7336300E+02, 7.4055000E+02, 7.4004200E+02, 5.9249600E+02, 4.5027500E+02, 4.2405400E+02, 4.0063800E+02, 3.8206200E+02, 3.9965500E+02,
# #             4.2632100E+02, 4.1682000E+02, 4.0618600E+02, 3.8142000E+02, 3.5282200E+02, 3.4947700E+02, 3.4653800E+02, 3.3933300E+02, 3.3000400E+02, 3.2199000E+02, 3.1382000E+02,
# #             2.9473400E+02, 2.8712000E+02, 2.6334300E+02, 2.5320700E+02, 2.4166600E+02, 2.2063800E+02, 2.0029300E+02, 1.7940400E+02, 1.6509400E+02, 1.5441100E+02, 1.3893500E+02, 
# #             1.2955500E+02, 1.0726400E+02, 9.8776000E+01, 9.0248000E+01, 8.3001000E+01, 7.2906000E+01, 6.8772000E+01, 6.6264000E+01, 5.9340000E+01, 5.5914000E+01, 5.2484000E+01,
# #             4.9114000E+01, 4.5818000E+01, 4.1669000E+01, 1.1453000E+01, 1.0319000E+01]

# R = np.linspace(91.5,153,len(BMassDen))

# # plt.rcParams['font.size'] = 30
# # fig,(ax1,ax2) = plt.subplots(1,2,figsize=(32,16),sharey=True)
# # ax1.plot(xE,zE,"-b",label="Deformed")
# # ax1.plot(xR,zR,"-r",label="Rigid")
# # ax1.set_xlabel("x' coordinate rotating frame of reference [m]")
# # ax1.set_title("Mean deflected blade position")
# # ax1.grid()
# # ax1.legend()
# # ax1.set_xlim([Rotor_coordinates[0]-5,Rotor_coordinates[0]+10]); ax1.set_ylim([80,160])

# # ax2.plot(BMassDen,R)
# # ax2.set_xlabel("Blade mass density [kg/m]")
# # ax2.grid()


# # fig.supylabel("z coordinate rotating frame of reference [m]")

# # plt.savefig("../../Thesis/Figures/Blade_mean_position.png")
# # plt.close(fig)

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(xD,dt,"xD")
# plt.loglog(frq,PSD)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Displacement $x_{\widehat{H}}$ relative to rigid blade [m]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Blade_x_displacement.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(yD,dt,"yD")
# plt.loglog(frq,PSD)
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD - Displacement $y_{\widehat{H}}$ relative to rigid blade [m]")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Blade_y_displacement.png")
# plt.close(fig)


# plt.rcParams['font.size'] = 16
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_E_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroFxh = np.array(OpenFAST_vars.variables["RtAeroFxh"][Time_start_idx:])/1000
# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)
# RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000

# RtAeroMxa = np.array(OpenFAST_vars.variables["RtAeroMxh"][Time_start_idx:])/1000
# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
# RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

# MR = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

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
# print(correlation_coef(Ux_LPF_shift,RtAeroFxh_LPF_shift))
# print(correlation_coef(Ux_LPF_shift,RtAeroMxa_LPF_shift))
# print(correlation_coef(RtAeroMxa,RtAeroFxh))



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
# plt.savefig("../../Thesis/Figures/Time_correlations_deformable.png")
# plt.close(fig)


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]
# Time_steps = np.arange(0,len(Time_OF)-1)

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
# RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

# MR_R = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

# LSShftFys = np.array(OpenFAST_vars.variables["LSShftFys"][Time_start_idx:])
# LSShftFzs = np.array(OpenFAST_vars.variables["LSShftFzs"][Time_start_idx:])

# FR_RR = np.sqrt(np.add(np.square(LSShftFys),np.square(LSShftFzs)))

# LSSTipMys = np.array(OpenFAST_vars.variables["LSSTipMys"][Time_start_idx:])
# LSSTipMzs = np.array(OpenFAST_vars.variables["LSSTipMzs"][Time_start_idx:])

# MR_RR = np.sqrt(np.add(np.square(LSSTipMys),np.square(LSSTipMzs)))

# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
# FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# FBR_RR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_E_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]
# Time_steps = np.arange(0,len(Time_OF)-1)

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)
# RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

# MR_E = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

# LSShftFys = np.array(OpenFAST_vars.variables["LSShftFys"][Time_start_idx:])
# LSShftFzs = np.array(OpenFAST_vars.variables["LSShftFzs"][Time_start_idx:])

# FR_EE = np.sqrt(np.add(np.square(LSShftFys),np.square(LSShftFzs)))

# LSSTipMys = np.array(OpenFAST_vars.variables["LSSTipMys"][Time_start_idx:])
# LSSTipMzs = np.array(OpenFAST_vars.variables["LSSTipMzs"][Time_start_idx:])

# MR_EE = np.sqrt(np.add(np.square(LSSTipMys),np.square(LSSTipMzs)))

# LPF_1_OOPBM = np.array(hard_filter(MR_EE,0.3,dt,"lowpass"))
# BPF_OOPBM = np.array(hard_filter(MR_EE,[0.3,0.9],dt,"bandpass"))
# HPF_OOPBM = np.array(hard_filter(MR_EE,[1.5,40],dt,"bandpass"))

# plt.rcParams['font.size'] = 16
# times = np.arange(200,1300,100)
# for i in np.arange(0,len(times)-1):
#     idx1 = np.searchsorted(Time_OF,times[i])
#     idx2 = np.searchsorted(Time_OF,times[i+1])
#     fig = plt.figure(figsize=(14,8))
#     plt.plot(Time_OF[idx1:idx2],MR_EE[idx1:idx2],"-k",label="Total $M_{H,\perp}$")
#     plt.plot(Time_OF[idx1:idx2],LPF_1_OOPBM[idx1:idx2],"-g",label="LPF 0.3Hz $M_{H,\perp}$")
#     plt.plot(Time_OF[idx1:idx2],BPF_OOPBM[idx1:idx2],"-r",label="BPF 0.3-0.9Hz $M_{H,\perp}$")
#     plt.plot(Time_OF[idx1:idx2],HPF_OOPBM[idx1:idx2]-1000,"-b",label="HPF 1.5-40Hz $M_{H,\perp}$\noffset: -1000kN-m")
#     plt.grid()
#     plt.xlabel("Time [s]",fontsize=22)
#     plt.ylabel("Magnitude aerodynamic OOPBM vector [kN-m]",fontsize=22)
#     plt.legend(fontsize=20)
#     plt.tight_layout()
#     plt.savefig("../../Thesis/Figures/Deform_{}_{}.png".format(times[i],times[i+1]))
#     plt.close(fig)


# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
# FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# FBR_EE = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(FBR_RR,dt,"FBR_RR")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# frq,PSD = temporal_spectra(FBR_EE,dt,"FBR_EE")
# plt.loglog(frq,PSD,"-b",label="Deform")
# plt.xlabel("Frequency [Hz]",fontsize=22)
# plt.ylabel("PSD - main bearing radial force magnitude\n$|F_{B,\perp}|$ [$kN^2$]",fontsize=22)
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/OOPMB_R_E_comparison.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(MR_RR,dt,"MR_RR")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# frq,PSD = temporal_spectra(MR_EE,dt,"MR_EE")
# plt.loglog(frq,PSD,"-b",label="Deform")
# plt.xlabel("Frequency [Hz]",fontsize=22)
# plt.ylabel("PSD - OOPBM magnitude $|M_{H,\perp}|$ [$kN-m^2$]",fontsize=22)
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/OOPBM_R_E_comparison.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(FR_EE,dt,"FR_EE")
# plt.loglog(frq,PSD,"-b",label="Deform")
# frq,PSD = temporal_spectra(FR_RR,dt,"FR_RR")
# plt.loglog(frq,PSD,"-r",label="Rigid")
# plt.xlabel("Frequency [Hz]",fontsize=22)
# plt.ylabel("PSD - Out-of-plane hub force magnitude\n$|F_{H,\perp}|$ [$kN^2$]",fontsize=22)
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/OOPF_R_E_comparison.png")
# plt.close(fig)


# df = Dataset(in_dir+"WTG01a.nc")


# uvelB1 = np.array(df.variables["uvel"][Time_start_idx:,1:301])
# vvelB1 = np.array(df.variables["vvel"][Time_start_idx:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1 = np.add(np.cos(np.radians(29.29))*uvelB1, np.sin(np.radians(29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][Time_start_idx:,301:601])
# vvelB2 = np.array(df.variables["vvel"][Time_start_idx:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2 = np.add(np.cos(np.radians(29.29))*uvelB2, np.sin(np.radians(29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][Time_start_idx:,601:901])
# vvelB3 = np.array(df.variables["vvel"][Time_start_idx:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3 = np.add(np.cos(np.radians(29.29))*uvelB3, np.sin(np.radians(29))*vvelB3)


# df = Dataset(in_dir+"WTG01b.nc")

# WT = df.groups["WTG01"]
# Rotor_coordinates = [np.float64(WT.variables["xyz"][0,0,0]),np.float64(WT.variables["xyz"][0,0,1]),np.float64(WT.variables["xyz"][0,0,2])]


# R = np.linspace(0,63,300)
# dr = R[1] - R[0]
# IyE = []
# IzE = []
# ix=0
# with Pool() as pool:
#     for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,Time_steps):
#         IyE.append(Iy_it); IzE.append(Iz_it)
#         print(ix)
#         ix+=1


# I = np.sqrt(np.add(np.square(IyE),np.square(IzE)))



# ccE = round(correlation_coef(MR[:-1],I),2)


# plt.rcParams['font.size'] = 16
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,MR,"-r")
# ax.set_ylabel("Out-of-plane bending moment deformable [kN-m]")
# ax.yaxis.label.set_color('red')  
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_OF[:-1],I,"-b")
# ax2.set_ylabel("Blade Asymmetry [$m^3/s$]")
# ax2.yaxis.label.set_color('blue')  
# ax2.grid()
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficient = {}".format(ccE))
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Blade_asymmetry_deformable.png")
# plt.close(fig)

# LPF_IB = hard_filter(I,0.3,dt,"lowpass")
# BPF_IB = hard_filter(I,[0.3,0.9],dt,"bandpass")
# HPF_IB = hard_filter(I,[1.5,40],dt,"bandpass")

# LPF_MR = hard_filter(MR,0.3,dt,"lowpass")
# BPF_MR = hard_filter(MR,[0.3,0.9],dt,"bandpass")
# HPF_MR = hard_filter(MR,[1.5,40],dt,"bandpass")

# print(correlation_coef(LPF_IB,LPF_MR[:-1]))
# print(correlation_coef(BPF_IB,BPF_MR[:-1]))
# print(correlation_coef(HPF_IB,HPF_MR[:-1]))

# fig,ax = plt.subplots(figsize=(14,8))
# frq,PSD = temporal_spectra(I,dt,"IB")
# ax.loglog(frq,PSD,"-b")
# ax.set_ylabel("Blade Asymmetry [$m^3/s$]")
# ax.yaxis.label.set_color('blue')  
# ax.grid()
# ax2=ax.twinx()
# frq,PSD = temporal_spectra(MR,dt,"MR")
# ax2.loglog(frq,PSD,"-r")
# ax2.set_ylabel("Out-of-plane bending moment deformable [kN-m]")
# ax2.yaxis.label.set_color('red')  
# ax2.grid()
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficient = {}".format(ccE))
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Spectra_Blade_asymmetry_deformable.png")
# plt.close(fig)



# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")


# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)
# Time_steps = np.arange(Time_start_idx,len(Time_OF)-1)
# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])/1000
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])/1000

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])/1000
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])/1000

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

# #Total radial bearing force FBR inc weight
# L1 = 1.912; L2 = 2.09

# #Total bearing force
# FBMy = RtAeroMzh/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# HPF_FBR = hard_filter(FBR,[1.5,40],dt,"bandpass")

# dHPF_FBR = dt_calc(HPF_FBR,dt)

# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]


# Time_sampling = np.array(df_OF.variables["Time_sampling"])

# Time_start = 200
# Time_sample_start_idx = np.searchsorted(Time_sampling,Time_start)
# Avg_rotor_vars = df_OF.groups["Rotor_Avg_Variables"]
# Avg_rotor_vars = Avg_rotor_vars.groups["5.5"]
# Ux_mean = np.average(Avg_rotor_vars.variables["Ux"][Time_sample_start_idx:])



# df = Dataset(in_dir+"WTG01a.nc")

# t = np.array(df.variables["time"][:])
# Tstart_idx = np.searchsorted(t,t[0]+200)
# uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
# vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1 = np.add(np.cos(np.radians(29.29))*uvelB1, np.sin(np.radians(29.29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
# vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2 = np.add(np.cos(np.radians(29.29))*uvelB2, np.sin(np.radians(29.29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
# vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3 = np.add(np.cos(np.radians(29.29))*uvelB3, np.sin(np.radians(29.29))*vvelB3)


# #HPF FBR calc
# dmag_HPF = []
# Time_mag_HPF = []
# dF_mag_HPF = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#     it_1 = zero_crossings_index_HPF_FBR[i]
#     it_2 = zero_crossings_index_HPF_FBR[i+1]

#     Time_mag_HPF.append(Time_OF[it_1])

#     dmag_HPF.append(HPF_FBR[it_1])

#     dF_mag_HPF.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1]))


# plt.rcParams['font.size'] = 16
# fig, ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF[:-1],hvelB1[:,225],"-b",label="B1")
# ax.plot(Time_OF[:-1],hvelB2[:,225],"-r",label="B2")
# ax.plot(Time_OF[:-1],hvelB3[:,225],"-g",label="B3")
# ax.axhline(y=Ux_mean,linestyle="--",color="k",label="Average rotor velocity")
# ax.legend(loc="lower left")
# ax2=ax.twinx()
# ax2.plot(Time_OF,HPF_FBR,"-k")
# sigma_events = 0
# for i in np.arange(0,len(dF_mag_HPF)):
#     if dF_mag_HPF[i] >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
#         ax2.plot(Time_mag_HPF[i],dmag_HPF[i],"ok")
#         sigma_events +=1
# ax.grid()
# ax.set_xlim([215,225])
# ax.set_ylabel("Streamwise velocity 75% span location [m/s]")
# ax2.set_ylabel("HPF (1.5-40Hz) main bearing radial force magnitude [kN]")
# fig.supxlabel("Time [s]")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/blade_velocity_HPF_FBR.png")
# plt.close(fig)


# def eddy_type(maxUx_1,maxUx_2):

#     if maxUx_1 <= Ux_mean and maxUx_2 <= Ux_mean:
#         return "LSS_only"
#     elif maxUx_1 <= Ux_mean and maxUx_2 >= Ux_mean:
#         return "LSS_HSR"
#     elif maxUx_2 <= Ux_mean and maxUx_1 >= Ux_mean:
#         return "LSS_HSR"
#     elif maxUx_1 > Ux_mean and maxUx_2 > Ux_mean:
#         return "HSR_only"


# LSS_array = []
# HSR_array = []
# LSS_HSR_array = []
# for perc in np.linspace(0.5,1.0,6):
#     LSS_only = 0
#     HSR_only = 0
#     LSS_HSR = 0
#     for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#         it_1 = zero_crossings_index_HPF_FBR[i]
#         it_2 = zero_crossings_index_HPF_FBR[i+1]


#         dF_FB_HPF_i = abs(HPF_FBR[it_2]-HPF_FBR[it_1])
#         if dF_FB_HPF_i >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):

#             dUxB1 = abs(hvelB1[it_2,225]-hvelB1[it_1,225])
#             dUxB2 = abs(hvelB2[it_2,225]-hvelB2[it_1,225])
#             dUxB3 = abs(hvelB3[it_2,225]-hvelB3[it_1,225])

#             max_dUx = np.max([dUxB1,dUxB2,dUxB3])

#             if dUxB1 >= perc*max_dUx:
#                 maxUx_1 = hvelB1[it_1,225]; maxUx_2 = hvelB1[it_2,225]
#                 eddyB1 = eddy_type(maxUx_1,maxUx_2)
#             else:
#                 eddyB1 = "None"
            
#             if dUxB2 >= perc*max_dUx:
#                 maxUx_1 = hvelB2[it_1,225]; maxUx_2 = hvelB2[it_2,225]
#                 eddyB2 = eddy_type(maxUx_1,maxUx_2)
#             else:
#                 eddyB2 = "None"

#             if dUxB3 >= perc*max_dUx:
#                 maxUx_1 = hvelB3[it_1,225]; maxUx_2 = hvelB3[it_2,225]
#                 eddyB3 = eddy_type(maxUx_1,maxUx_2)
#             else:
#                 eddyB3 = "None"
            

#             if eddyB1 == "LSS_only" and eddyB2 == "LSS_only" and eddyB3 == "LSS_only":
#                 LSS_only+=1
#             elif eddyB1 == "LSS_only" and eddyB2 == "None" and eddyB3 == "None":
#                 LSS_only+=1
#             elif eddyB2 == "LSS_only" and eddyB1 == "None" and eddyB3 == "None":
#                 LSS_only+=1
#             elif eddyB3 == "LSS_only" and eddyB2 == "None" and eddyB1 == "None":
#                 LSS_only+=1
#             elif eddyB1 == "HSR_only" and eddyB2 == "HSR_only" and eddyB3 == "HSR_only":
#                 HSR_only+=1
#             elif eddyB1 == "HSR_only" and eddyB2 == "None" and eddyB3 == "None":
#                 HSR_only+=1
#             elif eddyB2 == "HSR_only" and eddyB1 == "None" and eddyB3 == "None":
#                 HSR_only+=1
#             elif eddyB3 == "HSR_only" and eddyB2 == "None" and eddyB1 == "None":
#                 HSR_only+=1
#             else:
#                 LSS_HSR+=1
    
#     print(sigma_events)
#     print(LSS_only+HSR_only+LSS_HSR)

#     LSS_array.append(LSS_only/sigma_events); HSR_array.append(HSR_only/sigma_events); LSS_HSR_array.append(LSS_HSR/sigma_events)

#     if perc == 0.5 or perc == 1.0:
#         X = ["LSS only", "HSR only", "Both HSR and LSS"]
#         Y = [LSS_array[-1],HSR_array[-1],LSS_HSR_array[-1]]
#         c = ["b","r","g"]
#         fig = plt.figure(figsize=(14,8))
#         plt.bar(X,Y,color=c)
#         plt.ylabel("Fraction of $2 \sigma$ events in the HPF $F_{B\perp}$ [-]")
#         plt.title("Blade velocity threshold: {}\nTotal of {} $2\sigma$ events in 1000s".format(perc,sigma_events))
#         plt.grid()
#         plt.tight_layout()
#         plt.savefig("../../Thesis/Figures/Fraction_{}_eddy_type_velocity_threshold.png".format(perc))
#         plt.close(fig)


# print(LSS_array)
# print(HSR_array)
# print(LSS_HSR_array)

# X = np.linspace(0.5,1.0,6)
# fig = plt.figure(figsize=(14,8))
# plt.plot(X,LSS_array,"-b",label="LSS only")
# plt.plot(X,HSR_array,"-r",label="HSR only")
# plt.plot(X,LSS_HSR_array,"-g",label="LSS and HSR")
# plt.xlabel("Fraction threshold to include blade change in velocity [-]")
# plt.ylabel("Fraction of $2 \sigma$ events in the HPF $F_{B\perp}$ [-]")
# plt.title("Total of {} $2\sigma$ events in 1000s".format(sigma_events))
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Fraction_eddy_type_velocity_threshold.png")
# plt.close(fig)



# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# IHL_variables = df_OF.groups["IHL_Variables"]

# drUx_low = np.array(IHL_variables.variables["drUx_low"])
# drUx_high = np.array(IHL_variables.variables["drUx_high"])

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(drUx_low,5)
# plt.plot(X[1:],P[1:],"-b",label="LSS")
# P,X = probability_dist(drUx_high,5)
# plt.plot(X[1:],P[1:],"-r",label="HSR")
# plt.xlabel("Average gradient magnitude [1/s]")
# plt.ylabel("Probability [-]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/eddy_gradients.png")
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


# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])/1000
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])/1000

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)


# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])/1000
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])/1000

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

# #OOPBM
# OOPBM_R = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

# #FBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
# LPF_FBR = hard_filter(FBR,0.3,dt,"lowpass")
# BPF_FBR = hard_filter(FBR,[0.3,0.9],dt,"bandpass")
# HPF_FBR = hard_filter(FBR,[1.5,40],dt,"bandpass")

# print(np.mean(FBR)); print(np.std(FBR))
# print(np.mean(LPF_FBR)); print(np.std(LPF_FBR))
# print(np.mean(BPF_FBR)); print(np.std(BPF_FBR))
# print(np.mean(HPF_FBR)); print(np.std(HPF_FBR))

# Time_start_idx = np.searchsorted(Time_OF,700); Time_end_idx = np.searchsorted(Time_OF,800)

# print("T:700-800s")
# print(np.mean(FBR[Time_start_idx:Time_end_idx])); print(np.std(FBR[Time_start_idx:Time_end_idx]))
# print(np.mean(LPF_FBR[Time_start_idx:Time_end_idx])); print(np.std(LPF_FBR[Time_start_idx:Time_end_idx]))
# print(np.mean(BPF_FBR[Time_start_idx:Time_end_idx])); print(np.std(BPF_FBR[Time_start_idx:Time_end_idx]))
# print(np.mean(HPF_FBR[Time_start_idx:Time_end_idx])); print(np.std(HPF_FBR[Time_start_idx:Time_end_idx]))

# Time_start_idx = np.searchsorted(Time_OF,815); Time_end_idx = np.searchsorted(Time_OF,915)

# print("T:850-910s")
# print(np.mean(FBR[Time_start_idx:Time_end_idx])); print(np.std(FBR[Time_start_idx:Time_end_idx]))
# print(np.mean(LPF_FBR[Time_start_idx:Time_end_idx])); print(np.std(LPF_FBR[Time_start_idx:Time_end_idx]))
# print(np.mean(BPF_FBR[Time_start_idx:Time_end_idx])); print(np.std(BPF_FBR[Time_start_idx:Time_end_idx]))
# print(np.mean(HPF_FBR[Time_start_idx:Time_end_idx])); print(np.std(HPF_FBR[Time_start_idx:Time_end_idx]))

# Time_sampling = np.array(df_OF.variables["Time_sampling"])
# Time_start = 200
# Time_start_idx = np.searchsorted(Time_sampling,Time_start)
# Time_sampling = Time_sampling[Time_start_idx:]

# Rotor_avg_vars = df_OF.groups["Rotor_Avg_Variables"]

# Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]
# Ux = np.array(Rotor_avg_vars.variables["Ux"][Time_start_idx:])

# plt.rcParams['font.size'] = 16
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,FBR,"-r")
# ax.set_ylabel("Aerodynamic main bearing\nradial force magnitude [kN]",fontsize=22)
# ax.yaxis.label.set_color('red')
# ax2=ax.twinx()
# ax2.axhline(y=np.mean(Ux),color="b",linestyle="--",label="$\langle u_{x'} \\rangle_{A,T}$")
# ax2.legend(loc="upper right",fontsize=18)
# ax2.plot(Time_sampling,Ux,"-b")
# ax2.set_ylabel("Rotor averaged streamwise\nvelocity $\langle u_{x'} \\rangle_{A}$ [m/s]",fontsize=22)
# ax2.yaxis.label.set_color('blue')
# ax.grid()
# fig.supxlabel("Time [s]",fontsize=22)
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Ux_FBR.png")
# plt.close(fig)


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_sampling = np.array(df_OF.variables["Time_sampling"])
# Time_start = 200
# Time_start_idx = np.searchsorted(Time_sampling,Time_start)
# Time_sampling = Time_sampling[Time_start_idx:]

# IHL_vars = df_OF.groups["IHL_Variables"]

# Iy_high = np.array(IHL_vars.variables["Iy_high"][Time_start_idx:])
# Iy_low = np.array(IHL_vars.variables["Iy_low"][Time_start_idx:])
# Iy_int = np.array(IHL_vars.variables["Iy_int"][Time_start_idx:])

# print("Iy")
# print(np.mean(Iy_high)); print(np.mean(Iy_low)); print(np.mean(Iy_int))
# print(np.std(Iy_high)); print(np.std(Iy_low)); print(np.std(Iy_int))

# Iz_high = np.array(IHL_vars.variables["Iz_high"][Time_start_idx:])
# Iz_low = np.array(IHL_vars.variables["Iz_low"][Time_start_idx:])
# Iz_int = np.array(IHL_vars.variables["Iz_int"][Time_start_idx:])


# print("Iz")
# print(np.mean(Iz_high)); print(np.mean(Iz_low)); print(np.mean(Iz_int))
# print(np.std(Iz_high)); print(np.std(Iz_low)); print(np.std(Iz_int))

# Rotor_area = np.pi*63**2
# Area_high = np.array(IHL_vars.variables["Area_high"][Time_start_idx:])/Rotor_area
# Area_low = np.array(IHL_vars.variables["Area_low"][Time_start_idx:])/Rotor_area
# Area_int = np.array(IHL_vars.variables["Area_int"][Time_start_idx:])/Rotor_area

# Area = Area_high+Area_low+Area_int

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_sampling,Area_high,"-r",label="HSR")
# plt.plot(Time_sampling,Area_low,"-b",label="LSS")
# plt.plot(Time_sampling,Area_int,"-g",label="Intermediate")
# plt.plot(Time_sampling,Area,"--k",label="Total area")
# plt.xlabel("Time [s]",fontsize=22)
# plt.ylabel("Fraction of rotor disk [-]",fontsize=22)
# plt.legend(loc="upper left",fontsize=18)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Rotor_area_fraction.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(Iy_high,5)
# plt.plot(X,P,"-r",label="HSR")
# P,X = probability_dist(Iy_low,5)
# plt.plot(X,P,"-b",label="LSS")
# P,X = probability_dist(Iy_int,5)
# plt.plot(X,P,"-g",label="Intermediate")
# plt.yscale("log")
# plt.xlabel("Asymmetry around y axis [$m^4/s$]",fontsize=22)
# plt.ylabel("log() Probability [-]",fontsize=22)
# plt.legend(fontsize=18)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/PDF_Iy_IHL.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(Iz_high,5)
# plt.plot(X,P,"-r",label="HSR")
# P,X = probability_dist(Iz_low,5)
# plt.plot(X,P,"-b",label="LSS")
# P,X = probability_dist(Iz_int,5)
# plt.plot(X,P,"-g",label="Intermediate")
# plt.yscale("log")
# plt.xlabel("Asymmetry around z axis [$m^4/s$]",fontsize=22)
# plt.ylabel("log() Probability [-]",fontsize=22)
# plt.legend(fontsize=18)
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/PDF_Iz_IHL.png")
# plt.close(fig)


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/Steady_Rigid_blades_shear_0.0/"

# df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

# Time_OF = np.array(df["Time_[s]"])
# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)
# Time_OF = Time_OF[Time_start_idx:]

# LSShftFzs = np.array(df["LSShftFzs_[kN]"][Time_start_idx:])
# LSSTipMys = np.array(df["LSSTipMys_[kN-m]"][Time_start_idx:])
# LSSTipMys[np.abs(LSSTipMys)>0]=0

# fig = plt.figure()
# plt.plot(Time_OF,LSShftFzs)
# plt.xlabel("Time [s]")
# plt.ylabel("Hub force z component $F_{H,z}$")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/FHz_weight.png")
# plt.close(fig)

# fig = plt.figure()
# plt.plot(Time_OF,LSSTipMys)
# plt.xlabel("Time [s]")
# plt.ylabel("Hub moment y component $M_{H,y}$")
# plt.grid()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/MHy_weight.png")
# plt.close(fig)


# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df = Dataset(in_dir+"Dataset.nc")

# Time_sampling = np.array(df.variables["Time_sampling"])
# Time_start = 200
# Time_start_idx = np.searchsorted(Time_sampling,Time_start)
# Time_sampling = Time_sampling[Time_start_idx:]

# Rotor_vars = df.groups["Rotor_Avg_Variables"]
# Rotor_vars = Rotor_vars.groups["5.5"]
# Iy = np.array(Rotor_vars.variables["Iy"][Time_start_idx:])
# Iz = np.array(Rotor_vars.variables["Iz"][Time_start_idx:])

# I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))


# Ejection_vars = df.groups["Ejections_Variables"]

# ys = np.array(Ejection_vars.variables["ys"])

# #Thresholds = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.4]
# Thresholds = [1.4,2.0,2.5,3.0,3.5,4.0]

# plt.rcParams['font.size'] = 16
# fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
# fig.supxlabel("Time [s]")
# ax1.set_ylabel("Magnitude asymmetry\nvector [$m^4/s$]")
# ax1.grid()
# ax2.set_ylabel("Fraction of total asymmetry\n vector magnitude [-]")
# ax2.grid()
# for threshold in Thresholds:
#     T = np.zeros(len(Time_sampling))
#     group = Ejection_vars.groups["{}".format(threshold)]
#     Heights = np.array(group.variables["Height_ejection"][Time_start_idx:])
    
#     for it in np.arange(0,len(Time_sampling)):
#         Height = Heights[it]
#         for i in np.arange(0,len(Height)):
#             if isInside(ys[i],Height[i]) == True:
#                 T[it] = 1
    
#     perc = (np.sum(T)/len(Time_sampling))*100

#     print(threshold)
#     print(perc)

#     IyE = np.array(group.variables["Iy"][Time_start_idx:])
#     IzE = np.array(group.variables["Iz"][Time_start_idx:])

#     IE = np.sqrt(np.add(np.square(IyE),np.square(IzE)))

#     IE_frac = IE/I

#     ax1.plot(Time_sampling,IE,label="{}m/s".format(threshold))
#     ax2.plot(Time_sampling,IE_frac,label="{}m/s".format(threshold))

# ax1.plot(Time_sampling,I,"-k",label="$|I|$")
# ax1.legend()
# ax2.legend()
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Ejection_asymmetry_contribution.png")
# plt.close(fig)




# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)
# Time_steps = np.arange(Time_start_idx,len(Time_OF)-1)
# Time_OF = Time_OF[Time_start_idx:]


# Time_sampling = np.array(df_OF.variables["Time_sampling"])
# dt_sampling = Time_sampling[1] - Time_sampling[0]
# Time_start = 200
# Time_sampling_start_idx = np.searchsorted(Time_sampling,Time_start)

# Time_sampling = Time_sampling[Time_sampling_start_idx:]

# Rotor_avg_vars = df_OF.groups["Rotor_Avg_Variables"]
# Rotor_avg_vars_63 = Rotor_avg_vars.groups["5.5"]

# IA = np.array(Rotor_avg_vars_63.variables["IA"][Time_sampling_start_idx:])
# LPF_IA = hard_filter(IA,0.3,dt_sampling,"lowpass")


# df_WT = Dataset(in_dir+"WTG01b.nc")

# WT = df_WT.groups["WTG01"]


# Rotor_coordinates = [np.float64(WT.variables["xyz"][0,0,0]),np.float64(WT.variables["xyz"][0,0,1]),np.float64(WT.variables["xyz"][0,0,2])]


# df = Dataset(in_dir+"WTG01a.nc")
# uvelB1 = np.array(df.variables["uvel"][:,1:301])
# vvelB1 = np.array(df.variables["vvel"][:,1:301])
# uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
# hvelB1 = np.add(np.cos(np.radians(29.29))*uvelB1, np.sin(np.radians(29.29))*vvelB1)
# uvelB2 = np.array(df.variables["uvel"][:,301:601])
# vvelB2 = np.array(df.variables["vvel"][:,301:601])
# uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
# hvelB2 = np.add(np.cos(np.radians(29.29))*uvelB2, np.sin(np.radians(29.29))*vvelB2)
# uvelB3 = np.array(df.variables["uvel"][:,601:901])
# vvelB3 = np.array(df.variables["vvel"][:,601:901])
# uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
# hvelB3 = np.add(np.cos(np.radians(29.29))*uvelB3, np.sin(np.radians(29.29))*vvelB3)


# R = np.linspace(0,63,300)
# dr = R[1] - R[0]

# Iy = []
# Iz = []
# ix=0
# with Pool() as pool:
#     for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,Time_steps):
#         Iy.append(Iy_it); Iz.append(Iz_it)
#         print(ix)
#         ix+=1
# Iy = np.array(Iy); Iz = -np.array(Iz)
# I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

# LPF_I = hard_filter(I,0.3,dt,"lowpass")

# f = interpolate.interp1d(Time_OF[:-1],LPF_I)

# LPF_I_interp = f(Time_sampling)

# plt.rcParams['font.size'] = 16
# cc = round(correlation_coef(LPF_I_interp,LPF_IA),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF[:-1],LPF_I,"-b")
# ax.set_ylabel("LPF (0.3Hz) Blade asymmetry [$m^3/s$]")
# ax.yaxis.label.set_color('blue')
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_sampling,LPF_IA,"-r")
# ax2.set_ylabel("LPF (0.3Hz) Asymmetry parameter [$m^4/s$]")
# ax2.yaxis.label.set_color('red')
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficient = {}".format(cc))
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/cc_LPF_IB_IA.png")
# plt.close(fig)



# appendix D
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_R_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])/1000
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])/1000

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])/1000
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])/1000

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)


# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# #FBR
# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))


# BPF_FBR = hard_filter(FBR,[0.3,0.9],dt,"bandpass")
# HPF_FBR = hard_filter(FBR,[1.5,40],dt,"bandpass")

# LPF_1 = low_pass_filter(FBR,0.3,dt)
# LPF_2 = low_pass_filter(FBR,0.9,dt)
# LPF_3 = low_pass_filter(FBR,1.5,dt)

# HPF = np.subtract(FBR,LPF_3)
# HPF_FBR_2 = low_pass_filter(HPF,40,dt)



# dBPF_FBR = dt_calc(BPF_FBR,dt)
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

# dHPF_FBR = dt_calc(HPF_FBR,dt)
# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]

# dHPF_FBR_2 = dt_calc(HPF_FBR_2,dt)
# zero_crossings_index_HPF_FBR_2 = np.where(np.diff(np.sign(dHPF_FBR_2)))[0]


# #BPF FBR calc
# FB_BPF = []
# dt_BPF = []
# Time_mag_BPF = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

#     it_1 = zero_crossings_index_BPF_FBR[i]
#     it_2 = zero_crossings_index_BPF_FBR[i+1]

#     Time_mag_BPF.append(Time_OF[it_1])

#     FB_BPF.append(BPF_FBR[it_1])
#     dt_BPF.append(Time_OF[it_2]-Time_OF[it_1])

# zero_crossings = np.zeros(len(zero_crossings_index_BPF_FBR)-1)

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,BPF_FBR,"-b",label="BPF $F_{B,\perp}$")
# plt.plot(Time_mag_BPF,FB_BPF,"or",label="Peaks in BPF $F_{B,\perp}$")
# plt.plot(Time_OF[:-1],dBPF_FBR,"-k",label="$dF_{B,\perp,BPF}/dt$")
# plt.plot(Time_mag_BPF,zero_crossings,"og",label="Zero crossings")
# plt.xlabel("Time [s]")
# plt.ylabel("BPF (0.3-0.9Hz) Bearing radial force magnitude [kN]")
# plt.legend()
# plt.grid()
# plt.xlim([200,220])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/BPF_FB_zero_crossings.png")
# plt.close(fig)




# #HPF FBR calc
# FB_HPF = []
# dt_HPF = []
# dF_HPF = []
# Time_mag_HPF = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#     it_1 = zero_crossings_index_HPF_FBR[i]
#     it_2 = zero_crossings_index_HPF_FBR[i+1]

#     Time_mag_HPF.append(Time_OF[it_1])

#     FB_HPF.append(HPF_FBR[it_1])
#     dF_HPF.append(abs(HPF_FBR[it_2]-HPF_FBR[it_1]))
#     dt_HPF.append(Time_OF[it_2]-Time_OF[it_1])

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,HPF_FBR,"-b",label="HPF $F_{B,\perp}$")
# plt.plot(Time_mag_HPF,FB_HPF,"or",label="Peaks in $F_{B,\perp,HPF}$")
# plt.xlabel("Time [s]")
# plt.ylabel("HPF (1.5-40Hz) Bearing radial force magnitude [kN]")
# plt.legend()
# plt.grid()
# plt.xlim([200,205])
# #plt.ylim([-6e03, 6e03])
# plt.tight_layout()


# #HPF FBR calc
# dF_HPF = []
# FB_HPF = []
# dt_HPF = []
# Time_mag_HPF = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR_2)-1):

#     it_1 = zero_crossings_index_HPF_FBR_2[i]
#     it_2 = zero_crossings_index_HPF_FBR_2[i+1]

#     Time_mag_HPF.append(Time_OF[it_1])

#     FB_HPF.append(HPF_FBR_2[it_1])
#     dF_HPF.append(abs(HPF_FBR_2[it_2]-HPF_FBR_2[it_1]))
#     dt_HPF.append(Time_OF[it_2]-Time_OF[it_1])

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_OF,HPF_FBR_2,"-b",label="HPF $F_{B,\perp}$")
# plt.plot(Time_mag_HPF,FB_HPF,"or",label="Peaks in $F_{B,\perp,HPF}$")
# plt.xlabel("Time [s]")
# plt.ylabel("HPF (1.5-40Hz) Bearing radial force magnitude [kN]")
# plt.legend()
# plt.grid()
# plt.xlim([200,205])
# plt.ylim([-300, 300])
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/HPF_FB_zero_crossings.png")
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

# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])/1000
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])/1000

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])/1000
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])/1000

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)


# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# #FBR
# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# WR = 1079*((L1+L2)/L2)

# LPF_FBR = hard_filter(FBR,0.3,dt,"lowpass")
# BPF_FBR = hard_filter(FBR,[0.3,0.9],dt,"bandpass")
# HPF_FBR = hard_filter(FBR,[1.5,40],dt,"bandpass")

# dLPF_FBR = dt_calc(LPF_FBR,dt)
# zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]


# dFBR = dt_calc(FBR,dt)
# zero_crossings_index_FBR = np.where(np.diff(np.sign(dFBR)))[0]

# dLPF_FBR = dt_calc(LPF_FBR,dt)
# zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]

# dBPF_FBR = dt_calc(BPF_FBR,dt)
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

# dHPF_FBR = dt_calc(HPF_FBR,dt)
# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]


# #FBR calc
# dF_mag = []
# dt_mag = []
# Time_mag = []
# FBR_mag = []
# for i in np.arange(0,len(zero_crossings_index_FBR)-1):

#     it_1 = zero_crossings_index_FBR[i]
#     it_2 = zero_crossings_index_FBR[i+1]

#     dt = Time_OF[it_2]-Time_OF[it_1]

#     Time_mag.append(Time_OF[it_1])
#     FBR_mag.append(FBR[it_1])

#     dF_mag.append(abs(FBR[it_2] - FBR[it_1])/WR)
#     dt_mag.append(dt)

# print("Total")
# PFBR,XFBR = probability_dist(dF_mag,5)
# Pdt_FBR,Xdt_FBR = probability_dist(dt_mag,5)

# mode_FBR, mean_FBR, std_FBR, skew_FBR, flat_FBR = moments(dF_mag,XFBR,PFBR)
# mode_T, mean_T, std_T, skew_T, flat_T = moments(dt_mag,Xdt_FBR,Pdt_FBR)

# threshold = np.mean(dF_mag)+2*np.std(dF_mag)
# dt_threshold = []
# dF_threshold = []

# for i in np.arange(0,len(dF_mag)):

#     if dF_mag[i] >= threshold:
#         dF_threshold.append(dF_mag[i])
#         dt_threshold.append(dt_mag[i])

# print(len(dF_threshold))


# #LPF FBR calc
# dF_LPF = []
# dt_LPF = []
# Time_mag_LPF = []
# FBR_LPF_mag = []
# for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-1):

#     it_1 = zero_crossings_index_LPF_FBR[i]
#     it_2 = zero_crossings_index_LPF_FBR[i+1]

#     dt = Time_OF[it_2]-Time_OF[it_1]

#     Time_mag_LPF.append(Time_OF[it_1])
#     FBR_LPF_mag.append(LPF_FBR[it_1])

#     dF_LPF.append(abs(LPF_FBR[it_2] - LPF_FBR[it_1])/WR)
#     dt_LPF.append(dt)


# print("LPF")
# PF_LPF,XF_LPF = probability_dist(dF_LPF,5)
# Pdt_LPF,Xdt_LPF = probability_dist(dt_LPF,5)

# mode_LPF, mean_LPF, std_LPF, skew_LPF, flat_LPF = moments(dF_LPF,XF_LPF,PF_LPF)
# mode_LPF_T, mean_LPF_T, std_LPF_T, skew_LPF_T, flat_LPF_T = moments(dt_LPF,Xdt_LPF,Pdt_LPF)

# #BPF FBR calc
# dF_BPF = []
# dt_BPF = []
# Time_mag_BPF = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

#     it_1 = zero_crossings_index_BPF_FBR[i]
#     it_2 = zero_crossings_index_BPF_FBR[i+1]

#     Time_mag_BPF.append(Time_OF[it_1])

#     dF_BPF.append(abs(BPF_FBR[it_2] - BPF_FBR[it_1])/WR)
#     dt_BPF.append(Time_OF[it_2]-Time_OF[it_1])


# print("BPF")
# PF_BPF,XF_BPF = probability_dist(dF_BPF,5)
# Pdt_BPF,Xdt_BPF = probability_dist(dt_BPF,5)

# mode_BPF, mean_BPF, std_BPF, skew_BPF, flat_BPF = moments(dF_BPF,XF_BPF,PF_BPF)
# mode_BPF_T, mean_BPF_T, std_BPF_T, skew_BPF_T, flat_BPF_T = moments(dt_BPF,Xdt_BPF,Pdt_BPF)

# #HPF FBR calc
# dF_HPF = []
# dt_HPF = []
# Time_mag_HPF = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#     it_1 = zero_crossings_index_HPF_FBR[i]
#     it_2 = zero_crossings_index_HPF_FBR[i+1]

#     Time_mag_HPF.append(Time_OF[it_1])

#     dF_HPF.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1])/WR)
#     dt_HPF.append(Time_OF[it_2]-Time_OF[it_1])


# print("HPF")
# PF_HPF,XF_HPF = probability_dist(dF_HPF,5)
# Pdt_HPF,Xdt_HPF = probability_dist(dt_HPF,5)

# mode_HPF, mean_HPF, std_HPF, skew_HPF, flat_HPF = moments(dF_HPF,XF_HPF,PF_HPF)
# mode_HPF_T, mean_HPF_T, std_HPF_T, skew_HPF_T, flat_HPF_T = moments(dt_HPF,Xdt_HPF,Pdt_HPF)



# dt = Time_OF[1] - Time_OF[0]

# LPF_2_FBR = hard_filter(FBR,0.9,dt,"lowpass")

# dLPF_2_FBR = dt_calc(LPF_2_FBR,dt)
# zero_crossings_index_LPF_2_FBR = np.where(np.diff(np.sign(dLPF_2_FBR)))[0]

# #LPF+BPF FBR calc
# LPF_BPF_mag = []
# FBR_mag = []
# dF_LPF_BPF = []
# dt_LPF_BPF = []
# Time_mag_LPF_BPF = []
# dFBR_mag = []
# for i in np.arange(0,len(zero_crossings_index_LPF_2_FBR)-1):

#     it_1 = zero_crossings_index_LPF_2_FBR[i]
#     it_2 = zero_crossings_index_LPF_2_FBR[i+1]

#     Time_mag_LPF_BPF.append(Time_OF[it_1])
#     FBR_mag.append(FBR[it_1])
#     LPF_BPF_mag.append(LPF_2_FBR[it_1])

#     dF_LPF_BPF.append(abs(LPF_2_FBR[it_2] - LPF_2_FBR[it_1]))
#     dFBR_mag.append(abs(FBR[it_2]-FBR[it_1]))



# #Deformable 
# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/NREL_5MW_MCBL_E_CRPM/76000/"

# df_OF = Dataset(in_dir+"Dataset.nc")

# Time_OF = np.array(df_OF.variables["Time_OF"])
# dt = Time_OF[1] - Time_OF[0]

# Time_start = 200
# Time_start_idx = np.searchsorted(Time_OF,Time_start)

# Time_OF = Time_OF[Time_start_idx:]

# OpenFAST_vars = df_OF.groups["OpenFAST_Variables"]

# Azimuth = np.radians(np.array(OpenFAST_vars.variables["Azimuth"][Time_start_idx:]))

# RtAeroFyh = np.array(OpenFAST_vars.variables["RtAeroFyh"][Time_start_idx:])/1000
# RtAeroFzh = np.array(OpenFAST_vars.variables["RtAeroFzh"][Time_start_idx:])/1000

# RtAeroFys,RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)

# RtAeroMyh = np.array(OpenFAST_vars.variables["RtAeroMyh"][Time_start_idx:])/1000
# RtAeroMzh = np.array(OpenFAST_vars.variables["RtAeroMzh"][Time_start_idx:])/1000

# RtAeroMys,RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)


# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

# #FBR
# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# WR = 1079*((L1+L2)/L2)

# LPF_FBR = hard_filter(FBR,0.3,dt,"lowpass")
# BPF_FBR = hard_filter(FBR,[0.3,0.9],dt,"bandpass")
# HPF_FBR = hard_filter(FBR,[1.5,40],dt,"bandpass")

# dLPF_FBR = dt_calc(LPF_FBR,dt)
# zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]


# dFBR = dt_calc(FBR,dt)
# zero_crossings_index_FBR = np.where(np.diff(np.sign(dFBR)))[0]

# dLPF_FBR = dt_calc(LPF_FBR,dt)
# zero_crossings_index_LPF_FBR = np.where(np.diff(np.sign(dLPF_FBR)))[0]

# dBPF_FBR = dt_calc(BPF_FBR,dt)
# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

# dHPF_FBR = dt_calc(HPF_FBR,dt)
# zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]


# #FBR calc
# dF_mag_E = []
# dt_mag_E = []
# Time_mag_E = []
# FBR_mag_E = []
# for i in np.arange(0,len(zero_crossings_index_FBR)-1):

#     it_1 = zero_crossings_index_FBR[i]
#     it_2 = zero_crossings_index_FBR[i+1]

#     dt = Time_OF[it_2]-Time_OF[it_1]

#     Time_mag_E.append(Time_OF[it_1])
#     FBR_mag_E.append(FBR[it_1])

#     dF_mag_E.append(abs(FBR[it_2] - FBR[it_1])/WR)
#     dt_mag_E.append(dt)

# print("Total")
# PFBR_E,XFBR_E = probability_dist(dF_mag_E,5)
# Pdt_FBR_E,Xdt_FBR_E = probability_dist(dt_mag_E,5)

# mode_FBR_E, mean_FBR_E, std_FBR_E, skew_FBR_E, flat_FBR_E = moments(dF_mag_E,XFBR_E,PFBR_E)
# mode_T_E, mean_T_E, std_T_E, skew_T_E, flat_T_E = moments(dt_mag_E,Xdt_FBR_E,Pdt_FBR_E)

# dt_threshold_E = []
# dF_threshold_E = []
# for i in np.arange(0,len(dF_mag_E)):

#     if dF_mag_E[i] >= threshold:
#         dF_threshold_E.append(dF_mag_E[i])
#         dt_threshold_E.append(dt_mag_E[i])

# print(len(dF_threshold_E))


# #LPF FBR calc
# dF_LPF_E = []
# dt_LPF_E = []
# Time_mag_LPF_E = []
# FBR_LPF_mag_E = []
# for i in np.arange(0,len(zero_crossings_index_LPF_FBR)-1):

#     it_1 = zero_crossings_index_LPF_FBR[i]
#     it_2 = zero_crossings_index_LPF_FBR[i+1]

#     dt = Time_OF[it_2]-Time_OF[it_1]

#     Time_mag_LPF_E.append(Time_OF[it_1])
#     FBR_LPF_mag_E.append(LPF_FBR[it_1])

#     dF_LPF_E.append(abs(LPF_FBR[it_2] - LPF_FBR[it_1])/WR)
#     dt_LPF_E.append(dt)


# print("LPF")
# PF_LPF_E,XF_LPF_E = probability_dist(dF_LPF_E,5)
# Pdt_LPF_E,Xdt_LPF_E = probability_dist(dt_LPF_E,5)

# mode_LPF_E, mean_LPF_E, std_LPF_E, skew_LPF_E, flat_LPF_E = moments(dF_LPF_E,XF_LPF_E,PF_LPF_E)
# mode_LPF_T_E, mean_LPF_T_E, std_LPF_T_E, skew_LPF_T_E, flat_LPF_T_E = moments(dt_LPF_E,Xdt_LPF_E,Pdt_LPF_E)

# #BPF FBR calc
# dF_BPF_E = []
# dt_BPF_E = []
# Time_mag_BPF_E = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR)-1):

#     it_1 = zero_crossings_index_BPF_FBR[i]
#     it_2 = zero_crossings_index_BPF_FBR[i+1]

#     Time_mag_BPF_E.append(Time_OF[it_1])

#     dF_BPF_E.append(abs(BPF_FBR[it_2] - BPF_FBR[it_1])/WR)
#     dt_BPF_E.append(Time_OF[it_2]-Time_OF[it_1])


# print("BPF")
# PF_BPF_E,XF_BPF_E = probability_dist(dF_BPF_E,5)
# Pdt_BPF_E,Xdt_BPF_E = probability_dist(dt_BPF_E,5)

# mode_BPF_E, mean_BPF_E, std_BPF_E, skew_BPF_E, flat_BPF_E = moments(dF_BPF_E,XF_BPF_E,PF_BPF_E)
# mode_BPF_T_E, mean_BPF_T_E, std_BPF_T_E, skew_BPF_T_E, flat_BPF_T_E = moments(dt_BPF_E,Xdt_BPF_E,Pdt_BPF_E)

# #HPF FBR calc
# dF_HPF_E = []
# dt_HPF_E = []
# Time_mag_HPF_E = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#     it_1 = zero_crossings_index_HPF_FBR[i]
#     it_2 = zero_crossings_index_HPF_FBR[i+1]

#     Time_mag_HPF_E.append(Time_OF[it_1])

#     dF_HPF_E.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1])/WR)
#     dt_HPF_E.append(Time_OF[it_2]-Time_OF[it_1])


# print("HPF")
# PF_HPF_E,XF_HPF_E = probability_dist(dF_HPF_E,5)
# Pdt_HPF_E,Xdt_HPF_E = probability_dist(dt_HPF_E,5)

# mode_HPF_E, mean_HPF_E, std_HPF_E, skew_HPF_E, flat_HPF_E = moments(dF_HPF_E,XF_HPF_E,PF_HPF_E)
# mode_HPF_T_E, mean_HPF_T_E, std_HPF_T_E, skew_HPF_T_E, flat_HPF_T_E = moments(dt_HPF_E,Xdt_HPF_E,Pdt_HPF_E)



# dt = Time_OF[1] - Time_OF[0]

# LPF_2_FBR = hard_filter(FBR,0.9,dt,"lowpass")

# dLPF_2_FBR = dt_calc(LPF_2_FBR,dt)
# zero_crossings_index_LPF_2_FBR = np.where(np.diff(np.sign(dLPF_2_FBR)))[0]

# #LPF+BPF FBR calc
# LPF_BPF_mag_E = []
# FBR_mag_E = []
# dF_LPF_BPF_E = []
# dt_LPF_BPF_E = []
# Time_mag_LPF_BPF_E = []
# dFBR_mag_E = []
# for i in np.arange(0,len(zero_crossings_index_LPF_2_FBR)-1):

#     it_1 = zero_crossings_index_LPF_2_FBR[i]
#     it_2 = zero_crossings_index_LPF_2_FBR[i+1]

#     Time_mag_LPF_BPF_E.append(Time_OF[it_1])
#     FBR_mag_E.append(FBR[it_1])
#     LPF_BPF_mag_E.append(LPF_2_FBR[it_1])

#     dF_LPF_BPF_E.append(abs(LPF_2_FBR[it_2] - LPF_2_FBR[it_1]))
#     dFBR_mag_E.append(abs(FBR[it_2]-FBR[it_1]))


# plt.rcParams['font.size'] = 16
# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8),sharey=False)
# label="Rigid: $X_{max}$"+"={}, $\mu_1$={}, $\mu_2$={}, $\mu_3$={}, $\mu_4$={}".format(mode_FBR,mean_FBR,std_FBR,skew_FBR,flat_FBR)
# ax1.plot(XFBR,PFBR,"-r",label=label)
# label="Deform: $X_{max}$"+"={}, $\mu_1$={}, $\mu_2$={}, $\mu_3$={}, $\mu_4$={}".format(mode_FBR_E,mean_FBR_E,std_FBR_E,skew_FBR_E,flat_FBR_E)
# ax1.plot(XFBR_E,PFBR_E,"-b",label=label)
# ax1.set_xlabel("$\\Delta F_{B,\perp}/(W_R L/L_2)$ [-]")
# ax1.set_yscale("log")
# ax1.grid()
# ax1.legend(loc="upper right",bbox_to_anchor=(1.0,1.15),fontsize=14)
# label="Rigid: $X_{max}$"+"={}s, $\mu_1$={}s, $\mu_2$={}s, $\mu_3$={}, $\mu_4$={}".format(mode_T,mean_T,std_T,skew_T,flat_T)
# ax2.plot(Xdt_FBR,Pdt_FBR,"-r",label=label)
# label="Deform: $X_{max}$"+"={}s, $\mu_1$={}s, $\mu_2$={}s, $\mu_3$={}, $\mu_4$={}".format(mode_T_E,mean_T_E,std_T_E,skew_T_E,flat_T_E)
# ax2.plot(Xdt_FBR_E,Pdt_FBR_E,"-b",label=label)
# ax2.set_xlabel("$\\Delta t$ [s]")
# ax2.set_yscale("log")
# ax2.grid()
# ax2.legend(loc="upper right",bbox_to_anchor=(1.25,1.15),fontsize=14)
# fig.supylabel("log() Probability [-]")
# plt.savefig("../../Thesis/Figures/PDF_dF_dt_rigid_deform_FBR.png")
# plt.close(fig)

# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8),sharey=False)
# label="Rigid: $X_{max}$"+"={}, $\mu_1$={}, $\mu_2$={}, $\mu_3$={}, $\mu_4$={}".format(mode_LPF,mean_LPF,std_LPF,skew_LPF,flat_LPF)
# ax1.plot(XF_LPF,PF_LPF,"-r",label=label)
# label="Deform: $X_{max}$"+"={}, $\mu_1$={}, $\mu_2$={}, $\mu_3$={}, $\mu_4$={}".format(mode_LPF_E,mean_LPF_E,std_LPF_E,skew_LPF_E,flat_LPF_E)
# ax1.plot(XF_LPF_E,PF_LPF_E,"-b",label=label)
# ax1.set_xlabel("$\\Delta F_{B,\perp,LPF}/(W_RL/L_2)$ [-]")
# ax1.set_yscale("log")
# ax1.grid()
# ax1.legend(loc="upper right",bbox_to_anchor=(1.0,1.15),fontsize=14)
# label="Rigid: $X_{max}$"+"={}s, $\mu_1$={}s, $\mu_2$={}s, $\mu_3$={}, $\mu_4$={}".format(mode_LPF_T,mean_LPF_T,std_LPF_T,skew_LPF_T,flat_LPF_T)
# ax2.plot(Xdt_LPF,Pdt_LPF,"-r",label=label)
# label="Deform: $X_{max}$"+"={}s, $\mu_1$={}s, $\mu_2$={}s, $\mu_3$={}, $\mu_4$={}".format(mode_LPF_T_E,mean_LPF_T_E,std_LPF_T_E,skew_LPF_T_E,flat_LPF_T_E)
# ax2.plot(Xdt_LPF_E,Pdt_LPF_E,"-b",label=label)
# ax2.set_xlabel("$\\Delta t_{LPF}$ [s]")
# ax2.set_yscale("log")
# ax2.grid()
# ax2.legend(loc="upper right",bbox_to_anchor=(1.25,1.15),fontsize=14)
# fig.supylabel("log() Probability [-]")
# plt.savefig("../../Thesis/Figures/PDF_dF_dt_rigid_deform_FBR_LPF.png")
# plt.close(fig)

# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8),sharey=False)
# label="Rigid: $X_{max}$"+"={}, $\mu_1$={}, $\mu_2$={}, $\mu_3$={}, $\mu_4$={}".format(mode_BPF,mean_BPF,std_BPF,skew_BPF,flat_BPF)
# ax1.plot(XF_BPF,PF_BPF,"-r",label=label)
# label="Deform: $X_{max}$"+"={}, $\mu_1$={}, $\mu_2$={}, $\mu_3$={}, $\mu_4$={}".format(mode_BPF_E,mean_BPF_E,std_BPF_E,skew_BPF_E,flat_BPF_E)
# ax1.plot(XF_BPF_E,PF_BPF_E,"-b",label=label)
# ax1.set_xlabel("$\\Delta F_{B,\perp,BPF}/(W_RL/L_2)$ [-]")
# ax1.set_yscale("log")
# ax1.grid()
# ax1.legend(loc="upper right",bbox_to_anchor=(1.0,1.15),fontsize=14)
# label="Rigid: $X_{max}$"+"={}s, $\mu_1$={}s, $\mu_2$={}s, $\mu_3$={}, $\mu_4$={}".format(mode_BPF_T,mean_BPF_T,std_BPF_T,skew_BPF_T,flat_BPF_T)
# ax2.plot(Xdt_BPF,Pdt_BPF,"-r",label=label)
# label="Deform: $X_{max}$"+"={}s, $\mu_1$={}s, $\mu_2$={}s, $\mu_3$={}, $\mu_4$={}".format(mode_BPF_T_E,mean_BPF_T_E,std_BPF_T_E,skew_BPF_T_E,flat_BPF_T_E)
# ax2.plot(Xdt_BPF_E,Pdt_BPF_E,"-b",label=label)
# ax2.set_xlabel("$\\Delta t_{BPF}$ [s]")
# ax2.set_yscale("log")
# ax2.grid()
# ax2.legend(loc="upper right",bbox_to_anchor=(1.25,1.15),fontsize=14)
# fig.supylabel("log() Probability [-]")
# plt.savefig("../../Thesis/Figures/PDF_dF_dt_rigid_deform_FBR_BPF.png")
# plt.close(fig)

# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8),sharey=False)
# label="Rigid: $X_{max}$"+"={}, $\mu_1$={}, $\mu_2$={}, $\mu_3$={}, $\mu_4$={}".format(mode_HPF,mean_HPF,std_HPF,skew_HPF,flat_HPF)
# ax1.plot(XF_HPF,PF_HPF,"-r",label=label)
# label="Deform: $X_{max}$"+"={}, $\mu_1$={}, $\mu_2$={}, $\mu_3$={}, $\mu_4$={}".format(mode_HPF_E,mean_HPF_E,std_HPF_E,skew_HPF_E,flat_HPF_E)
# ax1.plot(XF_HPF_E,PF_HPF_E,"-b",label=label)
# ax1.set_xlabel("$\\Delta F_{B,\perp,HPF}/(W_RL/L_2)$ [-]")
# ax1.set_yscale("log")
# ax1.grid()
# ax1.legend(loc="upper right",bbox_to_anchor=(1.0,1.15),fontsize=14)
# label="Rigid: $X_{max}$"+"={}s, $\mu_1$={}s, $\mu_2$={}s, $\mu_3$={}, $\mu_4$={}".format(mode_HPF_T,mean_HPF_T,std_HPF_T,skew_HPF_T,flat_HPF_T)
# ax2.plot(Xdt_HPF,Pdt_HPF,"-r",label=label)
# label="Deform: $X_{max}$"+"={}s, $\mu_1$={}s, $\mu_2$={}s, $\mu_3$={}, $\mu_4$={}".format(mode_HPF_T_E,mean_HPF_T_E,std_HPF_T_E,skew_HPF_T_E,flat_HPF_T_E)
# ax2.plot(Xdt_HPF_E,Pdt_HPF_E,"-b",label=label)
# ax2.set_xlabel("$\\Delta t_{HPF}$ [s]")
# ax2.set_yscale("log")
# ax2.grid()
# ax2.legend(loc="upper right",bbox_to_anchor=(1.25,1.15),fontsize=14)
# fig.supylabel("log() Probability [-]")
# plt.savefig("../../Thesis/Figures/PDF_dF_dt_rigid_deform_FBR_HPF.png")
# plt.close(fig)


# fig,(ax1) = plt.subplots(figsize=(14,8),sharey=False)
# diff_FB = np.subtract(dFBR_mag,dF_LPF_BPF)
# P,X = probability_dist(diff_FB,5)
# mode, mean, std, skew, flat = moments(diff_FB,X,P)
# label="Rigid: $X_{max}$"+"={}kN, $\mu_1$={}kN, $\mu_2$={}kN, $\mu_3$={}, $\mu_4$={}\n".format(mode,mean,std,skew,flat)
# ax1.plot(X,P,"-r",label=label)
# diff_FB = np.subtract(dFBR_mag_E,dF_LPF_BPF_E)
# P,X = probability_dist(diff_FB,5)
# mode, mean, std, skew, flat = moments(diff_FB,X,P)
# label="Deform: $X_{max}$"+"={}kN, $\mu_1$={}kN, $\mu_2$={}kN, $\mu_3$={}, $\mu_4$={}".format(mode,mean,std,skew,flat)
# ax1.plot(X,P,"-b",label=label)
# ax1.set_xlabel("$|\Delta F_{B,\perp}|-|\Delta F_{B,\perp,LPF+BPF}|$ [kN]")
# ax1.grid()
# ax1.legend(loc="upper center",bbox_to_anchor=(0.5,1.2))
# ax1.set_ylabel("Probability [-]")
# plt.tight_layout()
# plt.savefig("../../Thesis/Figures/Deform_PDF_dF_LPF_BPF.png")
# plt.close(fig)






