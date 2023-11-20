from math import ceil, floor
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
from scipy import interpolate

def dz_calc_2(u,dz):
    #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
    d_dz = []
    for i in np.arange(0,len(u)-4,1):
        if i == 0:
            d_dz_i = ((-(25/12)*u[i]+4*u[i+1]-3*u[i+2]+(4/3)*u[i+3]-(1/4)*u[i+4])/dz)
        else:
            d_dz_i = ((u[i+1] - u[i-1])/(2*dz))

        d_dz.append(d_dz_i)

    return d_dz

def dz_calc(u,dz):
    #compute graident to 2nd order accurate using central difference primarily and forward difference for the first cell
    d_dz = []
    for i in np.arange(0,len(u)-4,1):
        d_dz_i = ((-(25/12)*u[i]+4*u[i+1]-3*u[i+2]+(4/3)*u[i+3]-(1/4)*u[i+4])/dz)

        d_dz.append(d_dz_i)

    return d_dz


def dt_calc(u,dt):
    #compute time derivative using first order forward difference
    d_dt = []
    for i in np.arange(0,len(u)-1,1):
        d_dt.append( (u[i+1]-u[i])/dt )

    return d_dt


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


#directories
in_dir = "../../ABL_precursor_2/"
out_dir = in_dir + "plots/"


#loads statisitcs data
data = Dataset(in_dir+"abl_statistics60000.nc")
Mean_profiles = data.groups["mean_profiles"]

time = data.variables["time"]
z = np.array(Mean_profiles.variables["h"])
dz = z[1]-z[0]
u_star = data.variables["ustar"]
kappa = 0.41

u = np.array(Mean_profiles.variables["u"])
v = np.array(Mean_profiles.variables["v"])
hvelmag = []
for u_i, v_i in zip(u,v):
    hvelmag.append(np.add( np.multiply(np.cos(np.radians(29)),u_i), np.multiply(np.sin(np.radians(29)),v_i) ))

du_dz = []
for hvelmag_i in hvelmag:
    du_dz_i = dz_calc(hvelmag_i,dz)
    du_dz.append(du_dz_i)


phi_m = []
for i in np.arange(0,len(du_dz)):
    kappa = 0.41
    z_star_m = (z[0:-4]*kappa)/u_star[i]
    phi_m_i = np.multiply(z_star_m,du_dz[i])
    phi_m.append(phi_m_i)


Times = [32500, 33000, 34000, 34500]

plt.figure(figsize=(14,8))
for Time in Times:
    Time_idx = np.searchsorted(time,Time)
    hvelmag_i = hvelmag[Time_idx]
    plt.plot(hvelmag_i,z)
plt.ylabel("Distance from surface [m]",fontsize=16)
plt.xlabel("horizontal velocity [m/s]",fontsize=16)
plt.legend(Times)
plt.tight_layout()
plt.savefig(out_dir+"horizonal_velocity.png")
plt.close()


plt.figure(figsize=(14,8))
for Time in Times:
    Time_idx = np.searchsorted(time,Time)
    hvelmag_i = hvelmag[Time_idx]
    plt.plot(hvelmag_i,z)
plt.ylabel("Distance from surface [m]",fontsize=16)
plt.xlabel("horizontal velocity [m/s]",fontsize=16)
plt.legend(Times)
plt.ylim([0,200])
plt.tight_layout()
plt.savefig(out_dir+"horizonal_velocity_0.2.png")
plt.close()

plt.figure(figsize=(14,8))
for Time in Times:
    Time_idx = np.searchsorted(time,Time)

    plt.plot(phi_m[Time_idx],z[:-4])
plt.ylabel("Distance from surface [m]",fontsize=16)
plt.xlabel("phi_m [-]",fontsize=16)
plt.legend(Times)
plt.tight_layout()
plt.savefig(out_dir+"phi_m.png")
plt.close()

plt.figure(figsize=(14,8))
for Time in Times:
    Time_idx = np.searchsorted(time,Time)

    plt.plot(phi_m[Time_idx],z[:-4])
plt.ylabel("Distance from surface [m]",fontsize=16)
plt.xlabel("phi_m [-]",fontsize=16)
plt.legend(Times)
plt.ylim([0,200]); plt.xlim([0,2])
plt.tight_layout()
plt.savefig(out_dir+"phi_m_0.2.png")
plt.close()



Time = np.array(data.variables["time"])
dt = Time[1]-Time[0]

#quasi-stationarity
tstart = 32500
tstart_idx = np.searchsorted(Time,tstart)

#Time varying quantites
zi = np.array(data.variables["zi"])
u_star = np.array(data.variables["ustar"])
w_star = np.array(data.variables["wstar"])
T0 = np.array(data.variables["Tsurf"])
Q = np.array(data.variables["Q"])
L = np.array(data.variables["L"])
dzi_dt = dt_calc(zi,dt)
dzi_dt_u_star = np.true_divide(dzi_dt,u_star[:-1])
dzi_dt_w_star = np.true_divide(dzi_dt,w_star[:-1])

#moving statistics
ts_dzi_dt_u_star = pd.Series(dzi_dt_u_star, index=Time[:-1])
ts_dzi_dt_w_star = pd.Series(dzi_dt_w_star, index=Time[:-1])

zi_L = -np.true_divide(zi,L)
tau_u = np.true_divide(zi,u_star)
tau_w = np.true_divide(zi,w_star)

#Average global statistics
glob_zi = np.average(zi[tstart_idx:])
glob_L = np.average(L[tstart_idx:])
glob_zi_L = -np.true_divide(glob_zi,glob_L)
glob_u_star = np.average(u_star[tstart_idx:])
glob_w_star = np.average(w_star[tstart_idx:])
glob_tau_u = np.average(tau_u[tstart_idx:])
glob_tau_w = np.average(tau_w[tstart_idx:])
glob_Q = np.average(Q[tstart_idx:])


print("zi",glob_zi,"-L",-glob_L,"-zi/L",glob_zi_L,"u*",glob_u_star,"w*",glob_w_star,"tau_u",glob_tau_u/60,"tau_w",glob_tau_w/60)


#mean profiles averaged in time
z = np.array(Mean_profiles.variables["h"])
dz = z[1] - z[0]
z_zi = z/glob_zi
u = np.array(Mean_profiles.variables["u"])
v = np.array(Mean_profiles.variables["v"])
w = np.average(np.array(Mean_profiles.variables["w"][tstart_idx:]),axis=0)
hvelmag = []
for u_i, v_i in zip(u,v):
    hvelmag.append(np.add( np.multiply(np.cos(np.radians(29)),u_i), np.multiply(np.sin(np.radians(29)),v_i) ))
hvelmag = np.array(hvelmag)
#hub height
z_hub = 90
z_hub_idx = np.searchsorted(z,z_hub)
hvelmag_hub = hvelmag[:,z_hub_idx]
glob_hvelmag_hub = np.average(hvelmag_hub[tstart_idx:])

hvelmag = np.average(hvelmag[tstart_idx:],axis=0)

theta = np.average(np.array(Mean_profiles.variables["theta"][tstart_idx:]),axis=0)
w_theta = np.average(np.array(Mean_profiles.variables["w'theta'_r"][tstart_idx:]),axis=0)
u_u_r = np.average(np.array(Mean_profiles.variables["u'u'_r"][tstart_idx:]),axis=0)
v_v_r = np.average(np.array(Mean_profiles.variables["v'v'_r"][tstart_idx:]),axis=0)
w_w_r = np.average(np.array(Mean_profiles.variables["w'w'_r"][tstart_idx:]),axis=0)
u_w_r = np.average(np.array(Mean_profiles.variables["u'w'_r"][tstart_idx:]),axis=0)
v_w_r = np.average(np.array(Mean_profiles.variables["v'w'_r"][tstart_idx:]),axis=0)
u_w_sfs = np.average(np.array(Mean_profiles.variables["u'w'_sfs"][tstart_idx:]),axis=0)
v_w_sfs = np.average(np.array(Mean_profiles.variables["v'w'_sfs"][tstart_idx:]),axis=0)
du_dz = dz_calc(hvelmag,dz)
dtheta_dz = dz_calc(theta,dz)
u = np.average(u[tstart_idx:],axis=0); v = np.average(v[tstart_idx:],axis=0)
twist = coriolis_twist(u,v)

#phi_m
kappa = 0.41
z_star_m = (z[0:-4]*kappa)/glob_u_star
phi_m = np.multiply(z_star_m,du_dz)


#phi_h
T_star = glob_Q/glob_u_star
z_star_h = (z[0:-4]*kappa)/T_star
phi_h = np.multiply(z_star_h,dtheta_dz)

#precursor plots
with PdfPages(out_dir+'precursor_plots.pdf') as pdf:
    #plot Time varying quanities
    #horizontal velocity
    plt.figure(figsize=(14,8))
    plt.plot(Time,hvelmag_hub)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("horizontal velocity [m/s]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #zi
    plt.figure(figsize=(14,8))
    plt.plot(Time,zi)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$z_i$ [m]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #-zi/L
    plt.figure(figsize=(14,8))
    plt.plot(Time,zi_L)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$-z_i/L$ [m]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #ustar
    plt.figure(figsize=(14,8))
    plt.plot(Time,u_star)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$u_*$ [m]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #wstar
    plt.figure(figsize=(14,8))
    plt.plot(Time,w_star)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$w_*$ [m]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #tau_u
    plt.figure(figsize=(14,8))
    plt.plot(Time,tau_u/60)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$\\tau_u$ [min]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #tau_w
    plt.figure(figsize=(14,8))
    plt.plot(Time,tau_w/60)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$\\tau_w$ [min]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dzi/dt
    plt.figure(figsize=(14,8))
    plt.plot(Time[:-1],dzi_dt)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$dz_i/dt$ [m/s]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dzi/dt 1/u*
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
    pdf.savefig()
    plt.close()

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
    pdf.savefig()
    plt.close()

    #T0
    plt.figure(figsize=(14,8))
    plt.plot(Time,T0)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Surface Temperature [K]",fontsize=16)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    #plot Time varying quanities over quasi-stationary period
    #Horizotnal velocity
    plt.figure(figsize=(14,8))
    plt.plot(Time,hvelmag_hub)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("horizontal velocity [m/s]",fontsize=16)
    plt.title("quasi-stationary period",fontsize=18)
    plt.xlim(left=tstart)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #zi
    plt.figure(figsize=(14,8))
    plt.plot(Time,zi)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$z_i$ [m]",fontsize=16)
    plt.title("quasi-stationary period",fontsize=18)
    plt.xlim(left=tstart)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #tau_u
    plt.figure(figsize=(14,8))
    plt.plot(Time,tau_u/60)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$\\tau_u$ [min]",fontsize=16)
    plt.title("quasi-stationary period",fontsize=18)
    plt.xlim(left=tstart)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #tau_w
    plt.figure(figsize=(14,8))
    plt.plot(Time,tau_w/60)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$\\tau_w$ [min]",fontsize=16)
    plt.title("quasi-stationary period",fontsize=18)
    plt.xlim(left=tstart)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dzi/dt
    plt.figure(figsize=(14,8))
    plt.plot(Time[:-1],dzi_dt)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$dz_i/dt$ [m/s]",fontsize=16)#
    plt.title("quasi-stationary period",fontsize=18)
    plt.xlim(left=tstart)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dzi/dt 1/u*
    plt.figure(figsize=(14,8))
    plt.plot(Time[:-1],dzi_dt_u_star)
    window_idx = int((glob_tau_u)/dt)
    ts_dzi_dt_u_star.rolling(window=window_idx).mean().plot(style='k--')
    ts_dzi_dt_u_star.rolling(window=window_idx).std().plot(style='r--')
    plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$dz_i/dt 1/u_{star}$ [-]",fontsize=16)
    plt.title("quasi-stationary period",fontsize=18)
    plt.legend(["$dz_i/dt 1/u_{star}$","Mean","Std","0.01","-0.01"])
    plt.xlim(left=tstart)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dzi/dt 1/w*
    plt.figure(figsize=(14,8))
    plt.plot(Time[:-1],dzi_dt_w_star)
    window_idx = int((glob_tau_w)/dt)
    ts_dzi_dt_w_star.rolling(window=window_idx).mean().plot(style='k--')
    ts_dzi_dt_w_star.rolling(window=window_idx).std().plot(style='r--')
    plt.axhline(0.01,linestyle="--",color="g"); plt.axhline(-0.01,linestyle="--",color="g")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$dz_i/dt 1/w_{star}$ [-]",fontsize=16)
    plt.title("quasi-stationary period",fontsize=18)
    plt.legend(["$dz_i/dt 1/w_{star}$","Mean","Std","0.01","-0.01"])
    plt.xlim(left=tstart)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #T0
    plt.figure(figsize=(14,8))
    plt.plot(Time,T0)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Surface Temperature [K]",fontsize=16)
    plt.title("quasi-stationary period",fontsize=18)
    plt.xlim(left=tstart)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    #mean profiles
    #U
    plt.figure(figsize=(14,8))
    plt.plot(hvelmag,z_zi)
    plt.xlabel("Horizontal velocity [m/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dU/dz
    plt.figure(figsize=(14,8))
    plt.plot(du_dz,z_zi[:-4])
    plt.xlabel("Horizontal velocity gradient [1/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #W
    plt.figure(figsize=(14,8))
    plt.plot(w,z_zi)
    plt.xlabel("Vertical velocity [m/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #Coriolis twist
    plt.figure(figsize=(14,8))
    plt.plot(np.degrees(twist),z_zi)
    plt.xlabel("Flow angle [deg]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    f = interpolate.interp1d(z,twist)
    twist_hub = f(90)
    plt.text(26,90/glob_zi,"{}deg".format(round(np.degrees(twist_hub),0)),fontsize=14)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #u'u'_r
    plt.figure(figsize=(14,8))
    plt.plot(u_u_r,z_zi)
    plt.xlabel("$(u'u')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #v'v'_r
    plt.figure(figsize=(14,8))
    plt.plot(v_v_r,z_zi)
    plt.xlabel("$(v'v')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #w'w'_r
    plt.figure(figsize=(14,8))
    plt.plot(w_w_r,z_zi)
    plt.xlabel("$(w'w')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #u'w'_r
    plt.figure(figsize=(14,8))
    plt.plot(u_w_r,z_zi)
    plt.xlabel("$(u'w')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #Theta
    plt.figure(figsize=(14,8))
    plt.plot(theta,z_zi)
    plt.xlabel("Potential temperature [K]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dTheta/dz
    plt.figure(figsize=(14,8))
    plt.plot(dtheta_dz,z_zi[:-4])
    plt.xlabel("Potential temperature gradient [K/m]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #w'Theta'_r
    plt.figure(figsize=(14,8))
    plt.plot(w_theta,z_zi)
    plt.xlabel("$(w'\\theta')^r [Km/s]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    #mean profiles 0.2zi
    #U
    plt.figure(figsize=(14,8))
    plt.plot(hvelmag,z_zi)
    plt.xlabel("Horizontal velocity [m/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dU/dz
    plt.figure(figsize=(14,8))
    plt.plot(du_dz,z_zi[:-4])
    plt.xlabel("Horizontal velocity gradient [1/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #W
    plt.figure(figsize=(14,8))
    plt.plot(w,z_zi)
    plt.xlabel("Vertical velocity [m/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #Coriolis twist
    plt.figure(figsize=(14,8))
    plt.plot(np.degrees(twist),z_zi)
    plt.xlabel("Flow angle [deg]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    f = interpolate.interp1d(z,twist)
    twist_hub = f(90)
    plt.text(26,90/glob_zi,"{}deg".format(round(np.degrees(twist_hub),0)),fontsize=14)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #u'u'_r
    plt.figure(figsize=(14,8))
    plt.plot(u_u_r,z_zi)
    plt.xlabel("$(u'u')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #v'v'_r
    plt.figure(figsize=(14,8))
    plt.plot(v_v_r,z_zi)
    plt.xlabel("$(v'v')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #w'w'_r
    plt.figure(figsize=(14,8))
    plt.plot(w_w_r,z_zi)
    plt.xlabel("$(w'w')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #u'w'_r
    plt.figure(figsize=(14,8))
    plt.plot(u_w_r,z_zi)
    plt.xlabel("$(u'w')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #Theta
    plt.figure(figsize=(14,8))
    plt.plot(theta,z_zi)
    plt.xlabel("Potential temperature [K]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dTheta/dz
    plt.figure(figsize=(14,8))
    plt.plot(dtheta_dz,z_zi[:-4])
    plt.xlabel("Potential temperature gradient [K/m]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #w'Theta'_r
    plt.figure(figsize=(14,8))
    plt.plot(w_theta,z_zi)
    plt.xlabel("$(w'\\theta')^r [Km/s]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #Re_R_LES
    #phi_m
    plt.figure(figsize=(14,8))
    plt.plot(phi_m,z_zi[:-4])
    plt.xlabel("$\\phi_m$ [-]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #phi_m
    plt.figure(figsize=(14,8))
    plt.plot(phi_m,z_zi[:-4])
    plt.xlabel("$\\phi_m$ [-]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2]); plt.xlim([0,2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #phi_h
    plt.figure(figsize=(14,8))
    plt.plot(phi_h,z_zi[:-4])
    plt.xlabel("$\\phi_h$ [-]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #phi_h
    plt.figure(figsize=(14,8))
    plt.plot(phi_h,z_zi[:-4])
    plt.xlabel("$\\phi_h$ [-]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Horizontal average and time average over final 2400s, zi = {}".format(glob_zi),fontsize=18)
    plt.ylim([0,0.2]); plt.xlim([-1,0.5])
    plt.tight_layout()
    pdf.savefig()
    plt.close()



#comparing precursors
#quasi-stationarity
tstart = 32500
tstart_idx = np.searchsorted(Time,tstart)
tend = 32500+1200
tend_idx = np.searchsorted(Time,tend)

Time = Time[:]

#Time varying quantites
zi = np.array(data.variables["zi"][:])
u_star = np.array(data.variables["ustar"][:])
w_star = np.array(data.variables["wstar"][:])
T0 = np.array(data.variables["Tsurf"][:])
Q = np.array(data.variables["Q"][:])
L = np.array(data.variables["L"][:])
dzi_dt = dt_calc(zi,dt)
dzi_dt_u_star = np.true_divide(dzi_dt,u_star[:-1])
dzi_dt_w_star = np.true_divide(dzi_dt,w_star[:-1])

zi_L = -np.true_divide(zi,L)
tau_u = np.true_divide(zi,u_star)
tau_w = np.true_divide(zi,w_star)

#Average global statistics
glob_zi = np.average(zi[tstart_idx:])
glob_L = np.average(L[tstart_idx:])
glob_zi_L = -np.true_divide(glob_zi,glob_L)
glob_u_star = np.average(u_star[tstart_idx:])
glob_w_star = np.average(w_star[tstart_idx:])
glob_tau_u = np.average(tau_u[tstart_idx:])
glob_tau_w = np.average(tau_w[tstart_idx:])
glob_Q = np.average(Q[tstart_idx:])


#mean profiles averaged in time
z = np.array(Mean_profiles.variables["h"])
dz = z[1] - z[0]
z_zi = z/glob_zi
u = np.array(Mean_profiles.variables["u"])
v = np.array(Mean_profiles.variables["v"])
w = np.average(np.array(Mean_profiles.variables["w"][tstart_idx:]),axis=0)
hvelmag = []
for u_i, v_i in zip(u,v):
    hvelmag.append(np.add( np.multiply(np.cos(np.radians(29)),u_i), np.multiply(np.sin(np.radians(29)),v_i) ))
hvelmag = np.array(hvelmag)
#hub height
z_hub = 90
z_hub_idx = np.searchsorted(z,z_hub)
hvelmag_hub = hvelmag[:,z_hub_idx]
glob_hvelmag_hub = np.average(hvelmag_hub[tstart_idx:])

hvelmag = np.average(hvelmag[tstart_idx:],axis=0)

theta = np.average(np.array(Mean_profiles.variables["theta"][tstart_idx:]),axis=0)
w_theta = np.average(np.array(Mean_profiles.variables["w'theta'_r"][tstart_idx:]),axis=0)
u_u_r = np.average(np.array(Mean_profiles.variables["u'u'_r"][tstart_idx:]),axis=0)
v_v_r = np.average(np.array(Mean_profiles.variables["v'v'_r"][tstart_idx:]),axis=0)
w_w_r = np.average(np.array(Mean_profiles.variables["w'w'_r"][tstart_idx:]),axis=0)
u_w_r = np.average(np.array(Mean_profiles.variables["u'w'_r"][tstart_idx:]),axis=0)
v_w_r = np.average(np.array(Mean_profiles.variables["v'w'_r"][tstart_idx:]),axis=0)
u_w_sfs = np.average(np.array(Mean_profiles.variables["u'w'_sfs"][tstart_idx:]),axis=0)
v_w_sfs = np.average(np.array(Mean_profiles.variables["v'w'_sfs"][tstart_idx:]),axis=0)
du_dz = dz_calc(hvelmag,dz)
dtheta_dz = dz_calc(theta,dz)
u = np.average(u[tstart_idx:tend_idx],axis=0); v = np.average(v[tstart_idx:],axis=0)
twist = coriolis_twist(u,v)


#phi_m
kappa = 0.41
z_star_m = (z[0:-4]*kappa)/glob_u_star
phi_m = np.multiply(z_star_m,du_dz)


#phi_h
T_star = glob_Q/glob_u_star
z_star_h = (z[0:-4]*kappa)/T_star
phi_h = np.multiply(z_star_h,dtheta_dz)



#directories
in_dir = "../../ABL_precursor/post_processing/"


#loads statisitcs data
data = Dataset(in_dir+"abl_statistics60000.nc")
Mean_profiles = data.groups["mean_profiles"]


Time_2 = np.array(data.variables["time"])
dt_2 = Time_2[1]-Time_2[0]

#Time varying quantites
zi_2 = np.array(data.variables["zi"])
u_star_2 = np.array(data.variables["ustar"])
w_star_2 = np.array(data.variables["wstar"])
T0_2 = np.array(data.variables["Tsurf"])
Q_2 = np.array(data.variables["Q"])
L_2 = np.array(data.variables["L"])
dzi_dt_2 = dt_calc(zi_2,dt_2)
dzi_dt_u_star_2 = np.true_divide(dzi_dt_2,u_star_2[:-1])
dzi_dt_w_star_2 = np.true_divide(dzi_dt_2,w_star_2[:-1])
zi_L_2 = -np.true_divide(zi_2,L_2)
tau_u_2 = np.true_divide(zi_2,u_star_2)
tau_w_2 = np.true_divide(zi_2,w_star_2)


#quasi-stationarity
tstart = 32500
tstart_idx = np.searchsorted(Time,tstart)

#Average global statistics
glob_zi_2 = np.average(zi_2[tstart_idx:])
glob_L_2 = np.average(L_2[tstart_idx:])
glob_zi_L_2 = -np.true_divide(glob_zi_2,glob_L_2)
glob_u_star_2 = np.average(u_star_2[tstart_idx:])
glob_w_star_2 = np.average(w_star_2[tstart_idx:])
glob_tau_u_2 = np.average(tau_u_2[tstart_idx:])
glob_tau_w_2 = np.average(tau_w_2[tstart_idx:])
glob_Q_2 = np.average(Q_2[tstart_idx:])


print("zi",glob_zi,"-L",-glob_L,"-zi/L",glob_zi_L,"u*",glob_u_star,"w*",glob_w_star,"tau_u",glob_tau_u/60,"tau_w",glob_tau_w/60)
print("zi",glob_zi_2,"-L",-glob_L_2,"-zi/L",glob_zi_L_2,"u*",glob_u_star_2,"w*",glob_w_star_2,"tau_u",glob_tau_u_2/60,"tau_w",glob_tau_w_2/60)

#mean profiles averaged in time
z_2 = np.array(Mean_profiles.variables["h"])
dz_2 = z_2[1] - z_2[0]
z_zi_2 = z_2/glob_zi_2
u_2 = np.array(Mean_profiles.variables["u"])
v_2 = np.array(Mean_profiles.variables["v"])
w_2 = np.average(np.array(Mean_profiles.variables["w"][tstart_idx:]),axis=0)
hvelmag_2 = []
for u_i, v_i in zip(u_2,v_2):
    hvelmag_2.append(np.add( np.multiply(np.cos(np.radians(29)),u_i), np.multiply(np.sin(np.radians(29)),v_i) ))
hvelmag_2 = np.array(hvelmag_2)
#hub height
z_hub = 90
z_hub_idx = np.searchsorted(z,z_hub)
hvelmag_hub_2 = hvelmag_2[:,z_hub_idx]
glob_hvelmag_hub_2 = np.average(hvelmag_hub_2[tstart_idx:])

hvelmag_2 = np.average(hvelmag_2[tstart_idx:],axis=0)

theta_2 = np.average(np.array(Mean_profiles.variables["theta"][tstart_idx:]),axis=0)
w_theta_2 = np.average(np.array(Mean_profiles.variables["w'theta'_r"][tstart_idx:]),axis=0)
u_u_r_2 = np.average(np.array(Mean_profiles.variables["u'u'_r"][tstart_idx:]),axis=0)
v_v_r_2 = np.average(np.array(Mean_profiles.variables["v'v'_r"][tstart_idx:]),axis=0)
w_w_r_2 = np.average(np.array(Mean_profiles.variables["w'w'_r"][tstart_idx:]),axis=0)
u_w_r_2 = np.average(np.array(Mean_profiles.variables["u'w'_r"][tstart_idx:]),axis=0)
v_w_r_2 = np.average(np.array(Mean_profiles.variables["v'w'_r"][tstart_idx:]),axis=0)
u_w_sfs_2 = np.average(np.array(Mean_profiles.variables["u'w'_sfs"][tstart_idx:]),axis=0)
v_w_sfs_2 = np.average(np.array(Mean_profiles.variables["v'w'_sfs"][tstart_idx:]),axis=0)
du_dz_2 = dz_calc(hvelmag_2,dz_2)
dtheta_dz_2 = dz_calc(theta_2,dz_2)
u_2 = np.average(u_2[tstart_idx:],axis=0); v_2 = np.average(v_2[tstart_idx:],axis=0)
twist_2 = coriolis_twist(u_2,v_2)


#phi_m
kappa = 0.41
z_star_m_2 = (z_2[0:-4]*kappa)/glob_u_star_2
phi_m_2 = np.multiply(z_star_m_2,du_dz_2)


#phi_h
T_star_2 = glob_Q_2/glob_u_star_2
z_star_h_2 = (z_2[0:-4]*kappa)/T_star_2
phi_h_2 = np.multiply(z_star_h_2,dtheta_dz_2)

#comparing precursor plots
with PdfPages(out_dir+'comparing_precursor_plots_2.pdf') as pdf:
    #plot Time varying quanities
    #horizontal velocity
    plt.figure(figsize=(14,8))
    plt.plot(Time,hvelmag_hub,"r")
    plt.plot(Time_2,hvelmag_hub_2,"--b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("horizontal velocity [m/s]",fontsize=16)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #zi
    plt.figure(figsize=(14,8))
    plt.plot(Time,zi,"r")
    plt.plot(Time_2,zi_2,"--b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$z_i$ [m]",fontsize=16)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #-zi/L
    plt.figure(figsize=(14,8))
    plt.plot(Time,zi_L,"r")
    plt.plot(Time_2,zi_L_2,"--b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$-z_i/L$ [m]",fontsize=16)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #tau_u
    plt.figure(figsize=(14,8))
    plt.plot(Time,tau_u/60,"r")
    plt.plot(Time_2,tau_u_2/60,"--b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$\\tau_u$ [min]",fontsize=16)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #tau_w
    plt.figure(figsize=(14,8))
    plt.plot(Time,tau_w/60,"r")
    plt.plot(Time_2,tau_w_2/60,"--b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$\\tau_w$ [min]",fontsize=16)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dzi/dt
    plt.figure(figsize=(14,8))
    plt.plot(Time[:-1],dzi_dt,"r")
    plt.plot(Time_2[:-1],dzi_dt_2,"--b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("$dz_i/dt$ [m/s]",fontsize=16)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #T0
    plt.figure(figsize=(14,8))
    plt.plot(Time,T0,"r")
    plt.plot(Time_2,T0_2,"--b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Surface Temperature [K]",fontsize=16)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    #mean profiles
    #U
    plt.figure(figsize=(14,8))
    plt.plot(hvelmag,z_zi,"r")
    plt.plot(hvelmag_2,z_zi_2,"--b")
    plt.xlabel("Horizontal velocity [m/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dU/dz
    plt.figure(figsize=(14,8))
    plt.plot(du_dz,z_zi[:-4],"r")
    plt.plot(du_dz_2,z_zi_2[:-4],"--b")
    plt.xlabel("Horizontal velocity gradient [1/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #W
    plt.figure(figsize=(14,8))
    plt.plot(w,z_zi,"r")
    plt.plot(w_2,z_zi_2,"--b")
    plt.xlabel("Vertical velocity [m/s]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #Coriolis twist
    plt.figure(figsize=(14,8))
    plt.plot(np.degrees(twist),z_zi,"r")
    plt.plot(np.degrees(twist_2),z_zi_2,"--b")
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.xlabel("Flow angle [deg]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    f = interpolate.interp1d(z,twist)
    f2 = interpolate.interp1d(z_2,twist_2)
    twist_hub = f(90)
    twist_hub_2 = f2(90)
    plt.text(26,90/glob_zi,"{}deg".format(round(np.degrees(twist_hub),0)),fontsize=14)
    plt.text(26,100/glob_zi_2,"{}deg".format(round(np.degrees(twist_hub_2),0)),fontsize=14)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #u'u'_r
    plt.figure(figsize=(14,8))
    plt.plot(u_u_r,z_zi,"r")
    plt.plot(u_u_r_2,z_zi_2,"--b")
    plt.xlabel("$(u'u')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #v'v'_r
    plt.figure(figsize=(14,8))
    plt.plot(v_v_r,z_zi,"r")
    plt.plot(v_v_r_2,z_zi_2,"--b")
    plt.xlabel("$(v'v')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #w'w'_r
    plt.figure(figsize=(14,8))
    plt.plot(w_w_r,z_zi,"r")
    plt.plot(w_w_r_2,z_zi_2,"--b")
    plt.xlabel("$(w'w')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #u'w'_r
    plt.figure(figsize=(14,8))
    plt.plot(u_w_r,z_zi,"r")
    plt.plot(u_w_r_2,z_zi_2,"--b")
    plt.xlabel("$(u'w')^r [m^2/s^2]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #Theta
    plt.figure(figsize=(14,8))
    plt.plot(theta,z_zi,"r")
    plt.plot(theta_2,z_zi_2,"--b")
    plt.xlabel("Potential temperature [K]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #dTheta/dz
    plt.figure(figsize=(14,8))
    plt.plot(dtheta_dz,z_zi[:-4],"r")
    plt.plot(dtheta_dz_2,z_zi_2[:-4],"--b")
    plt.xlabel("Potential temperature gradient [K/m]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    #w'Theta'_r
    plt.figure(figsize=(14,8))
    plt.plot(w_theta,z_zi,"r")
    plt.plot(w_theta_2,z_zi_2,"--b")
    plt.xlabel("$(w'\\theta')^r [Km/s]$",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    plt.figure(figsize=(14,8))
    plt.plot(phi_m,z_zi[:-4],"r")
    plt.plot(phi_m_2,z_zi_2[:-4],"--b")
    plt.xlabel("$\\phi_m$ [-]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(phi_h,z_zi[:-4],"r")
    plt.plot(phi_h_2,z_zi_2[:-4],"--b")
    plt.xlabel("$\\phi_h$ [-]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(phi_m,z_zi[:-4],"r")
    plt.plot(phi_m_2,z_zi_2[:-4],"--b")
    plt.xlabel("$\\phi_m$ [-]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.ylim([0,0.2]); plt.xlim([0,2])
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(14,8))
    plt.plot(phi_h,z_zi[:-4],"r")
    plt.plot(phi_h_2,z_zi_2[:-4],"--b")
    plt.xlabel("$\\phi_h$ [-]",fontsize=16)
    plt.ylabel("$z/z_i$ [-]",fontsize=16)
    plt.title("Original time averaged over 1200s, \nnew precursor averaged over 2400s",fontsize=18)
    plt.legend(["new Precursor","Original Precursor"])
    plt.ylim([0,0.2]); plt.xlim([-1,0.5])
    plt.tight_layout()
    pdf.savefig()
    plt.close()