import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
from matplotlib.animation import FuncAnimation


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z

in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

a = Dataset(in_dir+"OF_Dataset.nc")


out_dir = in_dir + "polar_plots/"

Time_OF = np.array(a.variables["time_OF"])

Time_start = 100
Time_end = 1990

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

Azimuth = np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx])
Azimuth = np.radians(Azimuth)

RtAeroFyh = np.array(a.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(a.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)

RtAeroFR = np.sqrt( np.add( np.square(RtAeroFys), np.square(RtAeroFzs) ) )

RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 

LSSGagMys = np.array(a.variables["LSSGagMys"][Time_start_idx:Time_end_idx])
LSSGagMzs = np.array(a.variables["LSSGagMzs"][Time_start_idx:Time_end_idx])
LSSGagMR = np.sqrt( np.add(np.square(LSSGagMys), np.square(LSSGagMzs)) )

LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
LSShftFR = np.sqrt( np.add(np.square(LSShftFys), np.square(LSShftFzs)) )

Theta_AeroF = np.degrees(np.arctan2(RtAeroFzs,RtAeroFys))
Theta_AeroF = theta_360(Theta_AeroF)
Theta_AeroM = np.degrees(np.arctan2(RtAeroMzs,RtAeroMys))
Theta_AeroM = theta_360(Theta_AeroM)
Theta_LSSTipF = np.degrees(np.arctan2(LSShftFzs,LSShftFys))
Theta_LSSTipF = theta_360(Theta_LSSTipF)
Theta_LSSTipM = np.degrees(np.arctan2(LSSTipMzs,LSSTipMys))
Theta_LSSTipM = theta_360(Theta_LSSTipM)
Theta_GagM = np.degrees(np.arctan2(LSSGagMzs,LSSGagMys))
Theta_GagM = theta_360(Theta_GagM)

def update(i):
    plt.cla()
    c = ax.scatter(x_var[i], y_var[i], c="k", s=20)
    ax.arrow(0, 0, x_var[i], y_var[i], length_includes_head=True)
    ax.set_ylim(0,np.max(y_var))
    ax.set_title("{} {}\nTime = {}s".format(Ylabels[j],units[j],Time_OF[i]), va='bottom')


Variables = ["AeroF", "AeroM", "LSSTipF", "LSSTipM","LSSGagM"]
units = ["[kN]","[kN-m]","[kN]","[kN-m]","[kN-m]"]
Ylabels = ["Rotor Aerodynamic Force", "Rotor Aerodynamic Moment", "Rotor Aeroelastic Force", "Rotor Aeroelastic Moment", "LSS Moment"]
x_vars = [Theta_AeroF, Theta_AeroM, Theta_LSSTipF, Theta_LSSTipM, Theta_GagM]
y_vars = [RtAeroFR/1000, RtAeroMR/1000, LSShftFR, LSSTipMR, LSSGagMR]
for j in np.arange(0,len(x_vars)):


    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='polar')

    x_var = x_vars[j]; y_var = y_vars[j]

    ani = FuncAnimation(plt.gcf(), update, interval=1,frames=len(Time_OF))

    ani.save(filename=out_dir+"{}.gif".format(Variables[j]), writer="pillow")
