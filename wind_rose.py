from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from windrose import WindroseAxes

def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    return P,X, round(mu,2), round(sd,2)


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")
Time_OF = np.array(a.variables["time_OF"])

Azimuth = np.radians(np.array(a.variables["Azimuth"]))

RtAeroFyh = np.array(a.variables["RtAeroFyh"])
RtAeroFzh = np.array(a.variables["RtAeroFzh"])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)

RtAeroMyh = np.array(a.variables["RtAeroMyh"])
RtAeroMzh = np.array(a.variables["RtAeroMzh"])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

L1 = 1.912; L2 = 2.09

Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Aero_Theta = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Aero_Theta = theta_360(Aero_Theta)
#Aero_Theta = np.radians(np.array(Aero_Theta))

#plot windrose of Bearing Force

ax = WindroseAxes.from_ax()
ax.bar(Aero_Theta,Aero_FBR/1000,normed=True,opening=0.8,edgecolor="white")
ax.set_xticklabels(['0', '45', '90', '135',  '180', '225', '270', '315'])
ax.set_legend()

X,P,mu,std = probability_dist(Aero_Theta)
plt.figure(figsize=(14,8))
plt.plot(P,X)

plt.show()