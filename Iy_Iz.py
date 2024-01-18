import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy.signal import butter,filtfilt
from scipy import interpolate


def low_pass_filter(signal, cutoff):  
    
    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    dX = X[1] - X[0]
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    print(np.sum(P)*dX)
    return P,X, round(mu,2), round(sd,2)


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


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
Rel_Aero_FBy = np.true_divide(np.square(Aero_FBy),np.square(Aero_FBR))
Rel_Aero_FBz = np.true_divide(np.square(Aero_FBz),np.square(Aero_FBR))
add_Aero_RelFB = np.add(Rel_Aero_FBy,Rel_Aero_FBz)
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Theta_Aero_FB = theta_360(Theta_Aero_FB)

Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

offset = "5.5"
group = a.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Uz = np.array(group.variables["Uz"])
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

out_dir = in_dir+"PDFs/"

# fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(14,8),sharex=True)

# ax1.plot(Time_sampling,IA)
# ax1.set_ylabel("Asymmetry parameter")

# ax2.plot(Time_sampling,abs(Iy))

# ax2.plot(Time_sampling,abs(Iz))
# ax2.set_ylabel("Asymmetry")
# ax2.legend(["around y axis", "around z axis"])

# ax3.plot(Time_OF, Aero_FBR)
# ax3.set_ylabel("Bearing Force magnitude")
# #ax2.yaxis.label.set_color("red")

# ax1.grid()
# ax2.grid()
# ax3.grid()
# plt.xlabel("Time [s]",fontsize=16)
# plt.xticks(np.arange(0,1250,50))
# plt.suptitle("offset = -{}m".format(offset))
# plt.tight_layout()


fig = plt.figure(figsize=(14,8))
plt.plot(Time_OF, Theta_Aero_FB)
plt.xlabel("Time [s]",fontsize=16)
plt.ylabel("Direction bearing force vector [deg]",fontsize=16)
plt.title("0deg horizontal right",fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig(in_dir+"line_plots/theta.png")
plt.cla()

P,X,mu,std = probability_dist(Theta_Aero_FB)
fig = plt.figure(figsize=(14,8))
plt.plot(X,P)
plt.axvline(mu,linestyle="--")
plt.xlabel("Direction bearing force vector [deg]",fontsize=16)
plt.ylabel("PDF",fontsize=16)
plt.title("0deg horizontal right",fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"theta.png")
plt.cla()

theta_idx = np.searchsorted(X,180)
fig = plt.figure(figsize=(14,8))
plt.plot(X[theta_idx:],P[theta_idx:])
plt.xlabel("Direction bearing force vector [deg]",fontsize=16)
plt.ylabel("PDF",fontsize=16)
plt.title("0deg horizontal right",fontsize=14)
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"theta_180.png")
plt.cla()

dtheta = X[1] - X[0]
theta_225_idx = np.searchsorted(X,225); theta_315_idx = np.searchsorted(X,315); theta_359_idx = np.searchsorted(X,360)
prob_vert = np.sum(P[theta_225_idx:theta_315_idx])*dtheta
prob_horz = (np.sum(P[theta_idx:theta_225_idx])+np.sum(P[theta_315_idx:theta_359_idx]))*dtheta
prob_neg = np.sum(P[:theta_idx])*dtheta
print(prob_horz+prob_neg+prob_vert)
print("probability vector direction is primarily vertical = ",prob_vert)
print("probability vector direction is primarily horizontal = ",prob_horz)
print("probability vector direction is vertical = ",prob_neg)

Py,Xy,muy,stdy = probability_dist(Iy-np.mean(Iy))
Pz,Xz,muz,stdz = probability_dist(Iz-np.mean(Iz))

fig = plt.figure(figsize=(14,8))
plt.plot(Xy,Py,"-b")
plt.plot(Xz,Pz,"-r")
plt.axvline(stdy,linestyle="--",color="b")
plt.axvline(-stdy,linestyle="--",color="b")
plt.axvline(stdz,linestyle="--",color="r")
plt.axvline(-stdz,linestyle="--",color="r")
plt.xlabel("Asymmetry fluctuations",fontsize=16)
plt.ylabel("PDF",fontsize=16)
plt.legend(["around y", "around z"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"asymmetry_flucs.png")
plt.cla()

Py,Xy,muy,stdy = probability_dist(Iy)
dXy = Xy[1] - Xy[0]
prob = 0
for i in np.arange(0,len(Xy)):
    if Xy[i] >= 200000 or Xy[i] <= -200000:
        prob+=Py[i]*dXy
print("probability Iy greater than +/- 200,000 =", prob)

Pz,Xz,muz,stdz = probability_dist(Iz)
dXz = Xz[1] - Xz[0]
prob = 0
for i in np.arange(0,len(Xz)):
    if Xz[i] >= 200000 or Xz[i] <= -200000:
        prob+=Pz[i]*dXz
print("probability Iz greater than +/- 200,000 =", prob)

fig = plt.figure(figsize=(14,8))
plt.plot(Xy,Py,"-b")
plt.plot(Xz,Pz,"-r")
plt.axvline(muy,linestyle="--",color="b")
plt.axvline(muz,linestyle="--",color="r")
plt.xlabel("Asymmetry",fontsize=16)
plt.ylabel("PDF",fontsize=16)
plt.legend(["around y", "around z"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"asymmetry.png")
plt.cla()

Py,Xy,muy,stdy = probability_dist(Aero_FBy-np.mean(Aero_FBy))
Pz,Xz,muz,stdz = probability_dist(Aero_FBz-np.mean(Aero_FBz))

fig = plt.figure(figsize=(14,8))
plt.plot(Xy,Py,"-r")
plt.plot(Xz,Pz,"-b")
plt.axvline(stdy,linestyle="--",color="r")
plt.axvline(-stdy,linestyle="--",color="r")
plt.axvline(stdz,linestyle="--",color="b")
plt.axvline(-stdz,linestyle="--",color="b")
plt.xlabel("Bearing force components fluctuations",fontsize=16)
plt.ylabel("PDF",fontsize=16)
plt.legend(["FBy", "FBz"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"bearing_force_comps_fluc.png")
plt.cla()

Py,Xy,muy,stdy = probability_dist(Aero_FBy)
dXy = Xy[1] - Xy[0]
prob = 0
for i in np.arange(0,len(Xy)):
    if Xy[i] >= 1000000 or Xy[i] <= -1000000:
        prob+=Py[i]*dXy
print("probability Aero FBy greater than +/- 1MN =", prob)

Pz,Xz,muz,stdz = probability_dist(Aero_FBz)
dXz = Xz[1] - Xz[0]
prob = 0
for i in np.arange(0,len(Xz)):
    if Xz[i] >= 1000000 or Xz[i] <= -1000000:
        prob+=Pz[i]*dXz
print("probability Aero FBz greater than +/- 1MN =", prob)

fig = plt.figure(figsize=(14,8))
plt.plot(Xy,Py,"-r")
plt.plot(Xz,Pz,"-b")
plt.axvline(muy,linestyle="--",color="r")
plt.axvline(muz,linestyle="--",color="b")
plt.xlabel("Bearing force components",fontsize=16)
plt.ylabel("PDF",fontsize=16)
plt.legend(["FBy", "FBz"])
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"bearing_force_comps.png")
plt.cla()

out_dir = in_dir+"correlations/"

Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]

Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[:Time_end_idx]

f = interpolate.interp1d(Time_sampling,Iy)
Iy = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iz)
Iz = f(Time_OF)

corr_Iy = correlation_coef(Iy,Aero_FBz)
corr_Iz = correlation_coef(Iz,Aero_FBy)

fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,Iy,"b")
ax.set_ylabel("Asymmetry around y axis [$m^4/s$]")
ax.xaxis.label.set_color("b")
ax2 = ax.twinx()
ax2.plot(Time_OF,Aero_FBz,"r")
ax2.set_ylabel("Aerodynamic Bearing force z component",fontsize=16)
ax.xaxis.label.set_color("r")
plt.suptitle("correlation = {}".format(corr_Iy),fontsize=16)
plt.xlabel("Time [s]",fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig(out_dir,"Iy_FBz.png")
plt.cla()

fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,Iz,"b")
ax.set_ylabel("Asymmetry around z axis [$m^4/s$]")
ax.xaxis.label.set_color("b")
ax2 = ax.twinx()
ax2.plot(Time_OF,Aero_FBy,"r")
ax2.set_ylabel("Aerodynamic Bearing force y component",fontsize=16)
ax.xaxis.label.set_color("r")
plt.suptitle("correlation = {}".format(corr_Iz),fontsize=16)
plt.xlabel("Time [s]",fontsize=16)
plt.grid()
plt.tight_layout()
plt.savefig(out_dir,"Iz_FBy.png")
plt.cla()