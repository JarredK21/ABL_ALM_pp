from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import interpolate

def correlation_coef(x,y):

    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r

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
    return P,X


def low_pass_filter(signal, cutoff,dt):

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


plt.rcParams['font.size'] = 16
in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
out_dir=in_dir+"Asymmetry_analysis/Separating_eddy_types_analysis/"
a = Dataset(in_dir+"Asymmetry_Dataset.nc")
Time = np.array(a.variables["time"])
Time = Time - Time[0]
dt = Time[1]-Time[0]

Iy_high = np.array(a.variables["Iy_high"])
Iz_high = np.array(a.variables["Iz_high"])
I_high = np.sqrt(np.add(np.square(Iy_high),np.square(Iz_high)))
A_high = np.array(a.variables["Area_high"])

idx = np.searchsorted(Time,Time[0]+20)
cc = []
for it in np.arange(0,len(Time)-idx):
    cc.append(correlation_coef(I_high[it:it+idx],A_high[it:it+idx]))


A_rotor = np.pi * 63**2
Thresholds = [0,0.15,0.3,0.5,0.7,0.85,1.0]
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(Thresholds)-1):
    X = []
    for it in np.arange(0,len(Time)):
        if Thresholds[i] < A_high[it]/A_rotor <= Thresholds[i+1]:
            X.append(I_high[it])
    print(np.mean(X))
    P,X = probability_dist(X)
    plt.plot(X,P,label="Area {} - {}$m^2$".format(Thresholds[i],Thresholds[i+1]))
plt.xlabel("Magnitude Asymmetry vector\nHigh speed areas [$m^4/s$]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.savefig(out_dir+"PDF_threshold_High_Area_I.png")
plt.close()

fig,(ax,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax.plot(Time,I_high,"-r")
ax.yaxis.label.set_color('red')
ax.set_ylabel("Magnitude Asymmetry vector\n(high speed areas) [$m^4/s$]")
ax2=ax.twinx()
ax2.plot(Time,A_high,"-b")
ax2.yaxis.label.set_color('blue')
ax2.set_ylabel("Area (high speed) [$m^2$]")
ax.grid()
ax3.plot(Time[idx:],cc,"-k")
ax3.set_ylabel("Local correlation\ncoefficient T = 20s")
fig.supxlabel("Time [s]")
fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(I_high,A_high),2)))
plt.tight_layout()
plt.savefig(out_dir+"I_high_cc_A_high.png")
plt.close()


Iy_low = np.array(a.variables["Iy_low"])
Iz_low = np.array(a.variables["Iz_low"])
I_low = np.sqrt(np.add(np.square(Iy_low),np.square(Iz_low)))
A_low = np.array(a.variables["Area_low"])

idx = np.searchsorted(Time,Time[0]+20)
cc = []
for it in np.arange(0,len(Time)-idx):
    cc.append(correlation_coef(I_low[it:it+idx],A_low[it:it+idx]))

A_rotor = np.pi * 63**2
Thresholds = [0,0.15,0.3,0.5,0.7,0.85,1.0]
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(Thresholds)-1):
    X = []
    for it in np.arange(0,len(Time)):
        if Thresholds[i] < A_low[it]/A_rotor <= Thresholds[i+1]:
            X.append(I_low[it])
    print(np.mean(X))
    P,X = probability_dist(X)
    plt.plot(X,P,label="Area {} - {}$m^2$".format(Thresholds[i],Thresholds[i+1]))
plt.xlabel("Magnitude Asymmetry vector\nLow speed areas [$m^4/s$]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.savefig(out_dir+"PDF_threshold_Low_Area_I.png")
plt.close()

fig,(ax,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax.plot(Time,I_low,"-r")
ax.yaxis.label.set_color('red')
ax.set_ylabel("Magnitude Asymmetry vector\n(low speed areas) [$m^4/s$]")
ax2=ax.twinx()
ax2.plot(Time,A_low,"-b")
ax2.yaxis.label.set_color('blue')
ax2.set_ylabel("Area (low speed) [$m^2$]")
ax.grid()
ax3.plot(Time[idx:],cc,"-k")
ax3.set_ylabel("Local correlation\ncoefficient T = 20s")
fig.supxlabel("Time [s]")
fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(I_low,A_low),2)))
plt.tight_layout()
plt.savefig(out_dir+"I_low_cc_A_low.png")
plt.close()



Iy_int = np.array(a.variables["Iy_int"])
Iz_int = np.array(a.variables["Iz_int"])
I_int = np.sqrt(np.add(np.square(Iy_int),np.square(Iz_int)))
A_int = np.array(a.variables["Area_int"])

idx = np.searchsorted(Time,Time[0]+20)
cc = []
for it in np.arange(0,len(Time)-idx):
    cc.append(correlation_coef(I_int[it:it+idx],A_int[it:it+idx]))

A_rotor = np.pi * 63**2
Thresholds = [0,0.15,0.3,0.5,0.7,0.85,1.0]
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(Thresholds)-1):
    X = []
    for it in np.arange(0,len(Time)):
        if Thresholds[i] < A_int[it]/A_rotor <= Thresholds[i+1]:
            X.append(I_int[it])
    print(np.mean(X))
    P,X = probability_dist(X)
    plt.plot(X,P,label="Area {} - {}$m^2$".format(Thresholds[i],Thresholds[i+1]))
plt.xlabel("Magnitude Asymmetry vector\nIntermediate speed areas [$m^4/s$]")
plt.ylabel("Probability [-]")
plt.grid()
plt.legend()
plt.savefig(out_dir+"PDF_threshold_Int_Area_I.png")
plt.close()

fig,(ax,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax.plot(Time,I_int,"-r")
ax.yaxis.label.set_color('red')
ax.set_ylabel("Magnitude Asymmetry vector\n(Intermediate speed areas) [$m^4/s$]")
ax2=ax.twinx()
ax2.plot(Time,A_int,"-b")
ax2.yaxis.label.set_color('blue')
ax2.set_ylabel("Area (Intermediate speed) [$m^2$]")
ax.grid()
ax3.plot(Time[idx:],cc,"-k")
ax3.set_ylabel("Local correlation\ncoefficient T = 20s")
fig.supxlabel("Time [s]")
fig.suptitle("Correlation coefficient = {}".format(round(correlation_coef(I_int,A_int),2)))
plt.tight_layout()
plt.savefig(out_dir+"I_int_cc_A_int.png")
plt.close()


df_OF = Dataset(in_dir+"Dataset.nc")

Time_sampling = np.array(df_OF.variables["time_sampling"])
Time_OF = np.array(df_OF.variables["time_OF"])
Time_sampling[-1] = Time_OF[-1]
dt = Time_OF[1]-Time_OF[0]

Azimuth = np.radians(np.array(df_OF.variables["Azimuth"]))

RtAeroFyh = np.array(df_OF.variables["RtAeroFyh"])
RtAeroFzh = np.array(df_OF.variables["RtAeroFzh"])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(df_OF.variables["RtAeroMyh"])
RtAeroMzh = np.array(df_OF.variables["RtAeroMzh"])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

LPF_FBR = low_pass_filter(FBR,0.3,dt)

group = df_OF.groups["5.5"]

Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])
I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
f = interpolate.interp1d(Time_sampling,I)
I_interp = f(Time_OF)

print(correlation_coef(LPF_FBR,I_interp))

fig,(ax1,ax2,ax3,ax4) = plt.subplots(4,1,figsize=(14,8),sharex=True)
ax1.plot(Time,A_high/A_rotor)
ax1.axhline(y=0.3,linestyle="--")
ax1.axhline(y=0.7,linestyle="--")
ax1.grid()
ax1.set_title("High speed area [$m^2$]")
ax2.plot(Time,A_low/A_rotor)
ax2.axhline(y=0.3,linestyle="--")
ax2.axhline(y=0.7,linestyle="--")
ax2.set_title("Low s[eed area [$m^2$]")
ax2.grid()
ax3.plot(Time,A_int/A_rotor)
ax3.axhline(y=0.3,linestyle="--")
ax3.axhline(y=0.7,linestyle="--")
ax3.set_title("Intermediate Area [$m^2$]")
ax3.grid()
ax4.plot(Time_OF,LPF_FBR,"-r")
ax5=ax4.twinx()
ax5.plot(Time_sampling,I,"-b")
ax4.grid()
ax4.set_title("Low pass filtered $F_{B_R}$ [kN]")
plt.tight_layout()
plt.savefig(out_dir+"Area_FBR.png")