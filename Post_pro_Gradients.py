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


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time_OF)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt


def moments(y):
    mu = np.mean(y)
    std = np.std(y)
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    return mu, std, skewness,kurotsis


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset_gradients.nc")
Time = np.array(a.variables["time_sampling"])
Time_start_idx = np.searchsorted(Time,200)
Time = Time[Time_start_idx:]

group = a.groups["63.0"]

dyUx = np.array(group.variables["dyUx"][Time_start_idx:])
dzUx = np.array(group.variables["dzUx"][Time_start_idx:])
drUx = np.sqrt(np.add(np.square(dyUx),np.square(dzUx)))



df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["time_OF"])
Time_start_idx = np.searchsorted(Time_OF,200)
Time_OF = Time_OF[Time_start_idx:]
dt = Time_OF[1]-Time_OF[0]

Azimuth = np.radians(np.array(df_OF.variables["Azimuth"][Time_start_idx:]))

RtAeroFyh = np.array(df_OF.variables["RtAeroFyh"][Time_start_idx:])
RtAeroFzh = np.array(df_OF.variables["RtAeroFzh"][Time_start_idx:])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(df_OF.variables["RtAeroMyh"][Time_start_idx:])
RtAeroMzh = np.array(df_OF.variables["RtAeroMzh"][Time_start_idx:])

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

LPF_1_FBR = low_pass_filter(FBR,0.3,dt)
LPF_2_FBR = low_pass_filter(FBR,0.9,dt)
LPF_3_FBR = low_pass_filter(FBR,1.5,dt)

HPF_FBR = np.subtract(FBR,LPF_3_FBR)
HPF_FBR = np.array(low_pass_filter(HPF_FBR,40,dt))
BPF_FBR = np.subtract(LPF_2_FBR,LPF_1_FBR)



dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))

zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

BPF_FBR_2 = []
Times_2 = []
for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
    idx = zero_crossings_index_BPF_FBR[i]
    BPF_FBR_2.append(BPF_FBR[idx]); Times_2.append(Time_OF[idx])


f = interpolate.interp1d(Time,drUx)
dr_Ux_interp = f(Times_2)

f = interpolate.interp1d(Time,dyUx)
dy_Ux_interp = f(Times_2)

f = interpolate.interp1d(Time,dzUx)
dz_Ux_interp = f(Times_2)

print(correlation_coef(dy_Ux_interp,BPF_FBR_2))
print(correlation_coef(dz_Ux_interp,BPF_FBR_2))

fig = plt.figure(figsize=(14,8))
plt.plot(Time,dyUx)
plt.grid()
plt.ylabel("dyUx")

fig = plt.figure(figsize=(14,8))
plt.plot(Time,dzUx)
plt.grid()
plt.ylabel("dzUx")

fig,(ax,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax.plot(Time,drUx,"-b")
ax.grid()
ax.set_ylabel("drUx")

ax2.plot(Times_2,BPF_FBR_2,"-r")
ax2.set_ylabel("BPF FBR")
ax2.grid()
fig.suptitle("{}".format(correlation_coef(dr_Ux_interp,BPF_FBR_2)))

P,X=probability_dist(dyUx)
fig = plt.figure(figsize=(14,8))
plt.plot(X,P,label="dyUx\n{}".format(moments(dyUx)))
plt.grid()
plt.xlabel("Gradient")

P,X=probability_dist(dzUx)
plt.plot(X,P,label="dzUx\n{}".format(moments(dzUx)))
plt.grid()
plt.legend()

plt.show()