from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import interpolate


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

a = Dataset(in_dir+"Dataset.nc")

print(a)

Time = np.array(a.variables["Time_sampling"])
Time_start_idx = np.searchsorted(Time,200)
Time = Time[Time_start_idx:]
dt_sampling = Time[1] - Time[0]

Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]

Iy = np.array(Rotor_avg_vars.variables["Iy"][Time_start_idx:])
Iz = np.array(Rotor_avg_vars.variables["Iz"][Time_start_idx:])
I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

Rotor_gradients = a.groups["Rotor_Gradients"]

dyUx = np.array(Rotor_gradients.variables["dyUx"][Time_start_idx:])
dzUx = np.array(Rotor_gradients.variables["dzUx"][Time_start_idx:])
drUx = np.array(Rotor_gradients.variables["drUx"][Time_start_idx:])



Split_rotor_vars = a.groups["Split_rotor_Variables"]

IyL = np.array(Split_rotor_vars.variables["IyL"][Time_start_idx:])
IzL = np.array(Split_rotor_vars.variables["IzL"][Time_start_idx:])
IL = np.sqrt(np.add(np.square(IyL),np.square(IzL)))

IyM = np.array(Split_rotor_vars.variables["IyM"][Time_start_idx:])
IzM = np.array(Split_rotor_vars.variables["IzM"][Time_start_idx:])
IM = np.sqrt(np.add(np.square(IyM),np.square(IzM)))

IyH = np.array(Split_rotor_vars.variables["IyH"][Time_start_idx:])
IzH = np.array(Split_rotor_vars.variables["IzH"][Time_start_idx:])
IH = np.sqrt(np.add(np.square(IyH),np.square(IzH)))

dyUxL = np.array(Split_rotor_vars.variables["dyUxL"][Time_start_idx:])
dzUxL = np.array(Split_rotor_vars.variables["dzUxL"][Time_start_idx:])
drUxL = np.array(Split_rotor_vars.variables["drUxL"][Time_start_idx:])

dyUxM = np.array(Split_rotor_vars.variables["dyUxM"][Time_start_idx:])
dzUxM = np.array(Split_rotor_vars.variables["dzUxM"][Time_start_idx:])
drUxM = np.array(Split_rotor_vars.variables["drUxM"][Time_start_idx:])

dyUxH = np.array(Split_rotor_vars.variables["dyUxH"][Time_start_idx:])
dzUxH = np.array(Split_rotor_vars.variables["dzUxH"][Time_start_idx:])
drUxH = np.array(Split_rotor_vars.variables["drUxH"][Time_start_idx:])

print(correlation_coef(dyUx,dyUxL))
print(correlation_coef(dzUx,dzUxM))
print(correlation_coef(drUx,drUxH))


# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,IyL,"-b",label="lower 1/3\ncc = {}".format(correlation_coef(Iy,IyL)))
# plt.plot(Time,IyM,"-g",label="middle 1/3\ncc = {}".format(correlation_coef(Iy,IyM)))
# plt.plot(Time,IyH,"-r",label="upper 1/3\ncc = {}".format(correlation_coef(Iy,IyH)))
# plt.plot(Time,Iy,"-k",label="Total")
# plt.xlabel("Time [s]")
# plt.ylabel("Asymmetry around y axis [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()

# print(correlation_coef(IyL,IyM))
# print(correlation_coef(IyL,IyH))
# print(correlation_coef(IyM,IyH))

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time,IyM,"-g")
# ax.set_ylabel("middle 1/3 Asymmetry around y axis [$m^4/s$]")
# ax2=ax.twinx()
# ax2.plot(Time,Iy,"-k")
# ax2.set_ylabel("Total asymmetry around y axis [$m^4/s$]")
# fig.supxlabel("Time [s]")
# ax.grid()
# plt.tight_layout()


# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(IyL)
# plt.plot(X,P,"-b",label="lower 1/3\n{}".format(moments(IyL)))
# P,X = probability_dist(IyM)
# plt.plot(X,P,"-g",label="middle 1/3\n{}".format(moments(IyM)))
# P,X = probability_dist(IyH)
# plt.plot(X,P,"-r",label="upper 1/3\n{}".format(moments(IyH)))
# plt.ylabel("Probability [-]")
# plt.xlabel("Asymmetry around y axis [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(IyL,dt_sampling,Var="IyL")
# plt.loglog(frq,PSD,"-b",label="lower 1/3")
# frq,PSD = temporal_spectra(IyM,dt_sampling,Var="IyM")
# plt.loglog(frq,PSD,"-g",label="middle 1/3")
# frq,PSD = temporal_spectra(IyH,dt_sampling,Var="IyH")
# plt.loglog(frq,PSD,"-r",label="upper 1/3")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Asymmetry around y axis [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()


# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,IzL,"-b",label="lower 1/3\ncc = {}".format(correlation_coef(Iz,IzL)))
# plt.plot(Time,IzM,"-g",label="middle 1/3\ncc = {}".format(correlation_coef(Iz,IzM)))
# plt.plot(Time,IzH,"-r",label="upper 1/3\ncc = {}".format(correlation_coef(Iz,IzH)))
# plt.plot(Time,Iz,"-k",label="Total")
# plt.xlabel("Time [s]")
# plt.ylabel("Asymmetry around z axis [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()

# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(IzL)
# plt.plot(X,P,"-b",label="lower 1/3\n{}".format(moments(IzL)))
# P,X = probability_dist(IzM)
# plt.plot(X,P,"-g",label="middle 1/3\n{}".format(moments(IzM)))
# P,X = probability_dist(IzH)
# plt.plot(X,P,"-r",label="upper 1/3\n{}".format(moments(IzH)))
# plt.ylabel("Probability [-]")
# plt.xlabel("Asymmetry around z axis [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(IzL,dt_sampling,Var="IzL")
# plt.loglog(frq,PSD,"-b",label="lower 1/3")
# frq,PSD = temporal_spectra(IzM,dt_sampling,Var="IzM")
# plt.loglog(frq,PSD,"-g",label="middle 1/3")
# frq,PSD = temporal_spectra(IzH,dt_sampling,Var="IzH")
# plt.loglog(frq,PSD,"-r",label="upper 1/3")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Asymmetry around z axis [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,IL,"-b",label="lower 1/3\ncc = {}".format(correlation_coef(I,IL)))
# plt.plot(Time,IM,"-g",label="middle 1/3\ncc = {}".format(correlation_coef(I,IM)))
# plt.plot(Time,IH,"-r",label="upper 1/3\ncc = {}".format(correlation_coef(I,IH)))
# plt.plot(Time,I,"-k",label="Total")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude Asymmetry vector [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()

# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(IL)
# plt.plot(X,P,"-b",label="lower 1/3\n{}".format(moments(IL)))
# P,X = probability_dist(IM)
# plt.plot(X,P,"-g",label="middle 1/3\n{}".format(moments(IM)))
# P,X = probability_dist(IH)
# plt.plot(X,P,"-r",label="upper 1/3\n{}".format(moments(IH)))
# plt.ylabel("Probability [-]")
# plt.xlabel("Magnitude Asymmetry vector [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(IL,dt_sampling,Var="IL")
# plt.loglog(frq,PSD,"-b",label="lower 1/3")
# frq,PSD = temporal_spectra(IM,dt_sampling,Var="IM")
# plt.loglog(frq,PSD,"-g",label="middle 1/3")
# frq,PSD = temporal_spectra(IH,dt_sampling,Var="IH")
# plt.loglog(frq,PSD,"-r",label="upper 1/3")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Magnitude Asymmetry vector [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()



fig = plt.figure(figsize=(14,8))
plt.plot(Time,dyUxL,"-b",label="lower 1/3")
plt.plot(Time,dyUxM,"-g",label="middle 1/3")
plt.plot(Time,dyUxH,"-r",label="upper 1/3")
plt.xlabel("Time [s]")
plt.ylabel("Velocity gradient y component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()


fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dyUxL)
plt.plot(X,P,"-b",label="lower 1/3")
P,X = probability_dist(dyUxM)
plt.plot(X,P,"-g",label="middle 1/3")
P,X = probability_dist(dyUxH)
plt.plot(X,P,"-r",label="upper 1/3")
plt.ylabel("Probability [-]")
plt.xlabel("Velocity gradient y component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()

fig = plt.figure(figsize=(14,8))
plt.plot(Time,dzUxL,"-b",label="lower 1/3")
plt.plot(Time,dzUxM,"-g",label="middle 1/3")
plt.plot(Time,dzUxH,"-r",label="upper 1/3")
plt.xlabel("Time [s]")
plt.ylabel("Velocity gradient z component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dzUxL)
plt.plot(X,P,"-b",label="lower 1/3")
P,X = probability_dist(dzUxM)
plt.plot(X,P,"-g",label="middle 1/3")
P,X = probability_dist(dzUxH)
plt.plot(X,P,"-r",label="upper 1/3")
plt.ylabel("Probability [-]")
plt.xlabel("Velocity gradient z component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()

fig = plt.figure(figsize=(14,8))
plt.plot(Time,drUxL,"-b",label="lower 1/3")
plt.plot(Time,drUxM,"-g",label="middle 1/3")
plt.plot(Time,drUxH,"-r",label="upper 1/3")
plt.xlabel("Time [s]")
plt.ylabel("Velocity gradient Magntiude [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(drUxL)
plt.plot(X,P,"-b",label="lower 1/3")
P,X = probability_dist(drUxM)
plt.plot(X,P,"-g",label="middle 1/3")
P,X = probability_dist(drUxH)
plt.plot(X,P,"-r",label="upper 1/3")
plt.ylabel("Probability [-]")
plt.xlabel("Velocity gradient Magntiude [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()

plt.show()




# Rotor_Gradients = a.groups["Rotor_Gradients"]

# dyUx = np.array(Rotor_Gradients.variables["dyUx"][Time_start_idx:])
# dzUx = np.array(Rotor_Gradients.variables["dzUx"][Time_start_idx:])
# drUx = np.array(Rotor_Gradients.variables["drUx"][Time_start_idx:])

    
# OpenFAST_Variables = a.groups["OpenFAST_Variables"]

# Time_OF = np.array(a.variables["Time_OF"])
# Time_start_idx = np.searchsorted(Time_OF,200)
# Time_OF = Time_OF[Time_start_idx:]
# dt = Time_OF[1]-Time_OF[0]

# Azimuth = np.radians(np.array(OpenFAST_Variables.variables["Azimuth"][Time_start_idx:]))

# RtAeroFyh = np.array(OpenFAST_Variables.variables["RtAeroFyh"][Time_start_idx:])
# RtAeroFzh = np.array(OpenFAST_Variables.variables["RtAeroFzh"][Time_start_idx:])

# RtAeroFys = []; RtAeroFzs = []
# for i in np.arange(0,len(Time_OF)):
#     RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
#     RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
# RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


# RtAeroMyh = np.array(OpenFAST_Variables.variables["RtAeroMyh"][Time_start_idx:])
# RtAeroMzh = np.array(OpenFAST_Variables.variables["RtAeroMzh"][Time_start_idx:])

# RtAeroMys = []; RtAeroMzs = []
# for i in np.arange(0,len(Time_OF)):
#     RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
#     RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
# RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

# #Total radial aerodynamic bearing force aeroFBR
# L1 = 1.912; L2 = 2.09

# FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
# FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

# FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


# FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

# LPF_FBR = low_pass_filter(FBR,0.3,dt)

# LPF_1_FBR = low_pass_filter(FBR,0.3,dt)
# LPF_2_FBR = low_pass_filter(FBR,0.9,dt)
# LPF_3_FBR = low_pass_filter(FBR,1.5,dt)

# HPF_FBR = np.subtract(FBR,LPF_3_FBR)
# HPF_FBR = np.array(low_pass_filter(HPF_FBR,40,dt))
# BPF_FBR = np.subtract(LPF_2_FBR,LPF_1_FBR)


# dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))

# zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

# BPF_FBR_2 = []
# Times_2 = []
# for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
#     idx = zero_crossings_index_BPF_FBR[i]
#     BPF_FBR_2.append(BPF_FBR[idx]); Times_2.append(Time_OF[idx])


# f = interpolate.interp1d(Time,drUx)
# dr_Ux_interp = f(Times_2)

# f = interpolate.interp1d(Time,dyUx)
# dy_Ux_interp = f(Times_2)

# f = interpolate.interp1d(Time,dzUx)
# dz_Ux_interp = f(Times_2)


# # idx = np.searchsorted(Times_2,Times_2[0]+20)
# # cc = []
# # for it in np.arange(0,len(Times_2)-idx):
# #     cc.append(correlation_coef(BPF_FBR_2[it:it+idx],dr_Ux_interp[it:it+idx]))


# # fig = plt.figure(figsize=(14,8))
# # plt.plot(Times_2[idx:],cc,"-k")
# # plt.xlabel("Time [s]")
# # plt.ylabel("CC")
# # plt.grid()
# # plt.tight_layout()


# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,dyUx)
# plt.grid()
# plt.ylabel("dyUx")

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,dzUx)
# plt.grid()
# plt.ylabel("dzUx")

# fig,(ax,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
# ax.plot(Time,drUx,"-b")
# ax.grid()
# ax.set_ylabel("drUx")

# ax2.plot(Times_2,BPF_FBR_2,"-r")
# ax2.set_ylabel("BPF FBR")
# ax2.grid()
# fig.suptitle("{}".format(correlation_coef(dr_Ux_interp,BPF_FBR_2)))

# P,X=probability_dist(dyUx)
# fig = plt.figure(figsize=(14,8))
# plt.plot(X,P,label="dyUx\n{}".format(moments(dyUx)))
# plt.grid()
# plt.xlabel("Gradient")

# P,X=probability_dist(dzUx)
# plt.plot(X,P,label="dzUx\n{}".format(moments(dzUx)))
# plt.grid()
# plt.legend()


# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(dyUx,dt_sampling,Var="dyUx")
# plt.loglog(frq,PSD,"-r",label="$\langle du_{x'}/dy \\rangle_A$")
# frq,PSD = temporal_spectra(dzUx,dt_sampling,Var="dzUx")
# plt.loglog(frq,PSD,"-b",label="$\langle du_{x'}/dz \\rangle_A$")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Rotor averaged gradient [1/s]")
# plt.legend()
# plt.grid()
# plt.tight_layout()


# plt.show()