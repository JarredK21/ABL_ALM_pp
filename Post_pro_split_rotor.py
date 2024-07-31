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

    return round(mu,3), round(std,3), round(skewness,3),round(kurotsis,3)


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

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

plt.rcParams['font.size'] = 12

print(correlation_coef(IyL,IyM))
print(correlation_coef(IyL,IyH))
print(correlation_coef(IyM,IyH))

out_dir=in_dir+"Split_rotor_analysis/"

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,IyL,"-b",label="lower 1/3\ncc = {}".format(round(correlation_coef(Iy,IyL),2)))
# plt.plot(Time,IyM,"-g",label="middle 1/3\ncc = {}".format(round(correlation_coef(Iy,IyM),2)))
# plt.plot(Time,IyH,"-r",label="upper 1/3\ncc = {}".format(round(correlation_coef(Iy,IyH),2)))
# plt.plot(Time,Iy,"-k",label="Total")
# plt.xlabel("Time [s]")
# plt.ylabel("Asymmetry around y axis [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()
# plt.savefig(out_dir+"Iy_comp.png")
# plt.close()


# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time,IyM,"-g")
# ax.set_ylabel("middle 1/3 Asymmetry around y axis [$m^4/s$]")
# ax.yaxis.label.set_color('green')
# ax2=ax.twinx()
# ax2.plot(Time,Iy,"-k")
# ax2.set_ylabel("Total asymmetry around y axis [$m^4/s$]")
# ax2.yaxis.label.set_color('black')
# fig.supxlabel("Time [s]")
# ax.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Iy_IyM.png")
# plt.close()

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time,np.add(IyL,IyH),"-m")
# ax.set_ylabel("summation upper and lower 1/3's Asymmetry around y axis [$m^4/s$]")
# ax2=ax.twinx()
# ax2.plot(Time,Iy,"-k")
# ax2.set_ylabel("Total asymmetry around y axis [$m^4/s$]")
# fig.supxlabel("Time [s]")
# fig.suptitle("cc = {}".format(round(correlation_coef(np.add(IyL,IyH),Iy),2)))
# ax.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"IyL_plus_IyH_Iy.png")
# plt.close()



# # fig,ax = plt.subplots(figsize=(14,8))
# # ax.plot(Time,np.subtract(IyL,np.mean(IyL)),"-b")
# # ax.set_ylabel("lower 1/3 Asymmetry around y axis [$m^4/s$]")
# # ax2=ax.twinx()
# # ax2.plot(Time,np.subtract(IyH,np.mean(IyH)),"-r")
# # ax2.set_ylabel("upper 1/3 Asymmetry around y axis [$m^4/s$]")
# # fig.supxlabel("Time [s]")
# # ax.grid()
# # plt.tight_layout()



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
# plt.savefig(out_dir+"PDF_Iy_comp.png")
# plt.close()

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
# plt.savefig(out_dir+"spectra_Iy_comp.png")
# plt.close()

print(correlation_coef(IzL,IzM))
print(correlation_coef(IzL,IzH))
print(correlation_coef(IzM,IzH))

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,IzL,"-b",label="lower 1/3\ncc = {}".format(round(correlation_coef(Iz,IzL),2)))
# plt.plot(Time,IzM,"-g",label="middle 1/3\ncc = {}".format(round(correlation_coef(Iz,IzM),2)))
# plt.plot(Time,IzH,"-r",label="upper 1/3\ncc = {}".format(round(correlation_coef(Iz,IzH),2)))
# plt.plot(Time,Iz,"-k",label="Total")
# plt.xlabel("Time [s]")
# plt.ylabel("Asymmetry around z axis [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()
# plt.savefig(out_dir+"Iz_comp.png")
# plt.close()

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
# plt.savefig(out_dir+"PDF_Iz_comp.png")
# plt.close()

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
# plt.savefig(out_dir+"spectra_Iz_comp.png")
# plt.close()

print(correlation_coef(IL,IM))
print(correlation_coef(IL,IH))
print(correlation_coef(IM,IH))

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,IL,"-b",label="lower 1/3\ncc = {}".format(round(correlation_coef(I,IL),2)))
# plt.plot(Time,IM,"-g",label="middle 1/3\ncc = {}".format(round(correlation_coef(I,IM),2)))
# plt.plot(Time,IH,"-r",label="upper 1/3\ncc = {}".format(round(correlation_coef(I,IH),2)))
# plt.plot(Time,I,"-k",label="Total")
# plt.xlabel("Time [s]")
# plt.ylabel("Magnitude Asymmetry vector [$m^4/s$]")
# plt.grid()
# plt.tight_layout()
# plt.legend()
# plt.savefig(out_dir+"I_comp.png")
# plt.close()

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
# plt.savefig(out_dir+"PDF_I_comp.png")
# plt.close()

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
# plt.savefig(out_dir+"spectra_I_comp.png")
# plt.close()

print(correlation_coef(dyUxL,dyUxM))
print(correlation_coef(dyUxL,dyUxH))
print(correlation_coef(dyUxM,dyUxH))

fig = plt.figure(figsize=(14,8))
plt.plot(Time,dyUxL,"-b",label="lower 1/3")
plt.plot(Time,dyUxM,"-g",label="middle 1/3")
plt.plot(Time,dyUxH,"-r",label="upper 1/3")
plt.xlabel("Time [s]")
plt.ylabel("Velocity gradient y component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"dyUx_comp.png")
plt.close()


fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dyUxL)
plt.plot(X,P,"-b",label="lower 1/3\n{}".format(moments(dyUxL)))
P,X = probability_dist(dyUxM)
plt.plot(X,P,"-g",label="middle 1/3\n{}".format(moments(dyUxM)))
P,X = probability_dist(dyUxH)
plt.plot(X,P,"-r",label="upper 1/3\n{}".format(moments(dyUxH)))
plt.ylabel("Probability [-]")
plt.xlabel("Velocity gradient y component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"PDF_dyUx_comp.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(dyUxL,dt_sampling,Var="dyUxL")
plt.loglog(frq,PSD,"-b",label="lower 1/3")
frq,PSD = temporal_spectra(dyUxM,dt_sampling,Var="dyUxM")
plt.loglog(frq,PSD,"-g",label="middle 1/3")
frq,PSD = temporal_spectra(dyUxH,dt_sampling,Var="dyUxH")
plt.loglog(frq,PSD,"-r",label="upper 1/3")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Velocity gradient y component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"spectra_dyUx_comp.png")
plt.close()

print("dz correlations")
print(correlation_coef(dzUxL,dzUxM))
print(correlation_coef(dzUxL,dzUxH))
print(correlation_coef(dzUxM,dzUxH))

fig = plt.figure(figsize=(14,8))
plt.plot(Time,dzUxL,"-b",label="lower 1/3")
plt.plot(Time,dzUxM,"-g",label="middle 1/3")
plt.plot(Time,dzUxH,"-r",label="upper 1/3")
plt.xlabel("Time [s]")
plt.ylabel("Velocity gradient z component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"dzUx_comp.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dzUxL)
plt.plot(X,P,"-b",label="lower 1/3\n{}".format(moments(dzUxL)))
P,X = probability_dist(dzUxM)
plt.plot(X,P,"-g",label="middle 1/3\n{}".format(moments(dzUxM)))
P,X = probability_dist(dzUxH)
plt.plot(X,P,"-r",label="upper 1/3\n{}".format(moments(dzUxH)))
plt.ylabel("Probability [-]")
plt.xlabel("Velocity gradient z component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"PDF_dzUx_comp.png")
plt.close()


fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(dzUxL,dt_sampling,Var="dzUxL")
plt.loglog(frq,PSD,"-b",label="lower 1/3")
frq,PSD = temporal_spectra(dzUxM,dt_sampling,Var="dzUxM")
plt.loglog(frq,PSD,"-g",label="middle 1/3")
frq,PSD = temporal_spectra(dzUxH,dt_sampling,Var="dzUxH")
plt.loglog(frq,PSD,"-r",label="upper 1/3")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Velocity gradient z component [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"spectra_dzUx_comp.png")
plt.close()

print("dr corrleations")
print(correlation_coef(drUxL,drUxM))
print(correlation_coef(drUxL,drUxH))
print(correlation_coef(drUxM,drUxH))

fig = plt.figure(figsize=(14,8))
plt.plot(Time,drUxL,"-b",label="lower 1/3")
plt.plot(Time,drUxM,"-g",label="middle 1/3")
plt.plot(Time,drUxH,"-r",label="upper 1/3")
plt.xlabel("Time [s]")
plt.ylabel("Velocity gradient Magntiude [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"drUx_comp.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(drUxL)
plt.plot(X,P,"-b",label="lower 1/3\n{}".format(moments(drUxL)))
P,X = probability_dist(drUxM)
plt.plot(X,P,"-g",label="middle 1/3\n{}".format(moments(drUxM)))
P,X = probability_dist(drUxH)
plt.plot(X,P,"-r",label="upper 1/3\n{}".format(moments(drUxH)))
plt.ylabel("Probability [-]")
plt.xlabel("Velocity gradient Magntiude [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"PDF_drUx_comp.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(drUxL,dt_sampling,Var="drUxL")
plt.loglog(frq,PSD,"-b",label="lower 1/3")
frq,PSD = temporal_spectra(drUxM,dt_sampling,Var="drUxM")
plt.loglog(frq,PSD,"-g",label="middle 1/3")
frq,PSD = temporal_spectra(drUxH,dt_sampling,Var="drUxH")
plt.loglog(frq,PSD,"-r",label="upper 1/3")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Velocity gradient Magntiude [$1/s$]")
plt.grid()
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"spectra_drUx_comp.png")
plt.close()