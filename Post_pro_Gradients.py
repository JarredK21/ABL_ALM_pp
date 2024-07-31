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

Time = np.array(a.variables["Time_sampling"])
Time_start_idx = np.searchsorted(Time,200)
Time = Time[Time_start_idx:]
dt_sampling = Time[1] - Time[0]


Rotor_Gradients = a.groups["Rotor_Gradients"]

dyUx = np.array(Rotor_Gradients.variables["dyUx"][Time_start_idx:])
dzUx = np.array(Rotor_Gradients.variables["dzUx"][Time_start_idx:])
drUx = np.array(Rotor_Gradients.variables["drUx"][Time_start_idx:])

Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
Iy = np.array(Rotor_avg_vars.variables["Iy"][Time_start_idx:])
Iz = np.array(Rotor_avg_vars.variables["Iz"][Time_start_idx:])
I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

out_dir=in_dir+"three_frequency_analysis/OOPBM_analysis/"
plt.rcParams.update({'font.size': 18})
cc_1 = round(correlation_coef(I,drUx))
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,I,"-b")
ax.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,drUx,"-r")
ax2.set_ylabel("Magnitude velocity gradient vector averaged over rotor [1/s]")
fig.supxlabel("Time [s]")
fig.suptitle("correlation coefficient = {}".format(cc_1))
plt.tight_layout()
plt.savefig(out_dir+"I_cc_drUx.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(I,dt_sampling,Var="I")
plt.loglog(frq,PSD,"-b",label="Magntidue Asymmetry vector")
frq,PSD = temporal_spectra(drUx,dt_sampling,Var="drUx")
plt.loglog(frq,PSD,"-r",label="$du_{x'}/dr$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"spectra_I_drUx.png")
plt.close()

idx = np.searchsorted(Time,Time[0]+10)
cc = []
for it in np.arange(0,len(Time)-idx):
    cc.append(correlation_coef(I[it:it+idx],drUx[it:it+idx]))

fig,(ax1,ax3) = plt.subplots(2,1,figsize=(14,8))
ax1.plot(Time,I,"-b")
ax1.set_ylabel("Magnitude Asymmetry\nvector [$m^4/s$]")
ax1.grid()
ax2=ax1.twinx()
ax2.plot(Time,drUx,"-r")
ax2.set_ylabel("Rotor averaged\nvelocity gradient [1/s]")
ax1.set_title("correlation coefficient = {}".format(cc_1))
ax3.plot(Time[int(idx/2):int(-idx/2)],cc)
ax3.set_ylabel("Local correlation\ncoefficient T = 10s")
ax3.grid()
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"local_cc_I_drUx.png")
plt.close()


time_shift_idx = np.searchsorted(Time,4.65+Time[0])

dyUx_shifted = dyUx[:-time_shift_idx]
dzUx_shifted = dzUx[:-time_shift_idx]
drUx_shifted = drUx[:-time_shift_idx]

Time_shifted = Time[:-time_shift_idx]
    
OpenFAST_Variables = a.groups["OpenFAST_Variables"]

Time_OF = np.array(a.variables["Time_OF"])
Time_start_idx = np.searchsorted(Time_OF,200)
Time_OF = Time_OF[Time_start_idx:]
dt = Time_OF[1]-Time_OF[0]

time_shift_idx = np.searchsorted(Time_OF,4.65+Time_OF[0])
Time_OF_shifted = Time_OF[:-time_shift_idx]

Azimuth = np.radians(np.array(OpenFAST_Variables.variables["Azimuth"][Time_start_idx:]))

RtAeroFyh = np.array(OpenFAST_Variables.variables["RtAeroFyh"][Time_start_idx:])
RtAeroFzh = np.array(OpenFAST_Variables.variables["RtAeroFzh"][Time_start_idx:])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(OpenFAST_Variables.variables["RtAeroMyh"][Time_start_idx:])
RtAeroMzh = np.array(OpenFAST_Variables.variables["RtAeroMzh"][Time_start_idx:])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

OOPBM = np.sqrt(np.add(np.square(RtAeroMys),np.square(RtAeroMzs)))

LPF_1_OOPBM = low_pass_filter(OOPBM,0.3,dt)
LPF_2_OOPBM = low_pass_filter(OOPBM,0.9,dt)
LPF_3_OOPBM = low_pass_filter(OOPBM,1.5,dt)

HPF_OOPBM = np.subtract(OOPBM,LPF_3_OOPBM)
HPF_OOPBM = np.array(low_pass_filter(HPF_OOPBM,40,dt))
BPF_OOPBM = np.subtract(LPF_2_OOPBM,LPF_1_OOPBM)

LPF_1_OOPBM_shifted = LPF_1_OOPBM[time_shift_idx:]
BPF_OOPBM_shifted = BPF_OOPBM[time_shift_idx:]
HPF_OOPBM_shifted = HPF_OOPBM[time_shift_idx:]


dBPF_OOPBM = np.array(dt_calc(BPF_OOPBM,dt))

zero_crossings_index_BPF_OOPBM = np.where(np.diff(np.sign(dBPF_OOPBM)))[0]

Env_BPF_OOPBM = []
Env_Times = []
for i in np.arange(0,len(zero_crossings_index_BPF_OOPBM),2):
    idx = zero_crossings_index_BPF_OOPBM[i]
    Env_BPF_OOPBM.append(BPF_OOPBM[idx]); Env_Times.append(Time_OF[idx])

time_shift_idx = np.searchsorted(Env_Times,4.65+Env_Times[0])

Env_Times_shifted = Env_Times[:-time_shift_idx]
Env_BPF_OOPBM_shifted = Env_BPF_OOPBM[time_shift_idx:]

f = interpolate.interp1d(Env_Times_shifted,Env_BPF_OOPBM_shifted)
new_x = np.linspace(np.min(Env_Times_shifted), np.max(Env_Times_shifted), len(Time_shifted))

Env_BPF_OOPBM_shifted_interp = f(new_x)

cc = correlation_coef(drUx_shifted,Env_BPF_OOPBM_shifted_interp)

plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/OOPBM_analysis/"
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Env_Times_shifted,Env_BPF_OOPBM_shifted,"-b")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time_shifted,drUx_shifted,"-r")
ax.set_ylabel("Envelope Magnitude Aerodynamic Rotor moment vector [kN-m]\n3P frequency")
ax.yaxis.label.set_color('blue')
ax2.set_ylabel("Magnitude velocity gradient $du_{x'}/dr$ [1/s]")
ax2.yaxis.label.set_color('red') 
fig.supxlabel("Time [s]")
plt.suptitle("Correlation coefficient = {}".format(round(cc,2)))
plt.savefig(out_dir+"magnitude_gradient_cc_FBR_3P.png")
plt.close()


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.plot(Env_Times_shifted,Env_BPF_OOPBM_shifted,"-b")
ax1.yaxis.label.set_color('blue')
ax1.set_ylabel("Envelope BPF $M_{H}$")
ax3 = ax1.twinx()
ax3.plot(Time_shifted,drUx_shifted,"-r")
ax3.set_ylabel("$du_{x'}/dr$")
ax3.yaxis.label.set_color('red')

plt.suptitle("Correlation coefficient = {}".format(round(cc,2)))

cc = []
idx = np.searchsorted(Time_shifted,10+Time_shifted[0])
for i in np.arange(0,len(Time_shifted)-idx,1):
    cc.append(correlation_coef(drUx_shifted[i:i+idx],Env_BPF_OOPBM_shifted_interp[i:i+idx]))


ax2.plot(Time_shifted[:-idx],cc,"-k")
fig.supxlabel("Time [s]")
ax2.set_ylabel("Local correlation coefficient T = 10s [-]")
ax2.grid()
plt.tight_layout()
plt.savefig(out_dir+"Local_cc_drUx_BPF_FBR.png")
plt.close()


f = interpolate.interp1d(Time_OF_shifted,LPF_1_OOPBM_shifted)
new_x = np.linspace(np.min(Time_OF_shifted), np.max(Time_OF_shifted), len(Time_shifted))

LPF_1_OOPBM_shifted_interp = f(new_x)

cc = round(correlation_coef(LPF_1_OOPBM_shifted_interp,drUx_shifted),2)

fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF_shifted,LPF_1_OOPBM_shifted,"-r")
ax.set_ylabel("LPF OOPBM [kN-m]")
ax.yaxis.label.set_color('red')
ax2=ax.twinx()
ax2.plot(Time_shifted,drUx_shifted,"-b")
ax2.set_ylabel("$du_{x'}/dr$")
ax2.yaxis.label.set_color('blue')
ax.grid()
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"LPF_OOPBM_cc_du_dr.png")
plt.close()


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.plot(Time_OF_shifted,LPF_1_OOPBM_shifted,"-r")
ax1.yaxis.label.set_color('red')
ax1.set_ylabel("LPF $M_{H}$")
ax3 = ax1.twinx()
ax3.plot(Time_shifted,drUx_shifted,"-b")
ax3.set_ylabel("$du_{x'}/dr$")
ax3.yaxis.label.set_color('blue')

plt.suptitle("Correlation coefficient = {}".format(round(cc,2)))

cc = []
idx = np.searchsorted(Time_shifted,10+Time_shifted[0])
for i in np.arange(0,len(Time_shifted)-idx,1):
    cc.append(correlation_coef(drUx_shifted[i:i+idx],LPF_1_OOPBM_shifted_interp[i:i+idx]))


ax2.plot(Time_shifted[:-idx],cc,"-k")
fig.supxlabel("Time [s]")
ax2.set_ylabel("Local correlation coefficient T = 10s [-]")
ax2.grid()
plt.tight_layout()
plt.show()
# plt.savefig(out_dir+"Local_cc_drUx_LPF_FBR.png")
# plt.close()


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

BPF_FBR_shifted = BPF_FBR[time_shift_idx:]
HPF_FBR_shifted = HPF_FBR[time_shift_idx:]

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF_shifted,BPF_FBR_shifted,"-b")
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_shifted,abs(dyUx_shifted),"-r")

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF_shifted,BPF_FBR_shifted,"-b")
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_shifted,abs(dzUx_shifted),"-r")

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF_shifted,BPF_FBR_shifted,"-b")
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_shifted,drUx_shifted,"-r")





dBPF_FBR = np.array(dt_calc(BPF_FBR,dt))

zero_crossings_index_BPF_FBR = np.where(np.diff(np.sign(dBPF_FBR)))[0]

BPF_FBR_2 = []
Times_2 = []
for i in np.arange(0,len(zero_crossings_index_BPF_FBR),2):
    idx = zero_crossings_index_BPF_FBR[i]
    BPF_FBR_2.append(BPF_FBR[idx]); Times_2.append(Time_OF[idx])

time_shift_idx = np.searchsorted(Times_2,4.65+Times_2[0])

Times_2_shifted = Times_2[:-time_shift_idx]
BPF_FBR_2_shifted = BPF_FBR_2[time_shift_idx:]

f = interpolate.interp1d(Times_2_shifted,BPF_FBR_2_shifted)
new_x = np.linspace(np.min(Times_2_shifted), np.max(Times_2_shifted), len(Time_shifted))

BPF_FBR_2_shifted_interp = f(new_x)

cc = correlation_coef(drUx_shifted,BPF_FBR_2_shifted_interp)

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Times_2_shifted,BPF_FBR_2_shifted,"-b")
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_shifted,abs(dyUx_shifted),"-r")

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Times_2_shifted,BPF_FBR_2_shifted,"-b")
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_shifted,abs(dzUx_shifted),"-r")

# times = np.arange(200,1300,100)
# for i in np.arange(0,len(times)-1):

#     idx1 = np.searchsorted(Times_2_shifted,times[i]); idx2 = np.searchsorted(Times_2_shifted,times[i+1])
#     plt.rcParams['font.size'] = 16
#     out_dir=in_dir+"three_frequency_analysis/magnitude_gradient_all_times/"
#     fig,ax = plt.subplots(figsize=(14,8))
#     ax.plot(Times_2_shifted[idx1:idx2],BPF_FBR_2_shifted[idx1:idx2],"-b")
#     ax.grid()
#     ax2=ax.twinx()
#     idx1 = np.searchsorted(Time_shifted,times[i]); idx2 = np.searchsorted(Time_shifted,times[i+1])
#     ax2.plot(Time_shifted[idx1:idx2],drUx_shifted[idx1:idx2],"-r")
#     ax.set_ylabel("Envelope Magnitude Main bearing force vector [kN]\n3P frequency")
#     ax.yaxis.label.set_color('blue')
#     ax2.set_ylabel("Magnitude velocity gradient $du_{x'}/dr$ [1/s]")
#     ax2.yaxis.label.set_color('red') 
#     fig.supxlabel("Time [s]")
#     #plt.suptitle("Correlation coefficient = ")
#     plt.savefig(out_dir+"magnitude_gradient_cc_FBR_3P_{}_{}.png".format(times[i],times[i+1]))
#     plt.close()


plt.rcParams['font.size'] = 16
out_dir=in_dir+"three_frequency_analysis/"
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Times_2_shifted,BPF_FBR_2_shifted,"-b")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time_shifted,drUx_shifted,"-r")
ax.set_ylabel("Envelope Magnitude Main bearing force vector [kN]\n3P frequency")
ax.yaxis.label.set_color('blue')
ax2.set_ylabel("Magnitude velocity gradient $du_{x'}/dr$ [1/s]")
ax2.yaxis.label.set_color('red') 
fig.supxlabel("Time [s]")
plt.suptitle("Correlation coefficient = {}".format(round(cc,2)))
plt.savefig(out_dir+"magnitude_gradient_cc_FBR_3P.png")
plt.close()


fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
ax1.plot(Times_2_shifted,BPF_FBR_2_shifted,"-b")
ax1.yaxis.label.set_color('blue')
ax1.set_ylabel("BPF $F_{B_R}$")
ax3 = ax1.twinx()
ax3.plot(Time_shifted,drUx_shifted,"-r")
ax3.set_ylabel("$du_{x'}/dr$")
ax3.yaxis.label.set_color('red')

plt.suptitle("Correlation coefficient = {}".format(round(cc,2)))

cc = []
idx = np.searchsorted(Time_shifted,20+Time_shifted[0])
for i in np.arange(0,len(Time_shifted)-idx,1):
    cc.append(correlation_coef(drUx_shifted[i:i+idx],BPF_FBR_2_shifted_interp[i:i+idx]))


ax2.plot(Time_shifted[:-idx],cc,"-k")
fig.supxlabel("Time [s]")
ax2.set_ylabel("Local correlation coefficient T = 20s [-]")
ax2.grid()
plt.tight_layout()
plt.savefig(out_dir+"Local_cc_drUx_BPF_FBR.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.plot(Time_shifted[:-idx],cc,"-r",label="du/dr cc BPF FBR")


f = interpolate.interp1d(Time_OF,LPF_1_FBR)
LPF_1_FBR_interp = f(Times_2)

idx = np.searchsorted(Times_2,Times_2[0]+20)
cc = []
for it in np.arange(0,len(Times_2)-idx):
    cc.append(correlation_coef(BPF_FBR_2[it:it+idx],LPF_1_FBR_interp[it:it+idx]))

plt.plot(Times_2[idx:],cc,"-b",label="LPF FBR cc BPF FBR")
plt.grid()
plt.xlabel("Time [s]")
plt.ylabel("Local correlation coefficient T = 20s")
plt.tight_layout()
plt.legend()
plt.savefig(out_dir+"comparing_local_cc.png")
plt.close()


fig = plt.figure(figsize=(14,8))
plt.plot(Time,dyUx)
plt.grid()
plt.ylabel("Rotor averaged dyUx [1/s]")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"dyUx.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.plot(Time,dzUx)
plt.grid()
plt.ylabel("Rotor averaged dzUx [1/s]")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"dzUx.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.plot(Time,drUx)
plt.grid()
plt.ylabel("Rotor averaged drUx [1/s]")
plt.xlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"drUx.png")
plt.close()

plt.rcParams['font.size'] = 12
P,X=probability_dist(dyUx)
fig = plt.figure(figsize=(14,8))
plt.plot(X,P,label="dyUx\n{}".format(moments(dyUx)))
plt.grid()
plt.xlabel("Rotor averaged Velocity Gradient [1/s]",fontsize=16)

P,X=probability_dist(dzUx)
plt.plot(X,P,label="dzUx\n{}".format(moments(dzUx)))

P,X=probability_dist(drUx)
plt.plot(X,P,label="drUx\n{}".format(moments(drUx)))

plt.legend(loc="upper right")
plt.ylabel("Probability [-]",fontsize=16)
plt.tight_layout()
plt.savefig(out_dir+"PDF_velocity_gradients.png")
plt.close()

plt.rcParams['font.size'] = 16

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(dyUx,dt_sampling,Var="dyUx")
plt.loglog(frq,PSD,"-r",label="$\langle du_{x'}/dy \\rangle_A$")
frq,PSD = temporal_spectra(dzUx,dt_sampling,Var="dzUx")
plt.loglog(frq,PSD,"-b",label="$\langle du_{x'}/dz \\rangle_A$")
frq,PSD = temporal_spectra(drUx,dt_sampling,Var="drUx")
plt.loglog(frq,PSD,"-k",label="$\langle du_{x'}/dr \\rangle_A$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Rotor averaged gradient [1/s]")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"spectra_velocity_gradient.png")
plt.close()