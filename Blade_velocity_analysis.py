from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import interpolate
import pyFAST.input_output as io
import time
from multiprocessing import Pool


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z



def low_pass_filter(signal, cutoff,dt):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


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


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"


df = Dataset(in_dir+"WTG01.nc")

# num_act_points = 300

xco = np.array(df.variables["xco"])
yco = np.array(df.variables["yco"])
zco = np.array(df.variables["zco"])-90

R = np.linspace(0,63.0,300)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(xco, yco, zco, marker="o")
# ax.plot(xco[267], yco[267], zco[267],"*r",markersize=10)
# ax.plot(xco[567], yco[567], zco[567],"*r",markersize=10)
# ax.plot(xco[867], yco[867], zco[867],"*r",markersize=10)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')





Time = np.array(df.variables["time"])
dt = Time[1] - Time[0]
Tstart_idx = np.searchsorted(Time,200)
Time = Time[Tstart_idx:]


a = Dataset(in_dir+"Dataset.nc")
Time_sampling = np.array(a.variables["Time_sampling"])
T_start_sampling_idx = np.searchsorted(Time_sampling,200)
Time_sampling = Time_sampling[T_start_sampling_idx:]
dt_sampling = Time_sampling[1] - Time_sampling[0]

Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]
Iy = np.array(Rotor_avg_vars.variables["Iy"][T_start_sampling_idx:])
Iz = np.array(Rotor_avg_vars.variables["Iz"][T_start_sampling_idx:])
I_2 = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
I_2_LPF = low_pass_filter(I_2,0.3,dt_sampling)

Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
Rotor_avg_vars = Rotor_avg_vars.groups["63.0"]
Iy_3 = np.array(Rotor_avg_vars.variables["Iy"][T_start_sampling_idx:])
Iz_3 = np.array(Rotor_avg_vars.variables["Iz"][T_start_sampling_idx:])
I_3 = np.sqrt(np.add(np.square(Iy_3),np.square(Iz_3)))
I_3_LPF = low_pass_filter(I_3,0.3,dt_sampling)


a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["Time_OF"][Tstart_idx:-1])

OF_vars = a.groups["OpenFAST_Variables"]

Azimuth = np.radians(np.array(OF_vars.variables["Azimuth"][Tstart_idx:-1]))

RtAeroFyh = np.array(OF_vars.variables["RtAeroFyh"][Tstart_idx:-1])/1000
RtAeroFzh = np.array(OF_vars.variables["RtAeroFzh"][Tstart_idx:-1])/1000

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)


RtAeroMyh = np.array(OF_vars.variables["RtAeroMyh"][Tstart_idx:-1])/1000
RtAeroMzh = np.array(OF_vars.variables["RtAeroMzh"][Tstart_idx:-1])/1000

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 

LSSTipMys = np.array(OF_vars.variables["LSSTipMys"][Tstart_idx:-1])
LSSTipMzs = np.array(OF_vars.variables["LSSTipMzs"][Tstart_idx:-1])
LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))


# Time_sampling = np.array(a.variables["Time_sampling"])
# dt_sampling = Time_sampling[1] - Time_sampling[0]
# OF_vars = a.groups["OpenFAST_Variables"]
Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:-1])
# Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
# Tstart_idx_sampling = np.searchsorted(Time_sampling,200)
# Time_sampling = Time_sampling[Tstart_idx_sampling:]
# Ux_avg = np.array(Rotor_avg_vars.variables["Ux"][Tstart_idx_sampling:])

# for i in np.arange(0,len(Azimuth)-1):
#     if Azimuth[i+1] < Azimuth[i]:
#         Azimuth[i+1:]+=360

uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
#hvelB1 = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
#hvelB2 = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
#hvelB3 = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)

R = np.linspace(0,63,300)


def actuator_asymmetry_calc(it):
    hvelB1 = np.add(np.cos(np.radians(29))*uvelB1[it], np.sin(np.radians(29))*vvelB1[it])
    IyB1 = hvelB1*R*np.cos(np.radians(Azimuth[it]))
    IyB1_75 = IyB1[225]
    IyB1 = np.sum(IyB1)
    IzB1 = hvelB1*R*np.sin(np.radians(Azimuth[it]))
    IzB1_75 = IzB1[225]
    IzB1 = np.sum(IzB1)

    hvelB2 = np.add(np.cos(np.radians(29))*uvelB2[it], np.sin(np.radians(29))*vvelB2[it])
    AzB2 = Azimuth[it] + 120
    if AzB2 >= 360:
        AzB2-=360

    IyB2 = hvelB2*R*np.cos(np.radians(AzB2))
    IzB2 = hvelB2*R*np.sin(np.radians(AzB2))
    IyB2_75 = IyB2[225]
    IyB2 = np.sum(IyB2)
    IzB2_75 = IzB2[225]
    IzB2 = np.sum(IzB2)

    hvelB3 = np.add(np.cos(np.radians(29))*uvelB3[it], np.sin(np.radians(29))*vvelB3[it])
    AzB3 = Azimuth[it] + 240
    if AzB3 >= 360:
        AzB3-=360

    IyB3 = hvelB3*R*np.cos(np.radians(AzB3))
    IzB3 = hvelB3*R*np.sin(np.radians(AzB3))
    IyB3_75 = IyB3[225]
    IyB3 = np.sum(IyB3)
    IzB3_75 = IzB3[225]
    IzB3 = np.sum(IzB3)

    return IyB1+IyB2+IyB3, IzB1+IzB2+IzB3, IyB1_75+IyB2_75+IyB3_75, IzB1_75+IzB2_75+IzB3_75


dA = 0.3125*0.3125
Iy = []
Iz = []
Iy_75 = []
Iz_75 = []
ix=0
with Pool() as pool:
    for Iy_it, Iz_it, Iy_75_it, Iz_it_75 in pool.imap(actuator_asymmetry_calc,np.arange(0,len(Time))):
        Iy.append(Iy_it); Iz.append(Iz_it); Iy_75.append(Iy_75_it); Iz_75.append(Iz_it_75)
        print(ix)
        ix+=1


I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
I_75 = np.sqrt(np.add(np.square(Iy_75),np.square(Iz_75)))

out_dir=in_dir+"High_frequency_analysis/"
plt.rcParams.update({'font.size': 18})

cc = round(correlation_coef(I,I_75),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,I,"-b")
ax.yaxis.label.set_color('blue') 
ax.set_ylabel("Magnitude Asymmetry vector [$m^2/s$] $I_B$")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,I_75,"-r")
ax2.set_ylabel("Magnitude Asymmetry vector at 75% span [$m^2/s$] $I_B$")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"cc_IB_I_75.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(I_75,dt,Var="I_75")
plt.loglog(frq,PSD,"-r",label="$I_B$ 75% span")
frq,PSD = temporal_spectra(I,dt,Var="I_B")
plt.loglog(frq,PSD,"-b",label="$I_B$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title("Spectra $I_B at 47m span [m^2/s]\,I_B [m^2/s]$")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"spectra_IB_IB_75.png")
plt.close()

cc = round(correlation_coef(Iy,RtAeroMys),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,Iy,"-b")
ax.yaxis.label.set_color('blue') 
ax.set_ylabel("Asymmetry around y axis [$m^4/s$] $I_{B_y}$")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,LSSTipMys,"-r")
ax2.set_ylabel("Rotor Aerodynamic moment y component [kN-m]")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"cc_Iy_My.png")
plt.close()

cc = round(correlation_coef(Iz,RtAeroMzs),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,Iz,"-b")
ax.yaxis.label.set_color('blue') 
ax.set_ylabel("Asymmetry around z axis [$m^4/s$] $I_{B_z}$")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,LSSTipMzs,"-r")
ax2.set_ylabel("Rotor aerodynamic moment z component [kN-m]")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"cc_Iz_Mz.png")
plt.close()

cc = round(correlation_coef(I,RtAeroMR),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,I,"-b")
ax.yaxis.label.set_color('blue') 
ax.set_ylabel("Magnitude Asymmetry vector [$m^4/s$] $I_B$")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,RtAeroMR,"-r")
ax2.set_ylabel("Magnitude Aerodynamic Rotor moment [kN-m]")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"cc_I_MR.png")
plt.close()

cc = round(correlation_coef(I_75,RtAeroMR),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,I_75,"-b")
ax.yaxis.label.set_color('blue') 
ax.set_ylabel("Magnitude Asymmetry vector at 75% span [$m^2/s$] $I_B$")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,RtAeroMR,"-r")
ax2.set_ylabel("Magnitude Aerodynamic Rotor moment [kN-m]")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"cc_IB_75_MR.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(RtAeroMR,dt,Var="MR")
plt.loglog(frq,PSD,"-r",label="OOPBM")
frq,PSD = temporal_spectra(FBR,dt,Var="MR")
plt.loglog(frq,np.multiply(PSD,1e+05),"-g",label="$F_{B_R}$")
frq,PSD = temporal_spectra(I,dt,Var="I_B")
plt.loglog(frq,np.true_divide(PSD,1e+05),"-b",label="$I_B$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title("Spectra OOPBM $[kN-m]\,F_{B_R} [kN] \,I_B [m^4/s]$")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"spectra_IB_MR_FBR.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(I_3,dt_sampling,Var="I")
plt.loglog(frq,PSD,"-r",label="I")
frq,PSD = temporal_spectra(I,dt,Var="I_B")
plt.loglog(frq,PSD,"-b",label="$I_B$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title("Spectra $I [m^4/s]\, I_B [m^4/s]$")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"spectra_IB_I.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(I_3,dt_sampling,Var="I")
plt.loglog(frq,np.true_divide(PSD,2000),"-r",label="I")
frq,PSD = temporal_spectra(I,dt,Var="I_B")
plt.loglog(frq,np.multiply(PSD,100),"-b",label="$I_B$")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD")
plt.title("Spectra $I [m^4/s]\, I_B [m^4/s]$")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"spectra_IB_I_2.png")
plt.close()

#filtering 
LPF_1_MR = low_pass_filter(RtAeroMR,0.3,dt)
LPF_2_MR = low_pass_filter(RtAeroMR,0.9,dt)
LPF_3_MR = low_pass_filter(RtAeroMR,1.5,dt)

HPF_MR = np.subtract(RtAeroMR,LPF_3_MR)
HPF_MR = np.array(low_pass_filter(HPF_MR,40,dt))
BPF_MR = np.subtract(LPF_2_MR,LPF_1_MR)

LPF_1_I = low_pass_filter(I,0.3,dt)
LPF_2_I = low_pass_filter(I,0.9,dt)
LPF_3_I = low_pass_filter(I,1.5,dt)

HPF_I = np.subtract(I,LPF_3_I)
HPF_I = np.array(low_pass_filter(HPF_I,40,dt))
BPF_I = np.subtract(LPF_2_I,LPF_1_I)


cc = round(correlation_coef(LPF_1_MR,LPF_1_I),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,LPF_1_I,"-b")
ax.yaxis.label.set_color('blue') 
ax.set_ylabel("LPF 0.3Hz Magnitude Asymmetry vector [$m^4/s$] $I_B$")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,LPF_1_MR,"-r")
ax2.set_ylabel("LPF 0.3Hz Magnitude Rotor moment [kN-m]")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"LPF_cc_I_MR.png")
plt.close()

cc = round(correlation_coef(BPF_MR,BPF_I),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,BPF_I,"-b")
ax.yaxis.label.set_color('blue') 
ax.set_ylabel("BPF 0.3-0.9Hz Magnitude Asymmetry vector [$m^4/s$] $I_B$")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,BPF_MR,"-r")
ax2.set_ylabel("BPF 0.3-0.9Hz Magnitude Rotor moment [kN-m]")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"BPF_cc_I_MR.png")
plt.close()

cc = round(correlation_coef(HPF_MR,HPF_I),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time,HPF_I,"-b")
ax.yaxis.label.set_color('blue') 
ax.set_ylabel("HPF 1.5-40Hz Magnitude Asymmetry vector [$m^4/s$] $I_B$")
ax.grid()
ax2=ax.twinx()
ax2.plot(Time,HPF_MR,"-r")
ax2.set_ylabel("HPF 1.5-40Hz Magnitude Rotor moment [kN-m]")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"HPF_cc_I_MR.png")
plt.close()


times = np.arange(200,1300,100)
for i in np.arange(0,len(times)-1):
    idx1 = np.searchsorted(Time,times[i])
    idx2 = np.searchsorted(Time,times[i+1])
    fig,ax = plt.subplots(figsize=(14,8))
    #ax2=ax.twinx()

    ax.plot(Time[idx1:idx2],I[idx1:idx2],"-k",label="$I_B,tot$")

    #ax2.plot(Time[idx1:idx2],RtAeroMR[idx1:idx2],"--k",label="OOPBM")

    ax.plot(Time[idx1:idx2],LPF_1_I[idx1:idx2],"-b",label="LPF $I_B$")
    
    #ax2.plot(Time[idx1:idx2],LPF_1_MR[idx1:idx2],"--b",label="LPF OOPBM")

    ax.plot(Time[idx1:idx2],BPF_I[idx1:idx2],"-r",label="BPF $I_B$")

    #ax2.plot(Time[idx1:idx2],BPF_MR[idx1:idx2],"--r",label="BPF OOPBM")

    ax.plot(Time[idx1:idx2],np.subtract(HPF_I[idx1:idx2],10000),"-g",label="HPF $I_B -10,000m^2/s$")

    #ax2.plot(Time[idx1:idx2],HPF_MR[idx1:idx2],"--g",label="HPF OOPBM")

    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("Magnitude Asymmetry vector [$m^2/s$] $I_B$")
    ax.grid()
    #ax2.set_ylabel("Magnitude aerodynamic Rotor moment [kN-m]")
    #ax2.yaxis.label.set_color('red') 
    fig.supxlabel("Time [s]")
    ax.legend(loc="upper right")
    #ax2.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(out_dir+"I_all_times/I_{}_{}.png".format(times[i],times[i+1]))
    plt.close()


f = interpolate.interp1d(Time,LPF_1_I)
LPF_1_I_interp = f(Time_sampling)
cc = round(correlation_coef(LPF_1_I_interp,I_2_LPF),2)
fig,ax = plt.subplots(figsize=(14,8))
ax.plot(Time_sampling,LPF_1_I_interp,"-b")
ax.set_ylabel("LPF 0.3Hz Actuator Magnitude Asymmetry vector [$m^4/s$] $I_B$")
ax.yaxis.label.set_color('blue') 
ax2=ax.twinx()
ax2.plot(Time_sampling,I_2_LPF,"-r")
ax2.set_ylabel("LPF 0.3Hz Magnitude Asymmetry vector [$m^4/s$] $I$")
ax2.yaxis.label.set_color('red') 
fig.suptitle("correlation coefficient = {}".format(cc))
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"cc_I_I2.png")
plt.close()

times = np.arange(200,1300,100)
for i in np.arange(0,len(times)-1):
    idx1 = np.searchsorted(Time,times[i])
    idx2 = np.searchsorted(Time,times[i+1])
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time[idx1:idx2],I[idx1:idx2],"-b",label="$I_{B,tot}$")
    idx1 = np.searchsorted(Time_sampling,times[i])
    idx2 = np.searchsorted(Time_sampling,times[i+1])
    ax.plot(Time_sampling[idx1:idx2],LPF_1_I_interp[idx1:idx2],"-k",label="LPF $I_B$")
    ax.set_ylabel("Actuator Magnitude Asymmetry vector [$m^4/s$] $I_B$")
    ax.legend()
    ax.yaxis.label.set_color('blue') 
    ax2=ax.twinx()
    ax2.plot(Time_sampling[idx1:idx2],I_2[idx1:idx2],"-r")
    ax2.set_ylabel("Magnitude Asymmetry vector [$m^4/s$] $I$")
    ax2.yaxis.label.set_color('red') 
    ax.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"I_IB_all_times/cc_I_I2_{}_{}.png".format(times[i],times[i+1]))
    plt.close()



dHPF_MR = np.array(dt_calc(HPF_MR,dt))

zero_crossings_index_HPF_MR = np.where(np.diff(np.sign(dHPF_MR)))[0]
#HPF calc
dF_mag_HPF = []
dt_mag_HPF = []
dF_I_HPF = []
for i in np.arange(0,len(zero_crossings_index_HPF_MR)-1):

    it_1 = zero_crossings_index_HPF_MR[i]
    it_2 = zero_crossings_index_HPF_MR[i+1]

    dF_mag_HPF.append(HPF_MR[it_2] - HPF_MR[it_1])

    dt_mag_HPF.append(Time[it_2] - Time[it_1])

    dF_I_HPF.append(HPF_I[it_2] - HPF_I[it_1])


fig = plt.figure(figsize=(14,8))
plt.scatter(dF_I_HPF,dF_mag_HPF)
plt.xlabel("$dI_B [m^2/s]$")
plt.ylabel("dF [kN-m]")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"dIB_dF_HPF.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.scatter(dt_mag_HPF,dF_mag_HPF,c=dF_I_HPF,cmap="viridis")
plt.xlabel("$dt$ [s]")
plt.ylabel("$dF [kN-m]$")
plt.title("HPF $M_R$")
plt.grid()
plt.colorbar()
plt.tight_layout()
plt.savefig(out_dir+"MR_dF_dt_HPF_I.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dF_mag_HPF)
plt.plot(X,P)
plt.xlabel("Jump in Magnitude Rotor moment [kN-m]")
plt.ylabel("Probabilty [-]")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"PDF_dF_MR.png")
plt.close()

fig = plt.figure(figsize=(14,8))
P,X = probability_dist(dt_mag_HPF)
plt.plot(X,P)
plt.xlabel("Time step [s]")
plt.ylabel("Probabilty [-]")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"PDF_dt_MR.png")
plt.close()


#HPF calc
dF_mag_HPF_threshold = []
dt_mag_HPF_threshold = []
dF_I_HPF_threshold = []
for i in np.arange(0,len(zero_crossings_index_HPF_MR)-1):

    it_1 = zero_crossings_index_HPF_MR[i]
    it_2 = zero_crossings_index_HPF_MR[i+1]

    if HPF_MR[it_2] - HPF_MR[it_1] > 3*np.std(dF_mag_HPF) or HPF_MR[it_2] - HPF_MR[it_1] < -3*np.std(dF_mag_HPF):

        dF_mag_HPF_threshold.append(HPF_MR[it_2] - HPF_MR[it_1])

        dt_mag_HPF_threshold.append(Time[it_2] - Time[it_1])

        dF_I_HPF_threshold.append(HPF_I[it_2] - HPF_I[it_1])


fig = plt.figure(figsize=(14,8))
plt.scatter(dF_I_HPF_threshold,dF_mag_HPF_threshold)
plt.xlabel("$dI_B [m^2/s]$")
plt.ylabel("dF [kN-m]")
plt.title("HPF $M_R$ Threshold on 3x std")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"dIB_dF_HPF_threshold.png")
plt.close()

print(len(dF_mag_HPF_threshold))
fig = plt.figure(figsize=(14,8))
plt.scatter(dt_mag_HPF_threshold,dF_mag_HPF_threshold,c=dF_I_HPF_threshold,cmap="viridis")
plt.xlabel("$dt$ [s]")
plt.ylabel("$dF [kN-m]$")
plt.title("HPF $M_R$ Threshold on 3x std")
plt.grid()
plt.colorbar()
plt.tight_layout()
plt.savefig(out_dir+"MR_dF_dt_HPF_threshold_I.png")
plt.close()