from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import interpolate
import pyFAST.input_output as io
import time
from multiprocessing import Pool

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
zco = np.array(df.variables["zco"])

# print(xco[267],yco[267],zco[267])
# print(xco[567],yco[567],zco[567])
# print(xco[867],yco[867],zco[867])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(xco, yco, zco, marker="o")
# ax.plot(xco[267], yco[267], zco[267],"*r",markersize=10)
# ax.plot(xco[567], yco[567], zco[567],"*r",markersize=10)
# ax.plot(xco[867], yco[867], zco[867],"*r",markersize=10)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()

Time = np.array(df.variables["time"])
dt = Time[1] - Time[0]
Tstart_idx = np.searchsorted(Time,200)
Time = Time[Tstart_idx:]


a = Dataset(in_dir+"Dataset.nc")
Time_sampling = np.array(a.variables["Time_sampling"])
dt_sampling = Time_sampling[1] - Time_sampling[0]
OF_vars = a.groups["OpenFAST_Variables"]
Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:-1])
Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
Tstart_idx_sampling = np.searchsorted(Time_sampling,200)
Time_sampling = Time_sampling[Tstart_idx_sampling:]
Ux_avg = np.array(Rotor_avg_vars.variables["Ux"][Tstart_idx_sampling:])

for i in np.arange(0,len(Azimuth)-1):
    if Azimuth[i+1] < Azimuth[i]:
        Azimuth[i+1:]+=360

uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,267])
vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,267])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
hvelB1 = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
hvelB1_LPF = low_pass_filter(hvelB1,0.1,dt)
uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,567])
vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,567])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
hvelB2 = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,867])
vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,867])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
hvelB3 = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)


df_OF = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()


RootFxb1 = np.array(df_OF["RootFxb1_[kN]"][Tstart_idx:-1])
RootFyb1 = np.array(df_OF["RootFyb1_[kN]"][Tstart_idx:-1])
RootFzc1 = np.array(df_OF["RootFzc1_[kN]"][Tstart_idx:-1])
RootMxb1 = np.array(df_OF["RootMxb1_[kN-m]"][Tstart_idx:-1])
RootMyb1 = np.array(df_OF["RootMyb1_[kN-m]"][Tstart_idx:-1])
RootMzc1 = np.array(df_OF["RootMzc1_[kN-m]"][Tstart_idx:-1])

RootFxb2 = np.array(df_OF["RootFxb2_[kN]"][Tstart_idx:-1])
RootFyb2 = np.array(df_OF["RootFyb2_[kN]"][Tstart_idx:-1])
RootFzc2 = np.array(df_OF["RootFzc2_[kN]"][Tstart_idx:-1])
RootMxb2 = np.array(df_OF["RootMxb2_[kN-m]"][Tstart_idx:-1])
RootMyb2 = np.array(df_OF["RootMyb2_[kN-m]"][Tstart_idx:-1])
RootMzc2 = np.array(df_OF["RootMzc2_[kN-m]"][Tstart_idx:-1])

RootFxb3 = np.array(df_OF["RootFxb3_[kN]"][Tstart_idx:-1])
RootFyb3 = np.array(df_OF["RootFyb3_[kN]"][Tstart_idx:-1])
RootFzc3 = np.array(df_OF["RootFzc3_[kN]"][Tstart_idx:-1])
RootMxb3 = np.array(df_OF["RootMxb3_[kN-m]"][Tstart_idx:-1])
RootMyb3 = np.array(df_OF["RootMyb3_[kN-m]"][Tstart_idx:-1])
RootMzc3 = np.array(df_OF["RootMzc3_[kN-m]"][Tstart_idx:-1])

# del df_OF

# print("line 164")

# cc = []
# for i in np.arange(0,300):
#     uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,i])
#     vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,i])
#     uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
#     hvelB1 = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)

#     cc.append(correlation_coef(hvelB1,RootMyb1))
#     print(len(cc))



# R = np.linspace(0,1,300)
# fig = plt.figure(figsize=(14,8))
# plt.plot(R,cc)
# plt.show()


out_dir=in_dir+"High_frequency_analysis/"
# fig,ax = plt.subplots()
# ax.plot(Time,hvelB1,"-r")
# ax.set_ylabel("Streamwise velocity at 89% span [m/s]")
# ax2=ax.twinx()
# ax2.plot(Time,RootMyb1,"-b")
# ax2.set_ylabel("Edgewise bending Root moment [kN-m]")
# fig.suptitle("Blade 1: correlation coefficient = {}".format(round(correlation_coef(hvelB1,RootMyb1),2)))
# ax.grid()
# fig.supxlabel("Time [s]")
# plt.tight_layout()
# plt.savefig(out_dir+"B1Ux_cc_My.png")
# plt.close()

# fig,ax = plt.subplots()
# ax.plot(Time,hvelB2,"-r")
# ax.set_ylabel("Streamwise velocity at 89% span [m/s]")
# ax2=ax.twinx()
# ax2.plot(Time,RootMyb2,"-b")
# ax2.set_ylabel("Edgewise bending Root moment [kN-m]")
# fig.suptitle("Blade 2: correlation coefficient = {}".format(round(correlation_coef(hvelB2,RootMyb2),2)))
# ax.grid()
# fig.supxlabel("Time [s]")
# plt.tight_layout()
# plt.savefig(out_dir+"B2Ux_cc_My.png")
# plt.close()

# fig,ax = plt.subplots()
# ax.plot(Time,hvelB3,"-r")
# ax.set_ylabel("Streamwise velocity at 89% span [m/s]")
# ax2=ax.twinx()
# ax2.plot(Time,RootMyb3,"-b")
# ax2.set_ylabel("Edgewise bending Root moment [kN-m]")
# fig.suptitle("Blade 3: correlation coefficient = {}".format(round(correlation_coef(hvelB3,RootMyb3),2)))
# ax.grid()
# fig.supxlabel("Time [s]")
# plt.tight_layout()
# plt.savefig(out_dir+"B3Ux_cc_My.png")
# plt.close()

#spectra


RtFx = RootFxb1 + RootFxb2 + RootFxb3
RtMx = RootMxb1 + RootMxb2 + RootMxb3

#calculate rotor My and Mz from blades
#MH = sum(M_blades) + sum(r x F_blades)

LSShftFxa = np.array(df_OF["LSShftFxa_[kN]"][Tstart_idx:-1])
LSShftFys = np.array(df_OF["LSShftFys_[kN]"][Tstart_idx:-1])
LSShftFzs = np.array(df_OF["LSShftFzs_[kN]"][Tstart_idx:-1])

LSSTipMxa = np.array(df_OF["LSShftMxa_[kN-m]"][Tstart_idx:-1])
LSSTipMys = np.array(df_OF["LSSTipMys_[kN-m]"][Tstart_idx:-1])

# fig = plt.figure()
# plt.plot(Time,RtFx,"-r",label="summation over all 3 blades")
# plt.plot(Time,LSShftFxa,"-b",label="Elastodyn output")
# plt.xlabel("Time [s]")
# plt.ylabel("Rotor force x component [kN]")
# plt.title("correlation coefficient = {}".format(round(correlation_coef(RtFx,LSShftFxa),2)))
# plt.grid()
# plt.tight_layout()
# plt.legend()
# plt.savefig(out_dir+"Fx.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RtMx,"-r",label="Summation over all 3 blades")
# plt.plot(Time,LSSTipMxa,"-b",label="Elastodyn Mx")
# plt.xlabel("Time [s]")
# plt.ylabel("Rotor moment x component [kN-m]")
# plt.title("Correlation coefficient = {}".format(round(correlation_coef(RtMx,LSSTipMxa),2)))
# plt.grid()
# plt.legend()
# plt.tight_layout()
# plt.savefig(out_dir+"Mx.png")
# plt.close()

# out_dir=in_dir+"High_frequency_analysis/"
# fig = plt.figure()
# plt.plot(Time,RootFxb1)
# plt.ylabel("RootFxb1")
# plt.savefig(out_dir+"RootFxb1.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootFyb1)
# plt.ylabel("RootFyb1")
# plt.savefig(out_dir+"RootFyb1.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootFzc1)
# plt.ylabel("RootFzc1")
# plt.savefig(out_dir+"RootFzc1.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMxb1)
# plt.ylabel("RootMxb1")
# plt.savefig(out_dir+"RootMxb1.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMyb1)
# plt.ylabel("RootMyb1")
# plt.savefig(out_dir+"RootMyb1.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMzc1)
# plt.ylabel("RootMzc1")
# plt.savefig(out_dir+"RootMzc1.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootFxb2)
# plt.ylabel("RootFxb2")
# plt.savefig(out_dir+"RootFxb2.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootFyb2)
# plt.ylabel("RootFyb2")
# plt.savefig(out_dir+"RootFyb2.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootFzc2)
# plt.ylabel("RootFzc2")
# plt.savefig(out_dir+"RootFzc2.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMxb2)
# plt.ylabel("RootMxb2")
# plt.savefig(out_dir+"RootMxb2.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMyb2)
# plt.ylabel("RootMyb2")
# plt.savefig(out_dir+"RootMyb2.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMzc2)
# plt.ylabel("RootMzc2")
# plt.savefig(out_dir+"RootMzc2.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootFxb3)
# plt.ylabel("RootFxb3")
# plt.savefig(out_dir+"RootFxb3.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootFyb3)
# plt.ylabel("RootFyb3")
# plt.savefig(out_dir+"RootFyb3.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootFzc3)
# plt.ylabel("RootFzc3")
# plt.savefig(out_dir+"RootFzc3.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMxb3)
# plt.ylabel("RootMxb3")
# plt.savefig(out_dir+"RootMxb3.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMyb3)
# plt.ylabel("RootMyb3")
# plt.savefig(out_dir+"RootMyb3.png")
# plt.close()

# fig = plt.figure()
# plt.plot(Time,RootMzc3)
# plt.ylabel("RootMzc3")
# plt.savefig(out_dir+"RootMzc3.png")
# plt.close()



# fig = plt.figure(figsize=(14,8))
# plt.plot(Time,hvelB1_LPF,"-r")
# plt.plot(Time_sampling,Ux_avg,"-b")
# # plt.plot(Time,hvelB2,"-b")
# # plt.plot(Time,hvelB3,"-g")
# plt.xlabel("Time [s]")
# plt.ylabel("Streamwise velocity [m/s]")
# plt.legend(["LPF (0.1Hz) Blade 1 Local sampled streamwise\nblade velocity at 89% span [m/s]", "Rotor averaged velocity [m/s]"])
# plt.grid()
# plt.tight_layout()

# fig = plt.figure(figsize=(14,8))
# P,X = probability_dist(hvelB1)
# print(np.min(X),np.max(X))
# plt.plot(X,P,"-r",label="Blade 1")
# P,X = probability_dist(hvelB2)
# print(np.min(X),np.max(X))
# plt.plot(X,P,"-b",label="Blade 2")
# P,X = probability_dist(hvelB3)
# print(np.min(X),np.max(X))
# plt.plot(X,P,"-g",label="Blade 3")
# plt.ylabel("Probability [-]")
# plt.xlabel("Local sampled streamwise\nblade velocity at 89% span [m/s]")
# plt.legend()
# plt.grid()
# plt.tight_layout()


# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(hvelB1,dt,Var="B1")
# plt.loglog(frq,PSD,"-r")
# frq,PSD = temporal_spectra(hvelB2,dt,Var="B2")
# plt.loglog(frq,PSD,"-b")
# frq,PSD = temporal_spectra(hvelB3,dt,Var="B3")
# plt.loglog(frq,PSD,"-g")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("Local sampled streamwise blade velocity at 89% span [m/s]")
# plt.legend(["Blade 1", "Blade 2", "Blade 3"])
# plt.grid()
# plt.tight_layout()


# Azimuth = Azimuth-Azimuth[0]
# fig = plt.figure(figsize=(14,8))
# plt.plot(Azimuth,hvelB1,"-r")
# plt.plot(Azimuth,hvelB2,"-b")
# plt.plot(Azimuth,hvelB3,"-g")
# plt.xticks(np.arange(np.min(Azimuth), np.max(Azimuth), step=360))
# plt.xlabel("Azimuth position [deg]")
# plt.ylabel("Local sampled streamwise blade velocity at 89% span [m/s]")
# plt.legend(["Blade 1", "Blade 2", "Blade 3"])
# plt.grid()
# plt.tight_layout()
# plt.show()

# time_shift = 1.65
# Time_shift_idx = np.searchsorted(Time,time_shift+Time[0])

# Time_shifted = Time[:-2*Time_shift_idx]

# hvelB1_shifted = hvelB1[2*Time_shift_idx:]
# hvelB3_shifted = hvelB3[Time_shift_idx:-Time_shift_idx]
# hvelB2_shifted = hvelB2[:-2*Time_shift_idx]

# fig = plt.figure(figsize=(14,8))
# plt.plot(Time_shifted,hvelB1_shifted,"-r",label="Blade 1")
# plt.plot(Time_shifted,hvelB2_shifted,"-b",label="Blade 2")
# plt.plot(Time_shifted,hvelB3_shifted,"-g",label="Blade 3")



plt.show()