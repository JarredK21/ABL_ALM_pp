import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import statistics
from matplotlib.patches import Circle
from scipy.signal import butter,filtfilt
from scipy import interpolate


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


def moments(y):
    mu = np.mean(y)
    std = np.std(y)
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    return mu, std, skewness,kurotsis


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


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time_OF)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt



plot_all_MR_times = False
MR_analysis = False
plot_all_FBR_times = False
FBR_analysis = True
time_scale_analysis = False

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["time_OF"])
Time_start = 200
Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_OF = Time_OF[Time_start_idx:]
dt_OF = Time_OF[1] - Time_OF[0]

Time_sampling = np.array(df_OF.variables["time_sampling"])

Time_end = Time_sampling[-1]; Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

Time_steps = np.arange(0,len(Time_OF))


Azimuth = np.radians(np.array(df_OF.variables["Azimuth"][Time_start_idx:Time_end_idx]))

RtAeroFyh = np.array(df_OF.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(df_OF.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000

RtAeroFys_LPF_1 = low_pass_filter(RtAeroFys,0.3,dt_OF)
RtAeroFys_LPF_2 = low_pass_filter(RtAeroFys,0.9,dt_OF)
RtAeroFys_LPF_3 = low_pass_filter(RtAeroFys,1.5,dt_OF)

RtAeroFys_HPF = np.subtract(RtAeroFys,RtAeroFys_LPF_3)
RtAeroFys_BPF = np.subtract(RtAeroFys_LPF_2,RtAeroFys_LPF_1)

RtAeroFzs_LPF_1 = low_pass_filter(RtAeroFzs,0.3,dt_OF)
RtAeroFzs_LPF_2 = low_pass_filter(RtAeroFzs,0.9,dt_OF)
RtAeroFzs_LPF_3 = low_pass_filter(RtAeroFzs,1.5,dt_OF)

RtAeroFzs_HPF = np.subtract(RtAeroFzs,RtAeroFzs_LPF_3)
RtAeroFzs_BPF = np.subtract(RtAeroFzs_LPF_2,RtAeroFzs_LPF_1)

RtAeroMyh = np.array(df_OF.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(df_OF.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

RtAeroMys_LPF_1 = low_pass_filter(RtAeroMys,0.3,dt_OF)
RtAeroMys_LPF_2 = low_pass_filter(RtAeroMys,0.9,dt_OF)
RtAeroMys_LPF_3 = low_pass_filter(RtAeroMys,1.5,dt_OF)

RtAeroMys_HPF = np.subtract(RtAeroMys,RtAeroMys_LPF_3)
RtAeroMys_BPF = np.subtract(RtAeroMys_LPF_2,RtAeroMys_LPF_1)

RtAeroMzs_LPF_1 = low_pass_filter(RtAeroMzs,0.3,dt_OF)
RtAeroMzs_LPF_2 = low_pass_filter(RtAeroMzs,0.9,dt_OF)
RtAeroMzs_LPF_3 = low_pass_filter(RtAeroMzs,1.5,dt_OF)

RtAeroMzs_HPF = np.subtract(RtAeroMzs,RtAeroMzs_LPF_3)
RtAeroMzs_BPF = np.subtract(RtAeroMzs_LPF_2,RtAeroMzs_LPF_1)

LSSTipMys = np.array(df_OF.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(df_OF.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFys = np.array(df_OF.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(df_OF.variables["LSShftFzs"][Time_start_idx:Time_end_idx])

LSShftFys_LPF_1 = low_pass_filter(LSShftFys,0.3,dt_OF)
LSShftFys_LPF_2 = low_pass_filter(LSShftFys,0.9,dt_OF)
LSShftFys_LPF_3 = low_pass_filter(LSShftFys,1.5,dt_OF)

LSShftFys_HPF = np.subtract(LSShftFys,LSShftFys_LPF_3)
LSShftFys_BPF = np.subtract(LSShftFys_LPF_2,LSShftFys_LPF_1)

LSShftFzs_LPF_1 = low_pass_filter(LSShftFzs,0.3,dt_OF)
LSShftFzs_LPF_2 = low_pass_filter(LSShftFzs,0.9,dt_OF)
LSShftFzs_LPF_3 = low_pass_filter(LSShftFzs,1.5,dt_OF)

LSShftFzs_HPF = np.subtract(LSShftFzs,LSShftFzs_LPF_3)
LSShftFzs_BPF = np.subtract(LSShftFzs_LPF_2,LSShftFzs_LPF_1)

LSSTipMys_LPF_1 = low_pass_filter(LSSTipMys,0.3,dt_OF)
LSSTipMys_LPF_2 = low_pass_filter(LSSTipMys,0.9,dt_OF)
LSSTipMys_LPF_3 = low_pass_filter(LSSTipMys,1.5,dt_OF)

LSSTipMys_HPF = np.subtract(LSSTipMys,LSSTipMys_LPF_3)
LSSTipMys_BPF = np.subtract(LSSTipMys_LPF_2,LSSTipMys_LPF_1)

LSSTipMzs_LPF_1 = low_pass_filter(LSSTipMzs,0.3,dt_OF)
LSSTipMzs_LPF_2 = low_pass_filter(LSSTipMzs,0.9,dt_OF)
LSSTipMzs_LPF_3 = low_pass_filter(LSSTipMzs,1.5,dt_OF)

LSSTipMzs_HPF = np.subtract(LSSTipMzs,LSSTipMzs_LPF_3)
LSSTipMzs_BPF = np.subtract(LSSTipMzs_LPF_2,LSSTipMzs_LPF_1)



L1 = 1.912; L2 = 2.09

#Total bearing force
FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
dFBR_dt = np.array(dt_calc(FBR,dt_OF))

Theta_FB = np.degrees(np.arctan2(FBz,FBy))
Theta_FB = theta_360(Theta_FB)
dTheta_FB_dt = np.array(dt_calc(Theta_FB,dt_OF))


#Aerodynamic Bearing force LPF 0.3Hz
FBMy = RtAeroMzs_LPF_1/L2; FBFy = -RtAeroFys_LPF_1*((L1+L2)/L2)
FBMz = -RtAeroMys_LPF_1/L2; FBFz = -RtAeroFzs_LPF_1*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

Aero_FBR_LPF_1 = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
dFBR_LPF_1_dt = np.array(dt_calc(Aero_FBR_LPF_1,dt_OF))

Theta_Aero_FB_LPF_1 = np.degrees(np.arctan2(FBz,FBy))
Theta_Aero_FB_LPF_1 = theta_360(Theta_Aero_FB_LPF_1)
dTheta_Aero_FB_LPF_1_dt = np.array(dt_calc(Theta_Aero_FB_LPF_1,dt_OF))

#Bearing force LPF 0.3Hz
FBMy = LSSTipMzs_LPF_1/L2; FBFy = -LSShftFys_LPF_1*((L1+L2)/L2)
FBMz = -LSSTipMys_LPF_1/L2; FBFz = -LSShftFzs_LPF_1*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

FBR_LPF_1 = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
dFBR_LPF_1_dt = np.array(dt_calc(FBR_LPF_1,dt_OF))

Theta_FB_LPF_1 = np.degrees(np.arctan2(FBz,FBy))
Theta_FB_LPF_1 = theta_360(Theta_FB_LPF_1)
dTheta_FB_LPF_1_dt = np.array(dt_calc(Theta_FB_LPF_1,dt_OF))


#Bearing force BPF 0.3-0.9Hz
FBMy = LSSTipMzs_BPF/L2; FBFy = -LSShftFys_BPF*((L1+L2)/L2)
FBMz = -LSSTipMys_BPF/L2; FBFz = -LSShftFzs_BPF*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

FBR_BPF = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
dFBR_BPF_dt = np.array(dt_calc(FBR_BPF,dt_OF))

Theta_FB_BPF = np.degrees(np.arctan2(FBz,FBy))
Theta_FB_BPF = theta_360(Theta_FB_BPF)
dTheta_FB_BPF_dt = np.array(dt_calc(Theta_FB_BPF,dt_OF))

#Bearing force HPF 1.5Hz+
FBMy = LSSTipMzs_HPF/L2; FBFy = -LSShftFys_HPF*((L1+L2)/L2)
FBMz = -LSSTipMys_HPF/L2; FBFz = -LSShftFzs_HPF*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)

FBR_HPF = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
dFBR_HPF_dt = np.array(dt_calc(FBR_HPF,dt_OF))

Theta_FB_HPF = np.degrees(np.arctan2(FBz,FBy))
Theta_FB_HPF = theta_360(Theta_FB_HPF)
dTheta_FB_HPF_dt = np.array(dt_calc(Theta_FB_HPF,dt_OF))



#OOPBM
MR = np.sqrt(np.add(np.square(RtAeroMys), np.square(RtAeroMzs)))





#Asymmetry
group = df_OF.groups["63.0"]
Iy = np.array(group.variables["Iy"])
Iz = -np.array(group.variables["Iz"])

f = interpolate.interp1d(Time_sampling,Iy)
Iy = f(Time_OF)

Iy_LPF_1 = low_pass_filter(Iy,0.3,dt_OF)

f = interpolate.interp1d(Time_sampling,Iz)
Iz = f(Time_OF)

Iz_LPF_1 = low_pass_filter(Iz,0.3,dt_OF)

I_LPF_1 = np.sqrt(np.add(np.square(Iy_LPF_1),np.square(Iz_LPF_1)))

time_shift = Time_OF[0]+4.78; time_shift_idx = np.searchsorted(Time_OF,time_shift)


# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF[:-time_shift_idx],Iy_LPF_1[:-time_shift_idx])
# ax2=ax.twinx()
# ax2.plot(Time_OF[:-time_shift_idx],LSSTipMys_LPF_1[time_shift_idx:])
# plt.title("Signals low pass filtered 0.3Hz\ncorrelation coefficient = {}".format(round(correlation_coef(LSSTipMys_LPF_1[time_shift_idx:],Iy_LPF_1[:-time_shift_idx]),2)))
# fig.supxlabel("Time [s]")
# ax.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]")
# ax2.set_ylabel("Magnitude Aerodynamic main bearing force vector [kN]")
# plt.grid()
# plt.tight_layout()

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF[:-time_shift_idx],I_LPF_1[:-time_shift_idx])
# ax2=ax.twinx()
# ax2.plot(Time_OF[:-time_shift_idx],Aero_FBR_LPF_1[time_shift_idx:])
# plt.title("Signals low pass filtered 0.3Hz\ncorrelation coefficient = {}".format(round(correlation_coef(Aero_FBR_LPF_1[time_shift_idx:],I_LPF_1[:-time_shift_idx]),2)))
# fig.supxlabel("Time [s]")
# ax.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]")
# ax2.set_ylabel("Magnitude Aerodynamic main bearing force vector [kN]")
# plt.grid()
# plt.tight_layout()

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF[:-time_shift_idx],Iy_LPF_1[:-time_shift_idx])
# ax2=ax.twinx()
# ax2.plot(Time_OF[:-time_shift_idx],FBR_LPF_1[time_shift_idx:])
# plt.title("Signals low pass filtered 0.3Hz\ncorrelation coefficient = {}".format(round(correlation_coef(FBR_LPF_1[time_shift_idx:],Iy_LPF_1[:-time_shift_idx]),2)))
# fig.supxlabel("Time [s]")
# ax.set_ylabel("y component asymmetry vector [$m^4/s$]")
# ax2.set_ylabel("Magnitude bearing force vector [kN]")
# plt.grid()
# plt.tight_layout()
# plt.show()

if MR_analysis == True:
    MR_LPF_1 = low_pass_filter(MR,0.3,dt_OF)
    MR_LPF_2 = low_pass_filter(MR,0.9,dt_OF)
    MR_LPF_3 = low_pass_filter(MR,1.5,dt_OF)
    MR_LPF_4 = low_pass_filter(MR,6.0,dt_OF)

    MR_HPF = np.subtract(MR,MR_LPF_3)
    MR_BPF = np.subtract(MR_LPF_2,MR_LPF_1)
    MR_BPF_2 = np.subtract(MR_LPF_4,MR_LPF_3)


    dMR_HPF_dt = dt_calc(MR_HPF,dt_OF)
    dMR_BPF_dt = dt_calc(MR_BPF,dt_OF)

    print(np.std(MR_HPF))
    print(np.std(MR_BPF))

    MR_series = pd.Series(MR)
    window = int(10/dt_OF)
    MR_std = MR_series.rolling(window).std()

    MR_BPF_series = pd.Series(MR_BPF)
    window = int(10/dt_OF)
    MR_BPF_std = MR_BPF_series.rolling(window).std()

    MR_HPF_series = pd.Series(MR_HPF)
    window = int(10/dt_OF)
    MR_HPF_std = MR_HPF_series.rolling(window).std()    

    print(correlation_coef(MR_std,MR_BPF_std))
    print(correlation_coef(MR_std,MR_HPF_std))
    print(correlation_coef(MR_BPF_std,MR_HPF_std))

    bins_std = int(np.max(MR_HPF)/(np.std(MR_HPF)/10))
    X = np.linspace(0,np.max(MR_HPF)+1,bins_std)
    bin_idx_1 = np.searchsorted(X,np.std(MR_BPF))-1
    bin_idx_2 = np.searchsorted(X,np.std(MR_BPF)/2)-1
    bins_arr = []
    occ_arr = []
    time_1 = []

    for i in np.arange(0,len(X)-1):
        occ = 0
        bins_arr.append((X[i]+X[i+1])/2)
        for it in np.arange(0,len(MR_HPF)):
            if X[i]<= abs(MR_HPF[it]) < X[i+1]:
                occ+=1
                if i >= bin_idx_1:
                    time_1.append(it)
        occ_arr.append(occ) 

    time_2 = []
    for i in np.arange(0,len(X)-1):
        for it in np.arange(0,len(MR_HPF)):
            if X[i]<= abs(MR_HPF[it]) < X[i+1]:
                if i >= bin_idx_2:
                    time_2.append(it)

    print(np.sum(occ_arr))
    print(len(MR_HPF))
    print(np.sum(occ_arr[bin_idx_1:]))
    print(np.sum(occ_arr[bin_idx_2:]))
    print(len(time_1))
    print(len(time_2))
    Time_OF_time_1 = Time_OF[time_1]
    Time_OF_time_2 = Time_OF[time_2]
    MR_HPF_time_1 = MR_HPF[time_1]
    MR_HPF_time_2 = MR_HPF[time_2]

    print(correlation_coef(MR,MR_BPF))
    print(correlation_coef(MR,MR_HPF))

    out_dir = in_dir+"Internal_variations_analysis/MR_analysis/"
    plt.rcParams['font.size'] = 20

    fig = plt.figure(figsize=(28,16))
    plt.plot(Time_OF,MR_std,"-k")
    plt.plot(Time_OF,MR_BPF_std,"-r")
    plt.plot(Time_OF,MR_HPF_std,"-b")
    plt.xlabel("Time [s]")
    plt.ylabel("Local standard deviation T = 10s\nOut-of-plane bending moment [kN-m]")
    plt.legend(["Total", "Band-pass filter 0.3-0.9Hz", "High-pass filter 1.5Hz"])
    plt.title("Correlation coefficient Total-BPF = 0.9\nCorrelation coeeffcient between Total-HPF = 0.9")
    plt.xticks(np.arange(Time_OF[0],Time_OF[-1],10),fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"local_std_MR.png")
    plt.close()

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(28,16),sharex=True)
    ax1.plot(Time_OF,MR_std,"-k")
    ax1.plot(Time_OF,MR_BPF_std,"-r")
    ax1.plot(Time_OF,MR_HPF_std,"-b")
    ax2.plot(Time_OF[1:],dMR_HPF_dt)
    fig.supxlabel("Time [s]")
    ax1.set_ylabel("Local standard deviation T = 10s\nOut-of-plane bending moment [kN-m]")
    ax1.legend(["Total", "Band-pass filter 0.3-0.9Hz", "High-pass filter 1.5Hz"])
    ax2.set_ylabel("Out-of-plane bending moment derivative [kN-m/s]")
    ax1.set_xticks(np.arange(Time_OF[0],Time_OF[-1],20),fontsize=10)
    ax2.set_xticks(np.arange(Time_OF[0],Time_OF[-1],20),fontsize=10)
    ax1.grid()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"local_std_MR_w_dMR_HPF_dt.png")
    plt.close()

    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(14,8))
    P,X = probability_dist(dMR_BPF_dt)
    plt.plot(X,P,"-r")
    P,X = probability_dist(dMR_HPF_dt)
    plt.plot(X,P,"-b")
    plt.xlabel("Out-of-plane bending moment derivative $dM/dt$ [kN-m/s]")
    plt.ylabel("Probability [-]")
    plt.legend(["Band pass filter 0.3-0.9Hz", "High pass filter 1.5Hz"])
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_dM_dt.png")
    plt.close()



    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[1:],dMR_BPF_dt)
    plt.xlabel("Time [s]")
    plt.ylabel("Out-of-plane bending moment derivative $dM/dt$ [kN-m/s]")
    plt.title("Band pass filter 0.3-0.9Hz")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dM_BPF_dt.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[1:],dMR_HPF_dt)
    plt.xlabel("Time [s]")
    plt.ylabel("Out-of-plane bending moment derivative $dM/dt$ [kN-m/s]")
    plt.title("High pass filter 1.5Hz")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dM_HPF_dt.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,MR_BPF,"-r",label="Band pass filter 0.3-0.9Hz")

    plt.plot(Time_OF,MR_HPF,"-b",label="High pass filter 1.5Hz")
    plt.ylabel("Magnitude Out-of-plane bending moment vector [kN-m]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.xticks(np.arange(Time_OF[0],Time_OF[-1],0.2),fontsize=8)
    plt.xlim([361,367])
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"MR_filtered_3.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,MR_BPF,"-r",label="Band pass filter 0.3-0.9Hz")

    plt.plot(Time_OF,MR_BPF_2,"-b",label="Band pass filter 1.5-6.0Hz")
    plt.ylabel("Magnitude Out-of-plane bending moment vector [kN-m]")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.xticks(np.arange(Time_OF[0],Time_OF[-1],0.2),fontsize=8)
    plt.xlim([361,367])
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"MR_filtered_3_v2.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.plot(bins_arr,occ_arr,"k")
    plt.axvline(x=np.std(MR_BPF),linestyle="--",color="r")
    plt.axvline(x=np.std(MR_BPF)/2,linestyle="--",color="b")
    plt.ylabel("log ocurrances")
    plt.xlabel("Out-of-plane bending moment [kN-m]")
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir+"occurances_log.png")
    plt.close()


    bins_std = int(np.max(dMR_HPF_dt)/(np.std(dMR_HPF_dt)/10))
    X = np.linspace(0,np.max(dMR_HPF_dt)+1,bins_std)

    occ_arr = []
    bins_arr = []
    for i in np.arange(0,len(X)-1):
        occ = 0
        bins_arr.append((X[i]+X[i+1])/2)
        for it in np.arange(0,len(dMR_HPF_dt)):
            if X[i]<= abs(dMR_HPF_dt[it]) < X[i+1]:
                occ+=1
        occ_arr.append(occ) 

    fig = plt.figure(figsize=(14,8))
    plt.plot(bins_arr,occ_arr,"k")
    plt.ylabel("log ocurrances")
    plt.xlabel("Out-of-plane bending moment derivative [kN-m/s]")
    plt.title("High pass filter 1.5Hz")
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir+"dM_dt_occurances_log.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,MR_BPF,"-r",label="Band pass filter 0.3-0.9Hz")
    plt.plot(Time_OF,MR_HPF,"-b",label="High pass filter 1.5Hz")
    plt.plot(Time_OF_time_2,MR_HPF_time_2,"vg")
    plt.plot(Time_OF_time_1,MR_HPF_time_1,"ok")
    plt.ylabel("Magnitude Out-of-plane bending moment vector [kN-m]")
    plt.xlabel("Time [s]")
    plt.xlim([200,206])
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"MR_filtered_times.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))

    P,X = probability_dist(MR_HPF)
    plt.plot(X,P,"-r",label="High pass filtered 1.5Hz")
    plt.yscale("log")
    plt.axvline(np.mean(MR_HPF),linestyle="--",color="k")
    plt.axvline(np.mean(MR_HPF)+np.std(MR_HPF),linestyle="--",color="r")
    plt.axvline(np.mean(MR_HPF)-np.std(MR_HPF),linestyle="--",color="r")
    P,X = probability_dist(MR_BPF)
    plt.plot(X,P,"-b",label="Band pass filtered 0.3-0.9Hz")
    plt.axvline(np.mean(MR_BPF),linestyle="--",color="k")
    plt.axvline(np.mean(MR_BPF)+np.std(MR_BPF),linestyle="--",color="r")
    plt.axvline(np.mean(MR_BPF)-np.std(MR_BPF),linestyle="--",color="r")
    plt.xlabel("Magnitude Out-of-plane bending moment vector [kN]")
    plt.ylabel("Probability [-]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_MR_Filtered_logscale.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,MR)
    plt.title("Total signal")
    plt.ylabel("Magnitude Out-of-plane bending moment vector [kN]")
    plt.xlabel("Time [s]")
    plt.xticks(np.arange(Time_OF[0],Time_OF[-1],0.2),fontsize=8)
    plt.xlim([361,367])
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"MR_3.png")
    plt.close()

    P,X = probability_dist(MR)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,P)
    plt.axvline(np.mean(MR),linestyle="--",color="k")
    plt.axvline(np.mean(MR)+np.std(MR),linestyle="--",color="r")
    plt.axvline(np.mean(MR)-np.std(MR),linestyle="--",color="r")
    plt.title("Total signal")
    plt.xlabel("Magnitude Out-of-plane bending moment vector [kN]")
    plt.ylabel("Probability [-]")
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_MR.png")
    plt.close()


if FBR_analysis == True:

    print(np.std(FBR_HPF))
    print(np.std(FBR_BPF))

    FBR_series = pd.Series(FBR)
    window = int(10/dt_OF)
    FBR_std = FBR_series.rolling(window).std()

    FBR_LPF_series = pd.Series(FBR_LPF_1)
    window = int(10/dt_OF)
    FBR_LPF_std = FBR_LPF_series.rolling(window).std()

    FBR_BPF_series = pd.Series(FBR_BPF)
    window = int(10/dt_OF)
    FBR_BPF_std = FBR_BPF_series.rolling(window).std()

    FBR_HPF_series = pd.Series(FBR_HPF)
    window = int(10/dt_OF)
    FBR_HPF_std = FBR_HPF_series.rolling(window).std()    

    print(correlation_coef(FBR_std,FBR_LPF_std))
    print(correlation_coef(FBR_std,FBR_BPF_std))
    print(correlation_coef(FBR_std,FBR_HPF_std))
    print(correlation_coef(FBR_BPF_std,FBR_HPF_std))

    Theta_FB_series = pd.Series(Theta_FB)
    window = int(10/dt_OF)
    Theta_FB_std = Theta_FB_series.rolling(window).std()

    Theta_FB_BPF_series = pd.Series(Theta_FB_BPF)
    window = int(10/dt_OF)
    Theta_FB_BPF_std = Theta_FB_BPF_series.rolling(window).std()

    Theta_FB_HPF_series = pd.Series(Theta_FB_HPF)
    window = int(10/dt_OF)
    Theta_FB_HPF_std = Theta_FB_HPF_series.rolling(window).std()    

    print(correlation_coef(Theta_FB_std,Theta_FB_BPF_std))
    print(correlation_coef(Theta_FB_std,Theta_FB_HPF_std))
    print(correlation_coef(Theta_FB_BPF_std,Theta_FB_HPF_std))

    print(correlation_coef(FBR_std,Theta_FB_std))
    print(correlation_coef(FBR_BPF_std,Theta_FB_BPF_std))
    print(correlation_coef(FBR_HPF_std,Theta_FB_HPF_std))

    bins_std = int(np.max(FBR_HPF)/(np.std(FBR_HPF)/10))
    X = np.linspace(0,np.max(FBR_HPF)+1,bins_std)
    bin_idx_1 = np.searchsorted(X,np.std(FBR_BPF))-1
    bin_idx_2 = np.searchsorted(X,np.std(FBR_BPF)/2)-1
    bins_arr = []
    occ_arr = []
    time_1 = []

    for i in np.arange(0,len(X)-1):
        occ = 0
        bins_arr.append((X[i]+X[i+1])/2)
        for it in np.arange(0,len(FBR_HPF)):
            if X[i]<= abs(FBR_HPF[it]) < X[i+1]:
                occ+=1
                if i >= bin_idx_1:
                    time_1.append(it)
        occ_arr.append(occ) 

    time_2 = []
    for i in np.arange(0,len(X)-1):
        for it in np.arange(0,len(FBR_HPF)):
            if X[i]<= abs(FBR_HPF[it]) < X[i+1]:
                if i >= bin_idx_2:
                    time_2.append(it)

    print(np.sum(occ_arr))
    print(len(FBR_HPF))
    print(np.sum(occ_arr[bin_idx_1:]))
    print(np.sum(occ_arr[bin_idx_2:]))
    print(len(time_1))
    print(len(time_2))
    Time_OF_time_1 = Time_OF[time_1]
    Time_OF_time_2 = Time_OF[time_2]
    FBR_HPF_time_1 = FBR_HPF[time_1]
    FBR_HPF_time_2 = FBR_HPF[time_2]

    print(correlation_coef(FBR,FBR_BPF))
    print(correlation_coef(FBR,FBR_HPF))

    print(correlation_coef(dTheta_FB_dt,dFBR_HPF_dt))

    out_dir = in_dir+"Internal_variations_analysis/FBR_analysis/"
    plt.rcParams['font.size'] = 20
    

    fig = plt.figure(figsize=(28,16))
    plt.plot(Time_OF,FBR_std,"-k")
    plt.plot(Time_OF,FBR_LPF_std,"-g")
    plt.plot(Time_OF,FBR_BPF_std,"-r")
    plt.plot(Time_OF,FBR_HPF_std,"-b")
    plt.xlabel("Time [s]")
    plt.ylabel("Local standard deviation T = 10s\nMagnitude Main Bearing force vector [kN]")
    plt.legend(["Total", "Low-pass filter 0-0.3Hz","Band-pass filter 0.3-0.9Hz", "High-pass filter 1.5Hz"])
    plt.title("Correlation coefficient Total-LPF = {}\nCorrelation coefficient Total-BPF = {}\nCorrelation coeeffcient between Total-HPF = {}".format(round(correlation_coef(FBR_std,FBR_LPF_std),2),round(correlation_coef(FBR_std,FBR_BPF_std),2),round(correlation_coef(FBR_std,FBR_HPF_std),2)))
    plt.xticks(np.arange(Time_OF[0],Time_OF[-1],10),fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"local_std_FBR.png")
    plt.close()

    fig = plt.figure(figsize=(28,16))
    plt.plot(Time_OF,Theta_FB_std,"-k")
    plt.plot(Time_OF,Theta_FB_BPF_std,"-r")
    plt.plot(Time_OF,Theta_FB_HPF_std,"-b")
    plt.xlabel("Time [s]")
    plt.ylabel("Local standard deviation T = 10s\nDirection Main Bearing force vector [deg]")
    plt.legend(["Total", "Band-pass filter 0.3-0.9Hz", "High-pass filter 1.5Hz"])
    plt.title("Correlation coefficient Total-BPF = {}\nCorrelation coeeffcient between Total-HPF = {}".format(round(correlation_coef(FBR_std,FBR_BPF_std),2),round(correlation_coef(FBR_std,FBR_HPF_std),2)))
    plt.xticks(np.arange(Time_OF[0],Time_OF[-1],10),fontsize=10)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"local_std_Theta_FB.png")
    plt.close()

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(28,16),sharex=True)
    ax1.plot(Time_OF,FBR_std,"-k")
    ax1.plot(Time_OF,FBR_BPF_std,"-r")
    ax1.plot(Time_OF,FBR_HPF_std,"-b")
    ax2.plot(Time_OF[1:],dFBR_HPF_dt)
    fig.supxlabel("Time [s]")
    ax1.set_ylabel("Local standard deviation T = 10s\nMagnitude Main Bearing force vector [kN]")
    ax1.legend(["Total", "Band-pass filter 0.3-0.9Hz", "High-pass filter 1.5Hz"])
    ax2.set_ylabel("Magnitude Main Bearing force vector derivative [kN/s]")
    ax1.set_xticks(np.arange(Time_OF[0],Time_OF[-1],20),fontsize=10)
    ax2.set_xticks(np.arange(Time_OF[0],Time_OF[-1],20),fontsize=10)
    ax1.grid()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"local_std_FBR_w_dFBR_HPF_dt.png")
    plt.close()

    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(14,8))
    P,X = probability_dist(dFBR_BPF_dt)
    plt.plot(X,P,"-r")
    P,X = probability_dist(dFBR_HPF_dt)
    plt.plot(X,P,"-b")
    plt.xlabel("Magnitude Main Bearing force vector derivative $dF/dt$ [kN/s]")
    plt.ylabel("Probability [-]")
    plt.legend(["Band pass filter 0.3-0.9Hz", "High pass filter 1.5Hz"])
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_dFB_dt.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,Theta_FB,"-k")
    plt.plot(Time_OF,Theta_FB_BPF,"-r")
    plt.plot(Time_OF,Theta_FB_HPF,"-b")
    plt.xlabel("Time [s]")
    plt.ylabel("Direction main bearing force vector [deg]")
    plt.legend(["Total","Band pass filter 0.3-0.9Hz", "High pass filter 1.5Hz"])
    plt.xlim([200,205])
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Direction.png")
    plt.close()



    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[1:],dFBR_BPF_dt)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Main Bearing force vector derivative $dF/dt$ [kN/s]")
    plt.title("Band pass filter 0.3-0.9Hz")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_BPF_dt.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[1:],dFBR_HPF_dt)
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Main Bearing force vector derivative $dF/dt$ [kN/s]")
    plt.title("High pass filter 1.5Hz")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dF_HPF_dt.png")
    plt.close()


    fig = plt.figure(figsize=(14,8))
    plt.plot(bins_arr,occ_arr,"k")
    plt.axvline(x=np.std(FBR_BPF),linestyle="--",color="r")
    plt.axvline(x=np.std(FBR_BPF)/2,linestyle="--",color="b")
    plt.ylabel("log ocurrances")
    plt.xlabel("Magntiude Main Bearing force vector [kN]")
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir+"occurances_log.png")
    plt.close()


    bins_std = int(np.max(dFBR_HPF_dt)/(np.std(dFBR_HPF_dt)/10))
    X = np.linspace(0,np.max(dFBR_HPF_dt)+1,bins_std)

    occ_arr = []
    bins_arr = []
    for i in np.arange(0,len(X)-1):
        occ = 0
        bins_arr.append((X[i]+X[i+1])/2)
        for it in np.arange(0,len(dFBR_HPF_dt)):
            if X[i]<= abs(dFBR_HPF_dt[it]) < X[i+1]:
                occ+=1
        occ_arr.append(occ) 

    fig = plt.figure(figsize=(14,8))
    plt.plot(bins_arr,occ_arr,"k")
    plt.ylabel("log ocurrances")
    plt.xlabel("Magnitude Main Bearing force vector derivative [kN/s]")
    plt.title("High pass filter 1.5Hz")
    plt.grid()
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(out_dir+"dF_dt_occurances_log.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))

    P,X = probability_dist(FBR_HPF)
    plt.plot(X,P,"-r",label="High pass filtered 1.5Hz")
    plt.yscale("log")
    plt.axvline(np.mean(FBR_HPF),linestyle="--",color="k")
    plt.axvline(np.mean(FBR_HPF)+np.std(FBR_HPF),linestyle="--",color="r")
    plt.axvline(np.mean(FBR_HPF)-np.std(FBR_HPF),linestyle="--",color="r")
    P,X = probability_dist(FBR_BPF)
    plt.plot(X,P,"-b",label="Band pass filtered 0.3-0.9Hz")
    plt.axvline(np.mean(FBR_BPF),linestyle="--",color="k")
    plt.axvline(np.mean(FBR_BPF)+np.std(FBR_BPF),linestyle="--",color="r")
    plt.axvline(np.mean(FBR_BPF)-np.std(FBR_BPF),linestyle="--",color="r")
    plt.xlabel("Magnitude Main Bearing force vector [kN]")
    plt.ylabel("Probability [-]")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_FBR_Filtered_logscale.png")
    plt.close()

    P,X = probability_dist(FBR)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,P)
    plt.axvline(np.mean(FBR),linestyle="--",color="k")
    plt.axvline(np.mean(FBR)+np.std(FBR),linestyle="--",color="r")
    plt.axvline(np.mean(FBR)-np.std(FBR),linestyle="--",color="r")
    plt.title("Total signal")
    plt.xlabel("Magnitude Main Bearing force vector [kN]")
    plt.ylabel("Probability [-]")
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_FBR.png")
    plt.close()

    fig= plt.figure(figsize=(14,8))
    plt.plot(Time_OF,Theta_FB)
    plt.ylabel("Direction Main Bearing force vector [deg]")
    plt.xlabel("Time [s]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Theta_FB.png")
    plt.close()

    fig= plt.figure(figsize=(14,8))
    plt.plot(Time_OF[1:],dTheta_FB_dt)
    plt.ylabel("Direction Main Bearing force vector derivative [1/s]")
    plt.xlabel("Time [s]")
    plt.ylim([-4000,4000])
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dTheta_FB_dt.png")
    plt.close()


    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(28,16),sharex=True)
    ax1.plot(Time_OF[1:],dFBR_HPF_dt)
    ax2.plot(Time_OF[1:],dTheta_FB_dt)
    fig.supxlabel("Time [s]")
    ax1.set_ylabel("Magnitude Main Bearing force vector derivative [kN/s]")
    ax2.set_ylabel("Direction Main Bearing force vector derivative [1/s]")
    ax2.set_ylim([-4000,4000])
    ax1.set_xticks(np.arange(Time_OF[0],Time_OF[-1],20),fontsize=10)
    ax2.set_xticks(np.arange(Time_OF[0],Time_OF[-1],20),fontsize=10)
    ax1.grid()
    ax2.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Derivatives.png")
    plt.close()


if time_scale_analysis == True:

    out_dir = in_dir+"Internal_variations_analysis/FBR_analysis/"
    plt.rcParams['font.size'] = 12

    P,X = probability_dist(FBR_LPF_1)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,P)
    plt.axvline(x=np.mean(FBR_LPF_1)+np.std(FBR_LPF_1),linestyle="--",color="r")
    plt.axvline(x=np.mean(FBR_LPF_1)-np.std(FBR_LPF_1),linestyle="--",color="r")
    plt.text(np.max(X),np.min(P)*1.2,"{}".format(round(np.max(X),2)))
    plt.text(np.min(X),np.min(P)*1.2,"{}".format(round(np.min(X),2)))
    plt.xlabel("Magnitude Main Bearing force vector [kN]")
    plt.ylabel("Probability [-]")
    plt.title("Low pass filter 0.3Hz $F_{B,low}$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_FBR_low.png")
    plt.close()

    P,X = probability_dist(FBR_BPF)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,P)
    plt.axvline(x=np.mean(FBR_BPF)+np.std(FBR_BPF),linestyle="--",color="r")
    plt.axvline(x=np.mean(FBR_BPF)-np.std(FBR_BPF),linestyle="--",color="r")
    plt.text(np.max(X),np.min(P)*1.2,"{}".format(round(np.max(X),2)))
    plt.text(np.min(X),np.min(P)*1.2,"{}".format(round(np.min(X),2)))
    plt.xlabel("Magnitude Main Bearing force vector [kN]")
    plt.ylabel("Probability [-]")
    plt.title("Band pass filter 0.3-0.9Hz $F_{B,3P}$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_FBR_3P.png")
    plt.close()

    P,X = probability_dist(FBR_HPF)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,P)
    plt.axvline(x=np.mean(FBR_HPF)+np.std(FBR_HPF),linestyle="--",color="r")
    plt.axvline(x=np.mean(FBR_HPF)-np.std(FBR_HPF),linestyle="--",color="r")
    plt.text(np.max(X),np.min(P)*1.2,"{}".format(round(np.max(X),2)))
    plt.text(np.min(X),np.min(P)*1.2,"{}".format(round(np.min(X),2)))
    plt.xlabel("Magnitude Main Bearing force vector [kN]")
    plt.ylabel("Probability [-]")
    plt.title("High pass filter 1.5Hz $F_{B,high}$")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_FBR_high.png")
    plt.close()
    




if plot_all_MR_times == True:
    plt.rcParams['font.size'] = 20

    out_dir = in_dir+"Internal_variations_analysis/MR_plots/"

    times = np.arange(200,1210,10)
    for i in np.arange(0,len(times)-1):
        fig = plt.figure(figsize=(28,16))
        plt.plot(Time_OF,MR,"-k",label="Total signal")
        plt.plot(Time_OF,MR_BPF,"-r",label="Band pass filter 0.3-0.9Hz")
        plt.plot(Time_OF,MR_HPF,"-b",label="High pass filter 1.5Hz")
        plt.plot(Time_OF_time_1,MR_HPF_time_1,"ok")
        plt.axhline(y=np.std(MR_BPF),linestyle="--",color="r")
        plt.axhline(y=-np.std(MR_BPF),linestyle="--",color="r")
        plt.ylabel("Magnitude Out-of-plane bending moment vector [kN-m]")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.xticks(np.arange(Time_OF[0],Time_OF[-1],0.2),fontsize=12)
        plt.xlim([times[i],times[i+1]])
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"MR_{}_{}.png".format(times[i],times[i+1]))
        plt.close()


if plot_all_FBR_times == True:
    plt.rcParams['font.size'] = 20

    out_dir = in_dir+"Internal_variations_analysis/FBR_plots/"

    FBR_LPF_1 = np.subtract(FBR_LPF_1,np.mean(FBR_LPF_1))
    FBR_BPF = np.subtract(FBR_BPF,np.mean(FBR_BPF))
    FBR_HPF = np.subtract(FBR_HPF,np.mean(FBR_HPF))

    threshold_times = []
    FBR_low_times = []
    FBR_mid_times = []
    FBR_high_times = []
    for it in Time_steps:
        if FBR_LPF_1[it] >= 1.5*np.std(FBR_LPF_1) and FBR_BPF[it] >= 1.5*np.std(FBR_BPF) and FBR_HPF[it] >= 1.5*np.std(FBR_HPF):
            threshold_times.append(it)

        if FBR_LPF_1[it] >= 1.5*np.std(FBR_LPF_1):
            FBR_low_times.append(it)
        
        if FBR_BPF[it] >= 1.5*np.std(FBR_BPF):
            FBR_mid_times.append(it)

        if FBR_HPF[it] >= 1.5*np.std(FBR_HPF):
            FBR_high_times.append(it)

        print(it)

    Time_OF_times = Time_OF[threshold_times]
    FBR_LPF_times = FBR_LPF_1[threshold_times]
    FBR_BPF_times = FBR_BPF[threshold_times]
    FBR_HPF_times = FBR_HPF[threshold_times]

    Time_OF_LPF_times = Time_OF[FBR_low_times]
    Time_OF_mid_times = Time_OF[FBR_mid_times]
    Time_OF_high_times = Time_OF[FBR_high_times]

    FBR_LPF_only = FBR_LPF_1[FBR_low_times]
    FBR_BPF_only = FBR_BPF[FBR_mid_times]
    FBR_HPF_only = FBR_HPF[FBR_high_times]

    times = np.arange(200,1220,20)
    for i in np.arange(0,len(times)-1):
        fig = plt.figure(figsize=(28,16))
        plt.plot(Time_OF,FBR,"-k",label="Total signal")
        plt.plot(Time_OF,FBR_LPF_1,"g",label="Low pass filter 0.3Hz")
        plt.plot(Time_OF,FBR_BPF,"-r",label="Band pass filter 0.3-0.9Hz")
        plt.plot(Time_OF,FBR_HPF,"-b",label="High pass filter 1.5Hz")
        plt.plot(Time_OF_LPF_times,FBR_LPF_only,"om")
        plt.plot(Time_OF_mid_times,FBR_BPF_only,"*c")
        plt.plot(Time_OF_high_times,FBR_HPF_only,"vr")
        plt.plot(Time_OF_times,FBR_HPF_times,"ok")
        plt.plot(Time_OF_times,FBR_BPF_times,"*k")
        plt.plot(Time_OF_times,FBR_LPF_times,"vk")
        plt.ylabel("Fluctuating Magnitude Main Bearing force vector [kN]")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.xticks(np.arange(Time_OF[0],Time_OF[-1],1),fontsize=20)
        plt.xlim([times[i],times[i+1]])
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"FBR_{}_{}.png".format(times[i],times[i+1]))
        plt.close()


    Theta_FBR_LPF_times = dTheta_FB_LPF_1_dt[threshold_times]
    Theta_FBR_BPF_times = dTheta_FB_BPF_dt[threshold_times]
    Theta_FBR_HPF_times = dTheta_FB_HPF_dt[threshold_times]

    Theta_FBR_LPF_only = dTheta_FB_LPF_1_dt[FBR_low_times]
    Theta_FBR_BPF_only = dTheta_FB_BPF_dt[FBR_mid_times[:-1]]
    Theta_FBR_HPF_only = dTheta_FB_HPF_dt[FBR_high_times[:-1]]
    out_dir = in_dir+"Internal_variations_analysis/Theta_FBR_plots/"
    times = np.arange(200,1220,20)
    for i in np.arange(0,len(times)-1):
        fig = plt.figure(figsize=(28,16))
        plt.plot(Time_OF[:-1],dTheta_FB_dt,"-k",label="Total signal")
        plt.plot(Time_OF[:-1],dTheta_FB_LPF_1_dt,"g",label="Low pass filter 0.3Hz")
        plt.plot(Time_OF[:-1],dTheta_FB_BPF_dt,"-r",label="Band pass filter 0.3-0.9Hz")
        plt.plot(Time_OF[:-1],dTheta_FB_HPF_dt,"-b",label="High pass filter 1.5Hz")
        plt.plot(Time_OF_LPF_times,Theta_FBR_LPF_only,"om")
        plt.plot(Time_OF_mid_times[:-1],Theta_FBR_BPF_only,"*c")
        plt.plot(Time_OF_high_times[:-1],Theta_FBR_HPF_only,"vr")
        plt.plot(Time_OF_times,Theta_FBR_HPF_times,"ok")
        plt.plot(Time_OF_times,Theta_FBR_BPF_times,"*k")
        plt.plot(Time_OF_times,Theta_FBR_LPF_times,"vk")
        plt.ylabel("Direction Main Bearing force vector - derivative [1/s]")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.xticks(np.arange(Time_OF[0],Time_OF[-1],1),fontsize=20)
        plt.xlim([times[i],times[i+1]])
        plt.ylim([-1000,1000])
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"dTheta_FBR_{}_{}.png".format(times[i],times[i+1]))
        plt.close()