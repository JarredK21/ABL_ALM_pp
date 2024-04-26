import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import statistics
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


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"


a = Dataset(in_dir+"Asymmetry_Dataset.nc")

Time = np.array(a.variables["time"])
Time = Time - Time[0]

Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)
Time = Time[Time_start_idx:]
Time_steps = np.arange(0,len(Time))


df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["time_OF"])

Time_start = 200

Time_start_idx = np.searchsorted(Time_OF,Time_start)

Time_OF = Time_OF[Time_start_idx:]
dt_OF = Time_OF[1] - Time_OF[0]


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


L1 = 1.912; L2 = 2.09; L = L1 + L2

Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz
Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Aero_theta = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Aero_theta = theta_360(Aero_theta)
Aero_theta = np.radians(np.array(Aero_theta))


LSSTipMys = np.array(df_OF.variables["LSSTipMys"][Time_start_idx:])
LSSTipMzs = np.array(df_OF.variables["LSSTipMzs"][Time_start_idx:])

LSShftFys = np.array(df_OF.variables["LSShftFys"][Time_start_idx:])
LSShftFzs = np.array(df_OF.variables["LSShftFzs"][Time_start_idx:])


L1 = 1.912; L2 = 2.09
FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz

FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))
Theta_FB = theta_360(Theta_FB)
Theta_FB = np.radians(np.array(Theta_FB))


WR = 1079.1 #kN

RB = L2**2 * np.square(FBR)
RB_tilde = L2**2 * np.square(Aero_FBR)
MR = np.add(np.square(RtAeroMys), np.square(RtAeroMzs))
FR = np.add(np.square(RtAeroFys * L), np.square(RtAeroFzs * L))
Fy_Mz = 2*L*RtAeroFys*RtAeroMzs
Fz_My = 2*L*RtAeroFzs*RtAeroMys
WR_My = 2*L*WR*RtAeroMys
WRsq = (WR*L)**2


plot_pdf = True
polar_analysis = False
polar_Aero_analysis = False

plt.rcParams['font.size'] = 14

if plot_pdf == True:
    out_dir = in_dir+"Bearing_force_analysis/"
    with PdfPages(out_dir+'Bearing_force_analysis.pdf') as pdf:


        cc = round(correlation_coef(RB,MR),2)
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,MR,"-r")
        ax2=ax.twinx()
        ax2.plot(Time_OF,RB,"-b")
        fig.supxlabel("Time [s]")
        ax.set_ylabel("$(M_y^2 + M_z^2)$")
        ax.yaxis.label.set_color('red')
        ax2.set_ylabel("$L_2^2|R_B|^2 [kN^2-m^2]$")
        ax2.yaxis.label.set_color('blue')
        plt.title("cc(FBR,MR) = {}".format(cc))
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        cc = round(correlation_coef(RB_tilde,MR),2)
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,MR,"-r")
        ax2=ax.twinx()
        ax2.plot(Time_OF,Aero_FBR,"-b")
        fig.supxlabel("Time [s]")
        ax.set_ylabel("$(M_y^2 + M_z^2) [kN^2-m^2]$")
        ax.yaxis.label.set_color('red')
        ax2.set_ylabel("Aerodynamic - $L_2^2|R_B|^2 [kN^2-m^2]$")
        ax2.yaxis.label.set_color('blue')
        plt.title("cc(Aero_FBR,MR) = {}".format(cc))
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        cc = round(correlation_coef(RB,LSSTipMys),2)
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,LSSTipMys,"-r")
        ax2=ax.twinx()
        ax2.plot(Time_OF,RB,"-b")
        fig.supxlabel("Time [s]")
        ax.set_ylabel("Moment at rotor y direction [kN-m]")
        ax.yaxis.label.set_color('red')
        ax2.set_ylabel("$L_2^2|R_B|^2 [kN^2-m^2]$")
        ax2.yaxis.label.set_color('blue')
        plt.title("cc(FBR,My) = {}".format(cc))
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        
        cc = round(correlation_coef(RB,FR),2)
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,FR,"-r")
        ax2=ax.twinx()
        ax2.plot(Time_OF,RB,"-b")
        fig.supxlabel("Time [s]")
        ax.set_ylabel("$(L^2 F_y^2 + L^2 F_z^2) [kN^2-m^2]$")
        ax.yaxis.label.set_color('red')
        ax2.set_ylabel("$L_2^2|R_B|^2 [kN^2-m^2]$")
        ax2.yaxis.label.set_color('blue')
        plt.title("cc(FBR,FR) = {}".format(cc))
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        cc = round(correlation_coef(RB_tilde,FR),2)
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,FR,"-r")
        ax2=ax.twinx()
        ax2.plot(Time_OF,RB_tilde,"-b")
        fig.supxlabel("Time [s]")
        ax.set_ylabel("$(L^2 F_y^2 + L^2 F_z^2) [kN^2-m^2]$")
        ax.yaxis.label.set_color('red')
        ax2.set_ylabel("Aerodynamic - $L_2^2|R_B|^2 [kN^2-m^2]$")
        ax2.yaxis.label.set_color('blue')
        plt.title("cc(Aero_FBR,FR) = {}".format(cc))
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        cc = round(correlation_coef(RB,Fy_Mz),2)
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,Fy_Mz,"-r")
        ax2=ax.twinx()
        ax2.plot(Time_OF,RB,"-b")
        fig.supxlabel("Time [s]")
        ax.set_ylabel("$2LF_yM_z [kN^2-m^2]$")
        ax.yaxis.label.set_color('red')
        ax2.set_ylabel("$L_2^2|R_B|^2 [kN^2-m^2]$")
        ax2.yaxis.label.set_color('blue')
        plt.title("cc(FBR,$2LF_yM_z$) = {}".format(cc))
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        cc = round(correlation_coef(RB,Fz_My),2)
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,Fz_My,"-r")
        ax2=ax.twinx()
        ax2.plot(Time_OF,RB,"-b")
        fig.supxlabel("Time [s]")
        ax.set_ylabel("$2LF_zM_y [kN^2-m^2]$")
        ax.yaxis.label.set_color('red')
        ax2.set_ylabel("$L_2^2|R_B|^2 [kN^2-m^2]$")
        ax2.yaxis.label.set_color('blue')
        plt.title("cc(FBR,$2LF_zM_y$) = {}".format(cc))
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        cc = round(correlation_coef(RB,WR_My),2)
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,WR_My,"-r")
        ax2=ax.twinx()
        ax2.plot(Time_OF,RB,"-b")
        fig.supxlabel("Time [s]")
        ax.set_ylabel("$2LW_R M_y [kN^2-m^2]$")
        ax.yaxis.label.set_color('red')
        ax2.set_ylabel("$L_2^2|R_B|^2 [kN^2-m^2]$")
        ax2.yaxis.label.set_color('blue')
        plt.title("cc(FBR,$2LW_R M_y$) = {}".format(cc))
        ax.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()



        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
        ax1.plot(Time_OF,MR)
        ax2.plot(Time_OF,FR)
        fig.supxlabel("Time [s]")
        ax1.set_ylabel("$(M_y^2 + M_z^2)$")
        ax2.set_ylabel("$(L^2F_y^2 + L^2F_z^2)$")

        it = 0
        ic = 0
        MR_it = []
        print(len(MR))
        for i in np.arange(0,len(Time_OF)):

            if FR[i] >= MR[i]/10:
                MR_it.append(MR[i])
                ax1.plot(Time_OF[i],MR[i],"ob",markersize=2)
                ax2.plot(Time_OF[i],FR[i],"or",markersize=2)
                it+=dt_OF
                ic+=1

        print(ic) 
        ax1.grid()
        ax2.grid()
        ax1.set_title("Average $M_R$ when $F_R$ is within 1 order of magnitude {} $kN^2m^2$".format(round(np.average(MR_it))))
        ax2.set_title("Time $F_R \geq M_R/10$ = {}s".format(round(it,2)))
        plt.tight_layout()
        pdf.savefig()
        plt.close()


MR = np.add(np.square(RtAeroMys/L2), np.square(RtAeroMzs/L2))
Theta_MR = np.degrees(np.arctan2(RtAeroMzs,RtAeroMys))
Theta_MR = theta_360(Theta_MR)
Theta_MR = np.radians(np.array(Theta_MR))

Aero_FR = np.add(np.square(RtAeroFys * (L/L2)), np.square(RtAeroFzs * (L/L2)))
Theta_Aero_FR = np.degrees(np.arctan2(RtAeroFzs,RtAeroFys))
Theta_Aero_FR = theta_360(Theta_Aero_FR)
Theta_Aero_FR = np.radians(np.array(Theta_Aero_FR))

FR = np.add(np.square(LSShftFys * (L/L2)), np.square(LSShftFzs * (L/L2)))
Theta_FR = np.degrees(np.arctan2(LSShftFzs,LSShftFys))
Theta_FR = theta_360(Theta_FR)
Theta_FR = np.radians(np.array(Theta_FR))


f = interpolate.interp1d(Time_OF,Aero_FBR)
Aero_FBR = f(Time)
f = interpolate.interp1d(Time_OF,Aero_theta)
Aero_theta = f(Time)


f = interpolate.interp1d(Time_OF,MR)
MR = f(Time)
f = interpolate.interp1d(Time_OF,Theta_MR)
Theta_MR = f(Time)

f = interpolate.interp1d(Time_OF,FR)
FR = f(Time)
f = interpolate.interp1d(Time_OF,Theta_FR)
Theta_FR = f(Time)

f = interpolate.interp1d(Time_OF,Aero_FR)
Aero_FR = f(Time)
f = interpolate.interp1d(Time_OF,Theta_Aero_FR)
Theta_Aero_FR = f(Time)


f = interpolate.interp1d(Time_OF,FBR)
FBR = f(Time)
f = interpolate.interp1d(Time_OF,Theta_FB)
Theta_FB = f(Time)


def Update_polar(it):

    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(Theta_FB[it], FBR[it]/np.max(FBR), c="k", s=20)
    d = ax.scatter(Theta_FR[it],FR[it]/np.max(FR), c="b", s=20)
    f = ax.scatter(Theta_MR[it], MR[it]/np.max(MR), c="r", s=20)
    ax.arrow(0, 0, Theta_FB[it], FBR[it]/np.max(FBR), length_includes_head=True, color="k")
    ax.arrow(0, 0, Theta_FR[it], FR[it]/np.max(FR), length_includes_head=True, color="b")
    ax.arrow(0, 0, Theta_MR[it], MR[it]/np.max(MR), length_includes_head=True, color="r")
    ax.set_ylim([0,1])
    ax.set_title("Normalized Force vectors [kN]\nTime = {}s".format(Time[it]), va='bottom')
    plt.legend(["Main Bearing Force", "Rotor Force", "Rotor Moment"],loc="lower right")
    T = Time[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T



def Update_Aero_polar(it):

    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(Aero_theta[it], Aero_FBR[it], c="k", s=20)
    d = ax.scatter(Theta_Aero_FR[it],Aero_FR[it], c="b", s=20)
    f = ax.scatter(Theta_MR[it], MR[it], c="r", s=20)
    ax.arrow(0, 0, Aero_theta[it], Aero_FBR[it], length_includes_head=True, color="k")
    ax.arrow(0, 0, Theta_Aero_FR[it], Aero_FR[it], length_includes_head=True, color="b")
    ax.arrow(0, 0, Theta_MR[it], MR[it], length_includes_head=True, color="r")
    ax.set_ylim([0,np.max([np.max(Aero_FBR),np.max(MR),np.max(Aero_FR)])])
    ax.set_title("Aerodynamic Force vectors [kN]\nTime = {}s".format(Time[it]), va='bottom')
    plt.legend(["Main Bearing Force", "Rotor Force", "Rotor Moment"],loc="lower right")
    T = Time[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T


if polar_analysis == True:
    out_dir = in_dir+"Bearing_force_analysis/polar_OOPBM/"
    with Pool() as pool:
        for T in pool.imap(Update_polar,Time_steps):

            print(T)


if polar_Aero_analysis == True:
    out_dir = in_dir+"Bearing_force_analysis/polar_Aero_OOPBM/"
    with Pool() as pool:
        for T in pool.imap(Update_Aero_polar,Time_steps):

            print(T)