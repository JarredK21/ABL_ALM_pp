import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
import pyFAST.input_output as io
from matplotlib.backends.backend_pdf import PdfPages

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


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


def moving_stats(x,y,Ylabelx,unitsx,Ylabely,unitsy,time_window):

    window = int(time_window/dt)
    df = pd.Series(x)
    var = df.rolling(window=window).std()
    corr = correlation_coef(var[window:],abs(y[window:])) 

    fig,ax1 = plt.subplots(figsize=(14,8))
    ax1.plot(Time_OF,x,"k")
    ax1.plot(Time_OF,var,"b")
    ax2 = ax1.twinx()
    ax2.plot(Time_OF,abs(y),"r")
    ax1.set_ylabel("{}{}".format(Ylabelx,unitsx),fontsize=16)
    ax1.yaxis.label.set_color("blue")
    ax1.legend(["{}".format(Ylabelx),"Variance"])
    ax2.set_ylabel("{}{}".format(Ylabely,unitsy),fontsize=16)
    ax2.yaxis.label.set_color("red")
    fig.suptitle("correlation = {}".format(corr),fontsize=16)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"variance_{}.png".format(i))
    plt.cla()


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

out_dir = in_dir + "Role_of_forces/"

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

Azimuth = np.radians(np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx]))

RtAeroFxh = np.array(a.variables["RtAeroFxh"][Time_start_idx:Time_end_idx])
RtAeroFyh = np.array(a.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(a.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)

RtAeroFR = np.sqrt( np.add( np.square(RtAeroFys), np.square(RtAeroFzs) ) )

RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])
RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 

LSShftMxa = np.array(a.variables["LSShftMxa"][Time_start_idx:Time_end_idx])
LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

LSShftFxa = np.array(a.variables["LSShftFxa"][Time_start_idx:Time_end_idx])
LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
LSShftFR = np.sqrt( np.add(np.square(LSShftFys), np.square(LSShftFzs)) )

LSSGagMys = np.array(a.variables["LSSGagMys"][Time_start_idx:Time_end_idx])
LSSGagMzs = np.array(a.variables["LSSGagMzs"][Time_start_idx:Time_end_idx])
LSSGagMR = np.sqrt( np.add(np.square(LSSGagMys), np.square(LSSGagMzs)) )

L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))


Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Rel_Aero_FBy = np.true_divide(np.square(Aero_FBy),np.square(Aero_FBR))
Rel_Aero_FBz = np.true_divide(np.square(Aero_FBz),np.square(Aero_FBR))
add_Aero_RelFB = np.add(Rel_Aero_FBy,Rel_Aero_FBz)
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))

Theta_FB = np.degrees(np.arctan2(FBz,FBy))
#Theta_FB = theta_360(Theta_FB)

offset = "5.5"
group = a.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Uz = np.array(group.variables["Uz"])
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

f = interpolate.interp1d(Time_sampling,Ux)
Ux = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Uz)
Uz = f(Time_OF)

f = interpolate.interp1d(Time_sampling,IA)
IA = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iy)
Iy = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iz)
Iz = f(Time_OF)

#plotting options
plot_forces = False
plot_correlations = False
plot_PDF = False
plot_relative_contribution = True


if plot_forces == True:

    Vars = [[FBMy,FBFy],[FBMz,FBFz],[Aero_FBMy/1000,Aero_FBFy/1000],[Aero_FBMz/1000, Aero_FBFz/1000]]
    Ylabels = [["[$M_z/L_2$]","[$-F_yL/L_2$]"],["[$-M_y/L_2$]","[$-F_zL/L_2$]"],["[$\widetilde{M}_z/L_2$]","[$-\widetilde{F}_yL/L_2$]"],
               ["[$-\widetilde{M}_y/L_2$]","[$-\widetilde{F}_zL/L_2$]"]]
    units = [["[kN]","[kN]"],["[kN]","[kN]"],["[kN]","[kN]"],["[kN]","[kN]"]]
    for i in np.arange(0,len(Vars)):
        M_var = round(np.var(Vars[i][0]),2); F_var = round(np.var(Vars[i][1]),2)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time_OF,Vars[i][0],"b")
        plt.plot(Time_OF,Vars[i][1],"r")
        plt.xlabel("Time [s]",fontsize=16)
        plt.ylabel("Bearing force component contributions {}".format(units[i][0]),fontsize=16)
        plt.title("Variance of {} = {} \nVariance of {} = {}".format(Ylabels[i][0], M_var, Ylabels[i][1], F_var))
        plt.legend(Ylabels[i],fontsize=14)
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"{}.png".format(i))
        plt.close()


if plot_correlations == True:
    corr = correlation_coef(Aero_FBy,Aero_FBFy)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,Aero_FBy/1000,"b")
    ax.set_ylabel("Aerodynamic bearing force y component [kN]",fontsize=16)
    ax.yaxis.label.set_color("b")
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(Time_OF,Aero_FBFy/1000,"r")
    ax2.set_ylabel("$\widetilde{F}_y L/L_2 [kN]$",fontsize=16)
    ax2.yaxis.label.set_color("r")
    
    plt.title("correlation = {}".format(round(corr,2)),fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir+"corr_Fy")
    plt.close()

    corr = correlation_coef(Aero_FBy,Aero_FBMy)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,Aero_FBy/1000,"b")
    ax.set_ylabel("Aerodynamic bearing force y component [kN]",fontsize=16)
    ax.yaxis.label.set_color("b")
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(Time_OF,Aero_FBMy/1000,"r")
    ax2.set_ylabel("$\widetilde{M}_z 1/L_2 [kN]$",fontsize=16)
    ax2.yaxis.label.set_color("r")
    
    plt.title("correlation = {}".format(round(corr,2)),fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir+"corr_Mz")
    plt.close()


    corr = correlation_coef(Aero_FBz,Aero_FBFz)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,Aero_FBz/1000,"b")
    ax.set_ylabel("Aerodynamic bearing force z component [kN]",fontsize=16)
    ax.yaxis.label.set_color("b")
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(Time_OF,Aero_FBFz/1000,"r")
    ax2.set_ylabel("$\widetilde{F}_z L/L_2 [kN]$",fontsize=16)
    ax2.yaxis.label.set_color("r")
    
    plt.title("correlation = {}".format(round(corr,2)),fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir+"corr_Fz")
    plt.close()

    corr = correlation_coef(Aero_FBz,Aero_FBMz)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,Aero_FBz/1000,"b")
    ax.set_ylabel("Aerodynamic bearing force z component [kN]",fontsize=16)
    ax.yaxis.label.set_color("b")
    ax.grid()

    ax2 = ax.twinx()
    ax2.plot(Time_OF,Aero_FBMz/1000,"r")
    ax2.set_ylabel("$\widetilde{M}_y 1/L_2 [kN]$",fontsize=16)
    ax2.yaxis.label.set_color("r")
    
    plt.title("correlation = {}".format(round(corr,2)),fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir+"corr_My")
    plt.close()


if plot_PDF == True:
    out_dir=in_dir+"role_of_forces/"
    PMy,XMy,muMy,stdMy = probability_dist(Aero_FBMy/1000)
    PMz,XMz,muMz,stdMz = probability_dist(Aero_FBMz/1000)

    fig = plt.figure(figsize=(14,8))
    plt.plot(XMy,PMy)
    plt.xlabel("$\widetilde{M}_z/L_2$ [kN] ",fontsize=16)
    plt.ylabel("PDF",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"My.png")
    plt.cla()

    dX = XMy[1] - XMy[0]
    xMax = np.searchsorted(XMy,10*np.max(Aero_FBFy/1000))
    xMin = np.searchsorted(XMy,10*np.min(Aero_FBFy/1000))
    prob = np.sum(PMy[xMin:xMax])*dX
    print(np.max(Aero_FBFy/1000),np.min(Aero_FBFy/1000))
    print("probability rotor force is contributing to bearing force component = {}".format(prob))


    fig = plt.figure(figsize=(14,8))
    plt.plot(XMz,PMz)
    plt.xlabel("$\widetilde{M}_y/L_2$ [kN]",fontsize=16)
    plt.ylabel("PDF",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Mz.png")
    plt.cla()

    dX = XMz[1] - XMz[0]
    xMax = np.searchsorted(XMz,10*np.max(Aero_FBFz/1000))
    xMin = np.searchsorted(XMz,10*np.min(Aero_FBFz/1000))
    prob = np.sum(PMz[xMin:xMax])*dX
    print(np.max(Aero_FBFz/1000),np.min(Aero_FBFz/1000))
    print("probability rotor force is contributing to bearing force component = {}".format(prob))


if plot_relative_contribution == True:
    rel_cont = np.true_divide((RtAeroFzs*(L1+L2))/1000,RtAeroMys/1000)
    P,X,mu,std = probability_dist(rel_cont)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,P)
    plt.ylabel("PDF",fontsize=16)
    plt.xlabel("$\widetilde{F}_zL/\widetilde{M}_y$",fontsize=16)
    plt.title("$F_{B_z} = -1/L_2[\widetilde{M}_y + \widetilde{F}_z L]$",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"relative_contribution_FBZ.png")
    plt.close()

    rel_cont = np.true_divide((RtAeroFys*(L1+L2))/1000,RtAeroMzs/1000)
    P,X,mu,std = probability_dist(rel_cont)
    fig = plt.figure(figsize=(14,8))
    plt.plot(X,P)
    plt.ylabel("PDF ",fontsize=16)
    plt.xlabel("$\widetilde{F}_yL/\widetilde{M}_z$",fontsize=16)
    plt.title("$F_{B_y} = -1/L_2[\widetilde{M}_z + \widetilde{F}_y L]$",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"relative_contribution_FBY.png")
    plt.close()