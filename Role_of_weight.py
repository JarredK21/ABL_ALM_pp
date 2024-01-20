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

out_dir = in_dir + "Role_of_weight/"

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
plotting_comparisons = False
plotting_vars = False
plotting_contributions = False
rotor_weight_change = True
role_of_forces = False

if plotting_comparisons == True:
    Vars = [[RtAeroFys/1000,LSShftFys],[RtAeroFzs/1000,LSShftFzs],[RtAeroMys/1000,LSSTipMys],[RtAeroMzs/1000,LSSTipMzs],
            [Aero_FBy/1000,FBy],[Aero_FBz/1000,FBz],[Aero_FBR/1000, FBR],[Theta_Aero_FB, Theta_FB]]
    Ylabels = [["Aerodynamic rotor force y", "Rotor force y"], ["Aerodynamic rotor force z", "Rotor force z"], 
               ["Aerodynamic rotor moment y", "Rotor moment y"], ["Aerodynamic rotor moment z", "Rotor moment z"],
               ["Aerodynamic bearing force y", "Bearing force y"], ["Aerodynamic bearing force z", "Bearing force z"],
               ["Aerodynamic Bearing \nforce magnitude", "Bearing \nforce magnitude"], 
               ["Aerodynamic Bearing \nforce direction", "Bearing \nforce direction"]]
    units = [["[kN]","[kN]"],["[kN]","[kN]"],["[kN-m]","[kN-m]"],["[kN-m]","[kN-m]"],["[kN]","[kN]"],["[kN]","[kN]"],["[kN]", "[kN]"], ["[deg]", "[deg]"]]
    for i in np.arange(0,len(Vars)):
        corr = correlation_coef(Vars[i][0],Vars[i][1])
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time_OF,Vars[i][0],"b")
        ax.set_ylabel("{} {}".format(Ylabels[i][0],units[i][0]),fontsize=16)
        ax.yaxis.label.set_color("b")

        ax2 = ax.twinx()
        ax2.plot(Time_OF,Vars[i][1],"r")
        ax2.set_ylabel("{} {}".format(Ylabels[i][1],units[i][1]),fontsize=16)
        ax2.yaxis.label.set_color("r")

        fig.supxlabel("Time [s]",fontsize=16)
        fig.suptitle("correlation = {}".format(round(corr,2)))
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"{}.png".format(i))
        plt.cla()

        phi = Vars[i][1]
        variance = round(np.var(phi),2)
        range = round(np.max(phi)-np.min(phi),2)

        fig,(ax1,ax2) = plt.subplots(2,figsize=(14,8))
        ax1.plot(Time_OF,Vars[i][1])
        ax1.set_ylabel("{} {}".format(Ylabels[i][1],units[i][1]),fontsize=16)
        ax2.plot(Time_OF,np.subtract(Vars[i][1],Vars[i][0]))
        ax2.set_ylabel("{}-\n{} {}".format(Ylabels[i][1],Ylabels[i][0], units[i][1]),fontsize=16)

        ax1.set_title("Variance = {} \nRange = {}".format(variance,range))

        remainder = np.subtract(Vars[i][1],Vars[i][0])
        variance = round(np.var(remainder),4)
        range = round(np.max(remainder)-np.min(remainder),2)

        ax2.set_title("Variance = {} \nRange = {}".format(variance,range))

        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"remainer_{}.png".format(i))
        plt.cla()


if plotting_vars == True:
    Vars = [[Aero_FBR/1000, FBR],[Theta_Aero_FB, Theta_FB],[Aero_FBz/1000,FBz]]
    Ylabels = [["Aerodynamic Bearing \nforce magnitude", "Bearing \nforce magnitude"], 
               ["Aerodynamic Bearing \nforce direction", "Bearing \nforce direction"],
               ["Aerodynamic Bearing \n force z", "Bearing \nforce z"]]
    units = [["[kN]", "[kN]"], ["[deg]", "[deg]"],["[kN]","[kN]"]]
    for i in np.arange(0,len(Vars)):
        corr = correlation_coef(Vars[i][0],Vars[i][1])
        fig,(ax1,ax2) = plt.subplots(2,figsize=(14,8))
        ax1.plot(Time_OF,Vars[i][0],"b")
        ax1.set_ylabel("{} {}".format(Ylabels[i][0],units[i][0]),fontsize=16)
        ax1.yaxis.label.set_color("b")

        ax2.plot(Time_OF,Vars[i][1],"r")
        ax2.set_ylabel("{} {}".format(Ylabels[i][1],units[i][1]),fontsize=16)
        ax2.yaxis.label.set_color("r")

        fig.supxlabel("Time [s]",fontsize=16)
        fig.suptitle("correlation = {}".format(round(corr,2)))
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"{}.png".format(Ylabels[i][1]))
        plt.cla()


if rotor_weight_change == True:
    out_dir = in_dir+"Role_of_weight/perc_rotor_weight/"
    rotor_weight = -1079.1
    percentage = [0.3, 0.5, 0.7, 0.9]

    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc
        FBR_perc = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
        Theta_FB_perc = np.degrees(np.arctan2(FBz_perc,FBy))

        corr1 = correlation_coef(LSShftFzs,Fzs)
        corr2 = correlation_coef(RtAeroFzs/1000,Fzs)
        fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(14,8))
        ax1.plot(Time_OF,LSShftFzs,"b")
        ax1.set_ylabel("Original rotor \nforce z [kN]",fontsize=12)
        ax1.yaxis.label.set_color("b")
        ax1.grid()

        ax2.plot(Time_OF,Fzs,"r")
        ax2.set_ylabel("Rotor force z with \n{} reduction in weight [kN]".format(perc),fontsize=12)
        ax2.yaxis.label.set_color("r")
        ax2.set_title("correlation with original = {}".format(round(corr1,2)),fontsize=16)
        ax2.grid()

        ax3.plot(Time_OF,RtAeroFzs/1000,"g")
        ax3.set_ylabel("Aerodynamic \n rotor force z [kN]".format(perc),fontsize=12)
        ax3.yaxis.label.set_color("g")
        ax3.set_title("correlation with above = {}".format(round(corr2,2)),fontsize=16)
        ax3.grid()

        plt.tight_layout()
        plt.savefig(out_dir+"Fzs_perc_{}.png".format(perc))
        plt.close()

        corr1 = correlation_coef(FBz,FBz_perc)
        corr2 = correlation_coef(Aero_FBz,FBz_perc)
        fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(14,8))
        ax1.plot(Time_OF,FBz,"b")
        ax1.set_ylabel("Original Bearing \nforce z [kN]",fontsize=12)
        ax1.yaxis.label.set_color("b")
        ax1.grid()

        ax2.plot(Time_OF,FBz_perc,"r")
        ax2.set_ylabel("Bearing force z with \n{} reduction in weight [kN]".format(perc),fontsize=12)
        ax2.yaxis.label.set_color("r")
        ax2.set_title("correlation with original = {}".format(round(corr1,2)),fontsize=16)
        ax2.grid()

        ax3.plot(Time_OF,Aero_FBz/1000,"g")
        ax3.set_ylabel("Aerodynamic \n Bearing force z [kN]".format(perc),fontsize=12)
        ax3.yaxis.label.set_color("g")
        ax3.set_title("correlation with above = {}".format(round(corr2,2)),fontsize=16)
        ax3.grid()

        plt.tight_layout()
        plt.savefig(out_dir+"FBz_perc_{}.png".format(perc))
        plt.close()



        corr1 = correlation_coef(np.square(FBz),np.square(FBz_perc))
        corr2 = correlation_coef(np.square(Aero_FBz),np.square(FBz_perc))
        fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(14,8))
        ax1.plot(Time_OF,np.square(FBz),"b")
        ax1.set_ylabel("Original Bearing \nforce z squared [$kN^2$]",fontsize=12)
        ax1.yaxis.label.set_color("b")
        ax1.grid()

        ax2.plot(Time_OF,np.square(FBz_perc),"r")
        ax2.set_ylabel("Bearing force z squared with \n {} reduction in weight [$kN^2$]".format(perc),fontsize=12)
        ax2.yaxis.label.set_color("r")
        ax2.set_title("correlation with original = {}".format(round(corr1,2)),fontsize=16)
        ax2.grid()

        ax3.plot(Time_OF,np.square(Aero_FBz/1000),"g")
        ax3.set_ylabel("Aerodynamic \n Bearing force z squared [$kN^2$]".format(perc),fontsize=12)
        ax3.yaxis.label.set_color("g")
        ax3.set_title("correlation with above = {}".format(round(corr2,2)),fontsize=16)
        ax3.grid()

        plt.tight_layout()
        plt.savefig(out_dir+"FBz_squared_perc_{}.png".format(perc))
        plt.close()


        corr1 = correlation_coef(FBR,FBR_perc)
        corr2 = correlation_coef(Aero_FBR,FBR_perc)
        fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(14,8))
        ax1.plot(Time_OF,FBR,"b")
        ax1.set_ylabel("Original Bearing \nforce magnitude [kN]",fontsize=12)
        ax1.yaxis.label.set_color("b")
        ax1.grid()

        ax2.plot(Time_OF,FBR_perc,"r")
        ax2.set_ylabel("Bearing force magnitude with \n{} reduction in weight [kN]".format(perc),fontsize=12)
        ax2.yaxis.label.set_color("r")
        ax2.set_title("correlation with original = {}".format(round(corr1,2)),fontsize=16)
        ax2.grid()

        ax3.plot(Time_OF,Aero_FBR/1000,"g")
        ax3.set_ylabel("Aerodynamic \n Bearing force magnitude [kN]".format(perc),fontsize=12)
        ax3.yaxis.label.set_color("g")
        ax3.set_title("correlation with above = {}".format(round(corr2,2)),fontsize=16)
        ax3.grid()

        plt.tight_layout()
        plt.savefig(out_dir+"FBR_perc_{}.png".format(perc))
        plt.close()


        corr1 = correlation_coef(Theta_FB,Theta_FB_perc)
        corr2 = correlation_coef(Theta_Aero_FB,Theta_FB_perc)
        fig,(ax1,ax2,ax3) = plt.subplots(3,figsize=(14,8))
        ax1.plot(Time_OF,Theta_FB,"b")
        ax1.set_ylabel("Original Bearing \nforce direction [deg]",fontsize=12)
        ax1.yaxis.label.set_color("b")
        ax1.grid()

        ax2.plot(Time_OF,Theta_FB_perc,"r")
        ax2.set_ylabel("Bearing force direction with \n{} reduction in weight [deg]".format(perc),fontsize=12)
        ax2.yaxis.label.set_color("r")
        ax2.set_title("correlation with original = {}".format(round(corr1,2)),fontsize=16)
        ax2.grid()

        ax3.plot(Time_OF,Theta_Aero_FB,"g")
        ax3.set_ylabel("Aerodynamic \n Bearing force direction [deg]".format(perc),fontsize=12)
        ax3.yaxis.label.set_color("g")
        ax3.set_title("correlation with above = {}".format(round(corr2,2)),fontsize=16)
        ax3.grid()

        plt.tight_layout()
        plt.savefig(out_dir+"Theta_perc_{}.png".format(perc))
        plt.close()

        fig, (ax1,ax2) = plt.subplots(2,figsize=(14,8))
        ax1.plot(Time_OF,FBz_perc)
        ax1.set_ylabel("Bearing force z component with \n{} reduction in weight [kN]".format(perc),fontsize=12)
        ax1.axhline(0,linestyle="--",color="k")
        ax1.grid()
        ax2.plot(Time_OF,Theta_FB_perc)
        ax2.axhline(0,linestyle="--",color="k")
        ax2.set_ylabel("Bearing force direction with \n{} reduction in weight [deg]".format(perc),fontsize=12)
        ax2.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"FB_vector_{}.png".format(perc))
        plt.close()




