from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from windrose import WindroseAxes
from scipy import stats
import pandas as pd
from matplotlib.animation import FuncAnimation
from scipy.signal import butter,filtfilt
from scipy import interpolate
import pylab as pl

def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    skew = stats.skew(y)
    kurtosis = stats.kurtosis(y,fisher=False)
    no_bin = 10000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    dX = X[1] - X[0]
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    print(np.sum(P)*dX)
    return P,X, round(mu,2), round(sd,2), round(skew,2), round(kurtosis,2)


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


def low_pass_filter(signal, cutoff,dt):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time_OF)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

time_steps = np.arange(0,len(Time_OF),100)

Azimuth = np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx])
Azimuth = np.radians(Azimuth)

RtAeroFyh = np.array(a.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(a.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)

RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])

L1 = 1.912; L2 = 2.09
FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz

FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))
Theta_FB_360 = theta_360(Theta_FB)
Theta_FB_rad = np.radians(np.array(Theta_FB))

Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))
Theta_Aero_FB_360 = theta_360(Theta_Aero_FB)
Theta_Aero_FB_rad = np.radians(np.array(Theta_Aero_FB))




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
PDFs = False
lineplots = False
plot_FB_stats = False
plot_stats = False
plot_F_comp_stats = False
FB_correlations = False
moving_stats = False
random_plots = False
plot_trajectory = False
plot_filtered_trajectory = False
plot_windrose = False
Iy_Iz = False
correlations_weight = False
time_shift_analysis = False
mean_trajectories = False


if plot_stats == True:

    out_dir = in_dir+"Direction/PDFs/"


    mean_magnitude = []; mean_direction = []
    variance = []
    labels = []
    rotor_weight = -1079.1
    percentage = np.linspace(0,1.0,11)
    
    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc

        mu_y = np.mean(FBy); mu_z = np.mean(FBz_perc)
        variance.append(np.mean(np.square(np.subtract(FBz_perc,mu_z))) + np.mean(np.square(np.subtract(FBy,mu_y))))

        mean_magnitude.append(np.sqrt(mu_y**2 + mu_z**2))
        mean_direction.append(np.degrees(np.arctan2(mu_z,mu_y)))

        labels.append("{} reduction".format(round(perc,1)))


    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,mean_magnitude,"o-k")
    plt.ylabel("Magnitude of mean of \nMain bearing vector [kN]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"magnitude_mean_FBR.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,mean_direction,"o-k")
    plt.ylabel("Direction of mean of \nMain bearing vector [deg]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"direction_mean_FBR.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,variance,"o-k")
    plt.ylabel("Variance of \nMain bearing force vector [kN]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"variance_FBR.png")
    plt.close()


if plot_FB_stats == True:
    out_dir = in_dir+"Direction/PDFs/"

    vars = []
    labels = []
    rotor_weight = -1079.1
    percentage = np.linspace(0,1.0,11)
    
    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc
        FBR_perc = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
        Theta_FB_perc = np.degrees(np.arctan2(FBz_perc,FBy))
        Theta_FB_perc_360 = theta_360(Theta_FB_perc)

        vars.append(Theta_FB_perc_360); vars.append(FBR_perc)
        labels.append("{} reduction".format(round(perc,1)))


    means = []
    std_dev = []
    skewness = []
    kurtosis = []
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(0,len(vars),2):

        var = vars[i]
        P,X,mu,std,skew,kr = probability_dist(var)
        means.append(mu)
        std_dev.append(std)
        skewness.append(skew)
        kurtosis.append(kr)


    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,means,"o-k")
    plt.ylabel("Mean of \nDirection of main bearing force vector [deg]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"means_direction.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,std_dev,"o-k")
    plt.ylabel("Standard deviations of \nDirection of main bearing force vector [deg]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"std_direction.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,skewness,"o-k")
    plt.ylabel("Skewness of \nDirection of main bearing force vector [-]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"skew_direction.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,skewness,"o-k")
    plt.ylabel("Kurtosis of \nDirection of main bearing force vector [-]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Kurtosis_direction.png")
    plt.close()


    means = []
    std_dev = []
    skewness = []
    kurtosis = []
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(1,len(vars),2):

        var = vars[i]
        P,X,mu,std,skew,kr = probability_dist(var)
        means.append(mu)
        std_dev.append(std)
        skewness.append(skew)
        kurtosis.append(kr)


    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,means,"o-k")
    plt.ylabel("Mean of \nMagnitude of main bearing force [kN]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"means_FBR.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,std_dev,"o-k")
    plt.ylabel("Standard deviations of \nMagnitude of main bearing force [kN]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"std_FBR.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,skewness,"o-k")
    plt.ylabel("Skewness of \nMagnitude of main bearing force [-]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"skew_FBR.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,kurtosis,"o-k")
    plt.ylabel("Kurtosis of \nMagnitude of main bearing force [-]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Kurtosis_FBR.png")
    plt.close()


if plot_F_comp_stats == True:
    out_dir = in_dir+"Direction/Tendency/"

    fig,(ax1,ax2)=plt.subplots(2,figsize=(14,8))
    ax1.plot(Time_OF,FBy)
    ax1.set_ylabel("Bearing force y compoment [kN]",fontsize=16)
    ax1.grid()

    ax2.plot(Time_OF,FBz)
    ax2.set_ylabel("Bearing force z component [kN]",fontsize=16)
    ax2.grid()

    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir+"Bearing_force_comp.png")
    plt.close()

    fig,(ax1,ax2)=plt.subplots(2,1,figsize=(14,8))
    P,X,mu,std,skew,kr = probability_dist(FBy)
    ax1.plot(X,P)
    ax1.axvline(mu,linestyle="--",color="k")
    ax1.set_ylabel("Bearing force y compoment [kN]",fontsize=16)
    ax1.set_title("mean = {}[kN], standard deviation = {}kN,\nskewness = {}, kurtosis = {}".format(mu,std,skew,kr))
    ax1.grid()

    P,X,mu,std,skew,kr = probability_dist(FBz)
    ax2.plot(X,P)
    ax2.axvline(mu,linestyle="--",color="k")
    ax2.set_ylabel("Bearing force z component [kN]",fontsize=16)
    ax2.set_title("mean = {}[kN], standard deviation = {}kN,\nskewness = {}, kurtosis = {}".format(mu,std,skew,kr))
    ax2.grid()

    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.savefig(out_dir+"Bearing_force_comp_PDF.png")
    plt.close()


if PDFs == True:
    units = ["[deg]","[kN]"]
    Ylabels = []
    vars = []
    names = ["Theta","FBR"]
    filenames = []

    rotor_weight = -1079.1
    percentage = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]

    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc
        FBR_perc = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
        Theta_FB_perc = np.degrees(np.arctan2(FBz_perc,FBy))
        Theta_FB_perc_360 = theta_360(Theta_FB_perc)

        vars.append(Theta_FB_perc_360); vars.append(FBR_perc)
        Ylabels.append("Direction main bearing force vector with \n{} reduction in weight {}".format(perc, units[0]))
        Ylabels.append("Magnitude Main Bearing Force with \n{} reduction in weight {}".format(perc, units[1]))
        filenames.append("{}_perc_{}.png".format(names[0],perc));filenames.append("{}_perc_{}.png".format(names[1],perc))


    #PDFs
    out_dir = in_dir+"Direction/PDFs/"
    for i in np.arange(0,len(vars)):
        var = vars[i];Ylabel = Ylabels[i]; filename = filenames[i]
        P,X,mu,std,skew,kr = probability_dist(var)
        fig = plt.figure(figsize=(14,8))
        plt.plot(X,P)
        plt.axvline(mu,linestyle="--",color="k")
        plt.axvline(mu+std,linestyle="--",color="r")
        plt.axvline(mu-std,linestyle="--",color="r")
        plt.ylabel("PDF",fontsize=16)
        plt.xlabel(Ylabel,fontsize=16)
        plt.title("Mean = {}, Standard deviation = {},\nskewness = {}, Kurtosis = {}".format(mu,std,skew,kr))
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+filename)
        plt.close()


    #PDFs
    out_dir = in_dir+"Direction/PDFs/"
    labels = ["Total", "0.3 reduction", "0.5 reduction", "0.7 reduction", "0.9 reduction", "Aero"]
    means = []
    std_dev = []
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(0,len(vars),2):

        var = vars[i]
        P,X,mu,std,skew,kr = probability_dist(var)
        means.append(mu)
        std_dev.append(std)
        
        plt.plot(X,P)

    plt.ylabel("PDF",fontsize=16)
    plt.xlabel("Direction of main bearing force vector [deg]",fontsize=16)
    plt.legend(labels)

    for mu in means:
        plt.axvline(mu,linestyle="--",color="k")

    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"joint_Theta.png")
    plt.close()
        

    means = []
    std_dev = []
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(1,len(vars),2):

        var = vars[i]
        P,X,mu,std,skew,kr = probability_dist(var)
        means.append(mu)
        std_dev.append(std)
        
        plt.plot(X,P)

    plt.ylabel("PDF",fontsize=16)
    plt.xlabel("Magnitude main bearing force vector [kN]",fontsize=16)
    plt.legend(labels)

    for mu in means:
        plt.axvline(mu,linestyle="--",color="k")

    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"joint_FBR.png")
    plt.close()


if lineplots == True:
    #lineplots
    vars = []; Ylabels = []; units = "[kN]"
    rotor_weight = -1079.1
    percentage = [0.0, 0.3, 0.7, 1.0]

    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc
        FBR_perc = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
        Theta_FB_perc = np.degrees(np.arctan2(FBz_perc,FBy))
        Theta_FB_perc = theta_360(Theta_FB_perc)
        #Theta_FB_perc = np.radians(np.array(Theta_FB_perc))

        vars.append(FBR_perc)
        Ylabels.append("Magnitude Main Bearing Force with \n{} reduction in weight {}".format(perc, units))

    out_dir = in_dir+"Direction/"
    fig,axs = plt.subplots(4,1,figsize=(14,10))
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    for i in np.arange(0,len(vars)):
        var = vars[i];Ylabel = Ylabels[i]
        axs[i].plot(Time_OF,var)
        axs[i].set_title("{}".format(Ylabel))
        axs[i].grid()

    fig.supxlabel("Time [s]")
    plt.savefig(out_dir+"joint_FBR_lineplot.png")
    plt.close()

    fig,axs = plt.subplots(4,1,figsize=(14,10))
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    for i in np.arange(0,len(vars)):
        var = vars[i];Ylabel = Ylabels[i]
        frq, PSD = temporal_spectra(var,dt,Ylabel)
        axs[i].loglog(frq,PSD)
        axs[i].axvline(0.605,linestyle="--",color="k")
        axs[i].set_title("{}".format(Ylabel))
        axs[i].grid()

    fig.supxlabel("Frequency [Hz]")
    plt.savefig(out_dir+"joint_spectra_FBR_lineplot.png")
    plt.close()




    #lineplots
    vars = []; Ylabels = []; units = "[deg]"

    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc
        FBR_perc = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
        Theta_FB_perc = np.degrees(np.arctan2(FBz_perc,FBy))
        Theta_FB_360_perc = theta_360(Theta_FB_perc)

        vars.append(Theta_FB_360_perc)
        Ylabels.append("Theta: Main bearing force with \n{} reduction in weight {}".format(perc, units))


    out_dir = in_dir+"Direction/"
    fig,axs = plt.subplots(4,1,figsize=(14,10))
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    for i in np.arange(0,len(vars)):
        var = vars[i];Ylabel = Ylabels[i]
        axs[i].plot(Time_OF,var)
        axs[i].set_title("{}".format(Ylabel))
        axs[i].grid()

    fig.supxlabel("Time [s]")
    plt.savefig(out_dir+"joint_Theta_lineplot.png")
    plt.close()

    fig,axs = plt.subplots(4,1,figsize=(14,10))
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    for i in np.arange(0,len(vars)):
        var = vars[i];Ylabel = Ylabels[i]
        frq, PSD = temporal_spectra(var,dt,Ylabel)
        axs[i].loglog(frq,PSD)
        axs[i].axvline(0.605,linestyle="--",color="k")
        axs[i].set_title("{}".format(Ylabel))
        axs[i].grid()

    fig.supxlabel("Frequency [Hz]")
    plt.savefig(out_dir+"joint_spectra_theta_lineplot.png")
    plt.close()


if FB_correlations == True:

    out_dir = in_dir+"Direction/FB_correlations/"

    corr = correlation_coef(FBR,FBz) 
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,FBR,"b")
    plt.plot(Time_OF,FBz,"r")
    plt.legend(["Magnitude Main bearing force","Main bearing force z component"],fontsize=12)
    plt.xlabel("Time [s]",fontsize=16)
    plt.axhline(0,linestyle="--",color="k")
    plt.ylabel("Main bearing forces [kN]",fontsize=16)
    plt.grid()
    plt.title("correlation = {}".format(round(corr,2)),fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_FBz_corr.png")
    plt.close()


    corr = correlation_coef(FBR,FBy) 
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,FBR,"b")
    plt.plot(Time_OF,FBy,"r")
    plt.legend(["Magnitude Main bearing force","Main bearing force y component"],fontsize=12)
    plt.xlabel("Time [s]",fontsize=16)
    plt.axhline(0,linestyle="--",color="k")
    plt.ylabel("Main bearing forces [kN]",fontsize=16)
    plt.grid()
    plt.title("correlation = {}".format(round(corr,2)),fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_FBy_corr.png")
    plt.close()


    corr = correlation_coef(FBR,Aero_FBz) 
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,FBR,"b")
    plt.plot(Time_OF,Aero_FBz/1000,"r")
    plt.legend(["Magnitude Main bearing force","Aerodynamic Main bearing \nforce z component"],fontsize=12)
    plt.xlabel("Time [s]",fontsize=16)
    plt.axhline(0,linestyle="--",color="k")
    plt.ylabel("Main bearing forces [kN]",fontsize=16)
    plt.grid()
    plt.title("correlation = {}".format(round(corr,2)),fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_Aero_FBz_corr.png")
    plt.close()


    corr = correlation_coef(Theta_FB_360,FBy)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,Theta_FB_360,"b")
    ax.axhline(0,linestyle="--",color="b")
    ax.set_ylabel("Theta: direction of main bearing force vector [deg]",fontsize=16)
    ax.yaxis.label.set_color('b')
    ax2 = ax.twinx()
    ax2.plot(Time_OF,FBy,"r")
    ax2.set_ylabel("Main bearing force y component [kN]",fontsize=16)
    ax2.axhline(0,linestyle="--",color="r")
    ax2.yaxis.label.set_color('r')
    fig.supxlabel("Time [s]",fontsize=16)
    fig.suptitle("Correlations = {}".format(round(corr,2)),fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Theta_FBy.png")
    plt.close()

    corr = correlation_coef(Theta_FB_360,FBz)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,Theta_FB_360,"b")
    ax.axhline(0,linestyle="--",color="b")
    ax.set_ylabel("Theta: direction of main bearing force vector [deg]",fontsize=16)
    ax.yaxis.label.set_color('b')
    ax2 = ax.twinx()
    ax2.plot(Time_OF,FBz,"r")
    ax2.axhline(0,linestyle="--",color="r")
    ax2.set_ylabel("Main bearing force z component [kN]",fontsize=16)
    ax2.yaxis.label.set_color('r')
    fig.supxlabel("Time [s]",fontsize=16)
    fig.suptitle("Correlations = {}".format(round(corr,2)),fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Theta_FBz.png")
    plt.close()


    corr = correlation_coef(Theta_FB_360,Aero_FBy)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,Theta_FB_360,"b")
    ax.axhline(0,linestyle="--",color="b")
    ax.set_ylabel("Theta: direction of main bearing force vector [deg]",fontsize=16)
    ax.yaxis.label.set_color('b')
    ax2 = ax.twinx()
    ax2.plot(Time_OF,Aero_FBy/1000,"r")
    ax2.set_ylabel("Aerodynamic Main bearing force y component [kN]",fontsize=16)
    ax2.axhline(0,linestyle="--",color="r")
    ax2.yaxis.label.set_color('r')
    fig.supxlabel("Time [s]",fontsize=16)
    fig.suptitle("Correlations = {}".format(round(corr,2)),fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Theta_Aero_FBy.png")
    plt.close()

    Py,Xy,muy,stdy,skewy,kry = probability_dist(FBy)
    Pz,Xz,muz,stdz,skewz,krz = probability_dist(FBz)
    corr = correlation_coef(FBy,FBz) 
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,FBy,"b")
    plt.axhline(muy,linestyle="--",color="k")
    plt.axhline(stdy+muy,linestyle="-.",color="k")
    plt.plot(Time_OF,FBz,"r")
    plt.axhline(muz,linestyle="--",color="k")
    plt.axhline(stdz+muz,linestyle="-.",color="k")
    plt.legend(["Main bearing force component y","$<F_{B_y}>_{T=1000}$", "$\sigma_{F_{B_y},T=1000}$","Main bearing force z component",
                "$<F_{B_z}>_{T=1000}$", "$\sigma_{F_{B_z},T=1000}$"],fontsize=12)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Main bearing forces [kN]",fontsize=16)
    plt.grid()
    plt.title("Correlation = "+str(round(corr,2))+"\n$\sigma ^2_{F_{B_y},T=1000}$ = "+"{:e}".format(stdy**2)+
              "\n$\sigma ^2_{F_{B_z},T=1000}$ = "+"{:e}".format(stdz**2),fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir+"FBy_FBz_corr.png")
    plt.close()


    corr = correlation_coef(np.square(FBR),np.square(FBz)) 
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.square(FBR),"b")
    plt.plot(Time_OF,np.square(FBz),"r")
    plt.legend(["Magnitude Main bearing force squared","Main bearing force z component squared"],fontsize=12)
    plt.xlabel("Time [s]",fontsize=16)
    plt.axhline(0,linestyle="--",color="k")
    plt.ylabel("Main bearing forces squared [$kN^2$]",fontsize=16)
    plt.grid()
    plt.title("correlation = {}".format(round(corr,2)),fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_FBz_squared_corr.png")
    plt.close()


    corr = correlation_coef(np.square(FBR),np.square(FBy)) 
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.square(FBR),"b")
    plt.plot(Time_OF,np.square(FBy),"r")
    plt.legend(["Magnitude Main bearing force squared","Main bearing force y component squared"],fontsize=12)
    plt.xlabel("Time [s]",fontsize=16)
    plt.axhline(0,linestyle="--",color="k")
    plt.ylabel("Main bearing forces squared [$kN^2$]",fontsize=16)
    plt.grid()
    plt.title("correlation = {}".format(round(corr,2)),fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir+"FBR_FBy_squared_corr.png")
    plt.close()


    Py,Xy,muy,stdy,skewy,kry = probability_dist(np.square(FBy))
    Pz,Xz,muz,stdz,skewz,krz = probability_dist(np.square(FBz))
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.square(FBy),"b")
    plt.axhline(muy,linestyle="--",color="k")
    plt.axhline(stdy+muy,linestyle="-.",color="k")
    plt.plot(Time_OF,np.square(FBz),"r")
    plt.axhline(muz,linestyle="--",color="k")
    plt.axhline(stdz+muz,linestyle="-.",color="k")
    plt.legend(["Main bearing force component y squared","$<(F_{B_y})^2>_{T=1000}$", "$\sigma_{F_{B_y}^2,T=1000}$",
                "Main bearing force z component squared", "$<(F_{B_z})^2>_{T=1000}$", "$\sigma_{F_{B_z}^2,T=1000}$"],fontsize=12)
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Main bearing forces squared [$kN^2$]",fontsize=16)
    plt.grid()
    plt.title("$\sigma ^2_{F_{B_y}^2,T=1000}$ = "+"{:e}".format(stdy**2)+"\n$\sigma ^2_{F_{B_z}^2,T=1000}$ = "+"{:e}".format(stdz**2),fontsize=14)
    plt.tight_layout()
    plt.savefig(out_dir+"FBy_FBz_squared_corr.png")
    plt.close()


if moving_stats == True:

    out_dir = in_dir+"Direction/FB_correlations/"
    time_window = 50
    window = int(time_window/dt)
    dy = pd.Series(np.square(FBy))
    dz = pd.Series(np.square(FBz))
    vary = dy.rolling(window=window).var()
    varz = dz.rolling(window=window).var()


    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,vary)
    plt.plot(Time_OF,varz)
    plt.yscale("log")
    plt.ylabel("Local variance of main bearing force components squared \ntime window = 50s",fontsize=16)
    plt.xlabel("Time [s]",fontsize=16)
    plt.legend(["[$\sigma ^2 _{F_{B_y}^2, T=50s}$]", "[$\sigma ^2 _{F_{B_z}^2, T=1000s}$]"],fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"variance_FBy_FBz.png")
    plt.close()


if random_plots == True:
    Py,Xy,muy,stdy,skewy,kry = probability_dist(np.square(FBy))
    Pz,Xz,muz,stdz,skewz,krz = probability_dist(np.square(FBz))


    fig=plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.square(FBy))
    plt.plot(Time_OF,np.square(FBz))
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Main bearing force components squared [$kN^2$]",fontsize=16)
    plt.title("FBy Variance = {}\nFBz Variance = {}".format(round(stdy**2,0),round(stdz**2,0)),fontsize=16)
    plt.legend(["$FBy^2$", "$FBz^2$"])
    plt.grid()
    plt.tight_layout()
    plt.savefig(in_dir+"Direction/FB_comps_squared.png")
    plt.close()


    fig=plt.figure(figsize=(14,8))
    plt.plot(Xy,Py)
    plt.plot(Xz,Pz)
    plt.xlabel("Main bearing force components squared [$kN^2$]",fontsize=16)
    plt.ylabel("PDF",fontsize=16)
    plt.legend(["$FBy^2$", "$FBz^2$"])
    plt.axvline(muy,linestyle="--",color="k")
    plt.axvline(muz,linestyle="--",color="k")
    plt.grid()
    plt.tight_layout()
    plt.savefig(in_dir+"Direction/PDF_FB_comps_squared.png")
    plt.close()


    plt.rcParams.update({'font.size': 16})
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.subtract(Ux,np.mean(Ux)))
    plt.axhline(0,color="k")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor averaged horizontal velocity [m/s]")
    plt.title("Mean = {}".format(round(np.mean(Ux),2)))
    labels = np.arange(200,1300,100)
    plt.xticks(labels)
    plt.grid()
    plt.tight_layout()
    plt.savefig(in_dir+"Ux_fluc.png")
    plt.close()


if plot_trajectory == True:

    out_dir = in_dir+"Direction/Trajectories/"

    percentage = [0.0]
    start_times = [200]
    end_times = [1200]

    for time_start,time_end in zip(start_times,end_times):

        for perc in percentage:

            rotor_weight = -1079.1

            Fzs = LSShftFzs-(perc*rotor_weight)
            FBFz_perc = -Fzs*((L1+L2)/L2)
            FBz_perc = FBMz + FBFz_perc

            FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
            Theta = np.degrees(np.arctan2(FBz_perc,FBy))
            Theta_FB_360 = theta_360(Theta)
            Theta_rad = np.radians(np.array(Theta_FB_360))
            


            Time_start_idx = np.searchsorted(Time_OF,time_start); Time_end_idx = np.searchsorted(Time_OF,time_end)
            time_steps = np.arange(Time_start_idx,Time_end_idx)

            mean_y = np.mean(FBy[Time_start_idx:Time_end_idx]); mean_z = np.mean(FBz_perc[Time_start_idx:Time_end_idx])
            mean_mag = np.sqrt(mean_y**2 + mean_z**2)
            mean_dir = np.arctan2(mean_z,mean_y)
            std = np.sqrt( np.mean( np.square( np.subtract(FBy[Time_start_idx:Time_end_idx],mean_y ))) + 
                           np.mean( np.square( np.subtract(FBz_perc[Time_start_idx:Time_end_idx],mean_z ))))
            print(time_start,time_end,std)
            fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8))
            ax1.plot(Time_OF[Time_start_idx:Time_end_idx],FBR[Time_start_idx:Time_end_idx])
            ax1.set_ylabel("Magnitude main bearing force [kN]]")
            ax1.grid()
            ax2.plot(Time_OF[Time_start_idx:Time_end_idx],Theta_FB_360[Time_start_idx:Time_end_idx])
            ax2.set_ylabel("Theta: Direction main bearing force vector [deg]")
            ax2.grid()
            fig.supxlabel("Time [s]")
            fig.suptitle("{} - {}s".format(time_start,time_end))
            plt.tight_layout()
            plt.savefig(out_dir+"lineplot_{}-{}s_{}_reduction.png".format(time_start,time_end,perc))
            plt.close()

            fig,(ax1,ax2) = plt.subplots(1,2,figsize=(14,8))
            P,X,mu,std,skew,kr = probability_dist(FBR[Time_start_idx:Time_end_idx])
            
            ax1.plot(X,P)
            ax1.set_xlabel("Magnitude main bearing force [kN]")
            ax1.set_xlim([0,3500])
            ax1.grid()
            P,X,mu,std,skew,kr = probability_dist(Theta_FB_360[Time_start_idx:Time_end_idx])
            ax2.plot(X,P)
            ax2.set_xlabel("Theta: Direction main bearing force vector [deg]")
            ax2.set_xlim([0,360])
            ax2.grid()
            fig.supylabel("PDF")
            fig.suptitle("{} - {}s".format(time_start,time_end))
            plt.tight_layout()
            plt.savefig(out_dir+"PDF_{}-{}s_{}_reduction.png".format(time_start,time_end,perc))
            plt.close()

            

            fig = plt.figure(figsize=(6,6))
            ax = plt.subplot(111,polar=True)
            ax.set_ylim(top=3500)
            ax.plot(Theta_rad[Time_start_idx:Time_end_idx],FBR[Time_start_idx:Time_end_idx],marker="o",color="b",markersize=0.1)
            ax.plot(mean_dir,mean_mag,marker="o",color="k",markersize=8)

            ax.set_title("Main bearing force vector trajectory\n{} reduction in weight\nT={} - {}s".format(perc,round(Time_OF[Time_start_idx],0),round(Time_OF[Time_end_idx-1],0)))
            plt.tight_layout()
            plt.savefig(out_dir+"{}-{}s_{}_reduction.png".format(time_start,time_end,perc))
            plt.close()


if Iy_Iz == True:

    out_dir = in_dir+"Direction/FB_correlations/"
    fig,(ax1,ax3) = plt.subplots(2,figsize=(14,8))

    ax1.plot(Time_OF,FBy,"b")
    ax1.set_ylabel("Main bearing\nforce y component [kN]",fontsize=12)
    ax1.yaxis.label.set_color('b')
    ax2=ax1.twinx()
    ax2.plot(Time_OF,Iz,"r")
    ax2.set_ylabel("Asymmetry\naround z axis [$m^4/s]$",fontsize=12)
    ax2.yaxis.label.set_color('r')

    ax3.plot(Time_OF,FBz,"b")
    ax3.set_ylabel("Main bearing\nforce z component [kN]",fontsize=12)
    ax3.yaxis.label.set_color('b')
    ax4=ax3.twinx()
    ax4.plot(Time_OF,Iy,"r")
    ax4.set_ylabel("Asymmetry\naround y axis [$m^4/s]$",fontsize=12)
    ax4.yaxis.label.set_color('r')
    
    LPF_Iy = low_pass_filter(Iy,0.3,dt)
    LPF_Iz = low_pass_filter(Iz,0.3,dt)
    LPF_FBy = low_pass_filter(FBy,0.3,dt)
    LPF_FBz = low_pass_filter(FBz,0.3,dt)

    ax1.plot(Time_OF,LPF_FBy,"--r")
    ax1.legend(["Total signal","Trend, LPF 0.3Hz"])
    ax2.plot(Time_OF,LPF_Iz,"--b")
    ax2.legend(["Total signal","Trend, LPF 0.3Hz"])
    ax3.plot(Time_OF,LPF_FBz,"--r")
    ax3.legend(["Total signal","Trend, LPF 0.3Hz"])
    ax4.plot(Time_OF,LPF_Iy,"--b")
    ax4.legend(["Total signal","Trend, LPF 0.3Hz"])

    corr_Iy_FBz = correlation_coef(Iy,FBz)
    corr_Iz_FBy = correlation_coef(Iz,FBy)

    LPF_corr_Iy_FBz = correlation_coef(LPF_Iy,LPF_FBz)
    LPF_corr_Iz_FBy = correlation_coef(LPF_Iz,LPF_FBy)

    ax1.set_title("Total correlation = {}\nTrend correlation = {}".format(round(corr_Iz_FBy,2),round(LPF_corr_Iz_FBy,2)),fontsize=12)
    ax3.set_title("Total correlation = {}\nTrend correlation = {}".format(round(corr_Iy_FBz,2),round(LPF_corr_Iy_FBz,2)),fontsize=12)
    ax1.grid()
    ax3.grid()
    fig.supxlabel("Time [s]",fontsize=12)

    plt.tight_layout()
    plt.savefig(out_dir+"Iy_Iz_correlations.png")
    plt.close()

    time_window = 50
    window = int(time_window/dt)

    fig,(ax1,ax3) = plt.subplots(2,figsize=(14,8))

    FBy_s = pd.Series(FBy)
    FBz_s = pd.Series(FBz)
    FBy_var = FBy_s.rolling(window=window).var()
    FBz_var = FBz_s.rolling(window=window).var()

    ax1.plot(Time_OF,FBy_var,"b")
    ax1.set_ylabel("Local variance Main bearing\nforce y component [kN], T = 50s",fontsize=12)
    ax1.yaxis.label.set_color('b')
    ax2=ax1.twinx()
    ax2.plot(Time_OF,abs(Iz),"r")
    ax2.set_ylabel("Absolule asymmetry\naround z axis [$m^4/s]$",fontsize=12)
    ax2.yaxis.label.set_color('r')

    ax3.plot(Time_OF,FBz_var,"b")
    ax3.set_ylabel("Local variance Main bearing\nforce z component [kN], T = 50s",fontsize=12)
    ax3.yaxis.label.set_color('b')
    ax4=ax3.twinx()
    ax4.plot(Time_OF,abs(Iy),"r")
    ax4.set_ylabel("Absolule asymmetry\naround y axis [$m^4/s]$",fontsize=12)
    ax4.yaxis.label.set_color('r')

    corr_Iy_FBz = correlation_coef(FBz_var[window:],abs(Iy[window:]))
    corr_Iz_FBy = correlation_coef(FBy_var[window:],abs(Iz[window:]))

    ax1.set_title("Correlation = {}".format(round(corr_Iz_FBy,2)),fontsize=16)
    ax3.set_title("Correlation = {}".format(round(corr_Iy_FBz,2)),fontsize=16)
    ax1.grid()
    ax3.grid()
    fig.supxlabel("Time [s]",fontsize=16)

    plt.tight_layout()
    plt.savefig(out_dir+"Iy_Iz_variance_correlations.png")
    plt.close()


    fig,(ax1,ax3) = plt.subplots(2,figsize=(14,8))

    LPF_FBy_s = pd.Series(LPF_FBy)
    LPF_FBz_s = pd.Series(LPF_FBz)
    LPF_FBy_var = LPF_FBy_s.rolling(window=window).var()
    LPF_FBz_var = LPF_FBz_s.rolling(window=window).var()

    ax1.plot(Time_OF,LPF_FBy_var,"b")
    ax1.set_ylabel("Local trend variance Main bearing\nforce y component [kN], T = 50s",fontsize=12)
    ax1.yaxis.label.set_color('b')
    ax2=ax1.twinx()
    ax2.plot(Time_OF,abs(Iz),"r")
    ax2.set_ylabel("Absolule asymmetry\naround z axis [$m^4/s]$",fontsize=12)
    ax2.yaxis.label.set_color('r')

    ax3.plot(Time_OF,LPF_FBz_var,"b")
    ax3.set_ylabel("Local trend variance Main bearing\nforce z component [kN], T = 50s",fontsize=12)
    ax3.yaxis.label.set_color('b')
    ax4=ax3.twinx()
    ax4.plot(Time_OF,abs(Iy),"r")
    ax4.set_ylabel("Absolule asymmetry\naround y axis [$m^4/s]$",fontsize=12)
    ax4.yaxis.label.set_color('r')

    corr_Iy_FBz = correlation_coef(abs(Iy[window:]),LPF_FBz_var[window:])
    corr_Iz_FBy = correlation_coef(Iz[window:],LPF_FBy_var[window:])

    ax1.set_title("Correlation = {}".format(round(corr_Iz_FBy,2)),fontsize=16)
    ax3.set_title("Correlation = {}".format(round(corr_Iy_FBz,2)),fontsize=16)
    ax1.grid()
    ax3.grid()
    fig.supxlabel("Time [s]",fontsize=16)

    plt.tight_layout()
    plt.savefig(out_dir+"trend_variance_Iy_Iz.png")
    plt.close()
    

if plot_windrose == True:

    #plot windrose of Bearing Force

    ax = WindroseAxes.from_ax()
    ax.bar(Aero_Theta,Aero_FBR/1000,normed=True,opening=0.8,edgecolor="white")
    ax.set_xticklabels(['0', '45', '90', '135',  '180', '225', '270', '315'])
    ax.set_legend()

    X,P,mu,std = probability_dist(Aero_Theta)
    plt.figure(figsize=(14,8))
    plt.plot(P,X)

    plt.show()


if correlations_weight == True:

    out_dir = in_dir+"Direction/PDFs/"
    percentage = np.linspace(0,1,11)

    corr_FBR_FBz = []
    corr_Theta_FBy = []
    corr_FBR_FBy = []
    corr_Theta_FBz = []
    corr_FBy_FBz = []
    for perc in percentage:
        rotor_weight = -1079.1

        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc

        FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
        Theta = np.degrees(np.arctan2(FBz_perc,FBy))
        Theta_FB_360 = theta_360(Theta)


        corr_FBR_FBz.append(correlation_coef(FBR,FBz_perc))
        corr_Theta_FBy.append(correlation_coef(Theta_FB_360,FBy))
        corr_FBR_FBy.append(correlation_coef(FBR,FBy))
        corr_Theta_FBz.append(correlation_coef(Theta_FB_360,FBz_perc))
        corr_FBy_FBz.append(correlation_coef(FBy,FBz_perc))


    fig = plt.figure(figsize=(14,8))
    plt.plot(percentage,corr_FBR_FBz)
    plt.xlabel("Percentage reduction",fontsize=16)
    plt.ylabel("Correlation coefficient magnitude main bearing force,\nmain bearing force z component",fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.savefig(out_dir+"cc_FBR_FBz.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(percentage,corr_FBR_FBy)
    plt.xlabel("Percentage reduction",fontsize=16)
    plt.ylabel("Correlation coefficient magnitude main bearing force,\nmain bearing force y component",fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.savefig(out_dir+"cc_FBR_FBy.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(percentage,corr_Theta_FBy)
    plt.xlabel("Percentage reduction",fontsize=16)
    plt.ylabel("Correlation coefficient Theta:direction main bearing force,\nmain bearing force y component",fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.savefig(out_dir+"cc_Theta_FBy.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(percentage,corr_Theta_FBz)
    plt.xlabel("Percentage reduction",fontsize=16)
    plt.ylabel("Correlation coefficient Theta:direction main bearing force,\nmain bearing force z component",fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.savefig(out_dir+"cc_Theta_FBz.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(percentage,corr_FBy_FBz)
    plt.xlabel("Percentage reduction",fontsize=16)
    plt.ylabel("Correlation coefficient main bearing force y component,\nmain bearing force z component",fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.savefig(out_dir+"cc_FBy_FBz.png")
    plt.close()


if time_shift_analysis == True:

    out_dir = in_dir+"Direction/lineplots/"
    # LPF_My = low_pass_filter(LSSTipMys,0.3,dt)
    # LPF_Mz = low_pass_filter(LSSTipMzs,0.3,dt)

    # My = np.subtract(LSSTipMys,LPF_My)
    # Mz = np.subtract(LSSTipMzs,LPF_Mz)

    My = low_pass_filter(LSSTipMys,1.0,dt)
    Mz = low_pass_filter(LSSTipMzs,1.0,dt)

    corr = correlation_coef(My,Mz)

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,My,"r")
    plt.plot(Time_OF,Mz,"b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Rotor moments [kN-m]\nlow pass filtered 1.0Hz",fontsize=16)
    plt.legend(["$\widetilde{M}_y$","$\widetilde{M}_z$"],fontsize=14)
    plt.xlim([200,240])
    plt.title("Correlation = {}".format(round(corr,2)),fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"My_Mz.png")
    plt.close()


    dMy_dt = dt_calc(My,dt)
    dMz_dt = dt_calc(Mz,dt)

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[:-1],dMy_dt,"r")
    plt.plot(Time_OF[:-1],dMz_dt,"b")
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Time derivative Rotor moments [kN-m/s]",fontsize=16)
    plt.legend(["$d\widetilde{M}_y/dt$","$d\widetilde{M}_z/dt$"],fontsize=14,loc="upper right")
    plt.xlim([200,240])
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dMy_dMz_dt.png")
    plt.close()

    zero_crossings_My = np.where(np.diff(np.sign(dMy_dt)))[0]
    zero_crossings_Mz = np.where(np.diff(np.sign(dMz_dt)))[0]
    zero_crossings_Mz = zero_crossings_Mz[1:]

    time_shift = []
    Time = []
    for i,j in zip(zero_crossings_My,zero_crossings_Mz):

        time_shift.append(Time_OF[j]-Time_OF[i])
        Time.append(np.average([Time_OF[j],Time_OF[i]]))

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time,time_shift)
    plt.title("Mean time shift = {}s".format(round(np.mean(time_shift),2)))
    plt.xlabel("Time [s]",fontsize=16)
    plt.ylabel("Time shift [s]",fontsize=16)
    plt.grid()
    plt.axhline(np.mean(time_shift),linewidth=1,color="r")
    plt.axhline(0.4125,color="k",linestyle="--")
    plt.tight_layout()
    plt.savefig(out_dir+"time_shift.png")
    plt.savefig(out_dir+"time_shift.png")
    plt.close()

    # fig = plt.figure(figsize=(14,8))
    # plt.plot(Time_OF,My,"r")
    # plt.plot(Time_OF,Mz,"b")
    # plt.xlabel("Time [s]",fontsize=16)
    # plt.ylabel("Trend Fluctuating Rotor moments [kN-m]\nlow pass filtered 1.0Hz",fontsize=16)
    # plt.legend(["$\widetilde{M}'_y$","$\widetilde{M}'_z$"],fontsize=14)
    # #plt.xlim([200,240])
    # plt.title("Correlation = {}".format(round(corr,2)),fontsize=14)
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(out_dir+"Trend_Fluc_My_Mz.png")
    # plt.close()


    # dMy_dt = dt_calc(My,dt)
    # dMz_dt = dt_calc(Mz,dt)

    # fig = plt.figure(figsize=(14,8))
    # plt.plot(Time_OF[:-1],dMy_dt,"r")
    # plt.plot(Time_OF[:-1],dMz_dt,"b")
    # plt.xlabel("Time [s]",fontsize=16)
    # plt.ylabel("Trend fluctuating\nTime derivative Rotor moments [kN-m/s]",fontsize=16)
    # plt.legend(["$d\widetilde{M}'_y/dt$","$d\widetilde{M}'_z/dt$"],fontsize=14,loc="upper right")
    # plt.xlim([200,240])
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(out_dir+"Trend_fluc_dMy_dMz_dt.png")
    # plt.close()

    # zero_crossings_My = np.where(np.diff(np.sign(dMy_dt)))[0]
    # zero_crossings_Mz = np.where(np.diff(np.sign(dMz_dt)))[0]
    # zero_crossings_Mz = zero_crossings_Mz[1:]

    # time_shift = []
    # Time = []
    # for i,j in zip(zero_crossings_My,zero_crossings_Mz):

    #     time_shift.append(Time_OF[j]-Time_OF[i])
    #     Time.append(np.average([Time_OF[j],Time_OF[i]]))

    # fig = plt.figure(figsize=(14,8))
    # plt.plot(Time,time_shift)
    # plt.title("Mean time shift = {}s".format(round(np.mean(time_shift),2)))
    # plt.xlabel("Time [s]",fontsize=16)
    # plt.ylabel("Time shift [s]",fontsize=16)
    # plt.grid()
    # plt.axhline(np.mean(time_shift),linewidth=1,color="r")
    # plt.axhline(0.4125,color="k",linestyle="--")
    # plt.tight_layout()
    # plt.savefig(out_dir+"Trend_fluc_time_shift.png")
    # plt.savefig(out_dir+"Trend_fluc_time_shift.png")
    # plt.close()


if plot_filtered_trajectory == True:

    out_dir = in_dir+"Direction/Trajectories/"

    start_times = [200]
    end_times = [1200]

    for time_start,time_end in zip(start_times,end_times):

        FBz = low_pass_filter(FBz,0.3,dt)
        FBy = low_pass_filter(FBy,0.3,dt)

        FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
        Theta = np.degrees(np.arctan2(FBz,FBy))
        Theta_FB_360 = theta_360(Theta)
        Theta_rad = np.radians(np.array(Theta_FB_360))

        Time_start_idx = np.searchsorted(Time_OF,time_start); Time_end_idx = np.searchsorted(Time_OF,time_end)
        time_steps = np.arange(Time_start_idx,Time_end_idx)

        mean_y = np.mean(FBy[Time_start_idx:Time_end_idx]); mean_z = np.mean(FBz[Time_start_idx:Time_end_idx])
        mean_mag = np.sqrt(mean_y**2 + mean_z**2)
        mean_dir = np.arctan2(mean_z,mean_y)
        std = np.sqrt( np.mean( np.square( np.subtract(FBy[Time_start_idx:Time_end_idx],mean_y ))) + 
                        np.mean( np.square( np.subtract(FBz[Time_start_idx:Time_end_idx],mean_z ))))
        print(time_start,time_end,std)
        

        fig = plt.figure(figsize=(6,6))
        ax = plt.subplot(111,polar=True)
        ax.set_ylim(top=3500)
        ax.plot(Theta_rad[Time_start_idx:Time_end_idx],FBR[Time_start_idx:Time_end_idx],marker="o",color="b",markersize=0.1)
        ax.plot(mean_dir,mean_mag,marker="o",color="k",markersize=8)

        ax.set_title("Main bearing force vector trajectory\nFiltered at 0.3Hz \nT={} - {}s".format(round(Time_OF[Time_start_idx],0),round(Time_OF[Time_end_idx-1],0)))
        plt.tight_layout()
        plt.savefig(out_dir+"Trend_{}-{}s.png".format(time_start,time_end))
        plt.close()


if mean_trajectories == True:

    out_dir = in_dir+"Direction/Trajectories/"
    percentage = [0.0,0.3,0.8,1.0]
    colors = ["r","b","g","c"]
    fig = plt.figure(figsize=(6,6))
    ax = plt.subplot(111,polar=True)
    for perc,color in zip(percentage,colors):

        rotor_weight = -1079.1

        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc

        FBR = np.sqrt(np.add(np.square(np.mean(FBy)),np.square(np.mean(FBz_perc))))
        Theta = np.arctan2(np.mean(FBz_perc),np.mean(FBy))
        plt.plot(Theta,FBR,marker="o",markersize=10,color=color)

    plt.legend(["0.0 reduction", "0.3 reduction", "0.8 reduction", "1.0 reduction"])   

    for perc,color in zip(percentage,colors):
        rotor_weight = -1079.1

        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc

        FBR = np.sqrt(np.add(np.square(np.mean(FBy)),np.square(np.mean(FBz_perc))))
        Theta = np.arctan2(np.mean(FBz_perc),np.mean(FBy))
        plt.arrow(0,0,Theta,FBR, width = 0.02,
                    color=color, lw = 2, zorder = 5)
        
    plt.title("Mean main bearing force vector [kN,deg]",fontsize=16)    
    plt.tight_layout()
    plt.savefig(out_dir+"Trajectories_means.png")
    plt.close()


out_dir=in_dir+"Direction/stats/"
times = [200,300,375,440,540,650,810,925,990,1100,1160,1200]
variance = [575.2,561.3,501.9,588.3,533.8,915.1,949.0,808.3,607.9,424.8,588.5,588.5]
LPF_1_ratio = [0.73,0.89,0.68,0.78,0.77,0.76,0.88,0.67,0.74,0.53,0.83,0.83]
HPF_1_ratio = [0.66,0.44,0.69,0.59,0.58,0.59,0.42,0.67,0.55,0.74,0.54,0.54]
HPF_2_ratio = [0.14,0.14,0.17,0.15,0.17,0.15,0.10,0.16,0.12,0.15,0.16,0.16]


plt.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(14,8))
plt.plot(times,variance,"-o")
plt.axhline(758.4)
plt.xlabel("Time [s]")
plt.ylabel("Local variance over period T\nMain bearing force vector")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"variance.png")
plt.close()

plt.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(14,8))
plt.plot(times,LPF_1_ratio,"-o")
plt.axhline(0.82)
plt.xlabel("Time [s]")
plt.ylabel("Ratio Local variance low pass filterd f = 0.3Hz,\nLocal variance over period T\nMain bearing force vector")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ratio_1_variance.png")
plt.close()

plt.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(14,8))
plt.plot(times,HPF_1_ratio,"-o")
plt.axhline(0.56)
plt.xlabel("Time [s]")
plt.ylabel("Ratio Local variance high pass filterd f = 0.3Hz,\nLocal variance over period T\nMain bearing force vector")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ratio_2_variance.png")
plt.close()

plt.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(14,8))
plt.plot(times,HPF_2_ratio,"-o")
plt.axhline(0.19)
plt.xlabel("Time [s]")
plt.ylabel("Ratio Local variance high pass filterd f = 1.0Hz,\nLocal variance over period T\nMain bearing force vector")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ratio_3_variance.png")
plt.close()


times = [200,300,375,440,540,650,810,925,990,1100,1160,1200]
T = []
for i in np.arange(0,len(times)-1):
    T.append(times[i+1]-times[i])

variance = [575.2,561.3,501.9,588.3,533.8,915.1,949.0,808.3,607.9,424.8,588.5]
LPF_1_ratio = [0.73,0.89,0.68,0.78,0.77,0.76,0.88,0.67,0.74,0.53,0.83]
HPF_1_ratio = [0.66,0.44,0.69,0.59,0.58,0.59,0.42,0.67,0.55,0.74,0.54]
HPF_2_ratio = [0.14,0.14,0.17,0.15,0.17,0.15,0.10,0.16,0.12,0.15,0.16]

sort_T = []; sort_var = []; sort_LPF_1_ratio = []; sort_HPF_1_ratio = []; sort_HPF_2_ratio = []

for i in np.argsort(T):
    sort_T.append(T[i])
    sort_var.append(variance[i])
    sort_LPF_1_ratio.append(LPF_1_ratio[i])
    sort_HPF_1_ratio.append(HPF_1_ratio[i])
    sort_HPF_2_ratio.append(HPF_2_ratio[i])


plt.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(14,8))
plt.plot(sort_T,sort_var,"-o")
plt.xlabel("Period T [s]")
plt.ylabel("Local variance over period T\nMain bearing force vector")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"sorted_variance.png")
plt.close()

plt.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(14,8))
plt.plot(sort_T,sort_LPF_1_ratio,"-o")
plt.xlabel("Period T [s]")
plt.ylabel("Ratio Local variance low pass filterd f = 0.3Hz,\nLocal variance over period T\nMain bearing force vector")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"sorted_Ratio_1_variance.png")
plt.close()

plt.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(14,8))
plt.plot(sort_T,sort_HPF_1_ratio,"-o")
plt.xlabel("Period T [s]")
plt.ylabel("Ratio Local variance high pass filterd f = 0.3Hz,\nLocal variance over period T\nMain bearing force vector")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"sorted_Ratio_2_variance.png")
plt.close()

plt.rcParams.update({'font.size': 16})
fig=plt.figure(figsize=(14,8))
plt.plot(sort_T,sort_HPF_2_ratio,"-o")
plt.xlabel("Period T [s]")
plt.ylabel("Ratio Local variance high pass filterd f = 1.0Hz,\nLocal variance over period T\nMain bearing force vector")
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"sorted_Ratio_3_variance.png")
plt.close()

# time_start = 200; time_end = 300
# Time_start_idx = np.searchsorted(Time_OF,time_start); Time_end_idx = np.searchsorted(Time_OF,time_end)
# time_steps = np.arange(Time_start_idx,Time_end_idx,10)

# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(1,2,1)
# ax.set_ylim([np.min(FBz),np.max(FBz)])
# ax.set_xlim([np.min(FBy),np.max(FBy)])
# ax.set_xlabel("Main bearing force y direction [kN]",fontsize=16)
# ax.set_ylabel("Main bearing force z direction [kN]",fontsize=16)
# ax.grid()

# ax2 = plt.subplot(1,2,2,projection='polar')
# ax2.set_ylim(top=np.max(FBR))

# def frame(i):

#     ax.plot(FBy[:i],FBz[:i],linestyle="-",marker="o",color="b",markersize=0.5)
#     ax.set_title("{}s".format(Time_OF[i]),fontsize=14)

#     ax2.plot(Theta_FB_rad[:i],FBR[:i],linestyle="-",marker="o",color="r",markersize=0.5)
#     ax2.set_title("{}s\nMagnitude and Direction\nMain bearing force vector [kN,deg]".format(Time_OF[i]),fontsize=14)


# animation = FuncAnimation(fig, func=frame, frames=time_steps, interval=1)
# plt.show()

# time_start = 200; time_end = 300
# Time_start_idx = np.searchsorted(Time_OF,time_start); Time_end_idx = np.searchsorted(Time_OF,time_end)
# time_steps = np.arange(Time_start_idx,Time_end_idx,5)

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF[Time_start_idx:Time_end_idx],FBR[Time_start_idx:Time_end_idx],"r")
# ax.set_ylabel("FBR")
# ax.set_ylabel("Main bearing force magnitude [kN]",fontsize=16)
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_OF[Time_start_idx:Time_end_idx],Theta_FB_360[Time_start_idx:Time_end_idx],"b")
# ax2.set_ylabel("Theta: Direction main bearing force vector [deg]",fontsize=16)
# fig.supxlabel("Time [s]",fontsize=16)
# fig.suptitle("{} - {}s".format(time_start,time_end),fontsize=16)
# plt.tight_layout()

# fig = plt.figure(figsize=(6,6))
# ax = plt.subplot(111,polar=True)
# ax.set_ylim(top=np.max(FBR))

# def frame(i):

#     ax.plot(Theta_FB_rad[i],FBR[i],linestyle="-",marker="o",color="b",markersize=0.5)
#     ax.set_title("{}".format(Time_OF[i]))


# animation = FuncAnimation(fig, func=frame, frames=time_steps, interval=2)
# plt.show()


# def frame(i):

    

#     ax.plot(Theta_FB_rad[i],FBR[i],linestyle="-",marker="o",color="b",markersize=0.5)
#     ax.set_title("{}".format(Time_OF[i]))


# animation = FuncAnimation(fig, func=frame, frames=time_steps, interval=2)
# plt.show()