from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from windrose import WindroseAxes
from scipy import stats
import pandas as pd

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
    no_bin = 1000
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




in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])

dt = Time_OF[1] - Time_OF[0]

Time_start = 200
Time_end = 1200

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


#plotting options
PDFs = False
lineplots = False
plot_FB_stats = False
plot_Fz_stats = False
plot_F_comp_stats = False
FB_correlations = False
moving_stats = True
random_plots = False


if plot_Fz_stats == True:
    out_dir = in_dir+"Direction/PDFs/"

    vars = []
    labels = []
    rotor_weight = -1079.1
    percentage = np.linspace(0,1.0,11)
    
    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc

        vars.append(np.sqrt(np.square(FBz_perc)+np.square(FBy)))
        labels.append("{} reduction".format(round(perc,1)))


    means = []
    std_dev = []
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(0,len(vars)):

        var = vars[i]
        P,X,mu,std,skew,kr = probability_dist(var)
        means.append(mu)
        std_dev.append(std)


    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,means,"o-k")
    plt.ylabel("Mean of \nMain bearing force z squared [$kN^2$]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"means_squared_FBz.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,std_dev,"o-k")
    plt.ylabel("Standard deviations of \nMain bearing force z squared [$kN^2$]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"std_squared_FBz.png")
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
    corr = correlation_coef(FBR,RtAeroMys/1000) 
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,FBR,"b")
    ax.set_ylabel("Main bearing force magnitude [kN]",fontsize=16)
    ax.yaxis.label.set_color('b')
    ax.grid()

    ax2=ax.twinx()
    ax2.plot(Time_OF,RtAeroMys/1000,"r")
    ax2.set_ylabel("Aerodynamic Rotor moment y component [kN-m]",fontsize=16)
    ax2.yaxis.label.set_color("r")

    plt.xlabel("Time [s]",fontsize=16)
    plt.title("correlation = {}".format(round(corr,2)))
    plt.tight_layout()
    plt.savefig(out_dir+"AeroMy_corr.png")
    plt.close()


    corr = correlation_coef(Theta_FB_360,RtAeroMzs/1000) 
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_OF,Theta_FB_360,"b")
    ax.set_ylabel("Main bearing force direction [deg]",fontsize=16)
    ax.yaxis.label.set_color('b')
    ax.grid()

    ax2=ax.twinx()
    ax2.plot(Time_OF,RtAeroMzs/1000,"r")
    ax2.set_ylabel("Aerodynamic Rotor moment z component [kN-m]",fontsize=16)
    ax2.yaxis.label.set_color("r")

    plt.xlabel("Time [s]",fontsize=16)
    plt.title("correlation = {}".format(round(corr,2)))
    plt.tight_layout()
    plt.savefig(out_dir+"AeroMz_corr.png")
    plt.close()



if moving_stats == True:

    out_dir = in_dir+"Direction/"
    time_window = 50
    window = int(time_window/dt)
    dy = pd.Series(np.square(FBy))
    dz = pd.Series(np.square(FBz))
    vary = dy.rolling(window=window).var()
    varz = dz.rolling(window=window).var()


    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,vary)
    plt.plot(Time_OF,varz)
    plt.ylabel("variance of main bearing force components squared \ntime window = 50s",fontsize=16)
    plt.xlabel("Time [s]",fontsize=16)
    plt.legend(["[$F_{B_y}^2$]", "[$F_{B_z}^2$]"],fontsize=14)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"variance_FB.png")
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




# fig,ax = plt.subplots(figsize=(14,8))
# y = []
# for i in np.arange(0,len(Time_OF)):
#     if abs(FBy[i]) >= abs(0.2*FBz[i]):
#         y.append(abs(FBy[i]))
#         ax.plot(Time_OF[i],FBy[i],"o",markersize=1,color="k")

# #corr = correlation_coef(y,Theta_FB_360)

# ax.set_ylabel("0.2FBz<=FBy")
# ax.grid()
# ax2=ax.twinx()
# ax2.plot(Time_OF,Theta_FB_360)
# ax2.set_ylabel("Theta")
# #plt.title("correlation = {}".format(corr))




# plt.savefig(in_dir+"Direction/FB_comps_squared_ratio.png")
# plt.close()

# #plot windrose of Bearing Force

# ax = WindroseAxes.from_ax()
# ax.bar(Aero_Theta,Aero_FBR/1000,normed=True,opening=0.8,edgecolor="white")
# ax.set_xticklabels(['0', '45', '90', '135',  '180', '225', '270', '315'])
# ax.set_legend()

# X,P,mu,std = probability_dist(Aero_Theta)
# plt.figure(figsize=(14,8))
# plt.plot(P,X)

# plt.show()