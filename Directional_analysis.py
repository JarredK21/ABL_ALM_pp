from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from windrose import WindroseAxes

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
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    return P,X, round(mu,2), round(sd,2)


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])

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
plot_Fz_stats = True



if plot_Fz_stats == True:
    out_dir = in_dir+"Direction/PDFs/"

    vars = [np.square(FBz)]
    rotor_weight = -1079.1
    percentage = np.arange(0.1,1.0,0.1)
    labels = ["Total"]
    
    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc

        vars.append(np.square(FBz_perc))
        labels.append("{} reduction".format(round(perc,1)))

    vars.append(np.square(Aero_FBz/1000))
    labels.append("Aero")


    means = []
    std_dev = []
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(0,len(vars)):

        var = vars[i]
        P,X,mu,std = probability_dist(var)
        means.append(mu)
        std_dev.append(std)


    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,means,"o-k")
    plt.ylabel("Mean of \nMain bearing force z squared [$kN^2$]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"means_square_FBz.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(labels,std_dev,"o-k")
    plt.ylabel("Standard deviations of \nMain bearing force z squared [$kN^2$]",fontsize=16)
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"std_square_FBz.png")
    plt.close()



if plot_FB_stats == True:
    out_dir = in_dir+"Direction/PDFs/"

    vars = [Theta_FB_360,FBR]
    rotor_weight = -1079.1
    percentage = np.arange(0.1,1.0,0.1)
    labels = ["Total"]
    
    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc
        FBR_perc = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
        Theta_FB_perc = np.degrees(np.arctan2(FBz_perc,FBy))
        Theta_FB_perc_360 = theta_360(Theta_FB_perc)

        vars.append(Theta_FB_perc_360); vars.append(FBR_perc)
        labels.append("{} reduction".format(round(perc,1)))

    vars.append(Theta_Aero_FB_360)
    vars.append(Aero_FBR/1000)
    labels.append("Aero")


    means = []
    std_dev = []
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(0,len(vars),2):

        var = vars[i]
        P,X,mu,std = probability_dist(var)
        means.append(mu)
        std_dev.append(std)


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

    means = []
    std_dev = []
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(1,len(vars),2):

        var = vars[i]
        P,X,mu,std = probability_dist(var)
        means.append(mu)
        std_dev.append(std)


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



if PDFs == True:
    units = ["[deg]","[kN]"]
    Ylabels = ["Direction main bearing force vector {}".format(units[0]),"Magnitude Main Bearing Force {}".format(units[1])]
    vars = [Theta_FB_360,FBR]
    names = ["Theta","FBR"]
    filenames = ["{}_total.png".format(names[0]),"{}_total.png".format(names[1])]

    rotor_weight = -1079.1
    percentage = [0.3, 0.5, 0.7, 0.9]

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



    vars.append(Theta_Aero_FB_360)
    vars.append(Aero_FBR/1000)
    Ylabels.append("Direction aerodynamic main bearing force vector {}".format(units[0]))
    Ylabels.append("Aerodynamic magnitude Main Bearing Force".format(units[1]))
    filenames.append("{}_aero.png".format(names[0])); filenames.append("{}_aero.png".format(names[1]))


    #PDFs
    out_dir = in_dir+"Direction/PDFs/"
    for i in np.arange(0,len(vars)):
        var = vars[i];Ylabel = Ylabels[i]; filename = filenames[i]
        P,X,mu,std = probability_dist(var)
        fig = plt.figure(figsize=(14,8))
        plt.plot(X,P)
        plt.axvline(mu,linestyle="--",color="k")
        plt.axvline(mu+std,linestyle="--",color="r")
        plt.axvline(mu-std,linestyle="--",color="r")
        plt.ylabel("PDF",fontsize=16)
        plt.xlabel(Ylabel,fontsize=16)
        plt.title("Mean = {} \nStandard deviation = {}".format(mu,std))
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
        P,X,mu,std = probability_dist(var)
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
        P,X,mu,std = probability_dist(var)
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
    vars = [FBR]; Ylabels = ["Magnitude main bearing force [kN]"]; units = "[kN]"
    rotor_weight = -1079.1
    percentage = [0.3, 0.7]

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

    vars.append(Aero_FBR/1000); Ylabels.append("Aerodynamic magnitude main bearing force [kN]")

    out_dir = in_dir+"Direction/"
    fig,axs = plt.subplots(4,1,figsize=(14,10))
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    for i in np.arange(0,len(vars)):
        var = vars[i];Ylabel = Ylabels[i]
        axs[i].plot(Time_OF,var)
        axs[i].set_title("{}".format(Ylabel))
        axs[i].grid()

    plt.savefig(out_dir+"joint_FBR_lineplot.png")
    plt.close()




    #lineplots
    vars = [Theta_FB]; Ylabels = ["Direction main bearing force vector [deg]"]; units = "[deg]"
    rotor_weight = -1079.1
    percentage = [0.3, 0.7]

    for perc in percentage:
        Fzs = LSShftFzs-(perc*rotor_weight)
        FBFz_perc = -Fzs*((L1+L2)/L2)
        FBz_perc = FBMz + FBFz_perc
        FBR_perc = np.sqrt(np.add(np.square(FBy),np.square(FBz_perc)))
        Theta_FB_perc = np.degrees(np.arctan2(FBz_perc,FBy))

        vars.append(Theta_FB_perc)
        Ylabels.append("Direction main bearing force vector with \n{} reduction in weight {}".format(perc, units))

    vars.append(Theta_Aero_FB); Ylabels.append("Direction main bearing force vector [deg]")

    out_dir = in_dir+"Direction/"
    fig,axs = plt.subplots(4,1,figsize=(14,10))
    fig.subplots_adjust(hspace = .5)
    axs = axs.ravel()

    for i in np.arange(0,len(vars)):
        var = vars[i];Ylabel = Ylabels[i]
        axs[i].plot(Time_OF,var)
        axs[i].set_title("{}".format(Ylabel))
        axs[i].grid()

    plt.savefig(out_dir+"joint_Theta_lineplot.png")
    plt.close()



# #plot windrose of Bearing Force

# ax = WindroseAxes.from_ax()
# ax.bar(Aero_Theta,Aero_FBR/1000,normed=True,opening=0.8,edgecolor="white")
# ax.set_xticklabels(['0', '45', '90', '135',  '180', '225', '270', '315'])
# ax.set_legend()

# X,P,mu,std = probability_dist(Aero_Theta)
# plt.figure(figsize=(14,8))
# plt.plot(P,X)

# plt.show()