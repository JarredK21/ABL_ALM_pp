import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages


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



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Asymmetry_Dataset.nc")

Time = np.array(a.variables["time"])
Time = Time - Time[0]
dt = Time[1] - Time[0]
Time_steps = np.arange(0,len(Time))

Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)
Time = Time[Time_start_idx:]
Time_steps = np.arange(0,len(Time))

A_high = np.array(a.variables["Area_high"][Time_start_idx:])
mu,sd,sk,k = moments(A_high)
print("stats A_high ",mu,sd,sk,k)
A_low = np.array(a.variables["Area_low"][Time_start_idx:])
mu,sd,sk,k = moments(A_low)
print("stats A_low ",mu,sd,sk,k)
A_int = np.array(a.variables["Area_int"][Time_start_idx:])
mu,sd,sk,k = moments(A_int)
print("stats A_int ",mu,sd,sk,k)

A_high_frq, A_high_PSD = temporal_spectra(A_high,dt,Var="A_high")
A_low_frq, A_low_PSD = temporal_spectra(A_low,dt,Var="A_low")
A_int_frq, A_int_PSD = temporal_spectra(A_int,dt,Var="A_int")

Iy_high = np.array(a.variables["Iy_high"][Time_start_idx:])
mu,sd,sk,k = moments(Iy_high)
print("stats Iy_high ",mu,sd,sk,k)
Iy_low = np.array(a.variables["Iy_low"][Time_start_idx:])
mu,sd,sk,k = moments(Iy_low)
print("stats Iy_low ",mu,sd,sk,k)
Iy_int = np.array(a.variables["Iy_int"][Time_start_idx:])
mu,sd,sk,k = moments(Iy_int)
print("stats Iy_int ",mu,sd,sk,k)

Iy_high_frq, Iy_high_PSD = temporal_spectra(Iy_high,dt,Var="Iy_high")
Iy_low_frq, Iy_low_PSD = temporal_spectra(Iy_low,dt,Var="Iy_low")
Iy_int_frq, Iy_int_PSD = temporal_spectra(Iy_int,dt,Var="Iy_int")

Iz_high = np.array(a.variables["Iz_high"][Time_start_idx:])
mu,sd,sk,k = moments(Iz_high)
print("stats Iz_high ",mu,sd,sk,k)
Iz_low = np.array(a.variables["Iz_low"][Time_start_idx:])
mu,sd,sk,k = moments(Iz_low)
print("stats Iz_low ",mu,sd,sk,k)
Iz_int = np.array(a.variables["Iz_int"][Time_start_idx:])
mu,sd,sk,k = moments(Iz_int)
print("stats Iz_int ",mu,sd,sk,k)

Iz_high_frq, Iz_high_PSD = temporal_spectra(Iz_high,dt,Var="Iz_high")
Iz_low_frq, Iz_low_PSD = temporal_spectra(Iz_low,dt,Var="Iz_low")
Iz_int_frq, Iz_int_PSD = temporal_spectra(Iz_int,dt,Var="Iz_int")

Ux_high = np.array(a.variables["Ux_high"][Time_start_idx:])
Ux_low = np.array(a.variables["Ux_low"][Time_start_idx:])
Ux_int = np.array(a.variables["Ux_int"][Time_start_idx:])

Ux_high_frq, Ux_high_PSD = temporal_spectra(Ux_high,dt,Var="Ux_high")
Ux_low_frq, Ux_low_PSD = temporal_spectra(Ux_low,dt,Var="Ux_low")
Ux_int_frq, Ux_int_PSD = temporal_spectra(Ux_int,dt,Var="Ux_int")

Iy = np.array(a.variables["Iy"][Time_start_idx:])
Iz = np.array(a.variables["Iz"][Time_start_idx:])


A_rot = np.pi*63**2

Frac_high_area = np.true_divide(A_high,A_rot)
Frac_low_area = np.true_divide(A_low,A_rot)
Frac_int_area = np.true_divide(A_int,A_rot)
Tot_area = np.add(np.add(Frac_high_area,Frac_low_area),Frac_int_area)

P_high_Iy = np.true_divide(Iy_high,Iy)
P_low_Iy = np.true_divide(Iy_low,Iy)
P_int_Iy = np.true_divide(Iy_int,Iy)
P_Tot_Iy = np.add(np.add(P_high_Iy,P_low_Iy),P_int_Iy)

P_high_Iz = np.true_divide(Iz_high,Iz)
P_low_Iz = np.true_divide(Iz_low,Iz)
P_int_Iz = np.true_divide(Iz_int,Iz)
P_Tot_Iz = np.add(np.add(P_high_Iz,P_low_Iz),P_int_Iz)



df = Dataset(in_dir+"Dataset.nc")
Time_sampling = np.array(df.variables["time_sampling"])
Time_start = 200; Time_start_idx = np.searchsorted(Time_sampling,Time_start)
Time_sampling = Time_sampling[Time_start_idx:]
group = df.groups["63.0"]
Iy_df = np.array(group.variables["Iy"][Time_start_idx:])
Iz_df = np.array(group.variables["Iz"][Time_start_idx:])
    

print("Iy_high cc with Iy ",correlation_coef(Iy,Iy_high))
print("Iy_low cc with Iy ",correlation_coef(Iy,Iy_low))
print("Iy_int cc with Iy ",correlation_coef(Iy,Iy_int))

print("Iz_high cc with Iz ",correlation_coef(Iz,Iz_high))
print("Iz_low cc with Iz ",correlation_coef(Iz,Iz_low))
print("Iz_int cc with Iz ",correlation_coef(Iz,Iz_int))

print("A_high cc with A_int ",correlation_coef(A_high,A_int))
print("A_low cc with A_int ",correlation_coef(A_low,A_int))


out_dir = in_dir+"Asymmetry_analysis/"
with PdfPages(out_dir+'Eddy_analysis.pdf') as pdf:

    plt.rcParams['font.size'] = 14

    #correlating Iy from original with new data
    cc = round(correlation_coef(Iy,Iy_df),2)
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,Iy,'-b')
    ax.set_ylabel("Iy [$m^4/s$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_sampling,Iy_df,"-r")
    ax2.set_ylabel("Iy dataset [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #correlating Iz from original with new data
    cc = round(correlation_coef(Iz,Iz_df),2)
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,Iz,'-b')
    ax.set_ylabel("Iz [$m^4/s$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_sampling,Iz_df,"-r")
    ax2.set_ylabel("Iz Dataset [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #plotting joint areas
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,A_high,'-r')
    ax.plot(Time,A_low,"-b")
    ax.plot(Time,A_int,"-g")
    ax.set_ylabel("Area [$m^2$]",fontsize=14)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #plotting areas separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,A_high)
    ax1.axhline(y=np.mean(A_high),linestyle="--",color="k")
    ax2.plot(Time,A_low)
    ax2.axhline(y=np.mean(A_low),linestyle="--",color="k")
    ax3.plot(Time,A_int)
    ax3.axhline(y=np.mean(A_int),linestyle="--",color="k")
    ax1.set_title("High speed areas [$m^2$]",fontsize=14)
    ax2.set_title("Low speed areas [$m^2$]",fontsize=14)
    ax3.set_title("Intermediate speed areas [$m^2$]",fontsize=14)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

    #plotting area spectra joint
    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(A_high_frq,A_high_PSD,'-r')
    ax.loglog(A_low_frq,A_low_PSD,"-b")
    ax.loglog(A_int_frq,A_int_PSD,"-g")
    ax.set_ylabel("Area [$m^2$]",fontsize=14)
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #plotting area spectra separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.loglog(A_high_frq,A_high_PSD)
    ax2.loglog(A_low_frq,A_low_PSD)
    ax3.loglog(A_int_frq,A_int_PSD)
    ax1.set_title("High speed areas [$m^2$]",fontsize=14)
    ax2.set_title("Low speed areas [$m^2$]",fontsize=14)
    ax3.set_title("Intermediate speed areas [$m^2$]",fontsize=14)
    fig.supxlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()


    #plotting area fractions joint
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,Frac_high_area,'-r')
    ax.plot(Time,Frac_low_area,"-b")
    ax.plot(Time,Frac_int_area,"-g")
    ax.plot(Time,Tot_area,"--k")
    ax.set_ylabel("Fraction of rotor disk area [-]",fontsize=14)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area","Total area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #plotting area fractions separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,Frac_high_area)
    ax1.axhline(y=np.mean(Frac_high_area),linestyle="--",color="k")
    ax2.plot(Time,Frac_low_area)
    ax2.axhline(y=np.mean(Frac_low_area),linestyle="--",color="k")
    ax3.plot(Time,Frac_int_area)
    ax3.axhline(y=np.mean(Frac_int_area),linestyle="--",color="k")
    ax1.set_title("Fraction of rotor disk area - high speed areas [-]",fontsize=14)
    ax2.set_title("Fraction of rotor disk area - low speed areas [-]",fontsize=14)
    ax3.set_title("Fraction of rotor disk area - intermediate speed areas [-]",fontsize=14)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

    #PDF area joint
    fig,ax = plt.subplots(figsize=(14,8))

    P,X = probability_dist(A_high)
    ax.plot(X,P,'-r')
    P,X = probability_dist(A_low)
    ax.plot(X,P,'-b')
    P,X = probability_dist(A_int)
    ax.plot(X,P,'-g')
    ax.set_ylabel("Probability [-]",fontsize=14)
    ax.set_xlabel("Area [$m^2$]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #plotting PDF of area separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    P,X = probability_dist(A_high)
    ax1.plot(X,P)
    P,X = probability_dist(A_low)
    ax2.plot(X,P)
    P,X = probability_dist(A_int)
    ax3.plot(X,P)
    ax1.set_title("Area - high speed areas [$m^2$]",fontsize=14)
    ax2.set_title("Area - low speed areas [$m^2$]",fontsize=14)
    ax3.set_title("Area - intermediate speed areas [$m^2$]",fontsize=14)
    fig.supylabel("Probability [-]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()
    



    #Iy fractions plotted joint
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,P_high_Iy,'-r')
    ax.plot(Time,P_low_Iy,"-b")
    ax.plot(Time,P_int_Iy,"-g")
    ax.plot(Time,P_Tot_Iy,"--k")
    ax.set_ylabel("Fraction of Asymmetry around y axis [-]",fontsize=14)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area","Total"])
    ax.set_ylim([-5,5])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()
    
    #Iy fractions plotted separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,P_high_Iy)
    ax1.set_ylim([-5,5])
    ax1.axhline(y=np.mean(P_high_Iy),linestyle="--",color="k")
    ax2.plot(Time,P_low_Iy)
    ax2.set_ylim([-5,5])
    ax2.axhline(y=np.mean(P_low_Iy),linestyle="--",color="k")
    ax3.plot(Time,P_int_Iy)
    ax3.set_ylim([-5,5])
    ax3.axhline(y=np.mean(P_int_Iy),linestyle="--",color="k")
    ax1.set_title("Fraction of Asymmetry around y axis - high speed area [-]",fontsize=14)
    ax2.set_title("Fraction of Asymmetry around y axis - low speed area [-]",fontsize=14)
    ax3.set_title("Fraction of Asymmetry around y axis - intermediate speed area [-]",fontsize=14)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

    #Iy plotted separetely
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,Iy_high)
    ax1.axhline(y=np.mean(Iy_high),linestyle="--",color="k")
    ax1.axhline(y=np.mean(Iy_high)+np.std(Iy_high),color="r")
    ax1.axhline(y=np.mean(Iy_high)-np.std(Iy_high),color="r")
    ax2.plot(Time,Iy_low)
    ax2.axhline(y=np.mean(Iy_low),linestyle="--",color="k")
    ax2.axhline(y=np.mean(Iy_low)+np.std(Iy_low),color="r")
    ax2.axhline(y=np.mean(Iy_low)-np.std(Iy_low),color="r")
    ax3.plot(Time,Iy_int)
    ax3.axhline(y=np.mean(Iy_int),linestyle="--",color="k")
    ax3.axhline(y=np.mean(Iy_int)+np.std(Iy_int),color="r")
    ax3.axhline(y=np.mean(Iy_int)-np.std(Iy_int),color="r")
    ax1.set_title("Asymmetry around y axis - high speed area [$m^4/s$]",fontsize=14)
    ax2.set_title("Asymmetry around y axis - low speed area [$m^4/s$]",fontsize=14)
    ax3.set_title("Asymmetry around y axis - intermediate speed area [$m^4/s$]",fontsize=14)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

    #Iy spectra plotted joint
    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(Iy_high_frq,Iy_high_PSD,'-r')
    ax.loglog(Iy_low_frq,Iy_low_PSD,"-b")
    ax.loglog(Iy_int_frq,Iy_int_PSD,"-g")
    ax.set_ylabel("Asymmetry around y axis [$m^4/s$]",fontsize=14)
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #Iy spectra plotted separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.loglog(Iy_high_frq,Iy_high_PSD)
    ax2.loglog(Iy_low_frq,Iy_low_PSD)
    ax3.loglog(Iy_int_frq,Iy_int_PSD)
    ax1.set_title("Asymmetry around y axis - high speed area [$m^4/s$]",fontsize=14)
    ax2.set_title("Asymmetry around y axis - low speed area [$m^4/s$]",fontsize=14)
    ax3.set_title("Asymmetry around y axis - intermediate speed area [$m^4/s$]",fontsize=14)
    fig.supxlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()


    #PDF Iy joint
    fig,ax = plt.subplots(figsize=(14,8))

    P,X = probability_dist(Iy_high)
    ax.plot(X,P,'-r')
    P,X = probability_dist(Iy_low)
    ax.plot(X,P,'-b')
    P,X = probability_dist(Iy_int)
    ax.plot(X,P,'-g')
    ax.set_ylabel("Probability [-]",fontsize=14)
    ax.set_xlabel("Asymmetry around y axis [$m^4/s$]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #plotting PDF of Iy separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    P,X = probability_dist(Iy_high)
    ax1.plot(X,P)
    P,X = probability_dist(Iy_low)
    ax2.plot(X,P)
    P,X = probability_dist(Iy_int)
    ax3.plot(X,P)
    ax1.set_title("Asymmetry around y axis - high speed areas [$m^4/s$]",fontsize=14)
    ax2.set_title("Asymmetry around y axis - low speed areas [$m^4/s$]",fontsize=14)
    ax3.set_title("Asymmetry around y axis - intermediate speed areas [$m^4/s$]",fontsize=14)
    fig.supylabel("Probability [-]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

    #fractions of Iz plotted joint
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,P_high_Iz,'-r')
    ax.plot(Time,P_low_Iz,"-b")
    ax.plot(Time,P_int_Iz,"-g")
    ax.plot(Time,P_Tot_Iz,"--k")
    ax.set_ylabel("Fraction of Asymmetry around z axis [-]",fontsize=14)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area","Total"])
    ax.set_ylim([-2,2])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #fractions of Iz plotted separetely
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,P_high_Iz)
    ax1.set_ylim([-10,10])
    ax1.axhline(y=np.mean(P_high_Iz),linestyle="--",color="k")
    ax2.plot(Time,P_low_Iz)
    ax2.set_ylim([-10,10])
    ax2.axhline(y=np.mean(P_low_Iz),linestyle="--",color="k")
    ax3.plot(Time,P_int_Iz)
    ax3.set_ylim([-10,10])
    ax3.axhline(y=np.mean(P_int_Iz),linestyle="--",color="k")
    ax1.set_title("Fraction of Asymmetry around z axis - high speed area [-]",fontsize=14)
    ax2.set_title("Fraction of Asymmetry around z axis - low speed area [-]",fontsize=14)
    ax3.set_title("Fraction of Asymmetry around z axis - intermediate speed area [-]",fontsize=14)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

    #Iz plotted separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,Iz_high)
    ax1.axhline(y=np.mean(Iz_high),linestyle="--",color="k")
    ax1.axhline(y=np.mean(Iz_high)+np.std(Iz_high),color="r")
    ax1.axhline(y=np.mean(Iz_high)-np.std(Iz_high),color="r")
    ax2.plot(Time,Iz_low)
    ax2.axhline(y=np.mean(Iz_low),linestyle="--",color="k")
    ax2.axhline(y=np.mean(Iz_low)+np.std(Iy_low),color="r")
    ax2.axhline(y=np.mean(Iz_low)-np.std(Iz_low),color="r")
    ax3.plot(Time,Iz_int)
    ax3.axhline(y=np.mean(Iz_int),linestyle="--",color="k")
    ax3.axhline(y=np.mean(Iz_int)+np.std(Iz_int),color="r")
    ax3.axhline(y=np.mean(Iz_int)-np.std(Iz_int),color="r")
    ax1.set_title("Asymmetry around z axis - high speed area [$m^4/s$]",fontsize=14)
    ax2.set_title("Asymmetry around z axis - low speed area [$m^4/s$]",fontsize=14)
    ax3.set_title("Asymmetry around z axis - intermediate speed area [$m^4/s$]",fontsize=14)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()


    #PDF Iz joint
    fig,ax = plt.subplots(figsize=(14,8))

    P,X = probability_dist(Iz_high)
    ax.plot(X,P,'-r')
    P,X = probability_dist(Iz_low)
    ax.plot(X,P,'-b')
    P,X = probability_dist(Iz_int)
    ax.plot(X,P,'-g')
    ax.set_ylabel("Probability [-]",fontsize=14)
    ax.set_xlabel("Asymmetry around z axis [$m^4/s$]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #plotting PDF of Iz separately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    P,X = probability_dist(Iz_high)
    ax1.plot(X,P)
    P,X = probability_dist(Iz_low)
    ax2.plot(X,P)
    P,X = probability_dist(Iz_int)
    ax3.plot(X,P)
    ax1.set_title("Asymmetry around z axis - high speed areas [$m^4/s$]",fontsize=14)
    ax2.set_title("Asymmetry around z axis - low speed areas [$m^4/s$]",fontsize=14)
    ax3.set_title("Asymmetry around z axis - intermediate speed areas [$m^4/s$]",fontsize=14)
    fig.supylabel("Probability [-]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

    #Iz spectra plotted joint
    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(Iz_high_frq,Iz_high_PSD,'-r')
    ax.loglog(Iz_low_frq,Iz_low_PSD,"-b")
    ax.loglog(Iz_int_frq,Iz_int_PSD,"-g")
    ax.set_ylabel("Asymmetry around z axis [$m^4/s$]",fontsize=14)
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #Iz spectra plotted separetly
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.loglog(Iz_high_frq,Iz_high_PSD)
    ax2.loglog(Iz_low_frq,Iz_low_PSD)
    ax3.loglog(Iz_int_frq,Iz_int_PSD)
    ax1.set_title("Asymmetry around z axis - high speed area [$m^4/s$]",fontsize=14)
    ax2.set_title("Asymmetry around z axis - low speed area [$m^4/s$]",fontsize=14)
    ax3.set_title("Asymmetry around z axis - intermediate speed area [$m^4/s$]",fontsize=14)
    fig.supxlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

        
    #average velocity plotted joint
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,Ux_high,'-r')
    ax.plot(Time,Ux_low,"-b")
    ax.plot(Time,Ux_int,"-g")
    ax.set_ylabel("Average streamwise velocity [m/s]",fontsize=14)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area","Total"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #average velocity plotted sepately
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,Ux_high)
    ax1.axhline(y=np.mean(Ux_high),linestyle="--",color="k")
    ax2.plot(Time,Ux_low)
    ax2.axhline(y=np.mean(Ux_low),linestyle="--",color="k")
    ax3.plot(Time,Ux_int)
    ax3.axhline(y=np.mean(Ux_int),linestyle="--",color="k")
    ax1.set_title("average velocity - high speed area [m/s]",fontsize=14)
    ax2.set_title("average velocity - low speed area [m/s]",fontsize=14)
    ax3.set_title("average velocity - intermediate speed area [m/s]",fontsize=14)
    fig.supxlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()

    #average velocity spectra plotted joint
    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(Ux_high_frq,Ux_high_PSD,'-r')
    ax.loglog(Ux_low_frq,Ux_low_PSD,"-b")
    ax.loglog(Ux_int_frq,Ux_int_PSD,"-g")
    ax.set_ylabel("Average velocity [m/s]",fontsize=14)
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area"])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    #average velocity spectra plotted separely
    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.loglog(Ux_high_frq,Ux_high_PSD)
    ax2.loglog(Ux_low_frq,Ux_low_PSD)
    ax3.loglog(Ux_int_frq,Ux_int_PSD)
    ax1.set_title("average velocity - high speed area [m/s]",fontsize=14)
    ax2.set_title("average velocity - low speed area [m/s]",fontsize=14)
    ax3.set_title("average velocity - intermediate speed area [m/s]",fontsize=14)
    fig.supxlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    ax1.grid()
    ax2.grid()
    ax3.grid()
    pdf.savefig()
    plt.close()


    #eddy contribution to Iy
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iy,"-k")


    high = 0
    low = 0
    int = 0
    high_low = 0
    high_int = 0
    low_int = 0
    high_low_int = 0

    alpha = 10
    for it in Time_steps:

        if abs(Iy_high[it]) > alpha*abs(Iy_low[it]) and abs(Iy_high[it]) > alpha*abs(Iy_int[it]):
            plt.plot(Time[it],Iy[it],"or",markersize=4)
            high+=dt
        elif abs(Iy_low[it]) > alpha*abs(Iy_high[it]) and abs(Iy_low[it]) > alpha*abs(Iy_int[it]):
            plt.plot(Time[it],Iy[it],"ob",markersize=4)
            low+=dt
        elif abs(Iy_int[it]) > alpha*abs(Iy_high[it]) and abs(Iy_int[it]) > alpha*abs(Iy_low[it]):
            plt.plot(Time[it],Iy[it],"og",markersize=4)
            int+=dt
        elif abs(Iy_high[it]) <= alpha*abs(Iy_low[it]) and abs(Iy_high[it]) > alpha*abs(Iy_int[it]):
            plt.plot(Time[it],Iy[it],"*m",markersize=4)
            high_low+=dt
        elif abs(Iy_high[it]) <= alpha*abs(Iy_int[it]) and abs(Iy_high[it]) > alpha*abs(Iy_low[it]):
            plt.plot(Time[it],Iy[it],"vy",markersize=4)
            high_int+=dt
        elif abs(Iy_low[it]) <= alpha*abs(Iy_int[it]) and abs(Iy_low[it]) > alpha*abs(Iy_high[it]):
            plt.plot(Time[it],Iy[it],">c",markersize=5)
            low_int+=dt
        else:
            plt.plot(Time[it],Iy[it],"<k",markersize=5)
            high_low_int+=dt

    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry around y axis [$m^4/s$]")
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    t_check = high+low+int+high_low+high_int+low_int+high_low_int
    print("total time ",t_check)
    print("1. ",high)
    print("2. ",low)
    print("3. ",int)
    print("4. ",high_low)
    print("5. ",high_int)
    print("6. ",low_int)
    print("7. ",high_low_int)


    #eddy contribution to Iz
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iz,"-k")
    high = 0
    low = 0
    int = 0
    high_low = 0
    high_int = 0
    low_int = 0
    high_low_int = 0

    alpha = 10
    for it in Time_steps:

        if abs(Iz_high[it]) > alpha*abs(Iz_low[it]) and abs(Iz_high[it]) > alpha*abs(Iz_int[it]):
            plt.plot(Time[it],Iz[it],"or",markersize=4)
            high+=dt
        elif abs(Iz_low[it]) > alpha*abs(Iz_high[it]) and abs(Iz_low[it]) > alpha*abs(Iz_int[it]):
            plt.plot(Time[it],Iz[it],"ob",markersize=4)
            low+=dt
        elif abs(Iz_int[it]) > alpha*abs(Iz_high[it]) and abs(Iz_int[it]) > alpha*abs(Iz_low[it]):
            plt.plot(Time[it],Iz[it],"og",markersize=4)
            int+=dt
        elif abs(Iz_high[it]) <= alpha*abs(Iz_low[it]) and abs(Iz_high[it]) > alpha*abs(Iz_int[it]):
            plt.plot(Time[it],Iz[it],"*m",markersize=4)
            high_low+=dt
        elif abs(Iz_high[it]) <= alpha*abs(Iz_int[it]) and abs(Iz_high[it]) > alpha*abs(Iz_low[it]):
            plt.plot(Time[it],Iz[it],"vy",markersize=5)
            high_int+=dt
        elif abs(Iz_low[it]) <= alpha*abs(Iz_int[it]) and abs(Iz_low[it]) > alpha*abs(Iz_high[it]):
            plt.plot(Time[it],Iz[it],">c",markersize=5)
            low_int+=dt
        else:
            plt.plot(Time[it],Iz[it],"<k",markersize=4)
            high_low_int+=dt

    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry around z axis [$m^4/s$]")
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    t_check = high+low+int+high_low+high_int+low_int+high_low_int
    print("total time ",t_check)
    print("1. ",high)
    print("2. ",low)
    print("3. ",int)
    print("4. ",high_low)
    print("5. ",high_int)
    print("6. ",low_int)
    print("7. ",high_low_int)