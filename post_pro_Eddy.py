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


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Asymmetry_Dataset.nc")

Time = np.array(a.variables["time"])
Time = Time - Time[0]
Time_steps = np.arange(0,len(Time))

A_high = np.array(a.variables["Area_high"])
A_low = np.array(a.variables["Area_low"])
A_int = np.array(a.variables["Area_int"])

Iy_high = np.array(a.variables["Area_high"])
Iy_low = np.array(a.variables["Area_low"])
Iy_int = np.array(a.variables["Area_int"])

Iz_high = np.array(a.variables["Area_high"])
Iz_low = np.array(a.variables["Area_low"])
Iz_int = np.array(a.variables["Area_int"])

Ux_high = np.array(a.variables["Area_high"])
Ux_low = np.array(a.variables["Area_low"])
Ux_int = np.array(a.variables["Area_int"])
Ux = (np.add(np.add(Ux_high,Ux_low),Ux_int))/3

Iy = np.array(a.variables["Iy"])
Iz = np.array(a.variables["Iz"])


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
group = df.groups["63.0"]
Iy_df = np.array(group.variables["Iy"])
Iz_df = np.array(group.variables["Iz"])
    


out_dir = in_dir+"Asymmetry_analysis/"
with PdfPages(out_dir+'Eddy_analysis.pdf') as pdf:

    #new asymmetry parameters
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

    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,P_high_Iy,'-r')
    ax.plot(Time,P_low_Iy,"-b")
    ax.plot(Time,P_int_Iy,"-g")
    #ax.plot(Time,P_Tot_Iy,"--k")
    ax.set_ylabel("Fraction of Asymmetry around y axis [-]",fontsize=14)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area","Total"])
    ax.set_ylim([-1,1])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,P_high_Iy)
    ax1.set_ylim([-1,1])
    ax1.axhline(y=np.mean(P_high_Iy),linestyle="--",color="k")
    ax2.plot(Time,P_low_Iy)
    ax2.set_ylim([-1,1])
    ax2.axhline(y=np.mean(P_low_Iy),linestyle="--",color="k")
    ax3.plot(Time,P_int_Iy)
    ax3.set_ylim([-1,1])
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


    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,P_high_Iz,'-r')
    ax.plot(Time,P_low_Iz,"-b")
    ax.plot(Time,P_int_Iz,"-g")
    #ax.plot(Time,P_Tot_Iz,"--k")
    ax.set_ylabel("Fraction of Asymmetry around z axis [-]",fontsize=14)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area","Total"])
    ax.set_ylim([-1,1])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,P_high_Iz)
    ax1.set_ylim([-1,1])
    ax1.axhline(y=np.mean(P_high_Iz),linestyle="--",color="k")
    ax2.plot(Time,P_low_Iz)
    ax2.set_ylim([-1,1])
    ax2.axhline(y=np.mean(P_low_Iz),linestyle="--",color="k")
    ax3.plot(Time,P_int_Iz)
    ax3.set_ylim([-1,1])
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

        
        
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time,Ux_high,'-r')
    ax.plot(Time,Ux_low,"-b")
    ax.plot(Time,Ux_int,"-g")
    ax.plot(Time,P_Tot_Iz,"--k")
    ax.set_ylabel("Fraction of Asymmetry around z axis [-]",fontsize=14)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.legend(["High speed area", "Low speed area", "intermediate area","Total"])
    ax.set_ylim([-1,1])
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

    ax1.plot(Time,P_high_Iz)
    ax1.set_ylim([-1,1])
    ax1.axhline(y=np.mean(P_high_Iz),linestyle="--",color="k")
    ax2.plot(Time,P_low_Iz)
    ax2.set_ylim([-1,1])
    ax2.axhline(y=np.mean(P_low_Iz),linestyle="--",color="k")
    ax3.plot(Time,P_int_Iz)
    ax3.set_ylim([-1,1])
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