import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy import interpolate
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter,filtfilt


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


def low_pass_filter(signal, cutoff):  
    
    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

out_dir = in_dir + "Asymmetry_analysis/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]
dt_sampling = Time_sampling[1] - Time_sampling[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

time_shift_idx = np.searchsorted(Time_OF,4.78)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

cutoff = 0.3

LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMys_LPF = low_pass_filter(LSSTipMys,cutoff)
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
LSSTipMzs_LPF = low_pass_filter(LSSTipMzs,cutoff)

LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFys_LPF = low_pass_filter(LSShftFys,cutoff)
LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
LSShftFzs_LPF = low_pass_filter(LSShftFzs,cutoff)

L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

FBMy_LPF = LSSTipMzs_LPF/L2; FBFy_LPF = -LSShftFys_LPF*((L1+L2)/L2)
FBMz_LPF = -LSSTipMys_LPF/L2; FBFz_LPF = -LSShftFzs_LPF*((L1+L2)/L2)

FBy_LPF = FBMy_LPF + FBFy_LPF; FBz_LPF = FBMz_LPF + FBFz_LPF
FBR_LPF = np.sqrt(np.add(np.square(FBy_LPF),np.square(FBz_LPF)))


offset = "63.0"
group = a.groups["{}".format(offset)]
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

f = interpolate.interp1d(Time_sampling,IA)
IA_interp = f(Time_OF)
IA_LPF = low_pass_filter(IA_interp,cutoff)

f = interpolate.interp1d(Time_sampling,Iy)
Iy_interp = f(Time_OF)
Iy_LPF = low_pass_filter(Iy_interp,cutoff)

f = interpolate.interp1d(Time_sampling,Iz)
Iz_interp = f(Time_OF)
Iz_LPF = low_pass_filter(Iz_interp,cutoff)
Iz_LPF = -Iz_LPF

I_LPF = np.sqrt(np.add(np.square(Iy_LPF),np.square(Iz_LPF)))

Time_start_idx = np.searchsorted(Time_sampling,Time_start)
Time_end_idx = np.searchsorted(Time_sampling,Time_end)

Time_sampling = Time_sampling[Time_start_idx:Time_end_idx]
IA = IA[Time_start_idx:Time_end_idx]
Iy = Iy[Time_start_idx:Time_end_idx]
Iz = -Iz[Time_start_idx:Time_end_idx]
I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))



a = Dataset(in_dir+"Asymmetry_Dataset.nc")

Time_Asy = np.array(a.variables["time"])
Time_Asy = Time_Asy - Time_Asy[0]
Iy_Asy = np.array(a.variables["Iy"])
Iz_Asy = np.array(a.variables["Iz"])

f = interpolate.interp1d(Time_Asy,Iy_Asy)
Iy_Asy_interp = f(Time_OF)
Iy_Asy_LPF = low_pass_filter(Iy_Asy_interp,cutoff)

f = interpolate.interp1d(Time_Asy,Iz_Asy)
Iz_Asy_interp = f(Time_OF)
Iz_Asy_LPF = low_pass_filter(Iz_Asy_interp,cutoff)
Iz_Asy_LPF = -Iz_Asy_LPF

I_Asy_LPF = np.sqrt(np.add(np.square(Iy_Asy_LPF),np.square(Iz_Asy_LPF)))

Time_start_idx = np.searchsorted(Time_Asy,Time_start)
Time_end_idx = np.searchsorted(Time_Asy,Time_end)

Time_Asy = Time_Asy[Time_start_idx:Time_end_idx]
Iy_Asy = Iy_Asy[Time_start_idx:Time_end_idx]
Iz_Asy = -Iz_Asy[Time_start_idx:Time_end_idx]
I_Asy = np.sqrt(np.add(np.square(Iy_Asy),np.square(Iz_Asy)))


with PdfPages(out_dir+'Asymmetry_analysis.pdf') as pdf:
    #plot Time varying quanities
    #IA vs I(t)
    cc = round(correlation_coef(IA_LPF,I_LPF),2)
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time_sampling,IA,'-b')
    ax.set_ylabel("Lavely Asymmetry parameter (IA) [$m^4/s$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_sampling,I,"-r")
    ax2.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(Iy_LPF[:-time_shift_idx],LSSTipMys_LPF[time_shift_idx:]),2)
    fig,ax = plt.subplots(figsize=(14,8))

    ax.set_title("Iy: (-63m plane), time shifted 4.78s, Low pass filtered 0.3Hz")
    ax.plot(Time_OF[:-time_shift_idx],LSSTipMys_LPF[time_shift_idx:],'-b')
    ax.set_ylabel("Rotor moment around y axis (My) [$kN-m$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_OF[:-time_shift_idx],Iy_LPF[:-time_shift_idx],"-r")
    ax2.set_ylabel("Asymmetry vector component y [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(Iy_LPF[:-time_shift_idx],FBz_LPF[time_shift_idx:]),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.set_title("Iy: (-63m plane), time shifted 4.78s, Low pass filtered 0.3Hz")
    ax.plot(Time_OF[:-time_shift_idx],FBz_LPF[time_shift_idx:],'-b')
    ax.set_ylabel("Bearing force component z [$kN$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_OF[:-time_shift_idx],Iy_LPF[:-time_shift_idx],"-r")
    ax2.set_ylabel("Asymmetry vector component y [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(Iz_LPF[:-time_shift_idx],LSSTipMzs_LPF[time_shift_idx:]),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.set_title("Iz: (-63m plane), time shifted 4.78s, Low pass filtered 0.3Hz")
    ax.plot(Time_OF[:-time_shift_idx],LSSTipMzs_LPF[time_shift_idx:],'-b')
    ax.set_ylabel("Rotor moment around z axis (My) [$kN-m$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_OF[:-time_shift_idx],Iz_LPF[:-time_shift_idx],"-r")
    ax2.set_ylabel("Asymmetry vector component z [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(Iz_LPF[:-time_shift_idx],FBy_LPF[time_shift_idx:]),2)
    fig,ax = plt.subplots(figsize=(14,8))

    ax.set_title("Iz: (-63m plane), time shifted 4.78s, Low pass filtered 0.3Hz")
    ax.plot(Time_OF[:-time_shift_idx],FBy_LPF[time_shift_idx:],'-b')
    ax.set_ylabel("Bearing force component y [$kN$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_OF[:-time_shift_idx],Iz_LPF[:-time_shift_idx],"-r")
    ax2.set_ylabel("Asymmetry vector component z [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()



    #plot time spectra
    #IA vs I(t)
    frq_IA,PSD_IA = temporal_spectra(IA,dt_sampling,Var="IA")
    frq_I,PSD_I = temporal_spectra(I,dt_sampling,Var="I")

    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(frq_IA,PSD_IA,'-b')
    ax.set_ylabel("PSD Lavely Asymmetry parameter (IA) [$m^4/s$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.loglog(frq_I,PSD_I,"-r")
    ax2.set_ylabel("PSD Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    frq_Iy,PSD_Iy = temporal_spectra(Iy,dt_sampling,Var="Iy")
    frq_My,PSD_My = temporal_spectra(LSSTipMys,dt,Var="My")

    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(frq_My,PSD_My,'-b')
    ax.set_ylabel("PSD Rotor moment around y axis (My) [$kN-m$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.loglog(frq_Iy,PSD_Iy,"-r")
    ax2.set_ylabel("PSD Asymmetry vector component y [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    frq_FBz,PSD_FBz = temporal_spectra(FBz,dt,Var="FBz")

    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(frq_FBz,PSD_FBz,'-b')
    ax.set_ylabel("PSD Bearing force component z [$kN$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.loglog(frq_Iy,PSD_Iy,"-r")
    ax2.set_ylabel("PSD Asymmetry vector component y [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    frq_Iz,PSD_Iz = temporal_spectra(Iz,dt_sampling,Var="Iz")
    frq_Mz,PSD_Mz = temporal_spectra(LSSTipMzs,dt,Var="Mz")

    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(frq_Mz,PSD_Mz,'-b')
    ax.set_ylabel("PSD Rotor moment around z axis (My) [$kN-m$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.loglog(frq_Iz,PSD_Iz,"-r")
    ax2.set_ylabel("PSD Asymmetry vector component z [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    frq_FBy,PSD_FBy = temporal_spectra(FBy,dt,Var="FBy")

    fig,ax = plt.subplots(figsize=(14,8))

    ax.loglog(frq_FBy,PSD_FBy,'-b')
    ax.set_ylabel("PSD Bearing force component y [$kN$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.loglog(frq_Iz,PSD_Iz,"-r")
    ax2.set_ylabel("PSD Asymmetry vector component z [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    ax.set_xlabel("Frequency [Hz]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()


    #new asymmetry parameters
    cc = round(correlation_coef(Iy,Iy_Asy),2)
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time_sampling,Iy,'-b')
    ax.set_ylabel("Iy [$m^4/s$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_Asy,Iy_Asy,"-r")
    ax2.set_ylabel("Iy Asym [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(Iz,Iz_Asy),2)
    fig,ax = plt.subplots(figsize=(14,8))

    ax.plot(Time_sampling,Iz,'-b')
    ax.set_ylabel("Iz [$m^4/s$]",fontsize=14)
    ax.yaxis.label.set_color('blue')
    ax2 = ax.twinx()
    ax2.plot(Time_Asy,Iz_Asy,"-r")
    ax2.set_ylabel("Iz Asym [$m^4/s$]",fontsize=14)
    ax2.yaxis.label.set_color('red')
    plt.title("Correlation coefficient {}".format(cc),fontsize=16)
    ax.set_xlabel("Time [s]",fontsize=16)
    plt.tight_layout()
    plt.grid()
    pdf.savefig()
    plt.close()



