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


def low_pass_filter(signal,dt,cutoff):  
    
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


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

out_dir = in_dir + "Asymmetry_analysis/"

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["Time_OF"])
Time_sampling = np.array(a.variables["Time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]
dt_sampling = Time_sampling[1] - Time_sampling[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

time_shift = 4.78
time_shift_idx = np.searchsorted(Time_OF,time_shift)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

OF_Vars = a.groups["OpenFAST_Variables"]

LSSTipMys = np.array(OF_Vars.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(OF_Vars.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])

LSShftFys = np.array(OF_Vars.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(OF_Vars.variables["LSShftFzs"][Time_start_idx:Time_end_idx])

Azimuth = np.radians(np.array(OF_Vars.variables["Azimuth"][Time_start_idx:Time_end_idx]))

RtAeroFyh = np.array(OF_Vars.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(OF_Vars.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)


RtAeroMyh = np.array(OF_Vars.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(OF_Vars.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)


MR = np.sqrt(np.add(np.square(LSSTipMys),np.square(LSSTipMzs)))
Theta_MR = np.degrees(np.arctan2(-RtAeroMys,RtAeroMzs))
Theta_MR = theta_360(Theta_MR)
Theta_MR = np.radians(np.array(Theta_MR))

L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

LPF_FBR = low_pass_filter(FBR,dt,0.3)


Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = -(Aero_FBMy + Aero_FBFy); Aero_FBz = -(Aero_FBMz + Aero_FBFz)
Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))

LPF_Aero_FBR = low_pass_filter(Aero_FBR,dt,0.3)


Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
Rotor_avg_vars_63 = Rotor_avg_vars.groups["63.0"]

Time_start_idx = np.searchsorted(Time_sampling,Time_start)
Time_end_idx = np.searchsorted(Time_sampling,Time_end)

Time_sampling = Time_sampling[Time_start_idx:Time_end_idx]
Iy = np.array(Rotor_avg_vars_63.variables["Iy"][Time_start_idx:Time_end_idx])
Iz = np.array(Rotor_avg_vars_63.variables["Iz"][Time_start_idx:Time_end_idx])

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
LPF_I = low_pass_filter(I,dt_sampling,0.3)


f = interpolate.interp1d(Time_OF,LPF_Aero_FBR)
LPF_Aero_FBR_interp = f(np.linspace(Time_OF[0],Time_OF[-1],len(Time_sampling)))




# with PdfPages(out_dir+'Asymmetry_analysis.pdf') as pdf:
#     #plot Time varying quanities

#     plt.rcParams['font.size'] = 16

#     cc = round(correlation_coef(LPF_I[:-time_shift_idx],LPF_Aero_FBR_interp[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],LPF_I[:-time_shift_idx],'-b')
#     ax.set_ylabel("I [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],LPF_Aero_FBR[time_shift_idx:],"-r")
#     ax2.set_ylabel("Magnitude Aero Bearing force vector [kN]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Correlation coefficient {}".format(cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()


#     #IA vs OOPBM
#     cc = round(correlation_coef(I_LPF[:-time_shift_idx],OOPBM[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],I_LPF[:-time_shift_idx],'-b')
#     ax.set_ylabel("Magnitude Asymmetry Vector (I) [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],OOPBM[time_shift_idx:],"-r")
#     ax2.set_ylabel("Rotor OOPBM [kN]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Correlation coefficient {}".format(cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     #IA vs FBR aero
#     cc = round(correlation_coef(I_LPF[:-time_shift_idx],Aero_FBR_LPF[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],I_LPF[:-time_shift_idx],'-b')
#     ax.set_ylabel("Magnitude Asymmetry Vector (I) [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Aero_FBR_LPF[time_shift_idx:],"-r")
#     ax2.set_ylabel("Magnitude Aerodynamic main bearing force vector [kN]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("I: (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()


#     #IA vs FBR
#     cc = round(correlation_coef(I_LPF[:-time_shift_idx],FBR_LPF[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],I_LPF[:-time_shift_idx],'-b')
#     ax.set_ylabel("Magnitude Asymmetry Vector (I) [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],FBR_LPF[time_shift_idx:],"-r")
#     ax2.set_ylabel("Magnitude main bearing force vector [kN]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("I: (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     #IA vs I(t)
#     cc = round(correlation_coef(IA_LPF,I_LPF),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_sampling,IA,'-b')
#     ax.set_ylabel("Lavely Asymmetry parameter (IA) [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_sampling,I,"-r")
#     ax2.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Correlation coefficient {}".format(cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     cc = round(correlation_coef(Iy_LPF[:-time_shift_idx],LSSTipMys_LPF[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],LSSTipMys_LPF[time_shift_idx:],'-b')
#     ax.set_ylabel("Rotor moment around y axis (My) [$kN-m$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Iy_LPF[:-time_shift_idx],"-r")
#     ax2.set_ylabel("Asymmetry vector component y [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Iy: (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     cc = round(correlation_coef(Iy_LPF[:-time_shift_idx],FBz_LPF[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))
#     ax.plot(Time_OF[:-time_shift_idx],FBz_LPF[time_shift_idx:],'-b')
#     ax.set_ylabel("Bearing force component z [$kN$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Iy_LPF[:-time_shift_idx],"-r")
#     ax2.set_ylabel("Asymmetry vector component y [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Iy: (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     cc = round(correlation_coef(Iz_LPF[:-time_shift_idx],LSSTipMzs_LPF[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))
#     ax.plot(Time_OF[:-time_shift_idx],LSSTipMzs_LPF[time_shift_idx:],'-b')
#     ax.set_ylabel("Rotor moment around z axis (My) [$kN-m$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Iz_LPF[:-time_shift_idx],"-r")
#     ax2.set_ylabel("Asymmetry vector component z [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Iz: (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     cc = round(correlation_coef(Iz_LPF[:-time_shift_idx],FBy_LPF[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],FBy_LPF[time_shift_idx:],'-b')
#     ax.set_ylabel("Bearing force component y [$kN$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Iz_LPF[:-time_shift_idx],"-r")
#     ax2.set_ylabel("Asymmetry vector component z [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Iz: (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()


#     cc = round(correlation_coef(Theta[:-time_shift_idx],Aero_theta[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],Aero_theta[time_shift_idx:],'-b')
#     ax.set_ylabel("Direction Aerodynamic main bearing vector [rads]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Theta[:-time_shift_idx],"-r")
#     ax2.set_ylabel("Direction Asymmetry vector [rads]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("I(theta): (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     cc = round(correlation_coef(Theta_MR,Aero_theta),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF,Aero_theta,'-b')
#     ax.set_ylabel("Direction Aerodynamic main bearing vector [rads]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF,Theta_MR,"-r")
#     ax2.set_ylabel("Direction OOPBM vector [rads]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Correlation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()


#     cc = round(correlation_coef(Theta[:-time_shift_idx],Theta_FB[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],Theta_FB[time_shift_idx:],'-b')
#     ax.set_ylabel("Direction main bearing vector [rads]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Theta[:-time_shift_idx],"-r")
#     ax2.set_ylabel("Direction Asymmetry vector [rads]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("I(theta): (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     cc = round(correlation_coef(Iz_LPF[:-time_shift_idx],Theta_FB[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],Theta_FB[time_shift_idx:],'-b')
#     ax.set_ylabel("Direction main bearing vector [rads]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Iz_LPF[:-time_shift_idx],"-r")
#     ax2.set_ylabel("Asymmetry around z axis [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("I(theta): (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     cc = round(correlation_coef(Iy_LPF[:-time_shift_idx],Theta_FB[time_shift_idx:]),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_OF[:-time_shift_idx],Theta_FB[time_shift_idx:],'-b')
#     ax.set_ylabel("Direction main bearing vector [rads]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_OF[:-time_shift_idx],Iy_LPF[:-time_shift_idx],"-r")
#     ax2.set_ylabel("Asymmetry around y axis [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("I(theta): (-63m plane), time shifted {}s, Low pass filtered 0.3Hz \nCorrelation coefficient {}".format(time_shift,cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()



#     #plot time spectra
#     #IA vs I(t)
#     frq_IA,PSD_IA = temporal_spectra(IA,dt_sampling,Var="IA")
#     frq_I,PSD_I = temporal_spectra(I,dt_sampling,Var="I")

#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.loglog(frq_IA,PSD_IA,'-b')
#     ax.set_ylabel("PSD Lavely Asymmetry parameter (IA) [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.loglog(frq_I,PSD_I,"-r")
#     ax2.set_ylabel("PSD Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     ax.set_xlabel("Frequency [Hz]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     frq_Iy,PSD_Iy = temporal_spectra(Iy,dt_sampling,Var="Iy")
#     frq_My,PSD_My = temporal_spectra(LSSTipMys,dt,Var="My")

#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.loglog(frq_My,PSD_My,'-b')
#     ax.set_ylabel("PSD Rotor moment around y axis (My) [$kN-m$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.loglog(frq_Iy,PSD_Iy,"-r")
#     ax2.set_ylabel("PSD Asymmetry vector component y [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     ax.set_xlabel("Frequency [Hz]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     frq_FBz,PSD_FBz = temporal_spectra(FBz,dt,Var="FBz")

#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.loglog(frq_FBz,PSD_FBz,'-b')
#     ax.set_ylabel("PSD Bearing force component z [$kN$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.loglog(frq_Iy,PSD_Iy,"-r")
#     ax2.set_ylabel("PSD Asymmetry vector component y [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     ax.set_xlabel("Frequency [Hz]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     frq_Iz,PSD_Iz = temporal_spectra(Iz,dt_sampling,Var="Iz")
#     frq_Mz,PSD_Mz = temporal_spectra(LSSTipMzs,dt,Var="Mz")

#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.loglog(frq_Mz,PSD_Mz,'-b')
#     ax.set_ylabel("PSD Rotor moment around z axis (My) [$kN-m$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.loglog(frq_Iz,PSD_Iz,"-r")
#     ax2.set_ylabel("PSD Asymmetry vector component z [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     ax.set_xlabel("Frequency [Hz]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     frq_FBy,PSD_FBy = temporal_spectra(FBy,dt,Var="FBy")

#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.loglog(frq_FBy,PSD_FBy,'-b')
#     ax.set_ylabel("PSD Bearing force component y [$kN$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.loglog(frq_Iz,PSD_Iz,"-r")
#     ax2.set_ylabel("PSD Asymmetry vector component z [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     ax.set_xlabel("Frequency [Hz]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

 
#     #new asymmetry parameters
#     cc = round(correlation_coef(Iy,Iy_Asy),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_sampling,Iy,'-b')
#     ax.set_ylabel("Iy [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_Asy,Iy_Asy,"-r")
#     ax2.set_ylabel("Iy Asym [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Correlation coefficient {}".format(cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()

#     cc = round(correlation_coef(Iz,Iz_Asy),2)
#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_sampling,Iz,'-b')
#     ax.set_ylabel("Iz [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_Asy,Iz_Asy,"-r")
#     ax2.set_ylabel("Iz Asym [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     plt.title("Correlation coefficient {}".format(cc),fontsize=16)
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()


#     fig,ax = plt.subplots(figsize=(14,8))

#     ax.plot(Time_sampling,Iy,'-b')
#     ax.set_ylabel("Iy [$m^4/s$]",fontsize=14)
#     ax.yaxis.label.set_color('blue')
#     ax2 = ax.twinx()
#     ax2.plot(Time_sampling,Iz,"-r")
#     ax2.set_ylabel("Iz [$m^4/s$]",fontsize=14)
#     ax2.yaxis.label.set_color('red')
#     ax.set_xlabel("Time [s]",fontsize=16)
#     plt.tight_layout()
#     plt.grid()
#     pdf.savefig()
#     plt.close()





