from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt


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




in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

Rigid_df = Dataset(in_dir+"Dataset.nc")

print(Rigid_df)

Time_OF = np.array(Rigid_df.variables["Time_OF"])
dt = Time_OF[1]
Start_time_idx = np.searchsorted(Time_OF,200)
Time_OF = Time_OF[Start_time_idx:]

Time_sampling = np.array(Rigid_df.variables["Time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]
dt_sampling = Time_sampling[1]
Start_time_sampling_idx = np.searchsorted(Time_sampling,200)
Time_sampling = Time_sampling[Start_time_sampling_idx:]

OF_vars = Rigid_df.groups["OpenFAST_Variables"]
print(OF_vars)
LSShftFxa_R = np.array(OF_vars.variables["LSShftFxa"][Start_time_idx:])
LSShftFys_R = np.array(OF_vars.variables["LSShftFys"][Start_time_idx:])
LSShftFzs_R = np.array(OF_vars.variables["LSShftFzs"][Start_time_idx:])
LSShftMxa_R = np.array(OF_vars.variables["LSShftMxa"][Start_time_idx:])
LSSTipMys_R = np.array(OF_vars.variables["LSSTipMys"][Start_time_idx:])
#Filtering My
LPF_1_LSSTipMys_R = low_pass_filter(LSSTipMys_R,0.3,dt)
LPF_2_LSSTipMys_R = low_pass_filter(LSSTipMys_R,0.9,dt)
LPF_3_LSSTipMys_R = low_pass_filter(LSSTipMys_R,1.5,dt)
BPF_LSSTipMys_R = np.subtract(LPF_2_LSSTipMys_R,LPF_1_LSSTipMys_R)
HPF_LSSTipMys_R = np.subtract(LSSTipMys_R,LPF_3_LSSTipMys_R)
HPF_LSSTipMys_R = np.array(low_pass_filter(HPF_LSSTipMys_R,40,dt))

LSSTipMzs_R = np.array(OF_vars.variables["LSSTipMzs"][Start_time_idx:])
#Filtering Mz
LPF_1_LSSTipMzs_R = low_pass_filter(LSSTipMzs_R,0.3,dt)
LPF_2_LSSTipMzs_R = low_pass_filter(LSSTipMzs_R,0.9,dt)
LPF_3_LSSTipMzs_R = low_pass_filter(LSSTipMzs_R,1.5,dt)
BPF_LSSTipMzs_R = np.subtract(LPF_2_LSSTipMzs_R,LPF_1_LSSTipMzs_R)
HPF_LSSTipMzs_R = np.subtract(LSSTipMzs_R,LPF_3_LSSTipMzs_R)
HPF_LSSTipMzs_R = np.array(low_pass_filter(HPF_LSSTipMzs_R,40,dt))

LSSTipMR_R = np.sqrt(np.add(np.square(LSSTipMys_R),np.square(LSSTipMzs_R)))
#Filtering MR
LPF_1_LSSTipMR_R = low_pass_filter(LSSTipMR_R,0.3,dt)
LPF_2_LSSTipMR_R = low_pass_filter(LSSTipMR_R,0.9,dt)
LPF_3_LSSTipMR_R = low_pass_filter(LSSTipMR_R,1.5,dt)
BPF_LSSTipMR_R = np.subtract(LPF_2_LSSTipMR_R,LPF_1_LSSTipMR_R)
HPF_LSSTipMR_R = np.subtract(LSSTipMR_R,LPF_3_LSSTipMR_R)
HPF_LSSTipMR_R = np.array(low_pass_filter(HPF_LSSTipMR_R,40,dt))

Rotor_avg_vars = Rigid_df.groups["Rotor_Avg_Variables"]
Rotor_avg_vars_R = Rotor_avg_vars.groups["5.5"]
Ux_R_R = np.array(Rotor_avg_vars_R.variables["Ux"][Start_time_sampling_idx:])
Iy_R_R = np.array(Rotor_avg_vars_R.variables["Iy"][Start_time_sampling_idx:])
Iz_R_R = np.array(Rotor_avg_vars_R.variables["Iz"][Start_time_sampling_idx:])
I_R_R = np.sqrt(np.add(np.square(Iy_R_R),np.square(Iz_R_R)))
Rotor_avg_vars_D = Rotor_avg_vars.groups["63.0"]
Ux_D_R = np.array(Rotor_avg_vars_D.variables["Ux"][Start_time_sampling_idx:])
Iy_D_R = np.array(Rotor_avg_vars_D.variables["Iy"][Start_time_sampling_idx:])
Iz_D_R = np.array(Rotor_avg_vars_D.variables["Iz"][Start_time_sampling_idx:])
I_D_R = np.sqrt(np.add(np.square(Iy_R_R),np.square(Iz_R_R)))

in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

Elastic_df = Dataset(in_dir+"Dataset.nc")

OF_vars = Elastic_df.groups["OpenFAST_Variables"]

LSShftFxa_E = np.array(OF_vars.variables["LSShftFxa"][Start_time_idx:])
LSShftFys_E = np.array(OF_vars.variables["LSShftFys"][Start_time_idx:])
LSShftFzs_E = np.array(OF_vars.variables["LSShftFzs"][Start_time_idx:])
LSShftMxa_E = np.array(OF_vars.variables["LSShftMxa"][Start_time_idx:])
LSSTipMys_E = np.array(OF_vars.variables["LSSTipMys"][Start_time_idx:])

#Filtering My
LPF_1_LSSTipMys_E = low_pass_filter(LSSTipMys_E,0.3,dt)
LPF_2_LSSTipMys_E = low_pass_filter(LSSTipMys_E,0.9,dt)
LPF_3_LSSTipMys_E = low_pass_filter(LSSTipMys_E,1.5,dt)
BPF_LSSTipMys_E = np.subtract(LPF_2_LSSTipMys_E,LPF_1_LSSTipMys_E)
HPF_LSSTipMys_E = np.subtract(LSSTipMys_E,LPF_3_LSSTipMys_E)
HPF_LSSTipMys_E = np.array(low_pass_filter(HPF_LSSTipMys_E,40,dt))

LSSTipMzs_E = np.array(OF_vars.variables["LSSTipMzs"][Start_time_idx:])

#Filtering Mz
LPF_1_LSSTipMzs_E = low_pass_filter(LSSTipMzs_E,0.3,dt)
LPF_2_LSSTipMzs_E = low_pass_filter(LSSTipMzs_E,0.9,dt)
LPF_3_LSSTipMzs_E = low_pass_filter(LSSTipMzs_E,1.5,dt)
BPF_LSSTipMzs_E = np.subtract(LPF_2_LSSTipMzs_E,LPF_1_LSSTipMzs_E)
HPF_LSSTipMzs_E = np.subtract(LSSTipMzs_E,LPF_3_LSSTipMzs_E)
HPF_LSSTipMzs_E = np.array(low_pass_filter(HPF_LSSTipMzs_E,40,dt))

LSSTipMR_E = np.sqrt(np.add(np.square(LSSTipMys_E),np.square(LSSTipMzs_E)))

#Filtering MR
LPF_1_LSSTipMR_E = low_pass_filter(LSSTipMR_E,0.3,dt)
LPF_2_LSSTipMR_E = low_pass_filter(LSSTipMR_E,0.9,dt)
LPF_3_LSSTipMR_E = low_pass_filter(LSSTipMR_E,1.5,dt)
BPF_LSSTipMR_E = np.subtract(LPF_2_LSSTipMR_E,LPF_1_LSSTipMR_E)
HPF_LSSTipMR_E = np.subtract(LSSTipMR_E,LPF_3_LSSTipMR_E)
HPF_LSSTipMR_E = np.array(low_pass_filter(HPF_LSSTipMR_E,40,dt))



Rotor_avg_vars = Elastic_df.groups["Rotor_Avg_Variables"]
Rotor_avg_vars_R = Rotor_avg_vars.groups["5.5"]
Ux_R_E = np.array(Rotor_avg_vars_R.variables["Ux"][Start_time_sampling_idx:])
Iy_R_E = np.array(Rotor_avg_vars_R.variables["Iy"][Start_time_sampling_idx:])
Iz_R_E = np.array(Rotor_avg_vars_R.variables["Iz"][Start_time_sampling_idx:])
I_R_E = np.sqrt(np.add(np.square(Iy_R_R),np.square(Iz_R_R)))
Rotor_avg_vars_D = Rotor_avg_vars.groups["63.0"]
Ux_D_E = np.array(Rotor_avg_vars_D.variables["Ux"][Start_time_sampling_idx:])
Iy_D_E = np.array(Rotor_avg_vars_D.variables["Iy"][Start_time_sampling_idx:])
Iz_D_E = np.array(Rotor_avg_vars_D.variables["Iz"][Start_time_sampling_idx:])
I_D_E = np.sqrt(np.add(np.square(Iy_R_R),np.square(Iz_R_R)))

out_dir=in_dir
with PdfPages(out_dir+'Elastic_Rigid_comparisons.pdf') as pdf:
    plt.rcParams['font.size'] = 16
    #plot Time varying quanities
    #horizontal velocity
    cc = round(correlation_coef(LSShftFxa_E,LSShftFxa_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LSShftFxa_R,"-b",label="Rigid rotor\n{}".format(moments(LSShftFxa_R)))
    plt.plot(Time_OF,LSShftFxa_E,"-r",label="Elastic rotor\n{}".format(moments(LSShftFxa_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor thrust [kN]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(LSShftFys_E,LSShftFys_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LSShftFys_R,"-b",label="Rigid rotor\n{}".format(moments(LSShftFys_R)))
    plt.plot(Time_OF,LSShftFys_E,"-r",label="Elastic rotor\n{}".format(moments(LSShftFys_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor force y component [kN]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(LSShftFzs_E,LSShftFzs_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LSShftFzs_R,"-b",label="Rigid rotor\n{}".format(moments(LSShftFzs_R)))
    plt.plot(Time_OF,LSShftFzs_E,"-r",label="Elastic rotor\n{}".format(moments(LSShftFzs_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor force z component [kN]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(LSShftMxa_E,LSShftMxa_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LSShftMxa_R,"-b",label="Rigid rotor\n{}".format(moments(LSShftMxa_R)))
    plt.plot(Time_OF,LSShftMxa_E,"-r",label="Elastic rotor\n{}".format(moments(LSShftMxa_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor torque [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(LSShftMxa_R,dt,Var="Mx_R")
    plt.loglog(frq,PSD,"-b",label="Rigid rotor")
    frq,PSD = temporal_spectra(LSShftMxa_E,dt,Var="Mx_E")
    plt.loglog(frq,PSD,"-r",label="Elastic rotor")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor torque [kN-m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(LSShftMxa_R,dt,Var="Mx_R")
    plt.loglog(frq,PSD,"-b",label="Rigid rotor")
    frq,PSD = temporal_spectra(LSShftMxa_E,dt,Var="Mx_E")
    plt.loglog(frq,np.multiply(PSD,1e+05),"-r",label="Elastic rotor +1e+05kN")
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor torque [kN-m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(LSSTipMys_E,LSSTipMys_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LSSTipMys_R,"-b",label="Rigid rotor\n{}".format(moments(LSSTipMys_R)))
    plt.plot(Time_OF,LSSTipMys_E,"-r",label="Elastic rotor\n{}".format(moments(LSSTipMys_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor moment y component [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(LPF_1_LSSTipMys_E,LPF_1_LSSTipMys_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LPF_1_LSSTipMys_R,"-b",label="Rigid rotor\n{}".format(moments(LPF_1_LSSTipMys_R)))
    plt.plot(Time_OF,LPF_1_LSSTipMys_E,"-r",label="Elastic rotor\n{}".format(moments(LPF_1_LSSTipMys_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("LPF 0.3Hz Rotor moment y component [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(BPF_LSSTipMys_E,BPF_LSSTipMys_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,BPF_LSSTipMys_R,"-b",label="Rigid rotor\n{}".format(moments(BPF_LSSTipMys_R)))
    plt.plot(Time_OF,BPF_LSSTipMys_E,"-r",label="Elastic rotor\n{}".format(moments(BPF_LSSTipMys_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("BPF 0.3-0.9Hz Rotor moment y component [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(HPF_LSSTipMys_E,HPF_LSSTipMys_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,HPF_LSSTipMys_R,"-b",label="Rigid rotor\n{}".format(moments(HPF_LSSTipMys_R)))
    plt.plot(Time_OF,HPF_LSSTipMys_E,"-r",label="Elastic rotor\n{}".format(moments(HPF_LSSTipMys_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("HPF 1.5Hz Rotor moment y component [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()



    cc = round(correlation_coef(LSSTipMzs_E,LSSTipMzs_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LSSTipMzs_R,"-b",label="Rigid rotor\n{}".format(moments(LSSTipMzs_R)))
    plt.plot(Time_OF,LSSTipMzs_E,"-r",label="Elastic rotor\n{}".format(moments(LSSTipMzs_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor moment z component [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(LPF_1_LSSTipMzs_E,LPF_1_LSSTipMzs_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LPF_1_LSSTipMzs_R,"-b",label="Rigid rotor\n{}".format(moments(LPF_1_LSSTipMzs_R)))
    plt.plot(Time_OF,LPF_1_LSSTipMzs_E,"-r",label="Elastic rotor\n{}".format(moments(LPF_1_LSSTipMzs_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("LPF 0.3Hz Rotor moment z component [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(BPF_LSSTipMzs_E,BPF_LSSTipMzs_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,BPF_LSSTipMzs_R,"-b",label="Rigid rotor\n{}".format(moments(BPF_LSSTipMzs_R)))
    plt.plot(Time_OF,BPF_LSSTipMzs_E,"-r",label="Elastic rotor\n{}".format(moments(BPF_LSSTipMzs_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("BPF 0.3-0.9Hz Rotor moment z component [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(HPF_LSSTipMzs_E,HPF_LSSTipMzs_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,HPF_LSSTipMzs_R,"-b",label="Rigid rotor\n{}".format(moments(HPF_LSSTipMzs_R)))
    plt.plot(Time_OF,HPF_LSSTipMzs_E,"-r",label="Elastic rotor\n{}".format(moments(HPF_LSSTipMzs_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("HPF 1.5Hz Rotor moment z component [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(LSSTipMR_E,LSSTipMR_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LSSTipMR_R,"-b",label="Rigid rotor\n{}".format(moments(LSSTipMR_R)))
    plt.plot(Time_OF,LSSTipMR_E,"-r",label="Elastic rotor\n{}".format(moments(LSSTipMR_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Out-of-plane bending moment magnitude [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(LPF_1_LSSTipMR_E,LPF_1_LSSTipMR_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,LPF_1_LSSTipMR_R,"-b",label="Rigid rotor\n{}".format(moments(LPF_1_LSSTipMR_R)))
    plt.plot(Time_OF,LPF_1_LSSTipMR_E,"-r",label="Elastic rotor\n{}".format(moments(LPF_1_LSSTipMR_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("LPF 0.3Hz Out-of-plane bending moment magnitude [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(BPF_LSSTipMR_E,BPF_LSSTipMR_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,BPF_LSSTipMR_R,"-b",label="Rigid rotor\n{}".format(moments(BPF_LSSTipMR_R)))
    plt.plot(Time_OF,BPF_LSSTipMR_E,"-r",label="Elastic rotor\n{}".format(moments(BPF_LSSTipMR_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("BPF 0.3-0.9Hz Out-of-plane bending moment magnitude [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(HPF_LSSTipMR_E,HPF_LSSTipMR_R),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_OF,HPF_LSSTipMR_R,"-b",label="Rigid rotor\n{}".format(moments(HPF_LSSTipMR_R)))
    plt.plot(Time_OF,HPF_LSSTipMR_E,"-r",label="Elastic rotor\n{}".format(moments(HPF_LSSTipMR_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("HPF 1.5Hz Out-of-plane bending moment magnitude [kN-m]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(LSSTipMR_R,dt,Var="MR_R")
    plt.loglog(frq,PSD,"-b",label="Rigid rotor")
    frq,PSD = temporal_spectra(LSSTipMR_E,dt,Var="MR_E")
    plt.loglog(frq,PSD,"-r",label="Elastic rotor")
    plt.xlabel("Time [s]")
    plt.ylabel("Out-of-plane bending moment magnitude [kN-m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(LSSTipMR_R,dt,Var="MR_R")
    plt.loglog(frq,PSD,"-b",label="Rigid rotor")
    frq,PSD = temporal_spectra(LSSTipMR_E,dt,Var="MR_E")
    plt.loglog(frq,np.multiply(PSD,1e+05),"-r",label="Elastic rotor +1e+05kN")
    plt.xlabel("Time [s]")
    plt.ylabel("Out-of-plane bending moment magnitude [kN-m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    cc = round(correlation_coef(Ux_R_R,Ux_R_E),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Ux_R_R,"-b",label="Rigid rotor\n{}".format(moments(Ux_R_R)))
    plt.plot(Time_sampling,Ux_R_E,"-r",label="Elastic rotor\n{}".format(moments(Ux_R_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor averaged velocity at rotor plane [m/s]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(Ux_D_R,Ux_D_E),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Ux_D_R,"-b",label="Rigid rotor\n{}".format(moments(Ux_D_R)))
    plt.plot(Time_sampling,Ux_D_E,"-r",label="Elastic rotor\n{}".format(moments(Ux_D_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Rotor averaged velocity 1/2D in front of rotor plane [m/s]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    cc = round(correlation_coef(Iy_R_R,Iy_R_E),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iy_R_R,"-b",label="Rigid rotor\n{}".format(moments(Iy_R_R)))
    plt.plot(Time_sampling,Iy_R_E,"-r",label="Elastic rotor\n{}".format(moments(Iy_R_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry around y axis at rotor plane [m/s]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(Iy_D_R,Iy_D_E),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iy_D_R,"-b",label="Rigid rotor\n{}".format(moments(Iy_D_R)))
    plt.plot(Time_sampling,Iy_D_E,"-r",label="Elastic rotor\n{}".format(moments(Iy_D_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry around y axis 1/2D in front of rotor plane [m/s]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    cc = round(correlation_coef(Iz_R_R,Iz_R_E),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iz_R_R,"-b",label="Rigid rotor\n{}".format(moments(Iz_R_R)))
    plt.plot(Time_sampling,Iz_R_E,"-r",label="Elastic rotor\n{}".format(moments(Iz_R_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry around z axis at rotor plane [m/s]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(Iz_D_R,Iz_D_E),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,Iz_D_R,"-b",label="Rigid rotor\n{}".format(moments(Iz_D_R)))
    plt.plot(Time_sampling,Iz_D_E,"-r",label="Elastic rotor\n{}".format(moments(Iz_D_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry around z axis 1/2D in front of rotor plane [m/s]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()


    cc = round(correlation_coef(I_R_R,I_R_E),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,I_R_R,"-b",label="Rigid rotor\n{}".format(moments(I_R_R)))
    plt.plot(Time_sampling,I_R_E,"-r",label="Elastic rotor\n{}".format(moments(I_R_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry vector magnitude at rotor plane [m/s]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()

    cc = round(correlation_coef(I_D_R,I_D_E),2)
    plt.figure(figsize=(14,8))
    plt.plot(Time_sampling,I_D_R,"-b",label="Rigid rotor\n{}".format(moments(I_D_R)))
    plt.plot(Time_sampling,I_D_E,"-r",label="Elastic rotor\n{}".format(moments(I_D_E)))
    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry vector magnitude 1/2D in front of rotor plane [m/s]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    pdf.savefig()
    plt.close()