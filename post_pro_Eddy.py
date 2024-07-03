import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import statistics
from scipy.signal import butter,filtfilt
from scipy import interpolate


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


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def theta_cont(Iy,Iz,Theta,Time_steps):
    for it in Time_steps[:-1]:
        if Iz[it] > 0 and Iy[it] > 0 and Iy[it+1] > 0 and Iz[it+1] < 0:
            Theta[it+1:]=Theta[it+1:]-360
        elif Iz[it] < 0 and Iy[it] > 0 and Iy[it+1] > 0 and Iz[it+1] > 0:
            Theta[it+1:]=Theta[it+1:]+360  

    return Theta


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Asymmetry_Dataset.nc")

Time = np.array(a.variables["time"])
Time = Time - Time[0]
dt = Time[1] - Time[0]
Time_steps = np.arange(0,len(Time))

Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)
Time = Time[Time_start_idx:]
Time_steps = np.arange(0,len(Time))
cutoff = 1/60

A_high = np.array(a.variables["Area_high"][Time_start_idx:])
mu,sd,sk,k = moments(A_high)
print("stats A_high ",mu,sd,sk,k)
A_low = np.array(a.variables["Area_low"][Time_start_idx:])
mu,sd,sk,k = moments(A_low)
print("stats A_low ",mu,sd,sk,k)
A_int = np.array(a.variables["Area_int"][Time_start_idx:])
mu,sd,sk,k = moments(A_int)
print("stats A_int ",mu,sd,sk,k)



A_high_trend = low_pass_filter(A_high,cutoff,dt)
A_high_flucs = np.subtract(A_high,A_high_trend)
A_low_trend = low_pass_filter(A_low,cutoff,dt)
A_low_flucs = np.subtract(A_low,A_low_trend)
A_int_trend = low_pass_filter(A_int,cutoff,dt)
A_int_flucs = np.subtract(A_int,A_int_trend)


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

Iz_high = np.array(a.variables["Iz_high"][Time_start_idx:])
mu,sd,sk,k = moments(Iz_high)
print("stats Iz_high ",mu,sd,sk,k)
Iz_low = np.array(a.variables["Iz_low"][Time_start_idx:])
mu,sd,sk,k = moments(Iz_low)
print("stats Iz_low ",mu,sd,sk,k)
Iz_int = np.array(a.variables["Iz_int"][Time_start_idx:])
mu,sd,sk,k = moments(Iz_int)
print("stats Iz_int ",mu,sd,sk,k)

I_high_mag_mean = np.sqrt(np.add(np.square(np.mean(Iy_high)),np.square(np.mean(Iz_high))))
I_high_theta_mean = np.degrees(np.arctan2(np.mean(Iz_high),np.mean(Iy_high)))
if I_high_theta_mean < 0:
    I_high_theta_mean+=360
Iy_high_flucs = np.subtract(Iy_high,np.mean(Iy_high))
Iz_high_flucs = np.subtract(Iz_high,np.mean(Iz_high))
sigma_I_high = np.mean(np.square(Iy_high_flucs)) + np.mean(np.square(Iz_high_flucs))

print("I high: ",I_high_mag_mean,I_high_theta_mean,np.sqrt(sigma_I_high))

I_low_mag_mean = np.sqrt(np.add(np.square(np.mean(Iy_low)),np.square(np.mean(Iz_low))))
I_low_theta_mean = np.degrees(np.arctan2(np.mean(Iz_low),np.mean(Iy_low)))
if I_low_theta_mean < 0:
    I_low_theta_mean+=360
Iy_low_flucs = np.subtract(Iy_low,np.mean(Iy_low))
Iz_low_flucs = np.subtract(Iz_low,np.mean(Iz_low))
sigma_I_low = np.mean(np.square(Iy_low_flucs)) + np.mean(np.square(Iz_low_flucs))

print("I low: ",I_low_mag_mean,I_low_theta_mean,np.sqrt(sigma_I_low))

I_int_mag_mean = np.sqrt(np.add(np.square(np.mean(Iy_int)),np.square(np.mean(Iz_int))))
I_int_theta_mean = np.degrees(np.arctan2(np.mean(Iz_int),np.mean(Iy_int)))
if I_int_theta_mean < 0:
    I_int_theta_mean+=360
Iy_int_flucs = np.subtract(Iy_int,np.mean(Iy_int))
Iz_int_flucs = np.subtract(Iz_int,np.mean(Iz_int))
sigma_I_int = np.mean(np.square(Iy_int_flucs)) + np.mean(np.square(Iz_int_flucs))

print("I int: ",I_int_mag_mean,I_int_theta_mean,np.sqrt(sigma_I_int))

Iy_high_trend = low_pass_filter(Iy_high,cutoff,dt)
Iy_high_flucs = np.subtract(Iy_high,Iy_high_trend)
Iy_low_trend = low_pass_filter(Iy_low,cutoff,dt)
Iy_low_flucs = np.subtract(Iy_low,Iy_low_trend)
Iy_int_trend = low_pass_filter(Iy_int,cutoff,dt)
Iy_int_flucs = np.subtract(Iy_int,Iy_int_trend)

Iy_high_frq, Iy_high_PSD = temporal_spectra(Iy_high,dt,Var="Iy_high")
Iy_low_frq, Iy_low_PSD = temporal_spectra(Iy_low,dt,Var="Iy_low")
Iy_int_frq, Iy_int_PSD = temporal_spectra(Iy_int,dt,Var="Iy_int")

Iz_high_trend = low_pass_filter(Iz_high,cutoff,dt)
Iz_high_flucs = np.subtract(Iz_high,Iz_high_trend)
Iz_low_trend = low_pass_filter(Iz_low,cutoff,dt)
Iz_low_flucs = np.subtract(Iz_low,Iz_low_trend)
Iz_int_trend = low_pass_filter(Iz_int,cutoff,dt)
Iz_int_flucs = np.subtract(Iz_int,Iz_int_trend)

Iz_high_frq, Iz_high_PSD = temporal_spectra(Iz_high,dt,Var="Iz_high")
Iz_low_frq, Iz_low_PSD = temporal_spectra(Iz_low,dt,Var="Iz_low")
Iz_int_frq, Iz_int_PSD = temporal_spectra(Iz_int,dt,Var="Iz_int")


I_high = np.sqrt(np.add(np.square(Iy_high),np.square(Iz_high)))
mu,sd,sk,k = moments(I_high)
print("stats I_high ",mu,sd,sk,k)
I_low = np.sqrt(np.add(np.square(Iy_low),np.square(Iz_low)))
mu,sd,sk,k = moments(I_low)
print("stats I_low ",mu,sd,sk,k)
I_int = np.sqrt(np.add(np.square(Iy_int),np.square(Iz_int)))
mu,sd,sk,k = moments(I_int)
print("stats I_int ",mu,sd,sk,k)

I_high_trend = low_pass_filter(I_high,cutoff,dt)
I_high_flucs = np.subtract(I_high,I_high_trend)
I_low_trend = low_pass_filter(I_low,cutoff,dt)
I_low_flucs = np.subtract(I_low,I_low_trend)
I_int_trend = low_pass_filter(I_int,cutoff,dt)
I_int_flucs = np.subtract(I_int,I_int_trend)

I_high_frq, I_high_PSD = temporal_spectra(I_high,dt,Var="I_high")
I_low_frq, I_low_PSD = temporal_spectra(I_low,dt,Var="I_low")
I_int_frq, I_int_PSD = temporal_spectra(I_int,dt,Var="Iz_int")

Theta_high = np.degrees(np.arctan2(Iz_high,Iy_high))
print("stats Delta theta_high ",mu,sd,sk,k)
Theta_high = np.array(theta_360(Theta_high))
mu,sd,sk,k = moments(Theta_high)
print("stats theta_high ",mu,sd,sk,k)
Delta_Theta_high = np.subtract(Theta_high[1:],Theta_high[:-1])
mu,sd,sk,k = moments(Delta_Theta_high)

Theta_low = np.degrees(np.arctan2(Iz_low,Iy_low))
Delta_Theta_low = np.subtract(Theta_low[1:],Theta_low[:-1])
mu,sd,sk,k = moments(Delta_Theta_low)
print("stats Delta theta_low ",mu,sd,sk,k)
Theta_low = np.array(theta_360(Theta_low))
mu,sd,sk,k = moments(Theta_low)
print("stats theta_low ",mu,sd,sk,k)

Theta_int = np.degrees(np.arctan2(Iz_int,Iy_int))
Delta_Theta_int = np.subtract(Theta_int[1:],Theta_int[:-1])
mu,sd,sk,k = moments(Delta_Theta_int)
print("stats Delta theta_int ",mu,sd,sk,k)
Theta_int = np.array(theta_360(Theta_int))
mu,sd,sk,k = moments(Theta_int)
print("stats theta_int ",mu,sd,sk,k)


Theta_high_frq, Theta_high_PSD = temporal_spectra(Theta_high,dt,Var="Theta_high")
Theta_low_frq, Theta_low_PSD = temporal_spectra(Theta_low,dt,Var="Theta_low")
Theta_int_frq, Theta_int_PSD = temporal_spectra(Theta_int,dt,Var="Theta_int")


Ux_high = np.array(a.variables["Ux_high"][Time_start_idx:])
mu,sd,sk,k = moments(Ux_high)
print("stats Ux_high ",mu,sd,sk,k)
Ux_low = np.array(a.variables["Ux_low"][Time_start_idx:])
mu,sd,sk,k = moments(Ux_low)
print("stats Ux_low ",mu,sd,sk,k)
Ux_int = np.array(a.variables["Ux_int"][Time_start_idx:])
mu,sd,sk,k = moments(Ux_int)
print("stats Ux_int ",mu,sd,sk,k)

Ux_high_frq, Ux_high_PSD = temporal_spectra(Ux_high,dt,Var="Ux_high")
Ux_low_frq, Ux_low_PSD = temporal_spectra(Ux_low,dt,Var="Ux_low")
Ux_int_frq, Ux_int_PSD = temporal_spectra(Ux_int,dt,Var="Ux_int")

Iy = np.array(a.variables["Iy"][Time_start_idx:])
Iz = -np.array(a.variables["Iz"][Time_start_idx:])

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
mu,sd,sk,k = moments(I)
print("stats I ",mu,sd,sk,k)
I_frq, I_PSD = temporal_spectra(I,dt,Var="I")


Theta = np.degrees(np.arctan2(Iz,Iy))
Delta_Theta = np.subtract(Theta[1:],Theta[:-1])
mu,sd,sk,k = moments(Delta_Theta)
print("stats Delta theta ",mu,sd,sk,k)
Theta = theta_360(Theta)
Theta = np.radians(np.array(Theta))

Theta_frq, Theta_PSD = temporal_spectra(Theta,dt,Var="Delta_Theta")

A_rot = np.pi*63**2

Frac_high_area = np.true_divide(A_high,A_rot)
Frac_low_area = np.true_divide(A_low,A_rot)
Frac_int_area = np.true_divide(A_int,A_rot)
Tot_area = np.add(np.add(Frac_high_area,Frac_low_area),Frac_int_area)

Times_high = []
Times_low = []
Times_int = []
Times_high_low = []
Times_high_int = []
Times_low_int = []
Times_high_low_int = []
for it in Time_steps:
    if Frac_high_area[it] >= 0.7:
        Times_high.append(it)
    elif Frac_low_area[it] >= 0.7:
        Times_low.append(it)
    elif Frac_int_area[it] >= 0.7:
        Times_int.append(it)
    elif Frac_high_area[it] >= 0.4 and Frac_low_area[it] >= 0.4:
        Times_high_low.append(it)
    elif Frac_high_area[it] >= 0.4 and Frac_int_area[it] >= 0.4:
        Times_high_int.append(it)
    elif Frac_low_area[it] >= 0.4 and Frac_int_area[it] >= 0.4:
        Times_low_int.append(it)
    else:
        Times_high_low_int.append(it)


fig = plt.figure(figsize=(14,8))
plt.plot(Time,I,"-k")
plt.plot(Time[Times_high],I[Times_high],"or")
plt.plot(Time[Times_low],I[Times_low],"ob")
plt.plot(Time[Times_int],I[Times_int],"og")
plt.grid()
plt.tight_layout()



P_high_Iy = np.true_divide(Iy_high,Iy)
P_low_Iy = np.true_divide(Iy_low,Iy)
P_int_Iy = np.true_divide(Iy_int,Iy)
P_Tot_Iy = np.add(np.add(P_high_Iy,P_low_Iy),P_int_Iy)

P_high_Iz = np.true_divide(Iz_high,Iz)
P_low_Iz = np.true_divide(Iz_low,Iz)
P_int_Iz = np.true_divide(Iz_int,Iz)
P_Tot_Iz = np.add(np.add(P_high_Iz,P_low_Iz),P_int_Iz)

P_Tot_I = np.true_divide(np.square(Iy_high),np.square(I)) + np.true_divide(np.square(Iy_low),np.square(I)) + np.true_divide(np.square(Iy_int),np.square(I)) \
    + np.true_divide((2*Iy_high*Iy_low),np.square(I)) + np.true_divide((2*Iy_high*Iy_int),np.square(I)) + np.true_divide((2*Iy_low*Iy_int),np.square(I)) + \
    np.true_divide(np.square(Iz_high),np.square(I)) + np.true_divide(np.square(Iz_low),np.square(I)) + np.true_divide(np.square(Iz_int),np.square(I)) \
        + np.true_divide((2*Iz_high*Iz_low),np.square(I)) + np.true_divide((2*Iz_high*Iz_int),np.square(I)) + np.true_divide((2*Iz_low*Iz_int),np.square(I))


PIy_high = np.true_divide(np.square(Iy_high),np.square(I))
PIy_low = np.true_divide(np.square(Iy_low),np.square(I))
PIy_int = np.true_divide(np.square(Iy_int),np.square(I))
PIy_high_low = np.true_divide((2*Iy_high*Iy_low),np.square(I))
PIy_high_int = np.true_divide((2*Iy_high*Iy_int),np.square(I))
PIy_low_int = np.true_divide((2*Iy_low*Iy_int),np.square(I))
PIz_high = np.true_divide(np.square(Iz_high),np.square(I))
PIz_low = np.true_divide(np.square(Iz_low),np.square(I))
PIz_int = np.true_divide(np.square(Iz_int),np.square(I))
PIz_high_low = np.true_divide((2*Iz_high*Iz_low),np.square(I))
PIz_high_int = np.true_divide((2*Iz_high*Iz_int),np.square(I))
PIz_low_int = np.true_divide((2*Iz_low*Iz_int),np.square(I))

PI_high = PIy_high + PIz_high
PI_low = PIy_low + PIz_low
PI_int = PIy_int + PIz_int
PI_high_low = PIy_high_low + PIz_high_low
PI_high_int = PIy_high_int + PIz_high_int
PI_low_int = PIy_low_int + PIz_low_int

I_high = np.sqrt(np.square(Iy_high) + np.square(Iz_high))
I_low = np.sqrt(np.square(Iy_low) + np.square(Iz_low))
I_int = np.sqrt(np.square(Iy_int) + np.square(Iz_int))
I_high_low = 2*Iy_high*Iy_low + 2*Iz_high*Iz_low
I_high_int = 2*Iy_high*Iy_int + 2*Iz_high*Iz_int
I_low_int = 2*Iy_low*Iy_int + 2*Iz_low*Iz_int


df = Dataset(in_dir+"Dataset.nc")
Time_sampling = np.array(df.variables["time_sampling"])
Time_start = 200; Time_start_idx = np.searchsorted(Time_sampling,Time_start)
Time_sampling = Time_sampling[Time_start_idx:]
group = df.groups["63.0"]
Iy_df = np.array(group.variables["Iy"][Time_start_idx:])
Iz_df = np.array(group.variables["Iz"][Time_start_idx:])


# fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)

# ax1.plot(Time,Ux_high,'-r')
# ax1.plot(Time,Ux_low,"-b")
# ax1.plot(Time,Ux_int,"-g")
# ax1.set_ylabel("Average streamwise velocity [m/s]",fontsize=14)
# ax1.set_xlabel("Time [s]",fontsize=16)
# ax1.legend(["High speed area", "Low speed area", "intermediate area","Total"])
# ax1.grid()


# ax2.plot(Time,A_high,'-r')
# ax2.plot(Time,A_low,"-b")
# ax2.plot(Time,A_int,"-g")
# ax2.set_ylabel("Area [$m^2$]",fontsize=14)
# ax2.set_xlabel("Time [s]",fontsize=16)
# ax2.legend(["High speed area", "Low speed area", "intermediate area"])
# ax2.grid()

# plt.tight_layout()
# plt.show()



print("Iy_high cc with Iy ",correlation_coef(Iy,Iy_high))
print("Iy_low cc with Iy ",correlation_coef(Iy,Iy_low))
print("Iy_int cc with Iy ",correlation_coef(Iy,Iy_int))

print("Iz_high cc with Iz ",correlation_coef(Iz,Iz_high))
print("Iz_low cc with Iz ",correlation_coef(Iz,Iz_low))
print("Iz_int cc with Iz ",correlation_coef(Iz,Iz_int))

print("I_high cc with I ",correlation_coef(I,I_high))
print("I_low cc with I ",correlation_coef(I,I_low))
print("I_int cc with I ",correlation_coef(I,I_int))

print("I_high cc with I_low ",correlation_coef(I_high,I_low))
print("I_high cc with I_int ",correlation_coef(I_high,I_int))
print("I_low cc with I_int ",correlation_coef(I_low,I_int))

print("Theta_high cc with Theta_low ",correlation_coef(Theta_high,Theta_low))
print("Theta_high cc with Theta_int ",correlation_coef(Theta_high,Theta_int))
print("Theta_low cc with Theta_int ",correlation_coef(Theta_low,Theta_int))

print("A_high cc with A_low ",correlation_coef(A_high,A_low))
print("A_high cc with A_int ",correlation_coef(A_high,A_int))
print("A_low cc with A_int ",correlation_coef(A_low,A_int))

print("Iy_high cc with Iy_low ",correlation_coef(Iy_high,Iy_low))
print("Iy_high cc with Iy_int ",correlation_coef(Iy_high,Iy_int))
print("Iy_low cc with Iy_int ",correlation_coef(Iy_low,Iy_int))

print("Iz_high cc with Iz_low ",correlation_coef(Iz_high,Iz_low))
print("Iz_high cc with Iz_int ",correlation_coef(Iz_high,Iz_int))
print("Iz_low cc with Iz_int ",correlation_coef(Iz_low,Iz_int))

Delta_I_high_low = np.subtract(I_high,I_low)
mu,sd,sk,k = moments(Delta_I_high_low)
print("stats Delta_I_high_low ",mu,sd,sk,k)
Delta_I_high_int = np.subtract(I_high,I_int)
mu,sd,sk,k = moments(Delta_I_high_int)
print("stats Delta_I_high_int ",mu,sd,sk,k)
Delta_I_low_int = np.subtract(I_low,I_int)
mu,sd,sk,k = moments(Delta_I_low_int)
print("stats Delta_I_low_int ",mu,sd,sk,k)


print("mode Delta I High-low = ",statistics.mode(Delta_I_high_low))
print("mode Delta I High-int = ",statistics.mode(Delta_I_high_int))
print("mode Delta I low_int = ",statistics.mode(Delta_I_low_int))

Delta_Theta_high_low = abs(np.subtract(Theta_high,Theta_low))
Delta_Theta_high_int = abs(np.subtract(Theta_high,Theta_int))
Delta_Theta_low_int = abs(np.subtract(Theta_low,Theta_int))



mu,sd,sk,k = moments(Delta_Theta_high_low)
print("stats Delta_Theta_high_low ",mu,sd,sk,k)
mu,sd,sk,k = moments(Delta_Theta_high_int)
print("stats Delta_Theta_high_int ",mu,sd,sk,k)
mu,sd,sk,k = moments(Delta_Theta_low_int)
print("stats Delta_Theta_low_int ",mu,sd,sk,k)

print("mode Delta Theta High-low = ",statistics.mode(Delta_Theta_high_low))
print("mode Delta Theta High-int = ",statistics.mode(Delta_Theta_high_int))
print("mode Delta Theta low_int = ",statistics.mode(Delta_Theta_low_int))


update_polar_Asymmetry = False
update_polar_vectors = False
update_polar_Aero_vectors = False
update_pdf_plots = True


def Update_Asymmetry(it):

    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(Theta_high[it], I_high[it], c="r", s=20)
    d = ax.scatter(Theta_low[it],I_low[it], c="b", s=20)
    e = ax.scatter(Theta_int[it],I_int[it], c="g", s=20)
    f = ax.scatter(Theta[it], I[it], c="k", s=20)
    ax.arrow(0, 0, Theta_high[it], I_high[it], length_includes_head=True, color="r")
    ax.arrow(0, 0, Theta_low[it], I_low[it], length_includes_head=True, color="b")
    ax.arrow(0, 0, Theta_int[it], I_int[it], length_includes_head=True, color="g")
    ax.arrow(0, 0, Theta[it], I[it], length_includes_head="True", color="k")
    ax.set_ylim(0,np.max([np.max(I_high),np.max(I_low),np.max(I_int)]))
    ax.set_title("Asymmetry vector [$m^4/s$]\nTime = {}s".format(Time[it]), va='bottom')
    T = Time[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T


if update_polar_Asymmetry == True:
    out_dir = in_dir+"Asymmetry_analysis/polar_asymmetry/"
    with Pool() as pool:
        for T in pool.imap(Update_Asymmetry,Time_steps):

            print(T)


df_OF = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(df_OF.variables["time_OF"])

Time_start = 200

Time_start_idx = np.searchsorted(Time_OF,Time_start)

Time_OF = Time_OF[Time_start_idx:]
dt_OF = Time_OF[1] - Time_OF[0]


Azimuth = np.radians(np.array(df_OF.variables["Azimuth"][Time_start_idx:]))

RtAeroFyh = np.array(df_OF.variables["RtAeroFyh"][Time_start_idx:])
RtAeroFzh = np.array(df_OF.variables["RtAeroFzh"][Time_start_idx:])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(df_OF.variables["RtAeroMyh"][Time_start_idx:])
RtAeroMzh = np.array(df_OF.variables["RtAeroMzh"][Time_start_idx:])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000

f = interpolate.interp1d(Time_OF,RtAeroFys)
RtAeroFys = f(Time)
RtAeroFys_LPF = low_pass_filter(RtAeroFys,0.3,dt=dt)
f = interpolate.interp1d(Time_OF,RtAeroFzs)
RtAeroFzs = f(Time)
RtAeroFzs_LPF = low_pass_filter(RtAeroFzs,0.3,dt=dt)
f = interpolate.interp1d(Time_OF,RtAeroMys)
RtAeroMys = f(Time)
RtAeroMys_LPF = low_pass_filter(RtAeroMys,0.3,dt=dt)
f = interpolate.interp1d(Time_OF,RtAeroMzs)
RtAeroMzs = f(Time)
RtAeroMzs_LPF = low_pass_filter(RtAeroMzs,0.3,dt=dt)


LSShftFys = np.array(df_OF.variables["LSShftFys"][Time_start_idx:])
LSShftFzs = np.array(df_OF.variables["LSShftFzs"][Time_start_idx:])
LSSTipMys = np.array(df_OF.variables["LSSTipMys"][Time_start_idx:])
LSSTipMzs = np.array(df_OF.variables["LSSTipMzs"][Time_start_idx:])

L1 = 1.912; L2 = 2.09
FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))

f = interpolate.interp1d(Time_OF,FBR)
FBR_interp = f(Time)

print(moments(FBR_interp[Times_high]))
print(moments(FBR_interp[Times_low]))
print(moments(FBR_interp[Times_int]))
print(moments(FBR_interp[Times_high_low]))
print(moments(FBR_interp[Times_high_int]))
print(moments(FBR_interp[Times_low_int]))

fig = plt.figure(figsize=(14,8))
plt.plot(Time,FBR_interp,"-k")
plt.plot(Time[Times_high],FBR_interp[Times_high],"or")
plt.plot(Time[Times_low],FBR_interp[Times_low],"ob")
plt.plot(Time[Times_int],FBR_interp[Times_int],"og")
plt.grid()
plt.tight_layout()

plt.show()

# plt.rcParams['font.size'] = 12
# frq,PSD = temporal_spectra(FBR,dt_OF,"FBR")
# plt.figure(figsize=(14,8))
# plt.loglog(I_frq,I_PSD,"-b")
# plt.loglog(frq,PSD,"-r")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD")
# plt.legend(["Magnitude Asymmetry vector [$m^4/s$]", "Magnitude Main bearing force vector [kN]"])
# plt.grid()
# plt.tight_layout()
# plt.savefig(in_dir+"Bearing_force_analysis/FBR_I_spectra.png")
# plt.close()

f = interpolate.interp1d(Time_OF,LSShftFys)
LSShftFys = f(Time)
LSShftFys_LPF = low_pass_filter(LSShftFys,0.3,dt=dt)
f = interpolate.interp1d(Time_OF,LSShftFzs)
LSShftFzs = f(Time)
LSShftFzs_LPF = low_pass_filter(LSShftFzs,0.3,dt=dt)
f = interpolate.interp1d(Time_OF,LSSTipMys)
LSSTipMys = f(Time)
LSSTipMys_LPF = low_pass_filter(LSSTipMys,0.3,dt=dt)
f = interpolate.interp1d(Time_OF,LSSTipMzs)
LSSTipMzs = f(Time)
LSSTipMzs_LPF = low_pass_filter(LSSTipMzs,0.3,dt=dt)


L1 = 1.912; L2 = 2.09; L = L1 + L2

Aero_FBMy_LPF = RtAeroMzs_LPF/L2; Aero_FBFy_LPF = -RtAeroFys_LPF*((L1+L2)/L2)
Aero_FBMz_LPF = -RtAeroMys_LPF/L2; Aero_FBFz_LPF = -RtAeroFzs_LPF*((L1+L2)/L2)

Aero_FBy_LPF = -(Aero_FBMy_LPF + Aero_FBFy_LPF); Aero_FBz_LPF = -(Aero_FBMz_LPF + Aero_FBFz_LPF)
Aero_FBR_LPF = np.sqrt(np.add(np.square(Aero_FBy_LPF),np.square(Aero_FBz_LPF)))
Aero_theta_LPF = np.degrees(np.arctan2(Aero_FBz_LPF,Aero_FBy_LPF))
Aero_theta_LPF = theta_360(Aero_theta_LPF)
Aero_theta_LPF = np.radians(np.array(Aero_theta_LPF))

MR_LPF = np.add(np.square(RtAeroMys_LPF/L2), np.square(RtAeroMzs_LPF/L2))
Theta_MR_LPF = np.degrees(np.arctan2(-RtAeroMys_LPF,RtAeroMzs_LPF))
Theta_MR_LPF = theta_360(Theta_MR_LPF)
Theta_MR_LPF = np.radians(np.array(Theta_MR_LPF))

Aero_FR_LPF = np.add(np.square(RtAeroFys_LPF * (L/L2)), np.square(RtAeroFzs_LPF * (L/L2)))
Theta_Aero_FR_LPF = np.degrees(np.arctan2(RtAeroFzs_LPF,RtAeroFys_LPF))
Theta_Aero_FR_LPF = theta_360(Theta_Aero_FR_LPF)
Theta_Aero_FR_LPF = np.radians(np.array(Theta_Aero_FR_LPF))


FBMy_LPF = LSSTipMzs_LPF/L2; FBFy_LPF = -LSShftFys_LPF*((L1+L2)/L2)
FBMz_LPF = -LSSTipMys_LPF/L2; FBFz_LPF = -LSShftFzs_LPF*((L1+L2)/L2)

FBy_LPF = -(FBMy_LPF + FBFy_LPF); FBz_LPF = -(FBMz_LPF + FBFz_LPF)
FBR_LPF = np.sqrt(np.add(np.square(FBy_LPF),np.square(FBz_LPF)))
Theta_FB_LPF = np.degrees(np.arctan2(FBz_LPF,FBy_LPF))
Theta_FB_LPF = theta_360(Theta_FB_LPF)
Theta_FB_LPF = np.radians(np.array(Theta_FB_LPF))


# time_shift = Time[0]+4.78; time_shift_idx = np.searchsorted(Time,time_shift)
# Time = Time[:-time_shift_idx]
# dt = Time[1]-Time[0]
# Time_steps = np.arange(0,len(Time))

# I = I[:-time_shift_idx]
# Theta = Theta[:-time_shift_idx]

# Aero_FBR = Aero_FBR_LPF[time_shift_idx:]
# Aero_theta = Aero_theta_LPF[time_shift_idx:]

# FBR = FBR_LPF[time_shift_idx:]
# Theta_FB = Theta_FB_LPF[time_shift_idx:]

# MR = MR_LPF[time_shift_idx:]
# Theta_MR = Theta_MR_LPF[time_shift_idx:]


def Update_Aero_vector(it):
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(Aero_theta[it], Aero_FBR[it]/np.max(Aero_FBR), c="k", s=20)
    d = ax.scatter(Theta[it],I[it]/np.max(I), c="b", s=20)
    f = ax.scatter(Theta_MR[it], MR[it]/np.max(MR), c="r", s=20)
    plt.legend(["Aerodynamic Main Bearing Force", "Asymmetry", "Modified Rotor Moment"],loc="lower right")
    ax.arrow(0, 0, Aero_theta[it], Aero_FBR[it]/np.max(Aero_FBR), length_includes_head=True, color="k")
    ax.arrow(0, 0, Theta[it], I[it]/np.max(I), length_includes_head=True, color="b")
    ax.arrow(0, 0, Theta_MR[it], MR[it]/np.max(MR), length_includes_head=True, color="r")
    ax.set_ylim([0,1])
    ax.set_title("Normalized vectors \nTime = {}s".format(Time[it]), va='bottom')
    T = Time[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T


def Update_vector(it):
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='polar')
    c = ax.scatter(Theta_FB[it], FBR[it]/np.max(FBR), c="k", s=20)
    d = ax.scatter(Theta[it],I[it]/np.max(I), c="b", s=20)
    f = ax.scatter(Theta_MR[it], MR[it]/np.max(MR), c="r", s=20)
    plt.legend(["Main Bearing Force", "Asymmetry", "Modified Rotor Moment"],loc="lower right")
    ax.arrow(0, 0, Theta_FB[it], FBR[it]/np.max(FBR), length_includes_head=True, color="k")
    ax.arrow(0, 0, Theta[it], I[it]/np.max(I), length_includes_head=True, color="b")
    ax.arrow(0, 0, Theta_MR[it], MR[it]/np.max(MR), length_includes_head=True, color="r")
    ax.set_ylim([0,1])
    ax.set_title("Normalized vectors \nTime = {}s".format(Time[it]), va='bottom')
    T = Time[it]
    plt.savefig(out_dir+"polar_plot_{}.png".format(Time_idx))
    plt.close(fig)

    return T


if update_polar_Aero_vectors == True:
    out_dir = in_dir+"Asymmetry_analysis/polar_Aero_vectors/"
    with Pool() as pool:
        for T in pool.imap(Update_Aero_vector,Time_steps):

            print(T)


if update_polar_vectors == True:
    out_dir = in_dir+"Asymmetry_analysis/polar_vectors/"
    with Pool() as pool:
        for T in pool.imap(Update_vector,Time_steps):

            print(T)


if update_pdf_plots == True:
    out_dir = in_dir+"Asymmetry_analysis/"
    with PdfPages(out_dir+'Eddy_analysis.pdf') as pdf:

        plt.rcParams['font.size'] = 14

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
        ax1.axhline(y=np.mean(A_high)+np.std(A_high),color="r")
        ax2.plot(Time,A_low)
        ax2.axhline(y=np.mean(A_low),linestyle="--",color="k")
        ax2.axhline(y=np.mean(A_low)+np.std(A_low),color="r")
        ax3.plot(Time,A_int)
        ax3.axhline(y=np.mean(A_int),linestyle="--",color="k")
        ax3.axhline(y=np.mean(A_int)+np.std(A_int),color="r")
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


        #trend and trend fluctuations area plotted separately
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

        ax1.plot(Time,A_high,"-r")
        ax1.plot(Time,A_high_trend,"--k")
        ax1.plot(Time,A_high_flucs,"-.k")
        ax2.plot(Time,A_low,"-b")
        ax2.plot(Time,A_low_trend,"--k")
        ax2.plot(Time,A_low_flucs,"-.k")
        ax3.plot(Time,A_int,"-g")
        ax3.plot(Time,A_int_trend,"--k")
        ax3.plot(Time,A_int_flucs,"-.k")
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


        #individual A high spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(A_high_frq,A_high_PSD)
        plt.ylabel("PSD High speed areas [$m^2$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual A low spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(A_low_frq,A_low_PSD)
        plt.ylabel("PSD Low speed areas [$m^2$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual A int spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(A_int_frq,A_int_PSD)
        plt.ylabel("PSD Intermediate speed areas [$m^2$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
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


        #Iy trends and fluctuations plotted sepeately
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

        ax1.plot(Time,Iy_high,"-r")
        ax1.plot(Time,Iy_high_trend,"--k")
        ax1.plot(Time,Iy_high_flucs,"-.k")
        ax2.plot(Time,Iy_low,"-b")
        ax2.plot(Time,Iy_low_trend,"--k")
        ax2.plot(Time,Iy_low_flucs,"-.k")
        ax3.plot(Time,Iy_int,"-g")
        ax3.plot(Time,Iy_int_trend,"--k")
        ax3.plot(Time,Iy_int_flucs,"-.k")
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

        #individual Iy high spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Iy_high_frq,Iy_high_PSD)
        plt.ylabel("PSD Asymmetry around y axis \nHigh speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual Iy low spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Iy_low_frq,Iy_low_PSD)
        plt.ylabel("PSD Asymmetry around y axis \nLow speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual Iy low spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Iy_int_frq,Iy_int_PSD)
        plt.ylabel("PSD Asymmetry around y axis \nIntermediate speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
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

                
                
        #Iz trends and fluctuations plotted sepeately
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

        ax1.plot(Time,Iz_high,"-r")
        ax1.plot(Time,Iz_high_trend,"--k")
        ax1.plot(Time,Iz_high_flucs,"-.k")
        ax2.plot(Time,Iz_low,"-b")
        ax2.plot(Time,Iz_low_trend,"--k")
        ax2.plot(Time,Iz_low_flucs,"-.k")
        ax3.plot(Time,Iz_int,"-g")
        ax3.plot(Time,Iz_int_trend,"--k")
        ax3.plot(Time,Iz_int_flucs,"-.k")
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

        #individual Iz high spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Iz_high_frq,Iz_high_PSD)
        plt.ylabel("PSD Asymmetry around z axis \nHigh speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual Iz low spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Iz_low_frq,Iz_low_PSD)
        plt.ylabel("PSD Asymmetry around z axis \nLow speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual Iz int spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Iz_int_frq,Iz_int_PSD)
        plt.ylabel("PSD Asymmetry around z axis \nIntermediate speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #magnitude asymmetry vector
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,I)
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude Asymmetry vector [$m^4/s$]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #I components contributions plotted
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,PI_high,"-r")
        plt.plot(Time,PI_low,"-b")
        plt.plot(Time,PI_int,"-g")
        plt.plot(Time,PI_high_low,"-m")
        plt.plot(Time,PI_high_int,"-y")
        plt.plot(Time,PI_low_int,"-c")
        plt.plot(Time,P_Tot_I,"-k")
        plt.ylim([-10,10])
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Contribution to Magnitude Asymmetry vector [-]")
        plt.legend(["high", "low", "int", "highlow", "highint", "lowint","Total"])
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        #I components plotted
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,I_high,"-r")
        plt.plot(Time,I_low,"-b")
        plt.plot(Time,I_int,"-g")
        plt.plot(Time,I_high_low,"-m")
        plt.plot(Time,I_high_int,"-y")
        plt.plot(Time,I_low_int,"-c")
        plt.grid()
        plt.xlabel("Time [s]")
        plt.ylabel("Contribution to Magnitude Asymmetry vector squared [$m^8/s^2$]")
        plt.legend(["high", "low", "int", "highlow", "highint", "lowint"])
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        #I plotted separately
        fig,(ax1,ax2,ax3,ax4,ax5,ax6) = plt.subplots(6,1,figsize=(14,8))

        ax1.plot(Time,I_high)
        ax2.plot(Time,I_low)
        ax3.plot(Time,I_int)
        ax4.plot(Time,I_high_low)
        ax5.plot(Time,I_high_int)
        ax6.plot(Time,I_low_int)
        ax1.set_title("Magnitude Asymmetry vector squared - high speed area [$m^4/s$]",fontsize=14)
        ax2.set_title("Magnitude Asymmetry vector squared - low speed area [$m^4/s$]",fontsize=14)
        ax3.set_title("Magnitude Asymmetry vector squared - intermediate speed area [$m^4/s$]",fontsize=14)
        ax4.set_title("Magnitude Asymmetry vector squared - highlow cross area [$m^4/s$]",fontsize=14)
        ax5.set_title("Magnitude Asymmetry vector squared - highint cross area [$m^4/s$]",fontsize=14)
        ax6.set_title("Magnitude Asymmetry vector squared - lowint cross area [$m^4/s$]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=16)
        plt.tight_layout()
        ax1.grid()
        ax2.grid()
        ax3.grid()
        ax4.grid()
        ax5.grid()
        ax6.grid()
        pdf.savefig()
        plt.close()

        #I trends and fluctuations plotted sepeately
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

        ax1.plot(Time,I_high,"-r")
        ax1.plot(Time,I_high_trend,"--k")
        ax1.plot(Time,I_high_flucs,"-.k")
        ax2.plot(Time,I_low,"-b")
        ax2.plot(Time,I_low_trend,"--k")
        ax2.plot(Time,I_low_flucs,"-.k")
        ax3.plot(Time,I_int,"-g")
        ax3.plot(Time,I_int_trend,"--k")
        ax3.plot(Time,I_int_flucs,"-.k")
        ax1.set_title("Magnitude Asymmetry vecotr - high speed area [$m^4/s$]",fontsize=14)
        ax2.set_title("Magnitude Asymmetry vector - low speed area [$m^4/s$]",fontsize=14)
        ax3.set_title("Magnitude Asymmetry vector - intermediate speed area [$m^4/s$]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=16)
        plt.tight_layout()
        ax1.grid()
        ax2.grid()
        ax3.grid()
        pdf.savefig()
        plt.close()


        #PDF I joint
        fig,ax = plt.subplots(figsize=(14,8))

        P,X = probability_dist(I_high)
        ax.plot(X,P,'-r')
        P,X = probability_dist(I_low)
        ax.plot(X,P,'-b')
        P,X = probability_dist(I_int)
        ax.plot(X,P,'-g')
        P,X = probability_dist(I)
        ax.plot(X,P, "-k")
        ax.set_ylabel("Probability [-]",fontsize=14)
        ax.set_xlabel("Magnitude Asymmetry vector [$m^4/s$]",fontsize=16)
        plt.legend(["High speed area", "Low speed area", "intermediate area","Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        #I spectra plotted joint
        fig,ax = plt.subplots(figsize=(14,8))

        ax.loglog(I_high_frq,I_high_PSD,'-r')
        ax.loglog(I_low_frq,I_low_PSD,"-b")
        ax.loglog(I_int_frq,I_int_PSD,"-g")
        ax.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
        ax.set_xlabel("Frequency [Hz]",fontsize=16)
        plt.legend(["High speed area", "Low speed area", "intermediate area"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        #individual I high spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(I_high_frq,I_high_PSD)
        plt.ylabel("PSD Magnitude Asymmetry vector \nHigh speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual I low spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(I_low_frq,I_low_PSD)
        plt.ylabel("PSD Magnitude Asymmetry vector \nLow speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual I int spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(I_int_frq,I_int_PSD)
        plt.ylabel("PSD Magnitude Asymmetry vector \nIntermediate speed areas [$m^4/s$]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()



        #Change in direction asymmetry vector
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time[1:],Delta_Theta)
        plt.xlabel("Time [s]")
        plt.ylabel("Change in the Direction of the Asymmetry vector [deg]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #Spectra change in direction asymmetry vector
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Theta_frq,Theta_PSD)
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("PSD Change in the Direction of the Asymmetry vector [deg]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        #change in direction plotted separately
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

        ax1.plot(Time[1:],Delta_Theta_high)
        ax1.axhline(y=np.mean(Delta_Theta_high),linestyle="--",color="k")
        ax1.axhline(y=np.mean(Delta_Theta_high)+np.std(Delta_Theta_high),color="r")
        ax2.plot(Time[1:],Delta_Theta_low)
        ax2.axhline(y=np.mean(Delta_Theta_low),linestyle="--",color="k")
        ax2.axhline(y=np.mean(Delta_Theta_low)+np.std(Delta_Theta_low),color="r")
        ax3.plot(Time[1:],Delta_Theta_int)
        ax3.axhline(y=np.mean(Delta_Theta_int),linestyle="--",color="k")
        ax3.axhline(y=np.mean(Delta_Theta_int)+np.std(Delta_Theta_int),color="r")
        ax1.set_title("Change in the Direction of the Asymmetry vector\n high speed area [deg]",fontsize=14)
        ax2.set_title("Change in the Direction of the Asymmetry vector\n low speed area [deg]",fontsize=14)
        ax3.set_title("Change in the Direction of the Asymmetry vector\n intermediate speed area [deg]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=16)
        plt.tight_layout()
        ax1.grid()
        ax2.grid()
        ax3.grid()
        pdf.savefig()
        plt.close()


        #PDF change in direction joint
        fig,ax = plt.subplots(figsize=(14,8))

        P,X = probability_dist(Delta_Theta_high)
        ax.plot(X,P,'-r')
        P,X = probability_dist(Delta_Theta_low)
        ax.plot(X,P,'-b')
        P,X = probability_dist(Delta_Theta_int)
        ax.plot(X,P,'-g')
        P,X = probability_dist(Delta_Theta)
        ax.plot(X,P,"-k")
        ax.set_ylabel("Probability [-]",fontsize=14)
        ax.set_xlabel("Change in Direction of the Asymmetry vector [deg]",fontsize=16)
        plt.legend(["High speed area", "Low speed area", "intermediate area","Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        #Direction spectra plotted joint
        fig,ax = plt.subplots(figsize=(14,8))

        ax.loglog(Theta_high_frq,Theta_high_PSD,'-r')
        ax.loglog(Theta_low_frq,Theta_low_PSD,"-b")
        ax.loglog(Theta_int_frq,Theta_int_PSD,"-g")
        ax.set_ylabel("Change in the Direction of the Asymmetry vector [deg]",fontsize=14)
        ax.set_xlabel("Frequency [Hz]",fontsize=16)
        plt.legend(["High speed area", "Low speed area", "intermediate area"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        #individual Delta Theta high spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Theta_high_frq,Theta_high_PSD)
        plt.ylabel("PSD Change in the Direction of the Asymmetry vector \nHigh speed areas [deg]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual Delta Theta low spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Theta_low_frq,Theta_low_PSD)
        plt.ylabel("PSD Change in the Direction of the Asymmetry vector \nLow speed areas [deg]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual Theta int spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Theta_int_frq,Theta_int_PSD)
        plt.ylabel("PSD Change in the Direction of the Asymmetry vector \nIntermediate speed areas [deg]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
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

        #individual Ux high spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Ux_high_frq,Ux_high_PSD)
        plt.ylabel("PSD Average velocity \nHigh speed areas [m/s]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual Ux low spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Ux_low_frq,Ux_low_PSD)
        plt.ylabel("PSD Average velocity \nLow speed areas [m/s]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #individual Ux int spectra
        fig = plt.figure(figsize=(14,8))
        plt.loglog(Ux_int_frq,Ux_int_PSD)
        plt.ylabel("PSD Average velocity \nIntermediate speed areas [m/s]")
        plt.xlabel("Frequency [Hz]")
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        #eddy contribution to Iy
        for alpha in [2,5,10]:
            fig = plt.figure(figsize=(14,8))
            plt.plot(Time_sampling,Iy,"-k")


            high = 0
            low = 0
            int = 0
            high_low = 0
            high_int = 0
            low_int = 0
            high_low_int = 0

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
                elif abs(Iy_high[it]) <= alpha*abs(Iy_low[it]) and abs(Iy_high[it]) > alpha*abs(Iy_int[it]) or abs(Iy_low[it]) <= alpha*abs(Iy_high[it]) and abs(Iy_low[it]) > alpha*abs(Iy_int[it]):
                    plt.plot(Time[it],Iy[it],"*m",markersize=4)
                    high_low+=dt
                elif abs(Iy_high[it]) <= alpha*abs(Iy_int[it]) and abs(Iy_high[it]) > alpha*abs(Iy_low[it]) or abs(Iy_int[it]) <= alpha*abs(Iy_high[it]) and abs(Iy_int[it]) > alpha*abs(Iy_low[it]):
                    plt.plot(Time[it],Iy[it],"vy",markersize=4)
                    high_int+=dt
                elif abs(Iy_low[it]) <= alpha*abs(Iy_int[it]) and abs(Iy_low[it]) > alpha*abs(Iy_high[it]) or abs(Iy_int[it]) <= alpha*abs(Iy_low[it]) and abs(Iy_int[it]) > alpha*abs(Iy_high[it]):
                    plt.plot(Time[it],Iy[it],">c",markersize=5)
                    low_int+=dt
                else:
                    plt.plot(Time[it],Iy[it],"<k",markersize=5)
                    high_low_int+=dt

            plt.xlabel("Time [s]")
            plt.ylabel("Asymmetry around y axis [$m^4/s$]")
            plt.title("Alpha = {}".format(alpha))
            plt.grid()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            t_check = high+low+int+high_low+high_int+low_int+high_low_int
            print(alpha)
            print("total time ",t_check)
            print("1. ",high)
            print("2. ",low)
            print("3. ",int)
            print("4. ",high_low)
            print("5. ",high_int)
            print("6. ",low_int)
            print("7. ",high_low_int)


        #eddy contribution to Iz
        for alpha in [2,5,10]:
            fig = plt.figure(figsize=(14,8))
            plt.plot(Time_sampling,Iz,"-k")
            high = 0
            low = 0
            int = 0
            high_low = 0
            high_int = 0
            low_int = 0
            high_low_int = 0

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
                elif abs(Iz_high[it]) <= alpha*abs(Iz_low[it]) and abs(Iz_high[it]) > alpha*abs(Iz_int[it]) or abs(Iz_low[it]) <= alpha*abs(Iz_high[it]) and abs(Iz_low[it]) > alpha*abs(Iz_int[it]):
                    plt.plot(Time[it],Iz[it],"*m",markersize=4)
                    high_low+=dt
                elif abs(Iz_high[it]) <= alpha*abs(Iz_int[it]) and abs(Iz_high[it]) > alpha*abs(Iz_low[it]) or abs(Iz_int[it]) <= alpha*abs(Iz_high[it]) and abs(Iz_int[it]) > alpha*abs(Iz_low[it]):
                    plt.plot(Time[it],Iz[it],"vy",markersize=5)
                    high_int+=dt
                elif abs(Iz_low[it]) <= alpha*abs(Iz_int[it]) and abs(Iz_low[it]) > alpha*abs(Iz_high[it]) or abs(Iz_int[it]) <= alpha*abs(Iz_low[it]) and abs(Iz_int[it]) > alpha*abs(Iz_high[it]):
                    plt.plot(Time[it],Iz[it],">c",markersize=5)
                    low_int+=dt
                else:
                    plt.plot(Time[it],Iz[it],"<k",markersize=4)
                    high_low_int+=dt

            plt.xlabel("Time [s]")
            plt.ylabel("Asymmetry around z axis [$m^4/s$]")
            plt.title("$Alpha = {}".format(alpha))
            plt.grid()
            plt.tight_layout()
            pdf.savefig()
            plt.close()

            t_check = high+low+int+high_low+high_int+low_int+high_low_int
            print(alpha)
            print("total time ",t_check)
            print("1. ",high)
            print("2. ",low)
            print("3. ",int)
            print("4. ",high_low)
            print("5. ",high_int)
            print("6. ",low_int)
            print("7. ",high_low_int)


        #delta I plotted separely
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

        ax1.plot(Time,Delta_I_high_low)
        ax1.axhline(y=np.mean(Delta_I_high_low),linestyle="--",color="k")
        ax2.plot(Time,Delta_I_high_int)
        ax2.axhline(y=np.mean(Delta_I_high_int),linestyle="--",color="k")
        ax3.plot(Time,Delta_I_low_int)
        ax3.axhline(y=np.mean(Delta_I_low_int),linestyle="--",color="k")
        ax1.set_title("$\Delta I$ - high-low speed area [$m^4/s$]",fontsize=14)
        ax2.set_title("$\Delta I$ - high-intermediate speed area [$m^4/s$]",fontsize=14)
        ax3.set_title("$\Delta I$ - low-intermediate speed area [$m^4/s$]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=16)
        plt.tight_layout()
        ax1.grid()
        ax2.grid()
        ax3.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        P,X = probability_dist(Delta_I_high_low)
        plt.plot(X,P,"-r",label="$\Delta I = I_{high}-I_{low}$ [$m^4/s$]")
        P,X = probability_dist(Delta_I_high_int)
        plt.plot(X,P,"-g",label="$\Delta I = I_{high}-I_{int}$ [$m^4/s$]")
        P,X = probability_dist(Delta_I_low_int)
        plt.plot(X,P,"-b",label="$\Delta I = I_{low}-I_{int}$ [$m^4/s$]")
        plt.ylabel("Probability [-]",fontsize=14)
        plt.xlabel("Difference in Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #Magnitude asymmetry and difference in magntidue of asymmetry
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(Time,I_high,"-r",label="$|I_{high}|$")
        ax1.plot(Time,I_low,"-b",label="$|I_{low}|$")
        ax1.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
        ax2.plot(Time,Delta_I_high_low,"-k")
        ax2.set_ylabel("$\Delta I = I_{high}-I_{low}$ [$m^4/s$]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=14)
        ax1.legend()
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #Magnitude asymmetry and difference in magntidue of asymmetry
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(Time,I_high,"-r",label="$|I_{high}|$")
        ax1.plot(Time,I_int,"-g",label="$|I_{int}|$")
        ax1.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
        ax2.plot(Time,Delta_I_high_int,"-k")
        ax2.set_ylabel("$\Delta I = I_{high}-I_{int}$ [$m^4/s$]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=14)
        ax1.legend()
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #Magnitude asymmetry and difference in magntidue of asymmetry
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(Time,I_low,"-b",label="$|I_{low}|$")
        ax1.plot(Time,I_int,"-g",label="$|I_{int}|$")
        ax1.set_ylabel("Magnitude Asymmetry vector [$m^4/s$]",fontsize=14)
        ax2.plot(Time,Delta_I_low_int,"-k")
        ax2.set_ylabel("$\Delta I = I_{low}-I_{int}$ [$m^4/s$]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=14)
        ax1.legend()
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #delta Theta plotted separely
        fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))

        ax1.plot(Time,Delta_Theta_high_low)
        ax1.axhline(y=np.mean(Delta_Theta_high_low),linestyle="--",color="k")
        ax2.plot(Time,Delta_Theta_high_int)
        ax2.axhline(y=np.mean(Delta_Theta_high_int),linestyle="--",color="k")
        ax3.plot(Time,Delta_Theta_low_int)
        ax3.axhline(y=np.mean(Delta_Theta_low_int),linestyle="--",color="k")
        ax1.set_title("$\Delta \Theta$ - high-low speed area [deg]",fontsize=14)
        ax2.set_title("$\Delta \Theta$ - high-intermediate speed area [deg]",fontsize=14)
        ax3.set_title("$\Delta \Theta$ - low-intermediate speed area [deg]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=16)
        plt.tight_layout()
        ax1.grid()
        ax2.grid()
        ax3.grid()
        pdf.savefig()
        plt.close()


        #PDF Delta theta joint
        fig,ax = plt.subplots(figsize=(14,8))

        P,X = probability_dist(Delta_Theta_high_low)
        ax.plot(X,P,'-r')
        P,X = probability_dist(Delta_Theta_high_int)
        ax.plot(X,P,'-b')
        P,X = probability_dist(Delta_Theta_low_int)
        ax.plot(X,P,'-g')
        ax.set_ylabel("Probability [-]",fontsize=14)
        ax.set_xlabel("$\Delta \Theta$ [deg]",fontsize=16)
        plt.legend(["High-Low speed area", "High-Int speed area", "Low-Int area"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        #Direction asymmetry and difference in magntidue of asymmetry
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(Time,Theta_high,"-r",label="$\\theta_{high}$")
        ax1.plot(Time,Theta_low,"-b",label="$\\theta_{low}$")
        ax1.set_ylabel("Direction Asymmetry vector [$m^4/s$]",fontsize=14)
        ax2.plot(Time,Delta_Theta_high_low,"-k")
        ax2.set_ylabel("$\Delta \\theta = \\theta_{high}-\\theta_{low}$ [deg]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=14)
        ax1.legend()
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #Magnitude asymmetry and difference in magntidue of asymmetry
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(Time,Theta_high,"-r",label="$\\theta_{high}$")
        ax1.plot(Time,Theta_int,"-g",label="$\\theta_{int}$")
        ax1.set_ylabel("Direction Asymmetry vector [deg]",fontsize=14)
        ax2.plot(Time,Delta_Theta_high_int,"-k")
        ax2.set_ylabel("$\Delta \\theta = \\theta_{high}-\\theta_{int}$ [deg]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=14)
        ax1.legend()
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()

        #Magnitude asymmetry and difference in magntidue of asymmetry
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(Time,Theta_low,"-b",label="$\\theta_{low}$")
        ax1.plot(Time,Theta_int,"-g",label="$\\theta_{int}$")
        ax1.set_ylabel("Direction Asymmetry vector [deg]",fontsize=14)
        ax2.plot(Time,Delta_Theta_low_int,"-k")
        ax2.set_ylabel("$\Delta \\theta = \\theta_{low}-\\theta_{int}$ [deg]",fontsize=14)
        fig.supxlabel("Time [s]",fontsize=14)
        ax1.legend()
        ax1.grid()
        ax2.grid()
        plt.tight_layout()
        pdf.savefig()
        plt.close()


        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,I_high,"-r")
        plt.plot(Time,I,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude Asymmetry [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(I_high,I),4)))
        plt.legend(["High speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,I_low,"-b")
        plt.plot(Time,I,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude Asymmetry [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(I_low,I),4)))
        plt.legend(["Low speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,I_int,"-g")
        plt.plot(Time,I,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude Asymmetry [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(I_int,I),4)))
        plt.legend(["Intermediate speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iy_high,"-r")
        plt.plot(Time,Iy,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around y axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iy_high,Iy),4)))
        plt.legend(["High speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iy_low,"-b")
        plt.plot(Time,Iy,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around y axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iy_low,Iy),4)))
        plt.legend(["Low speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iy_int,"-g")
        plt.plot(Time,Iy,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around y axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iy_int,Iy),4)))
        plt.legend(["Intermediate speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iz_high,"-r")
        plt.plot(Time,Iz,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around z axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iz_high,Iz),4)))
        plt.legend(["High speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iz_low,"-b")
        plt.plot(Time,Iz,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around z axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iz_low,Iz),4)))
        plt.legend(["Low speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iz_int,"-g")
        plt.plot(Time,Iz,"-k")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around z axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iz_int,Iz),4)))
        plt.legend(["Intermediate speed regions", "Total"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,A_high,"-r")
        plt.plot(Time,A_low,"-b")
        plt.xlabel("Time [s]")
        plt.ylabel("Area [$m^2$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(A_high,A_low),4)))
        plt.legend(["High speed regions", "low speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,A_high,"-r")
        plt.plot(Time,A_int,"-g")
        plt.xlabel("Time [s]")
        plt.ylabel("Area [$m^2$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(A_high,A_int),4)))
        plt.legend(["High speed regions", "Intermediate speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,A_int,"-g")
        plt.plot(Time,A_low,"-b")
        plt.xlabel("Time [s]")
        plt.ylabel("Area [$m^2$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(A_int,A_low),4)))
        plt.legend(["Intermediate speed regions", "low speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iy_high,"-r")
        plt.plot(Time,Iy_low,"-b")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around y axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iy_high,Iy_low),4)))
        plt.legend(["High speed regions", "low speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iy_high,"-r")
        plt.plot(Time,Iy_int,"-g")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around y axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iy_high,Iy_int),4)))
        plt.legend(["High speed regions", "Intermediate speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iy_int,"-g")
        plt.plot(Time,Iy_low,"-b")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around y axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iy_int,Iy_low),4)))
        plt.legend(["Intermediate speed regions", "low speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iz_high,"-r")
        plt.plot(Time,Iz_low,"-b")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around z axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iz_high,Iz_low),4)))
        plt.legend(["High speed regions", "low speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iz_high,"-r")
        plt.plot(Time,Iz_int,"-g")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around z axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iz_high,Iz_int),4)))
        plt.legend(["High speed regions", "Intermediate speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Iz_int,"-g")
        plt.plot(Time,Iz_low,"-b")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around z axis [$m^4/s$]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Iz_int,Iz_low),4)))
        plt.legend(["Intermediate speed regions", "low speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()


        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Theta_high,"-r")
        plt.plot(Time,Theta_low,"-b")
        plt.xlabel("Time [s]")
        plt.ylabel("Direction of asymmetry vector [rads]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Theta_high,Theta_low),4)))
        plt.legend(["High speed regions", "low speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Theta_high,"-r")
        plt.plot(Time,Theta_int,"-g")
        plt.xlabel("Time [s]")
        plt.ylabel("Direction of asymmetry vector [rads]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Theta_high,Theta_int),4)))
        plt.legend(["High speed regions", "Intermediate speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,Theta_int,"-g")
        plt.plot(Time,Theta_low,"-b")
        plt.xlabel("Time [s]")
        plt.ylabel("Direction of asymmetry vector [rads]")
        plt.title("correlation coefficient = {}".format(round(correlation_coef(Theta_int,Theta_low),4)))
        plt.legend(["Intermediate speed regions", "low speed regions"])
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()


