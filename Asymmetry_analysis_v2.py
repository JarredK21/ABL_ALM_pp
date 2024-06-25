import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from scipy import interpolate
from matplotlib.backends.backend_pdf import PdfPages
from scipy.signal import butter,filtfilt
from multiprocessing import Pool


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


def probability_dist_Area(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    no_bin = 1000
    X = np.linspace(0,1.0,no_bin)
    dX = X[1] - X[0]
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    print(np.sum(P)*dX)
    return P,X


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


def Update(it):

    out_dir = in_dir+"Asymmetry_analysis/Asymmetry/"

    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)

    fig,(ax1,ax2,ax3) = plt.subplots(3,1,figsize=(14,8))
    ax1.plot(Time[:it],I[:it],"-k")
    ax2.plot(Time[:it],Iy[:it],"-k")
    ax3.plot(Time[:it],Iz[:it],"-k")
    ax3.set_ylabel("Magnitude Asymmetry vector")
    ax2.set_ylabel("Asymmetry around y axis")
    ax1.set_ylabel("Asymmetry around z axis")
    fig.supxlabel("Time [s]")
    fig.suptitle("A_high = {}, A_low = {}, A_int = {}\nTime = {}s".format(round(Frac_high_area[it],2),round(Frac_low_area[it],2),round(Frac_int_area[it],2),round(Time[it],4)))
    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax1.set_xlim([np.min(Time),np.max(Time)])
    ax2.set_xlim([np.min(Time),np.max(Time)])
    ax3.set_xlim([np.min(Time),np.max(Time)])
    ax1.set_ylim([np.min(I),np.max(I)])
    ax2.set_ylim([np.min(Iy),np.max(Iy)])
    ax3.set_ylim([np.min(Iz),np.max(Iz)])
    plt.tight_layout()
    plt.savefig(out_dir+"I_{}.png".format(Time_idx))
    plt.close()

    return Time_idx



in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

out_dir = in_dir + "Asymmetry_analysis/"

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
A_low = np.array(a.variables["Area_low"][Time_start_idx:])
A_int = np.array(a.variables["Area_int"][Time_start_idx:])

Iy = np.array(a.variables["Iy"][Time_start_idx:])
Iz = -np.array(a.variables["Iz"][Time_start_idx:])

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

A_rot = np.pi*63**2

Frac_high_area = np.true_divide(A_high,A_rot)
Frac_low_area = np.true_divide(A_low,A_rot)
Frac_int_area = np.true_divide(A_int,A_rot)


# with Pool() as pool:
#     for T in pool.imap(Update,Time_steps):

#         print(T)


T_high = []; T_low =[]; T_int = []
for it in np.arange(0,len(Time)):

    if Frac_high_area[it] >= 0.80:
        T_high.append(it)
    elif Frac_low_area[it] >= 0.80:
        T_low.append(it)
    elif Frac_int_area[it] >= 0.80:
        T_int.append(it)

T_high_low = []; T_high_int = []; T_low_int = []
for it in np.arange(0,len(Time)):

    if Frac_int_area[it] < 0.1 and Frac_high_area[it] > 0.2 and Frac_low_area[it] > 0.2:
        T_high_low.append(it)
    elif Frac_low_area[it] < 0.1 and Frac_high_area[it] > 0.2 and Frac_int_area[it] > 0.2:
        T_high_int.append(it)
    elif Frac_high_area[it] < 0.1 and Frac_low_area[it] > 0.2 and Frac_int_area[it] > 0.2:
        T_low_int.append(it)

I_high = I[T_high]
I_low = I[T_low]
I_int = I[T_int]

Iy_high = Iy[T_high]
Iy_low = Iy[T_low]
Iy_int = Iy[T_int]

Iz_high = Iz[T_high]
Iz_low = Iz[T_low]
Iz_int = Iz[T_int]

I_high_low = I[T_high_low]
I_high_int = I[T_high_int]
I_low_int = I[T_low_int]

Iy_high_low = Iy[T_high_low]
Iy_high_int = Iy[T_high_int]
Iy_low_int = Iy[T_low_int]

Iz_high_low = Iz[T_high_low]
Iz_high_int = Iz[T_high_int]
Iz_low_int = Iz[T_low_int]


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

out_dir = in_dir + "Asymmetry_analysis/"

plt.rcParams['font.size'] = 14

fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(I_high)
ax.plot(X,P,"r")
P,X = probability_dist(I_low)
ax.plot(X,P,"b")
P,X = probability_dist(I_int)
ax.plot(X,P,"g")
ax.set_title("Experiment 1\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Magnitude Asymmetry vector [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex1_PDF_I.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(I_high,dt,Var="I_high")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(I_low,dt,Var="I_low")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(I_int,dt,Var="I_int")
ax.loglog(frq,PSD,"g")
ax.set_title("Experiment 1\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Magnitude Asymmetry Vector [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex1_spectra_I.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(Iy_high)
ax.plot(X,P,"r")
P,X = probability_dist(Iy_low)
ax.plot(X,P,"b")
P,X = probability_dist(Iy_int)
ax.plot(X,P,"g")
ax.set_title("Experiment 1\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Asymmetry around y axis [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex1_PDF_Iy.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(Iy_high,dt,Var="Iy_high")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(Iy_low,dt,Var="Iy_low")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(Iy_int,dt,Var="Iy_int")
ax.loglog(frq,PSD,"g")
ax.set_title("Experiment 1\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Asymmetry around y axis [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex1_spectra_Iy.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(Iz_high)
ax.plot(X,P,"r")
P,X = probability_dist(Iz_low)
ax.plot(X,P,"b")
P,X = probability_dist(Iz_int)
ax.plot(X,P,"g")
ax.set_title("Experiment 1\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Asymmetry around z axis [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex1_PDF_Iz.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(Iz_high,dt,Var="Iz_high")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(Iz_low,dt,Var="Iz_low")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(Iz_int,dt,Var="Iz_int")
ax.loglog(frq,PSD,"g")
ax.set_title("Experiment 1\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Asymmetry around z axis [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex1_spectra_Iz.png")
plt.close()


in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

out_dir = in_dir + "Asymmetry_analysis/"

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
A_low = np.array(a.variables["Area_low"][Time_start_idx:])
A_int = np.array(a.variables["Area_int"][Time_start_idx:])

Iy_tot = np.concatenate((Iy,np.array(a.variables["Iy"][Time_start_idx:])))
Iz_tot = np.concatenate((Iz,np.array(a.variables["Iz"][Time_start_idx:])))

Iy = np.array(a.variables["Iy"][Time_start_idx:])
Iz = -np.array(a.variables["Iz"][Time_start_idx:])

I_tot = np.concatenate((I,np.sqrt(np.add(np.square(Iy),np.square(Iz)))))

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

A_rot = np.pi*63**2

Frac_high_area_tot = np.concatenate((Frac_high_area,np.true_divide(A_high,A_rot)))
Frac_low_area_tot = np.concatenate((Frac_low_area,np.true_divide(A_low,A_rot)))
Frac_int_area_tot = np.concatenate((Frac_int_area,np.true_divide(A_int,A_rot)))

Frac_high_area = np.true_divide(A_high,A_rot)
Frac_low_area = np.true_divide(A_low,A_rot)
Frac_int_area = np.true_divide(A_int,A_rot)


# with Pool() as pool:
#     for T in pool.imap(Update,Time_steps):

#         print(T)

T_high = []; T_low =[]; T_int = []
for it in np.arange(0,len(Time)):

    if Frac_high_area[it] >= 0.80:
        T_high.append(it)
    elif Frac_low_area[it] >= 0.80:
        T_low.append(it)
    elif Frac_int_area[it] >= 0.80:
        T_int.append(it)

T_high_low = []; T_high_int = []; T_low_int = []
for it in np.arange(0,len(Time)):

    if Frac_int_area[it] < 0.1 and Frac_high_area[it] > 0.2 and Frac_low_area[it] > 0.2:
        T_high_low.append(it)
    elif Frac_low_area[it] < 0.1 and Frac_high_area[it] > 0.2 and Frac_int_area[it] > 0.2:
        T_high_int.append(it)
    elif Frac_high_area[it] < 0.1 and Frac_low_area[it] > 0.2 and Frac_int_area[it] > 0.2:
        T_low_int.append(it)

I_high_low_tot = np.concatenate((I_high_low,I[T_high_low]))
I_high_int_tot = np.concatenate((I_high_int,I[T_high_int]))
I_low_int_tot = np.concatenate((I_low_int,I[T_low_int]))

Iy_high_low_tot = np.concatenate((Iy_high_low,Iy[T_high_low]))
Iy_high_int_tot = np.concatenate((Iy_high_int,Iy[T_high_int]))
Iy_low_int_tot = np.concatenate((Iy_low_int,Iy[T_low_int]))

Iz_high_low_tot = np.concatenate((Iz_high_low,Iz[T_high_low]))
Iz_high_int_tot = np.concatenate((Iz_high_int,Iz[T_high_int]))
Iz_low_int_tot = np.concatenate((Iz_low_int,Iz[T_low_int]))

I_high_tot = np.concatenate((I_high,I[T_high]))
I_low_tot = np.concatenate((I_low,I[T_low]))
I_int_tot = np.concatenate((I_int,I[T_int]))

Iy_high_tot = np.concatenate((Iy_high,Iy[T_high]))
Iy_low_tot = np.concatenate((Iy_low,Iy[T_low]))
Iy_int_tot = np.concatenate((Iy_int,Iy[T_int]))

Iz_high_tot = np.concatenate((Iz_high,Iz[T_high]))
Iz_low_tot = np.concatenate((Iz_low,Iz[T_low]))
Iz_int_tot = np.concatenate((Iz_int,Iz[T_int]))


fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(I_high_tot)
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("I_high_tot",P_neg*dX)

ax.plot(X,P,"r")
P,X = probability_dist(I_low_tot)
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("I_low_tot",P_neg*dX)
ax.plot(X,P,"b")
P,X = probability_dist(I_int_tot)
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("I_int_tot",P_neg*dX)
ax.plot(X,P,"g")
ax.set_title("Time high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(I_high_tot)*dt,2),round(len(I_low_tot)*dt,2), round(len(I_int_tot)*dt,2)))
ax.set_xlabel("Magnitude Asymmetry vector [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_PDF_I.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(I_high_tot,dt,Var="I_high_tot")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(I_low_tot,dt,Var="I_low_tot")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(I_int_tot,dt,Var="I_int_tot")
ax.loglog(frq,PSD,"g")
ax.set_title("Time high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(I_high_tot)*dt,2),round(len(I_low_tot)*dt,2), round(len(I_int_tot)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Magnitude Asymmetry Vector [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_spectra_I.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(Iy_high_tot)
ax.plot(X,P,"r")
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("Iy_high_tot",P_neg*dX)
P,X = probability_dist(Iy_low_tot)
ax.plot(X,P,"b")
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("Iy_low_tot",P_neg*dX)
P,X = probability_dist(Iy_int_tot)
ax.plot(X,P,"g")
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("Iy_int_tot",P_neg*dX)
ax.set_title("Time high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(I_high_tot)*dt,2),round(len(I_low_tot)*dt,2), round(len(I_int_tot)*dt,2)))
ax.set_xlabel("Asymmetry around y axis [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_PDF_Iy.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(Iy_high_tot,dt,Var="Iy_high_tot")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(Iy_low_tot,dt,Var="Iy_low_tot")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(Iy_int_tot,dt,Var="Iy_int_tot")
ax.loglog(frq,PSD,"g")
ax.set_title("Time high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(I_high_tot)*dt,2),round(len(I_low_tot)*dt,2), round(len(I_int_tot)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Asymmetry around y axis [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_spectra_Iy.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(Iz_high_tot)
ax.plot(X,P,"r")
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("Iz_high_tot",P_neg*dX)
P,X = probability_dist(Iz_low_tot)
ax.plot(X,P,"b")
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("Iz_low_tot",P_neg*dX)
P,X = probability_dist(Iz_int_tot)
ax.plot(X,P,"g")
P_neg = 0
dX = X[1] - X[0]
for i in np.arange(0,len(X)):
    if X[i] < 0:
        P_neg+=P[i]
print("Iz_int_tot",P_neg*dX)
ax.set_title("Time high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(I_high_tot)*dt,2),round(len(I_low_tot)*dt,2), round(len(I_int_tot)*dt,2)))
ax.set_xlabel("Asymmetry around z axis [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_PDF_Iz.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(Iz_high_tot,dt,Var="Iz_high_tot")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(Iz_low_tot,dt,Var="Iz_low_tot")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(Iz_int_tot,dt,Var="Iz_int_tot")
ax.loglog(frq,PSD,"g")
ax.set_title("Time high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(I_high_tot)*dt,2),round(len(I_low_tot)*dt,2), round(len(I_int_tot)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Asymmetry around z axis [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_spectra_Iz.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(I_high_low_tot)
ax.plot(X,P,"m")
P,X = probability_dist(I_high_int_tot)
ax.plot(X,P,"y")
P,X = probability_dist(I_low_int_tot)
ax.plot(X,P,"c")
ax.set_title("Time high_low = {0}, Time high_int = {1}, Time low_int = {2}".format(round(len(I_high_low_tot)*dt,2),round(len(I_high_int_tot)*dt,2), round(len(I_low_int_tot)*dt,2)))
ax.set_xlabel("Magnitude Asymmetry vector [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high_low", "Area_high_int", "Area_low_int"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_PDF_I_mix.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(Iy_high_low_tot)
ax.plot(X,P,"m")
P,X = probability_dist(Iy_high_int_tot)
ax.plot(X,P,"y")
P,X = probability_dist(Iy_low_int_tot)
ax.plot(X,P,"c")
ax.set_title("Time high_low = {0}, Time high_int = {1}, Time low_int = {2}".format(round(len(I_high_low_tot)*dt,2),round(len(I_high_int_tot)*dt,2), round(len(I_low_int_tot)*dt,2)))
ax.set_xlabel("Asymmetry around y axis [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high_low", "Area_high_int", "Area_low_int"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_PDF_Iy_mix.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(Iz_high_low_tot)
ax.plot(X,P,"m")
P,X = probability_dist(Iz_high_int_tot)
ax.plot(X,P,"y")
P,X = probability_dist(Iz_low_int_tot)
ax.plot(X,P,"c")
ax.set_title("Time high_low = {0}, Time high_int = {1}, Time low_int = {2}".format(round(len(I_high_low_tot)*dt,2),round(len(I_high_int_tot)*dt,2), round(len(I_low_int_tot)*dt,2)))
ax.set_xlabel("Asymmetry around z axis [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high_low", "Area_high_int", "Area_low_int"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Tot_PDF_Iz_mix.png")
plt.close()





I_high = I[T_high]
I_low = I[T_low]
I_int = I[T_int]

Iy_high = Iy[T_high]
Iy_low = Iy[T_low]
Iy_int = Iy[T_int]

Iz_high = Iz[T_high]
Iz_low = Iz[T_low]
Iz_int = Iz[T_int]

plt.rcParams['font.size'] = 14

fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(I_high)
ax.plot(X,P,"r")
P,X = probability_dist(I_low)
ax.plot(X,P,"b")
P,X = probability_dist(I_int)
ax.plot(X,P,"g")
ax.set_title("Experiment 2\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Magnitude Asymmetry vector [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex2_PDF_I.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(I_high,dt,Var="I_high")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(I_low,dt,Var="I_low")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(I_int,dt,Var="I_int")
ax.loglog(frq,PSD,"g")
ax.set_title("Experiment 2\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Magnitude Asymmetry Vector [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex2_spectra_I.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(Iy_high)
ax.plot(X,P,"r")
P,X = probability_dist(Iy_low)
ax.plot(X,P,"b")
P,X = probability_dist(Iy_int)
ax.plot(X,P,"g")
ax.set_title("Experiment 2\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Asymmetry around y axis [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex2_PDF_Iy.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(Iy_high,dt,Var="Iy_high")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(Iy_low,dt,Var="Iy_low")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(Iy_int,dt,Var="Iy_int")
ax.loglog(frq,PSD,"g")
ax.set_title("Experiment 2\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Asymmetry around y axis [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex2_spectra_Iy.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))
P,X = probability_dist(Iz_high)
ax.plot(X,P,"r")
P,X = probability_dist(Iz_low)
ax.plot(X,P,"b")
P,X = probability_dist(Iz_int)
ax.plot(X,P,"g")
ax.set_title("Experiment 2\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Asymmetry around z axis [$m^4/s$]")
ax.set_ylabel("probability [-]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex2_PDF_Iz.png")
plt.close()

fig,ax = plt.subplots(figsize=(14,8))
frq,PSD = temporal_spectra(Iz_high,dt,Var="Iz_high")
ax.loglog(frq,PSD,"r")
frq,PSD = temporal_spectra(Iz_low,dt,Var="Iz_low")
ax.loglog(frq,PSD,"b")
frq,PSD = temporal_spectra(Iz_int,dt,Var="Iz_int")
ax.loglog(frq,PSD,"g")
ax.set_title("Experiment 2\nTime high>0.80% = {0}, Time low>0.80% = {1}, Time int>0.80% = {2}".format(round(len(T_high)*dt,2),round(len(T_low)*dt,2), round(len(T_int)*dt,2)))
ax.set_xlabel("Frequency [Hz]")
ax.set_ylabel("PSD - Magnitude Asymmetry Vector [$m^4/s$]")
plt.legend(["Area_high >80%", "Area_low >80%", "Area_int >80%"])
ax.grid()
plt.tight_layout()
plt.savefig(out_dir+"Ex2_spectra_Iz.png")
plt.close()

