import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset


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
Time = np.array(a.variables["time_sampling"])
dt = Time[1] - Time[0]
group = a.groups["63.0"]
Iy_data = np.array(group.variables["Iy"])
Iz_data = np.array(group.variables["Iz"])

Time_steps = np.arange(0,3077)

Frac_pos_area = []
Frac_neg_area = []
ux_neg = []
ux_pos = []
Iy_high = []
Iy_low = []
Iz_high = []
Iz_low = []
Iy = []
Iz = []

area_g_rot = []
for it in Time_steps:
    filename = "csv_files/Eddies_0.7_{}.csv".format(it)

    df = pd.read_csv(in_dir+filename)

    # df["Centroid_x_pos"].dropna()
    # index = df["Ux_avg_pos"].isin([0.000000])
    # for i in np.arange(0,len(index.values)):
    #     if index.values[i] == True:
    #         df["Centroid_x_pos"].drop([i])


    Centroids_x_pos = np.array(df["Centroid_x_pos"].dropna())
    Centroids_y_pos = np.array(df["Centroid_y_pos"].dropna())
    Area_pos = np.array(df["Area_pos"].dropna())
    Ux_avg_pos = np.array(df["Ux_avg_pos"].dropna())

    Centroids_x_neg = np.array(df["Centroid_x_neg"].dropna())
    Centroids_y_neg = np.array(df["Centroid_y_neg"].dropna())
    Area_neg = np.array(df["Area_neg"].dropna())
    Ux_avg_neg = np.array(df["Ux_avg_neg"].dropna())

    Rot_Area = np.pi * 63**2

    Frac_pos_area.append(np.sum(Area_pos)/Rot_Area)
    Frac_neg_area.append(np.sum(Area_neg)/Rot_Area)  


    ux = 0
    Tot_area = np.sum(Area_pos)
    for i in np.arange(0,len(Area_pos)):
        frac_area = Area_pos[i]/Tot_area
        ux+=Ux_avg_pos[i]*frac_area
    
    ux_pos.append(ux)

    ux = 0
    Tot_area = np.sum(Area_neg)
    for i in np.arange(0,len(Area_neg)):
        frac_area = Area_neg[i]/Tot_area
        ux+=Ux_avg_neg[i]*frac_area
    
    ux_neg.append(ux)

    y = np.concatenate((np.subtract(Centroids_x_pos,2560),np.subtract(Centroids_x_neg,2560)))
    z = np.concatenate((np.subtract(Centroids_y_pos,90),np.subtract(Centroids_y_neg,90)))
    dA = np.concatenate((Area_pos,Area_neg))
    Ux_avg = np.concatenate((Ux_avg_pos,Ux_avg_neg))

    num = np.multiply(Ux_avg,(np.multiply(y,dA)))
    Iz.append(np.sum(num)/Rot_Area)
    num = np.multiply(Ux_avg,(np.multiply(z,dA)))
    Iy.append(np.sum(num)/Rot_Area)

    num = np.multiply(Ux_avg_neg,(np.multiply(np.subtract(Centroids_x_neg,2560),Area_neg)))
    Iz_low.append((np.sum(num)/Rot_Area)/Iz[it])

    num = np.multiply(Ux_avg_pos,(np.multiply(np.subtract(Centroids_x_pos,2560),Area_pos)))
    Iz_high.append((np.sum(num)/Rot_Area)/Iz[it])

    num = np.multiply(Ux_avg_neg,(np.multiply(np.subtract(Centroids_y_neg,90),Area_neg)))
    Iy_low.append((np.sum(num)/Rot_Area)/Iy[it])

    num = np.multiply(Ux_avg_pos,(np.multiply(np.subtract(Centroids_y_pos,90),Area_pos)))
    Iy_high.append((np.sum(num)/Rot_Area)/Iy[it])





fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time,Frac_pos_area,'-b')
ax.set_ylabel("Fraction of rotor disk covered by contours [-]",fontsize=14)
ax.plot(Time,Frac_neg_area,"-r")
ax.set_xlabel("Time [s]",fontsize=16)
ax.legend(["High speed areas (ux'>0.7m/s)","low speed areas (ux'<-0.7 m/s)"])
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"csv_files/plots/Area.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time,ux_pos,'-b')
ax.set_ylabel("Average velocity of contours weighted by area fraction [m/s]",fontsize=14)
ax.plot(Time,ux_neg,"-r")
ax.set_xlabel("Time [s]",fontsize=16)
ax.legend(["High speed areas (ux'>0.7m/s)","low speed areas (ux'<-0.7 m/s)"])
ax.axhline(y=np.mean(ux_pos),linestyle="--",color="k")
ax.axhline(y=np.mean(ux_neg),linestyle="--",color="k")
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"csv_files/plots/velocity.png")
plt.close()

Iy_corr = correlation_coef(Iy_data,Iy)
Iz_corr = correlation_coef(Iz,Iz_data)

fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time,Iy,'-b')
ax.axhline(y=np.mean(Iy),linestyle="--",color="b")
ax.set_ylabel("Asymmetry around y axis [m2/s]",fontsize=14)
ax2 = ax.twinx()
ax2.plot(Time,Iy_data,"-r")
ax2.axhline(y=np.mean(Iy_data),linestyle="--",color="r")
ax2.set_ylabel("Asymmetry around y axis [m4/s]",fontsize=14)
plt.title("Correlation coefficient {}".format(Iy_corr),fontsize=16)
ax.set_xlabel("Time [s]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"csv_files/plots/Iy.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time,Iz,'-b')
ax.axhline(y=np.mean(Iz),linestyle="--",color="b")
ax.set_ylabel("Asymmetry around z axis [m2/s]",fontsize=14)
ax2 = ax.twinx()
ax2.plot(Time,Iz_data,"-r")
ax2.axhline(y=np.mean(Iz_data),linestyle="--",color="r")
ax2.set_ylabel("Asymmetry around z axis [m4/s]",fontsize=14)
plt.title("Correlation coefficient {}".format(Iz_corr),fontsize=16)
ax.set_xlabel("Time [s]",fontsize=16)
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"csv_files/plots/Iz.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time,Iy_low,'-b')
ax.set_ylabel("Proportion of Asymmetry around y axis [m2/s]",fontsize=14)
ax.plot(Time,Iy_high,"-r")
ax.set_xlabel("Time [s]",fontsize=16)
ax.legend(["Low speed areas", "high speed areas"])
plt.ylim([-1,1])
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"csv_files/plots/Iy_prop.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time,Iz_low,'-b')
ax.set_ylabel("Proportion of Asymmetry around z axis [m2/s]",fontsize=14)
ax.plot(Time,Iz_high,"-r")
ax.set_xlabel("Time [s]",fontsize=16)
ax.legend(["Low speed areas", "high speed areas"])
plt.ylim([-1,1])
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"csv_files/plots/Iz_prop.png")
plt.close()