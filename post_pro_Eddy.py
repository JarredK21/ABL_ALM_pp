import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")
Time = np.array(a.variables["time_sampling"])

Time_steps = np.arange(0,3077)

Frac_pos_area = []
Frac_neg_area = []
ux_neg = []
ux_pos = []
Iy_pos = []
Iy_neg = []
Iz_pos = []
Iz_neg = []

area_g_rot = []
for it in Time_steps:
    filename = "csv_files/Eddies_0.7_{}.csv".format(it)

    df = pd.read_csv(in_dir+filename)

    df["Centroid_x_pos"].dropna()
    index = df["Ux_avg_pos"].isin([0.000000])
    for i in np.arange(0,len(index.values)):
        if index.values[i] == True:
            df["Centroid_x_pos"].drop([i])


    Centroids_x_pos = np.array(df["Centroid_x_pos"])
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

    if Frac_neg_area[it] > 1.0:
        area_g_rot.append(["neg"])
    elif Frac_pos_area[it] > 1.0:
        area_g_rot.append(["pos"])
    elif Frac_neg_area[it]+Frac_pos_area[it] > 1.0:
        area_g_rot.append(["both"])

    if area_g_rot[it] == "pos":
        Area_pos_sorted = np.sort(Area_pos)
        second_largest = Area_pos_sorted[-2]
        idx = np.where(Area_pos == second_largest)[0]
        np.delete(Area_pos,idx)
        np.delete(Centroids_x_pos,idx)
        np.delete(Centroids_y_pos,idx)
        np.delete(Ux_avg_pos,idx)

        Frac_pos_area[it] = (np.sum(Area_pos)/Rot_Area)
    elif area_g_rot[it] == "neg":
        Area_neg_sorted = np.sort(Area_neg)
        second_largest = Area_neg_sorted[-2]
        idx = np.where(Area_neg == second_largest)[0]
        np.delete(Area_neg,idx)
        np.delete(Centroids_x_neg,idx)
        np.delete(Centroids_y_neg,idx)
        np.delete(Ux_avg_neg,idx)

        Frac_neg_area[it] = (np.sum(Area_neg)/Rot_Area)
    #elif area_g_rot[it] == "both":


    


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

    Centroids_x_pos = np.subtract(Centroids_x_pos,2560)
    Iy_pos.append(ux_pos[it]*np.average(Centroids_x_pos))
    

    Centroids_x_neg = np.subtract(Centroids_x_neg,2560)
    Iy_neg.append(ux_neg[it]*np.average(Centroids_x_neg))

    Centroids_y_pos = np.subtract(Centroids_y_pos,90)
    Iz_pos.append(ux_pos[it]*np.average(Centroids_y_pos))
    

    Centroids_y_neg = np.subtract(Centroids_y_neg,90)
    Iz_neg.append(ux_neg[it]*np.average(Centroids_y_neg))
    
print("areas greater than rotor")
print(area_g_rot)

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


fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time,Iy_pos,'-b')
ax.set_ylabel("Asymmetry around y axis created by contours [m2/s]",fontsize=14)
ax.plot(Time,Iy_neg,"-r")
ax.set_xlabel("Time [s]",fontsize=16)
ax.legend(["High speed areas (ux'>0.7m/s)","low speed areas (ux'<-0.7 m/s)"])
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"csv_files/plots/Iy.png")
plt.close()


fig,ax = plt.subplots(figsize=(14,8))

ax.plot(Time,Iz_pos,'-b')
ax.set_ylabel("Asymmetry around z axis created by contours [m2/s]",fontsize=14)
ax.plot(Time,Iz_neg,"-r")
ax.set_xlabel("Time [s]",fontsize=16)
ax.legend(["High speed areas (ux'>0.7m/s)","low speed areas (ux'<-0.7 m/s)"])
plt.tight_layout()
plt.grid()
plt.savefig(in_dir+"csv_files/plots/Iz.png")
plt.close()