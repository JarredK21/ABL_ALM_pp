from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

out_dir = "../../ABL_precursor_2_restart/plots/"

offsets = [22.5,85,142.5]
filter_cutoffs = [6.5e-03,3e-03,1.1e-03]

filter_cutoff = [1,1.5,2,2.5,3]

colors = ["r","b","g"]

mean_high_D = np.array([[423,309,256,226,207],[813,492,391,320,277],[1912,1275,1142,891,733]])
mean_low_D = np.array([[407,299,255,232,219],[787,575,450,373,320],[1772,1178,994,901,798]])

std_high_D = np.array([[429,344,312,294,281],[673,423,366,321,292],[931,721,668,642,608]])
std_low_D = np.array([[402,339,293,278,271],[517,471,421,386,344],[1017,507,667,564,559]])

min_high_D = np.array([[154,103,77,62,51],[333,222,167,133,111],[909,606,455,364,303]])
min_low_D = np.array([[154,103,77,62,51],[333,222,167,133,111],[909,606,455,364,303]])

mean_high_Tau = np.array([[33.3,24.4,20.2,17.9,16.4],[58.4,35.4,28.1,23.0,19.9],[133.6,89.1,79.8,62.3,51.2]])
mean_low_Tau = np.array([[41.4,30.7,26.2,24.0,22.6],[68.4,50.4,39.6,32.8,28.2],[146.1,97.5,82.9,75.4,66.9]])

std_high_Tau = np.array([[33.0,26.4,24.0,22.6,21.6],[47.6,29.8,25.8,22.6,20.6],[64.8,49.9,46.0,44.3,42.0]])
std_low_Tau = np.array([[41.8,35.8,31.1,29.7,29.0],[45.7,42.2,37.8,34.8,31.1],[85.2,42.3,56.7,48.0,47.7]])

min_high_Tau = np.array([[11.7,7.8,6.0,4.8,4.0],[23.7,15.4,11.7,9.3,7.8],[64.1,42.4,31.3,25.2,20.9]])
min_low_Tau = np.array([[14.6,9.8,7.3,5.8,4.9],[27.8,18.6,14.0,11.1,9.3],[73.6,49.2,37.0,29.6,24.5]])

mean_high_ux = np.array([[12.61,12.58,12.53,12.49,12.46],[13.86,13.84,13.84,13.84,13.83],[14.29,14.29,14.27,14.25,14.25]])
mean_low_ux = np.array([[9.94,9.88,9.86,9.87,9.88],[11.57,11.51,11.48,11.47,11.47],[12.18,12.11,12.05,12.01,12.01]])

std_high_ux = np.array([[0.25,0.26,0.26,0.26,0.25],[0.18,0.17,0.19,0.18,0.18],[0.16,0.13,0.14,0.14,0.16]])
std_low_ux = np.array([[0.26,0.29,0.31,0.34,0.35],[0.19,0.22,0.25,0.26,0.27],[0.11,0.13,0.16,0.19,0.21]])


#Eddy length
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5

    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),mean_high_D[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),mean_low_D[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Mean Eddy length [m]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Mean_Eddy_length_2.png")
plt.close()


fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5


    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),std_high_D[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),std_low_D[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Standard deviation Eddy length [m]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Std_Eddy_length_2.png")
plt.close()


#Eddy velocity
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5

    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),mean_high_ux[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),mean_low_ux[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Mean Eddy velocity [m/s]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Mean_Eddy_velocity_2.png")
plt.close()


fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5


    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),std_high_ux[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),std_low_ux[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Standard deviation Eddy velocity [m/s]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Std_Eddy_velocity_2.png")
plt.close()


#Eddy passage time
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5

    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),mean_high_Tau[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),mean_low_Tau[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Mean Eddy passage time [s]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Mean_Eddy_passage_time_2.png")
plt.close()


fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5


    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),std_high_Tau[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(1/np.multiply(filter_cutoffs[i],filter_cutoff),std_low_Tau[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Standard deviation Eddy passage time [s]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Std_Eddy_passage_time_2.png")
plt.close()