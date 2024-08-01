from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

out_dir = "../../ABL_precursor_2_restart/plots/"

offsets = [22.5,85,142.5]

filter_cutoff = [1/(1*3e-03),1/(1.5*3e-03),1/(2*3e-03),1/(2.5*3e-03),1/(3*3e-03)]

colors = ["r","b","g"]

mean_high_D = np.array([[979,719,579,485,429],[1085,712,575,461,391],[1128,750,573,451,378]])
mean_low_D = np.array([[1162,805,610,492,426],[1170,836,634,526,450],[1175,884,665,535,456]])

std_high_D = np.array([[849,711,653,604,582],[951,768,721,632,567],[940,804,720,633,573]])
std_low_D = np.array([[1031,895,775,682,621],[1101,916,777,722,658],[1054,937,832,741,678]])

min_high_D = np.array([[333,222,167,133,111],[333,222,167,133,111],[333,222,167,133,111]])
min_low_D = np.array([[333,222,167,133,111],[333,222,167,133,111],[333,222,167,133,111]])

mean_high_Tau = np.array([[77.6,56.5,45.4,37.9,33.6],[77.6,50.9,41.1,33.0,27.9],[78.6,52.2,39.9,31.4,26.3]])
mean_low_Tau = np.array([[116.1,81.5,62.3,50.5,44.0],[102.0,73.5,56.0,46.6,39.9],[98.7,74.7,56.4,45.3,38.7]])

std_high_Tau = np.array([[66.4,55.0,50.2,46.3,44.6],[67.3,54.1,50.8,44.5,39.9],[64.8,55.2,49.5,43.5,39.4]])
std_low_Tau = np.array([[104.1,91.9,80.3,71.4,65.4],[96.8,81.6,69.6,64.9,59.4],[89.5,80.1,71.3,63.7,58.4]])

min_high_Tau = np.array([[25.8,17.0,12.1,9.8,8.2],[23.7,15.4,11.4,9.3,7.8],[23.0,15.1,11.5,9.1,7.6]])
min_low_Tau = np.array([[31.6,21.1,15.8,12.7,10.5],[27.8,18.5,13.9,11.1,9.3],[26.8,17.9,13.4,10.8,9.0]])

mean_high_ux = np.array([[12.54,12.61,12.63,12.63,12.62],[13.89,13.88,13.87,13.86,13.85],[14.28,14.27,14.26,14.25,14.24]])
mean_low_ux = np.array([[10.08,10.0,9.94,9.89,9.86],[11.55,11.49,11.46,11.45,11.45],[11.99,11.96,11.94,11.95,11.97]])

std_high_ux = np.array([[0.26,0.27,0.28,0.28,0.29],[0.23,0.22,0.22,0.21,0.21],[0.2,0.2,0.19,0.19,0.19]])
std_low_ux = np.array([[0.23,0.26,0.28,0.29,0.31],[0.24,0.26,0.27,0.29,0.30],[0.24,0.26,0.27,0.28,0.28]])


#Eddy length
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5

    plt.plot(filter_cutoff,mean_high_D[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(filter_cutoff,mean_low_D[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Mean Eddy length [m]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Mean_Eddy_length_3.png")
plt.close()

fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5


    plt.plot(filter_cutoff,std_high_D[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(filter_cutoff,std_low_D[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Standard deviation Eddy length [m]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Std_Eddy_length_3.png")
plt.close()


#Eddy velocity
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5

    plt.plot(filter_cutoff,mean_high_ux[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(filter_cutoff,mean_low_ux[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Mean Eddy velocity [m/s]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Mean_Eddy_velocity_3.png")
plt.close()


fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5


    plt.plot(filter_cutoff,std_high_ux[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(filter_cutoff,std_low_ux[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Standard deviation Eddy velocity [m/s]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Std_Eddy_velocity_3.png")
plt.close()


#Eddy passage time
fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5

    plt.plot(filter_cutoff,mean_high_Tau[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(filter_cutoff,mean_low_Tau[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Mean Eddy Passage time [s]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Mean_Eddy_passage_time_3.png")
plt.close()


fig = plt.figure(figsize=(14,8))
for i in np.arange(0,len(offsets)):

    height = offsets[i]+7.5


    plt.plot(filter_cutoff,std_high_Tau[i],"-o",color=colors[i],label="high speed: {}m".format(height))
    plt.plot(filter_cutoff,std_low_Tau[i],"--*",color=colors[i],label="Low speed: {}m".format(height))
plt.xlabel("Filter width [m]")
plt.ylabel("Standard deviation Eddy Passage time [s]")
plt.legend()
plt.tight_layout()
plt.grid()
plt.savefig(out_dir+"Std_Eddy_passage_time_3.png")
plt.close()

x = ["High speed\n30m","High speed\n92.5m","High speed\n150m","Low speed\n30m","Low speed\n92.5m","Low speed\n150m"]

D_min = [167,167,167,167,167,167]
D_mean = [579,575,573,610,634,665]
H = np.subtract(D_mean,D_min)
D_std = [653,721,720,775,775,832]

fig = plt.figure(figsize=(14,8))
plt.bar(x,D_std,bottom=H,color="b")
plt.bar(x,H,bottom=D_min,color="r")
plt.ylabel("Eddy length x' direction [m]")
plt.grid()
plt.title("Approach 3: filter cutoff 167m")
plt.savefig(out_dir+"eddy_length_summary_bar.png")
plt.close()


T_min = [12.1,11.4,11.5,15.8,13.9,13.4]
T_mean = [45.4,41.1,39.9,62.3,56.0,56.4]
H = np.subtract(T_mean,T_min)
T_std = [50.2,50.8,49.5,80.3,69.6,71.3]

fig = plt.figure(figsize=(14,8))
plt.bar(x,T_std,bottom=T_mean,color="b")
plt.bar(x,H,bottom=T_min,color="r")
plt.ylabel("Eddy passage time [s]")
plt.grid()
plt.title("Approach 3: filter cutoff 167m")
plt.savefig(out_dir+"eddy_passage_time_summary_bar.png")
plt.close()

Ux_mean = [12.63,13.87,14.26,9.94,11.46,11.94]
Ux_std = [0.28,0.22,0.19,0.28,0.27,0.27]
Ux_min = np.subtract(Ux_mean,Ux_std)

fig = plt.figure(figsize=(14,8))
plt.bar(x,Ux_std,bottom=Ux_mean,color="b")
plt.bar(x,Ux_std,bottom=Ux_min,color="r")
plt.ylabel("Average of Eddy velocity [m/s]")
plt.grid()
plt.title("Approach 3: filter cutoff 167m")
plt.savefig(out_dir+"eddy_velocity_summary_bar.png")
plt.close()

