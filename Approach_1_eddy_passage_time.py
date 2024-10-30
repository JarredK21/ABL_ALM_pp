from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt

out_dir = "../../ABL_precursor_2_restart/Eddy_passage_time_analysis/"

offsets = [22.5,85,142.5]

filter_cutoff = [1/(1*3e-03),1/(1.5*3e-03),1/(2*3e-03),1/(2.5*3e-03),1/(3*3e-03)]

colors = ["r","b","g"]

# mean_high_D = np.array([[774,586,451,373,326],[813,492,391,320,277],[836,511,386,310,264]])
# mean_low_D = np.array([[850,569,445,360,318],[787,575,450,373,320],[838,597,448,361,309]])

# std_high_D = np.array([[659,537,431,376,352],[673,423,366,321,292],[633,465,385,324,288]])
# std_low_D = np.array([[533,487,429,374,347],[517,471,421,386,344],[578,488,430,367,336]])

# min_high_D = np.array([[333,222,167,133,111],[333,222,167,133,111],[333,222,167,133,111]])
# min_low_D = np.array([[333,222,167,133,111],[333,222,167,133,111],[333,222,167,133,111]])

# mean_high_Tau = np.array([[61.5,46.2,35.5,29.4,25.7],[58.4,35.4,28.1,23.0,19.9],[58.4,35.7,27.0,21.7,18.5]])
# mean_low_Tau = np.array([[84.6,57.3,45.1,36.7,32.5],[68.4,50.4,39.6,32.8,28.2],[70.2,50.2,37.8,30.4,26.0]])

# std_high_Tau = np.array([[51.5,41.6,33.2,28.9,27.1],[47.6,29.8,25.8,22.6,20.6],[43.7,32.0,26.5,22.3,19.8]])
# std_low_Tau = np.array([[54.1,50.2,44.6,39.1,36.5],[45.7,42.2,37.8,34.8,31.1],[49.3,41.8,36.9,31.6,28.9]])

# min_high_Tau = np.array([[25.9,17.0,12.7,10.2,8.6],[23.7,15.4,11.7,9.3,7.8],[23.0,15.1,11.5,9.1,7.6]])
# min_low_Tau = np.array([[31.9,21.1,15.8,12.7,10.6],[27.8,18.6,14.0,11.1,9.3],[26.9,17.9,13.4,10.8,9.0]])

# mean_high_ux = np.array([[12.52,12.0,12.01,12.01,12.0],[13.86,13.84,13.84,13.84,13.83],[14.26,14.25,14.24,14.23,14.23]])
# mean_low_ux = np.array([[9.56,10.02,9.96,9.91,9.88],[11.57,11.51,11.48,11.47,11.47],[12.02,11.99,11.96,11.98,12.0]])

# std_high_ux = np.array([[0.22,0.24,0.25,0.26,0.26],[0.18,0.17,0.19,0.18,0.18],[0.15,0.15,0.16,0.16,0.16]])
# std_low_ux = np.array([[0.18,0.23,0.25,0.27,0.28],[0.19,0.22,0.25,0.26,0.27],[0.2,0.22,0.24,0.25,0.25]])

mean_high_D = np.array([[586,451,368,312,272],[610,387,296,242,207],[567,363,273,223,192]])
mean_low_D = np.array([[695,479,385,330,298],[699,480,366,303,261],[704,463,364,277,237]])

std_high_D = np.array([[751,633,572,529,493],[859,659,556,492,442],[811,623,538,469,422]])
std_low_D = np.array([[921,746,662,598,558],[959,770,649,585,530],[962,749,642,558,506]])

# min_high_D = np.array([[333,222,167,133,111],[333,222,167,133,111],[333,222,167,133,111]])
# min_low_D = np.array([[333,222,167,133,111],[333,222,167,133,111],[333,222,167,133,111]])

mean_high_Tau = np.array([[46.7,35.7,29.0,24.6,21.4],[43.9,27.8,21.3,17.4,14.9],[39.7,25.4,19.1,15.6,13.5]])
mean_low_Tau = np.array([[67.4,46.8,37.7,32.5,29.5],[59.4,41.0,31.4,26.0,22.3],[57.6,38.1,28.4,22.7,19.5]])

std_high_Tau = np.array([[58.9,49.1,44.1,40.7,37.9],[61.0,46.6,39.3,34.7,31.2],[56.2,43.0,37.1,32.4,29.1]])
std_low_Tau = np.array([[89.1,73.0,64.9,59.0,55.3],[81.0,65.6,55.5,50.1,45.5],[78.5,61.5,52.8,45.9,41.7]])

# min_high_Tau = np.array([[25.9,17.0,12.7,10.2,8.6],[23.7,15.4,11.7,9.3,7.8],[23.0,15.1,11.5,9.1,7.6]])
# min_low_Tau = np.array([[31.9,21.1,15.8,12.7,10.6],[27.8,18.6,14.0,11.1,9.3],[26.9,17.9,13.4,10.8,9.0]])

mean_high_ux = np.array([[12.33,12.37,12.38,12.37,12.36],[13.71,13.67,13.67,13.66,13.65],[14.1,14.08,14.07,14.07,14.06]])
mean_low_ux = np.array([[10.39,10.36,10.33,10.31,10.3],[11.85,11.83,11.83,11.84,11.85],[12.29,12.3,12.31,12.33,12.34]])

std_high_ux = np.array([[0.28,0.32,0.33,0.34,0.33],[0.24,0.23,0.22,0.22,0.32],[0.2,0.19,0.19,0.18,0.18]])
std_low_ux = np.array([[0.31,0.34,0.38,0.41,0.44],[0.3,0.32,0.33,0.34,0.35],[0.3,0.3,0.3,0.3,0.3]])


#Eddy length
plt.rcParams['font.size'] = 16
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
plt.savefig(out_dir+"Mean_Eddy_length.png")
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
plt.savefig(out_dir+"Std_Eddy_length.png")
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
plt.savefig(out_dir+"Mean_Eddy_velocity.png")
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
plt.savefig(out_dir+"Std_Eddy_velocity.png")
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
plt.savefig(out_dir+"Mean_Eddy_passage_time.png")
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
plt.savefig(out_dir+"Std_Eddy_passage_time.png")
plt.close()


x = ["High speed\n30m","High speed\n92.5m","High speed\n150m","Low speed\n30m","Low speed\n92.5m","Low speed\n150m"]

D_mean = [368,296,274,385,366,346]
D_std = [[0,0,0,0,0,0],[572,556,538,662,649,642]]

fig = plt.figure(figsize=(14,8))
plt.bar(x,D_mean)
plt.errorbar(x,D_mean,yerr=D_std,fmt="o", color="k",capsize=10)
plt.ylabel("Eddy length x' direction [m]")
plt.grid()
plt.title("Filter cutoff 167m")
plt.savefig(out_dir+"eddy_length_summary_bar.png")
plt.close()


T_mean = [29.0,21.3,19.1,37.7,31.4,28.4]
T_std = [[0,0,0,0,0,0],[44.1,39.3,37.1,64.9,55.5,52.8]]

fig = plt.figure(figsize=(14,8))
plt.bar(x,T_mean)
plt.errorbar(x,T_mean,yerr=T_std,fmt="o", color="k",capsize=10)
plt.ylabel("Eddy passage time [s]")
plt.grid()
plt.title("Filter cutoff 167m")
plt.savefig(out_dir+"eddy_passage_time_summary_bar.png")
plt.close()

Ux_mean = [12.38,13.67,14.07,10.33,11.84,12.31]
Ux_std = [0.33,0.22,0.19,0.38,0.33,0.3]

fig = plt.figure(figsize=(14,8))
plt.bar(x,Ux_mean)
plt.errorbar(x,Ux_mean,yerr=Ux_std,fmt="o", color="k",capsize=10)
plt.ylabel("Average of Eddy velocity [m/s]")
plt.grid()
plt.title("Filter cutoff 167m")
plt.savefig(out_dir+"eddy_velocity_summary_bar.png")
plt.close()
