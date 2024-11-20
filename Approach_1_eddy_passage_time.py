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

mean_high_D = np.array([[568,440,360,306,266],[593,380,289,235,203],[546,351,266,215,187]])
mean_low_D = np.array([[719,496,396,339,306],[738,495,375,308,263],[725,474,349,278,238]])

std_high_D = np.array([[719,603,550,509,473],[821,633,530,463,421],[777,590,519,442,403]])
std_low_D = np.array([[961,781,691,622,583],[1020,799,671,602,544],[992,775,659,573,518]])

# min_high_D = np.array([[333,222,167,133,111],[333,222,167,133,111],[333,222,167,133,111]])
# min_low_D = np.array([[333,222,167,133,111],[333,222,167,133,111],[333,222,167,133,111]])

mean_high_Tau = np.array([[45.1,34.7,28.3,24.0,20.9],[42.5,27.2,20.7,16.8,14.6],[38.1,24.5,18.6,15.0,13.1]])
mean_low_Tau = np.array([[70.9,49.3,39.7,34.2,31.0],[63.6,43.0,32.6,26.8,23.0],[60.4,39.6,29.2,23.3,19.9]])

std_high_Tau = np.array([[56.1,46.6,42.2,39.0,36.2],[58.1,44.6,37.3,32.6,29.6],[53.7,40.6,35.7,30.4,27.7]])
std_low_Tau = np.array([[96.7,79.7,71.0,64.5,60.8],[89.5,70.7,59.8,53.8,48.8],[83.8,65.9,56.3,49.1,44.4]])

# min_high_Tau = np.array([[25.9,17.0,12.7,10.2,8.6],[23.7,15.4,11.7,9.3,7.8],[23.0,15.1,11.5,9.1,7.6]])
# min_low_Tau = np.array([[31.9,21.1,15.8,12.7,10.6],[27.8,18.6,14.0,11.1,9.3],[26.9,17.9,13.4,10.8,9.0]])

mean_high_ux = np.array([[12.39,12.44,12.44,12.44,12.42],[13.77,13.73,13.73,13.72,13.71],[14.16,14.14,14.13,14.13,14.12]])
mean_low_ux = np.array([[10.37,10.33,10.29,10.26,10.25],[11.83,11.82,11.82,11.82,11.83],[12.27,12.29,12.3,12.32,12.34]])

std_high_ux = np.array([[0.28,0.31,0.33,0.33,0.33],[0.23,0.22,0.22,0.21,0.21],[0.2,0.19,0.18,0.18,0.18]])
std_low_ux = np.array([[0.27,0.31,0.34,0.37,0.39],[0.28,0.3,0.31,0.32,0.33],[0.28,0.29,0.29,0.29,0.28]])


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

D_mean = [360,289,266,396,375,349]
D_std = [[0,0,0,0,0,0],[550,530,519,691,671,659]]

fig = plt.figure(figsize=(14,8))
plt.bar(x,D_mean)
plt.errorbar(x,D_mean,yerr=D_std,fmt="o", color="k",capsize=10)
plt.ylabel("Eddy length x' direction [m]")
plt.grid()
plt.title("Filter cutoff 167m")
plt.savefig(out_dir+"eddy_length_summary_bar.png")
plt.close()


T_mean = [28.3,20.7,18.6,39.7,32.6,29.2]
T_std = [[0,0,0,0,0,0],[42.2,37.3,35.7,71.0,59.8,56.3]]

fig = plt.figure(figsize=(14,8))
plt.bar(x,T_mean)
plt.errorbar(x,T_mean,yerr=T_std,fmt="o", color="k",capsize=10)
plt.ylabel("Eddy passage time [s]")
plt.grid()
plt.title("Filter cutoff 167m")
plt.savefig(out_dir+"eddy_passage_time_summary_bar.png")
plt.close()

Ux_mean = [12.44,13.73,14.13,10.29,11.82,12.3]
Ux_std = [0.33,0.22,0.18,0.34,0.31,0.29]

fig = plt.figure(figsize=(14,8))
plt.bar(x,Ux_mean)
plt.errorbar(x,Ux_mean,yerr=Ux_std,fmt="o", color="k",capsize=10)
plt.ylabel("Average of Eddy velocity [m/s]")
plt.grid()
plt.title("Filter cutoff 167m")
plt.savefig(out_dir+"eddy_velocity_summary_bar.png")
plt.close()
