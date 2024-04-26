import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from netCDF4 import Dataset
from matplotlib.backends.backend_pdf import PdfPages
from multiprocessing import Pool
import statistics
from scipy.signal import butter,filtfilt

in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"
out_dir = in_dir + "Asymmetry_analysis/"

df = Dataset(in_dir+"Thresholding_Dataset.nc")

Time = np.array(df.variables["Time"])
Time = Time - Time[0]
dt = Time[1] - Time[0]
Time_steps = np.arange(0,len(Time))

Time_start = 200; Time_start_idx = np.searchsorted(Time,Time_start)
Time = Time[Time_start_idx:]

#thresholds = [10.0,8.7,7.3,6.0,4.7,3.4,2.0,0.7]
thresholds = [11.0,9.5,8.1,6.6,5.1,3.6,2.2,0.7]

update_pdf_plots = True

if update_pdf_plots == True:
    with PdfPages(out_dir+'Ejections_analysis.pdf') as pdf:


        #plotting joint areas
        fig,ax = plt.subplots(figsize=(14,8))
        plt.rcParams['font.size'] = 14
        for threshold in thresholds:

            print("line 293",-threshold)

            group = df.groups["{}".format(threshold)]

            Area = np.array(group.variables["Area_ejection"][Time_start_idx:])

            ax.plot(Time,Area,label="ux'<-{}".format(threshold))
            ax.set_ylabel("Area [$m^2$]",fontsize=14)
            ax.set_xlabel("Time [s]",fontsize=16)


        plt.legend()
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()


        #plotting joint average heights
        fig,ax = plt.subplots(figsize=(14,8))
        plt.rcParams['font.size'] = 14
        for threshold in thresholds:

            print("line 293",-threshold)

            group = df.groups["{}".format(threshold)]

            Avg_height = np.array(group.variables["Average_height"][Time_start_idx:])

            ax.plot(Time,Avg_height,label="ux'<-{}".format(threshold))
            ax.set_ylabel("Average height [m]",fontsize=14)
            ax.set_xlabel("Time [s]",fontsize=16)


        plt.legend()
        plt.tight_layout()
        plt.grid()
        pdf.savefig()
        plt.close()



for threshold in thresholds:
    print("line 293",-threshold)

    group = df.groups["{}".format(threshold)]

    Area = np.array(group.variables["Area_ejection"][Time_start_idx:])
    ic = 0
    for A in Area:
        if A > 0:
            ic+=1
    Time_perc = ic/len(Area)
    print("Threshold = -{} Perc time = {}".format(threshold,round(Time_perc,4)))

    