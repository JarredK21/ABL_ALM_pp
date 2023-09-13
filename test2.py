import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
from multiprocessing import Pool
import time
import pandas

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


in_dir = "../../NREL_5MW_MCBL_R_CRPM_2/post_processing/"


a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 100
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])
 

 
# reading the CSV file
csvFile = pandas.read_csv(in_dir+'Ux.csv')
print(csvFile)
velocityx = csvFile["Ux"].to_list()


f = interpolate.interp1d(Time_sampling,velocityx)
Ux = f(Time_OF)

fig,ax = plt.subplots(figsize=(14,8))


corr = correlation_coef(RtAeroMxh,Ux)
corr = round(corr,2)

ax.plot(Time_OF,RtAeroMxh,"-b")
ax.set_ylabel("Mx",fontsize=14)

ax2=ax.twinx()
ax2.plot(Time_OF,Ux,"-r")
ax2.set_ylabel("x velocity",fontsize=14)

plt.title("Correlation = {0}".format(corr),fontsize=16)

ax.set_xlabel("Time [s]",fontsize=16)
plt.tight_layout()
plt.show()
#plt.savefig(in_dir+"velocityx_Mx.png")