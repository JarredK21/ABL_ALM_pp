import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
from multiprocessing import Pool
import time

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def Ux_it_offset(it):

    Velocityx = velocityx[it]
    #Velocityx = np.reshape(Velocityx,(y,x))

    Ux_rotor = []
    ijk = 0
    for k in np.arange(0,len(zs)):
        for j in np.arange(0,len(ys)):
            r = np.sqrt(ys[j]**2 + zs[k]**2)
            if r <= 60 and r > 1.5:
                Ux_rotor.append(Velocityx[ijk])
            ijk+=1
    return np.average(Ux_rotor)




start_time = time.time()

in_dir = "../../NREL_5MW_MCBL_R_CRPM_2/post_processing/"

a = Dataset(in_dir+"sampling_r_0.0.nc")

Time_sample = np.array(a.variables["time"])
time_idx = len(Time_sample); del Time_sample

p_rotor = a.groups["p_r"]; del a

x = p_rotor.ijk_dims[0] #no. data points
y = p_rotor.ijk_dims[1] #no. data points

coordinates = np.array(p_rotor.variables["coordinates"])

xo = coordinates[0:x,0]
yo = coordinates[0:x,1]
zo = np.linspace(p_rotor.origin[2],p_rotor.axis2[2],y)

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-29)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
zs = zo - rotor_coordiates[2]


velocityx = np.array(p_rotor.variables["velocityx"]); del p_rotor

print("line 68",time.time()-start_time)

Ux = []
print("Ux calcs")
with Pool() as pool:
    it = 1
    for Ux_it in pool.imap(Ux_it_offset, np.arange(0,time_idx)):
        Ux.append(Ux_it)
        print(it,time.time()-start_time)
        it+=1
    Ux = np.array(Ux)

import csv 
    
# field names 
fields = ['velocityx'] 
    
# data rows of csv file 
rows = [Ux] 
    
# name of csv file 
filename = in_dir+"velocityx.csv"
    
# writing to csv file 
with open(filename, 'w') as csvfile: 
    # creating a csv writer object 
    csvwriter = csv.writer(csvfile) 
        
    # writing the fields 
    csvwriter.writerow(fields) 
        
    # writing the data rows 
    csvwriter.writerows(rows)

print("line 102",time.time()-start_time)

a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 0
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])

print("line 121",time.time()-start_time)


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
