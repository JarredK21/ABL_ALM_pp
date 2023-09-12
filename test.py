import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
from multiprocessing import Pool
import time
import pandas as pd

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
    

# name of csv file 
filename = in_dir+"velocityx.csv"
    
dq = dict()

dq["velocityx"] = Ux

df = pd.DataFrame(dq)

df.to_csv(filename)

print("line 102",time.time()-start_time)
