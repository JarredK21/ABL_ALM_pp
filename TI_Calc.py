from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/convective_precursor/70000/"
df = Dataset(in_dir+"sampling70000.nc")

Time = np.array(df.variables["time"])
tstart_idx = np.searchsorted(Time,38000); tend_idx = np.searchsorted(Time,40000)
Time = Time[tstart_idx:tend_idx]
p_l = df.groups["p_l"]

nx = p_l.ijk_dims[0]; ny = p_l.ijk_dims[1]

coordinates = np.array(p_l.variables["coordinates"][:262144])
x = coordinates[0:512,0]; y = coordinates[0:512,0]

uhub = []
for it in np.arange(tstart_idx,tend_idx,1,dtype=int):
    u = np.array(p_l.variables["velocityx"][it,:262144]).reshape(nx,ny)
    v = np.array(p_l.variables["velocityy"][it,:262144]).reshape(nx,ny)

    fu = interpolate.interp2d(x,y,u); fv = interpolate.interp2d(x,y,v)
    hvel = np.sqrt(fu(2555,2560)[0]**2 + fv(2555,2560)[0]**2)

    uhub.append(hvel)

Uhub = np.average(uhub)
ustd = np.std(uhub)

TI = (ustd/Uhub)

print(Uhub)
print(TI)