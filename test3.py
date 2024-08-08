from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate  
from multiprocessing import Pool

def actuator_asymmetry_calc(it):
    R = np.linspace(0,63,300)
    hvelB1 = np.add(np.cos(np.radians(29))*uvelB1[it], np.sin(np.radians(29))*vvelB1[it])
    IyB1 = hvelB1*R*np.cos(Azimuth[it])
    IzB1 = hvelB1*R*np.sin(Azimuth[it])
    IyB1 = np.sum(IyB1)
    IzB1 = np.sum(IzB1)

    hvelB2 = np.add(np.cos(np.radians(29))*uvelB2[it], np.sin(np.radians(29))*vvelB2[it])
    AzB2 = Azimuth[it] + 120
    if AzB2 >= 360:
        AzB2-=360

    IyB2 = hvelB2*R*np.cos(AzB2)
    IzB2 = hvelB2*R*np.sin(AzB2)
    IyB2 = np.sum(IyB2)
    IzB2 = np.sum(IzB2)

    hvelB3 = np.add(np.cos(np.radians(29))*uvelB3[it], np.sin(np.radians(29))*vvelB3[it])
    AzB3 = Azimuth[it] + 240
    if AzB3 >= 360:
        AzB3-=360

    IyB3 = hvelB3*R*np.cos(AzB3)
    IzB3 = hvelB3*R*np.sin(AzB3)
    IyB3 = np.sum(IyB3)
    IzB3 = np.sum(IzB3)

    return IyB1+IyB2+IyB3, IzB1+IzB2+IzB3


in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"


#blade asymmetry calc
df = Dataset(in_dir+"WTG01.nc")

Time = np.array(df.variables["time"])
Tstart_idx = np.searchsorted(Time,200)
Time = Time[Tstart_idx:]

uvelB1 = np.array(df.variables["uvel"][Tstart_idx:,1:301])
vvelB1 = np.array(df.variables["vvel"][Tstart_idx:,1:301])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities

uvelB2 = np.array(df.variables["uvel"][Tstart_idx:,301:601])
vvelB2 = np.array(df.variables["vvel"][Tstart_idx:,301:601])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities

uvelB3 = np.array(df.variables["uvel"][Tstart_idx:,601:901])
vvelB3 = np.array(df.variables["vvel"][Tstart_idx:,601:901])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities

a = Dataset(in_dir+"Dataset.nc")

OF_vars = a.groups["OpenFAST_Variables"]

Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:-1])

IyB = []
IzB = []
ix=0
with Pool() as pool:
    for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,np.arange(0,len(Time))):
        IyB.append(Iy_it); IzB.append(Iz_it)
        print(ix)
        ix+=1

IB = np.sqrt(np.add(np.square(IyB),np.square(IzB)))

a = Dataset(in_dir+"Dataset.nc")
Time_OF = np.array(a.variables["Time_OF"])
Time_sampling = np.array(a.variables["Time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]
OF_vars = a.groups["OpenFAST_Variables"]
Azimuth = np.radians(OF_vars.variables["Azimuth"])

a = Dataset(in_dir+"sampling_r_-63.0_0.nc")

p = a.groups["p_r"]

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points


normal = 29

#define plotting axes
coordinates = np.array(p.variables["coordinates"])


xo = coordinates[0:x,0]
yo = coordinates[0:x,1]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-normal)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
xs = xs + rotor_coordiates[0]
ys = ys + rotor_coordiates[1]
zs = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y)


u = np.array(p.variables["velocityx"])

u[u<0]=0


u_plane = u.reshape(y,x)

f = interpolate.interp2d(ys,zs,u_plane,kind="linear")
ix = 0
for it in np.arange(0,len(Time_sampling)-1):

    print("initial inner it",Time_OF[it*100])
    for it_inner in np.arange(it*100,(it+1)*100):
        Az = -Azimuth[it_inner]
        print(ix)
        ix+=1

    print("next it",Time_sampling[it+1]); print("final inner it",Time_OF[it_inner])

Tstart_idx = np.searchsorted(Time_OF,200)
Azy = Azimuth[Tstart_idx:it_inner]

Time_end = np.searchsorted(Time,Time_OF[it_inner])
IB = IB[:Time_end]

print(len(Azy),len(IB))
