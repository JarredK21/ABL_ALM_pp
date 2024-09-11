from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt



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


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r



start_time = time.time()

in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

#blade asymmetry calc
df = Dataset(in_dir+"WTG01.nc")

Time = np.array(df.variables["time"])
Tstart_idx = np.searchsorted(Time,200)
T_end_idx = np.searchsorted(Time,1199.6361)+1
Time = Time[Tstart_idx:T_end_idx]
print(Time)

uvelB1 = np.array(df.variables["uvel"][Tstart_idx:T_end_idx,1:301])
vvelB1 = np.array(df.variables["vvel"][Tstart_idx:T_end_idx,1:301])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities

uvelB2 = np.array(df.variables["uvel"][Tstart_idx:T_end_idx,301:601])
vvelB2 = np.array(df.variables["vvel"][Tstart_idx:T_end_idx,301:601])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities

uvelB3 = np.array(df.variables["uvel"][Tstart_idx:T_end_idx,601:901])
vvelB3 = np.array(df.variables["vvel"][Tstart_idx:T_end_idx,601:901])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities

a = Dataset(in_dir+"Dataset.nc")

OF_vars = a.groups["OpenFAST_Variables"]

Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:T_end_idx])

IyB = []
IzB = []
ix=0
with Pool() as pool:
    for Iy_it, Iz_it in pool.imap(actuator_asymmetry_calc,np.arange(0,len(Time))):
        IyB.append(Iy_it); IzB.append(Iz_it)
        print(ix)
        ix+=1

IB = np.sqrt(np.add(np.square(IyB),np.square(IzB)))

fig = plt.figure()
plt.plot(Time,IB)
plt.show()

print(len(IB))



a = Dataset(in_dir+"Dataset_Planar_asymmetry.nc")

Planar_asymmetry = a.groups["Planar_Asymmetry_Variables"]

Iy = np.array(Planar_asymmetry.variables["Iy"][Tstart_idx:T_end_idx])
Iz = np.array(Planar_asymmetry.variables["Iz"][Tstart_idx:T_end_idx])

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))


out_dir=in_dir+"High_frequency_analysis/planar_blade_asymmetry/"

cc = round(correlation_coef(Iy,IyB),2)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,Iy,"-r",label="Blade asymmetry\nfrom planar data")
plt.plot(Time,IyB,"-b",label="Blade asymmetry\nfrom actuator data")
plt.xlabel("Time [s]")
plt.ylabel("Asymmetry around y axis [$m^2/s$]")
plt.title("correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Iy_5.5.png")
plt.close()

cc = round(correlation_coef(Iz,IzB),2)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,Iy,"-r",label="Blade asymmetry\nfrom planar data")
plt.plot(Time,IzB,"-b",label="Blade asymmetry\nfrom actuator data")
plt.xlabel("Time [s]")
plt.ylabel("Asymmetry around z axis [$m^2/s$]")
plt.title("correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"Iz_5.5.png")
plt.close()

cc = round(correlation_coef(I,IB),2)
fig = plt.figure(figsize=(14,8))
plt.plot(Time,I,"-r",label="Blade asymmetry\nfrom planar data")
plt.plot(Time,IB,"-b",label="Blade asymmetry\nfrom actuator data")
plt.xlabel("Time [s]")
plt.ylabel("Magnitude Asymmetry [$m^2/s$]")
plt.title("correlation coefficient = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"I_5.5.png")
plt.close()



    


    