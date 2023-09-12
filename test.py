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

    velocityx = np.array(p_rotor.variables["velocityx"][it]); velocityy = np.array(p_rotor.variables["velocityx"][it])

    Ux_rotor = []
    ijk = 0
    for co in coords:
        r = np.sqrt(co[0]**2 + co[1]**2)
        if r <= 63 and r > 1.5:
            hvelmag = velocityx[ijk]*np.cos(np.radians(29)) + velocityy[ijk]*np.sin(np.radians(29))
            Ux_rotor.append(hvelmag)
        ijk+=1
    return np.average(Ux_rotor)


def IA_it_offset(it):

    IA = 0
    ijk = 0
    for co in coords:
        r = np.sqrt(co[0]**2 + co[1]**2)
        if r <= 63 and r > 1.5:
            delta_Ux_i = delta_Ux(it,r,ijk)
            IA += r * delta_Ux_i * dA
    return IA


def delta_Ux(it,r,ijk):

    velocityx = np.array(p_rotor.variables["velocityx"][it]); velocityy = np.array(p_rotor.variables["velocityx"][it])

    Y_0 = coords[ijk][0]

    theta = np.arccos(Y_0/r)

    if theta + ((2*np.pi)/3) > (2*np.pi):
        theta_1 = theta +(2*np.pi)/3 - (2*np.pi)
    else:
        theta_1 = theta + (2*np.pi)/3

    Y_1 = r*np.cos(theta_1)
    Z_1 = r*np.sin(theta_1)


    if theta - ((2*np.pi)/3) < 0:
        theta_2 = theta - ((2*np.pi)/3) + (2*np.pi)
    else:
        theta_2 = theta - ((2*np.pi)/3)

    Y_2 = r*np.cos(theta_2)
    Z_2 = r*np.sin(theta_2)

    Ux_0 =  velocityx[ijk]*np.cos(np.radians(29)) + velocityy[ijk]*np.sin(np.radians(29))
    Ux_1_idx = search_coordintes(Y_1,Z_1)
    Ux_1 = velocityx[Ux_1_idx]*np.cos(np.radians(29)) + velocityy[Ux_1_idx]*np.sin(np.radians(29))
    Ux_2_idx = search_coordintes(Y_2,Z_2)
    Ux_2 = velocityx[Ux_2_idx]*np.cos(np.radians(29)) + velocityy[Ux_2_idx]*np.sin(np.radians(29))

    delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

    return delta_Ux

def search_coordintes(y,z):
    for ico in np.arange(0,len(coords)):
        if y >= coords[ico][0] and y <= coords[ico+1][0] and z >= coords[ico][1] and z <= coords[ico+1][1]:
            break
    return ico


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

dy = ys[1]-ys[0]
dz = zs[1] - zs[0]
dA = dy * dz

coords = []
for k in zs:
    for j in ys:
        coords.append([j, k])


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


IA = []
print("IA calcs")
with Pool() as pool:
    it = 1
    for IA_it in pool.imap(IA_it_offset, np.arange(0,time_idx)):
        IA.append(IA_it)
        print(it,time.time()-start_time)
        it+=1
    IA = np.array(IA_it)
    

# name of csv file 
filename = in_dir+"test.csv"
    
dq = dict()

dq["Ux"] = Ux

dq["IA"] = IA

df = pd.DataFrame(dq)

df.to_csv(filename)

print("line 102",time.time()-start_time)
