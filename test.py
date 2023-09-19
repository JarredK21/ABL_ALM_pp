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

    Ux_rotor = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            Ux_rotor.append(velocityx[it,ijk]*np.cos(np.radians(29))+velocityy[it,ijk]*np.sin(np.radians(29)))
        ijk+=1
    return np.average(Ux_rotor)


def Ux_it_offset_2(it):

    velx = np.reshape(velocityx[it],(y,x)); vely = np.reshape(velocityy[it],(y,x))

    fx = interpolate.interp2d(Y,Z,velx,kind="linear"); fy = interpolate.interp2d(Y,Z,vely,kind="linear")
    Ux_rotor = []
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)
        if r <= 63 and r > 1.5:
            vx = fx(j,k); vy = fy(j,k)
            Ux_rotor.append(vx*np.cos(np.radians(29)) + vy*np.sin(np.radians(29)) )
    return np.average(Ux_rotor)


# def IA_it_offset(it):

#     IA = 0
#     ijk = 0
#     for k in np.arange(0,len(zs)):
#         for j in np.arange(0,len(ys)):
#             r = np.sqrt(ys[j]**2 + zs[k]**2)
#             if r <= 63 and r > 1.5:
#                 delta_Ux_i = delta_Ux(it,r,ijk,j)
#                 IA += r * delta_Ux_i * dA
#             ijk+=1
#             print(ijk)
#     return IA


# def delta_Ux(it,r,ijk,j):

#     Y_0 = ys[j]

#     theta = np.arccos(Y_0/r)

#     if theta + ((2*np.pi)/3) > (2*np.pi):
#         theta_1 = theta +(2*np.pi)/3 - (2*np.pi)
#     else:
#         theta_1 = theta + (2*np.pi)/3

#     Y_1 = r*np.cos(theta_1)
#     Z_1 = r*np.sin(theta_1)


#     if theta - ((2*np.pi)/3) < 0:
#         theta_2 = theta - ((2*np.pi)/3) + (2*np.pi)
#     else:
#         theta_2 = theta - ((2*np.pi)/3)

#     Y_2 = r*np.cos(theta_2)
#     Z_2 = r*np.sin(theta_2)

#     Ux_0 = velocityx[it,ijk]*np.cos(np.radians(29))+velocityy[it,ijk]*np.sin(np.radians(29))
#     Ux_1_idx = search_coordintes(Y_1,Z_1)
#     Ux_1 = velocityx[it,Ux_1_idx]*np.cos(np.radians(29))+velocityy[it,Ux_1_idx]*np.sin(np.radians(29))
#     Ux_2_idx = search_coordintes(Y_2,Z_2)
#     Ux_2 = velocityx[it,Ux_2_idx]*np.cos(np.radians(29))+velocityy[it,Ux_2_idx]*np.sin(np.radians(29))

#     delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

#     return delta_Ux

# def search_coordintes(y,z):
#     for ico in np.arange(0,len(coords)):
#         if y >= coords[ico,0] and y <= coords[(ico+1),0] and z >= coords[ico,1] and z <= coords[(ico+1),1]:
#             break
#     return ico


start_time = time.time()

in_dir = "../../NREL_5MW_MCBL_R_CRPM_2/post_processing/"

a = Dataset(in_dir+"sampling_r_0.0.nc")

Time_sample = np.array(a.variables["time"])
time_idx = len(Time_sample); del Time_sample

p_rotor = a.groups["p_r"]; del a

x = p_rotor.ijk_dims[0] #no. data points
y = p_rotor.ijk_dims[1] #no. data points

coordinates = np.array(p_rotor.variables["coordinates"])

xo = coordinates[:,0]
yo = coordinates[:,1]
zo = coordinates[:,2]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-29)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
zs = zo - rotor_coordiates[2]

Y = np.linspace(round(np.min(ys),0), round(np.max(ys),0),x )
Z = np.linspace(round(np.min(zs),0), round(np.max(zs),0),y )

del coordinates

dy = ys[1]-ys[0]
dz = zs[1] - zs[0]
dA = dy * dz

# coords = []
# for k in zs:
#     for j in ys:
#         coords.append([j, k])
# coords = np.array(coords)

velocityx = np.array(p_rotor.variables["velocityx"]); velocityy = np.array(p_rotor.variables["velocityy"]); del p_rotor

print("line 117",time.time()-start_time)


Ux = []
print("Ux calcs")
with Pool() as pool:
    it = 1
    for Ux_it in pool.imap(Ux_it_offset, np.arange(0,time_idx)):
        Ux.append(Ux_it)
        print(it,time.time()-start_time)
        it+=1
    Ux = np.array(Ux)


Ux_2 = []
print("Ux_2 calcs")
with Pool() as pool:
    it = 1
    for Ux_it in pool.imap(Ux_it_offset_2, np.arange(0,time_idx)):
        Ux_2.append(Ux_it)
        print(it,time.time()-start_time)
        it+=1
    Ux_2 = np.array(Ux_2)


# IA = []
# print("IA calcs")
# with Pool() as pool:
#     it = 1
#     for IA_it in pool.imap(IA_it_offset, np.arange(0,time_idx)):
#         IA.append(IA_it)
#         print(it,time.time()-start_time)
#         it+=1
#     IA = np.array(IA_it)
    

# name of csv file 
filename = in_dir+"Ux.csv"
    
dq = dict()

dq["Ux"] = Ux
dq["Ux_2"] = Ux_2

#dq["IA"] = IA

df = pd.DataFrame(dq)

df.to_csv(filename)

print("line 185",time.time()-start_time)
