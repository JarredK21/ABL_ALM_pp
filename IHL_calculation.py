from netCDF4 import Dataset
import numpy as np
import time
from multiprocessing import Pool
from scipy import interpolate
import math
import matplotlib.pyplot as plt
from matplotlib import cm
import os


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    mag_horz_vel = []
    for i in np.arange(0,len(zs)):
        u_i = u[it,i*x:(i+1)*x]; v_i = v[it,i*x:(i+1)*x]
        if zs[i] < h[0]:
            twist_h = f(h[0])

        elif zs[i] > h[-1]:
            twist_h = f(h[-1])
        else:
            twist_h = f(zs[i])

        mag_horz_vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
        mag_horz_vel.extend(mag_horz_vel_i)
    mag_horz_vel = np.array(mag_horz_vel)
    return mag_horz_vel


def Fluc_Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    mag_fluc_horz_vel = []
    for i in np.arange(0,len(zs)):
        u_i = u[it,i*x:(i+1)*x]; v_i = v[it,i*x:(i+1)*x]
        if zs[i] < h[0]:
            twist_h = f(h[0])
            ux_mean = f_ux(h[0])

        elif zs[i] > h[-1]:
            twist_h = f(h[-1])
            ux_mean = f_ux(h[-1])
        else:
            twist_h = f(zs[i])
            ux_mean = f_ux(zs[i])

        mag_horz_vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
        mag_fluc_horz_vel_i = np.subtract(mag_horz_vel_i,ux_mean)
        mag_fluc_horz_vel.extend(mag_fluc_horz_vel_i)
    mag_fluc_horz_vel = np.array(mag_fluc_horz_vel)
    return mag_fluc_horz_vel


def Update(it):

    U = u[it]
    U_pri = u_pri[it]

    u_plane = U.reshape(y,x)
    Xs,Ys = np.meshgrid(ys,zs)

    Z = u_plane

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(Xs,Ys,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    cb = plt.colorbar(cs)

    AH = 0; AL = 0; AI = 0
    IyH = 0; IyL = 0; IyI = 0; Iy = 0
    IzH = 0; IzL = 0; IzI = 0; Iz = 0
    UxH = []; UxL = []; UxI = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)

        if r <= 63 and r > 1.5:

            Uijk = U[ijk]
            U_pri_ijk = U_pri[ijk]
            if cmin<=Uijk<=cmax:
                Iy+=(Uijk*k*dA)
                Iz+=(Uijk*j*dA)

                if U_pri_ijk >= 0.7:
                    AH+=dA
                    IyH+=(Uijk*k*dA)
                    IzH+=(Uijk*j*dA)
                    UxH.append(Uijk)
                    plt.plot(ys[ijk],zs[ijk],"ok")
                elif U_pri_ijk <= -0.7:
                    AL+=dA
                    IyL+=(Uijk*k*dA)
                    IzL+=(Uijk*j*dA)
                    UxL.append(Uijk)
                    plt.plot(ys[ijk],zs[ijk],"+k")
                else:
                    AI+=dA
                    IyI+=(Uijk*k*dA)
                    IzI+=(Uijk*j*dA)
                    UxI.append(Uijk)
                    plt.plot(ys[ijk],zs[ijk],"sk")
        ijk+=1

    if len(UxH) > 0:
        UxH = np.average(UxH)
    else:
        UxH = 0

    if len(UxL) > 0:
        UxL = np.average(UxL)
    else:
        UxL = 0

    if len(UxI) > 0:
        UxI = np.average(UxI)
    else:
        UxI = 0


    plt.xlabel("y' axis (rotor frame of reference) [m]",fontsize=40)
    plt.ylabel("z' axis (rotor frame of reference) [m]",fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)


    #define titles and filenames for movie
    Title = "Rotor Plane. \nFluctuating horizontal velocity [m/s]: Offset = -63.0m, Time = {}[s]".format(it)
    filename = "Rotor_Fluc_Horz_-63.0_{}.png".format(it)
    
    x_c = [-1,-10]; y_c = [-1,-10]
    plt.plot(x_c,y_c,"-k",linewidth=5,label="$u_x' \geq 0.7 m/s$")
    plt.plot(x_c,y_c,"--k",linewidth=5,label="$u_x' \leq -0.7 m/s$")

    plt.xlim([ys[0],ys[-1]]);plt.ylim(zs[0],zs[-1])
    plt.legend()
    plt.title(Title)
    plt.tight_layout()
    plt.savefig(in_dir+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy,Iz,UxH,UxL,UxI



start_time = time.time()

#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38200)
u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v


print("line 67", time.time()-start_time)
print(ux_mean_profile)

#directories
in_dir = "./"
out_dir = in_dir+"ISOplots/"



a = Dataset("./sampling_r_-63.0.nc")

#time options
Time = np.array(a.variables["time"])
dt = Time[1] - Time[0]
tstart = 38000
tstart_idx = np.searchsorted(Time,tstart)
tend = 39200
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(tstart_idx, tend_idx)
Time = Time[tstart_idx:tend_idx]
print(Time_steps)

#rotor data
p = a.groups["p_r"]; del a

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points


normal = 29

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

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

print("line 165",ys)
time.sleep(10)
print("line 167",zs)
time.sleep(10)

dy = (max(ys) - min(ys))/x
dz = (max(zs) - min(zs))/y
dA = dy * dz

print("line 174",dA)
time.sleep(5)

#velocity field
u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
del p

u[u<0]=0; v[v<0] #remove negative velocities

#fluctuating streamwise velocity
with Pool() as pool:
    u_pri = []
    for u_fluc_hvel_it in pool.imap(Fluc_Horizontal_velocity,Time_steps):
        
        u_pri.append(u_fluc_hvel_it)
        print(len(u_pri),time.time()-start_time)
u_pri = np.array(u_pri)

cmin = math.floor(np.min(u_pri))
cmax = math.ceil(np.max(u_pri))
print("line 184",cmin,cmax)

#streamwise velocity
with Pool() as pool:
    u_hvel = []
    for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u = np.array(u_hvel); del u_pri; del v

print("line 139",time.time()-start_time)

cmin = math.floor(np.min(u))
cmax = math.ceil(np.max(u))
print("line 184",cmin,cmax)

nlevs = int((cmax-cmin)/2)
if nlevs>abs(cmin) or nlevs>cmax:
    nlevs = min([abs(cmin),cmax])+1

levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
levels = np.concatenate((levs_min,levs_max[1:]))
print("line 153", levels)

time.sleep(5)

folder = out_dir+"Rotor_Plane_Fluctutating_horz_-63.0_3/"
isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)

it = 0
A_High_arr = []; A_Low_arr = []; A_Int_arr = []
Iy_High_arr = []; Iy_Low_arr = []; Iy_Int_arr = []
Iz_High_arr = []; Iz_Low_arr = []; Iz_Int_arr = []
Ux_High_arr = []; Ux_Low_arr = []; Ux_Int_arr = []
Iy_arr = []; Iz_arr = []
#with Pool() as pool:
    #for AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy_it,Iz_it,UxH_it,UxL_it,UxI_it in pool.imap(Update,Time_steps):        
    
print("time step",it)
AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy_it,Iz_it,UxH_it,UxL_it,UxI_it = Update(it)
A_High_arr.append(AH); A_Low_arr.append(AL); A_Int_arr.append(AI)
Iy_High_arr.append(IyH); Iy_Low_arr.append(IyL); Iy_Int_arr.append(IyI)
Iz_High_arr.append(IzH); Iz_Low_arr.append(IzL); Iz_Int_arr.append(IzI)
Ux_High_arr.append(UxH_it); Ux_Low_arr.append(UxL_it); Ux_Int_arr.append(UxI_it)
Iy_arr.append(Iy_it); Iz_arr.append(Iz_it)

print(AH)
print(AL)
print(AI)
print(IyH)
print(IyL)
print(IyI)
print(IzH)
print(IzL)
print(IzI)
print(Iy_it)
print(Iz_it)
print(UxH_it)
print(UxL_it)
print(UxI_it)

print("line 188",time.time()-start_time)
it+=1


ncfile = Dataset(out_dir+"Asymmetry_Dataset.nc",mode="w",format='NETCDF4')
ncfile.title = "Asymmetry data sampling output"

#create global dimensions
sampling_dim = ncfile.createDimension("sampling",None)

#create variables
Time_sampling = ncfile.createVariable("time", np.float64, ('sampling',),zlib=True)
Time_sampling[:] = np.array(Time)

Area_high = ncfile.createVariable("Area_high", np.float64, ('sampling',),zlib=True)
Area_low = ncfile.createVariable("Area_low", np.float64, ('sampling',),zlib=True)
Area_int = ncfile.createVariable("Area_int", np.float64, ('sampling',),zlib=True)

Iy_high = ncfile.createVariable("Iy_high", np.float64, ('sampling',),zlib=True)
Iy_low = ncfile.createVariable("Iy_low", np.float64, ('sampling',),zlib=True)
Iy_int = ncfile.createVariable("Iy_int", np.float64, ('sampling',),zlib=True)

Iz_high = ncfile.createVariable("Iz_high", np.float64, ('sampling',),zlib=True)
Iz_low = ncfile.createVariable("Iz_low", np.float64, ('sampling',),zlib=True)
Iz_int = ncfile.createVariable("Iz_int", np.float64, ('sampling',),zlib=True)

Ux_high = ncfile.createVariable("Ux_high", np.float64, ('sampling',),zlib=True)
Ux_low = ncfile.createVariable("Ux_low", np.float64, ('sampling',),zlib=True)
Ux_int = ncfile.createVariable("Ux_int", np.float64, ('sampling',),zlib=True)

Iy = ncfile.createVariable("Iy", np.float64, ('sampling',),zlib=True)
Iz = ncfile.createVariable("Iz", np.float64, ('sampling',),zlib=True)

Area_high[:] = np.array(A_High_arr); del A_High_arr
Area_low[:] = np.array(A_Low_arr); del A_Low_arr
Area_int[:] = np.array(A_Int_arr); del A_Int_arr

Iy_high[:] = np.array(Iy_High_arr); del Iy_High_arr
Iy_low[:] = np.array(Iy_Low_arr); del Iy_Low_arr
Iy_int[:] = np.array(Iy_Int_arr); del Iy_Int_arr

Iz_high[:] = np.array(Iz_High_arr); del Iz_High_arr
Iz_low[:] = np.array(Iz_Low_arr); del Iz_Low_arr
Iz_int[:] = np.array(Iz_Int_arr); del Iz_Int_arr

Ux_high[:] = np.array(Ux_High_arr); del Ux_High_arr
Ux_low[:] = np.array(Ux_Low_arr); del Ux_Low_arr
Ux_int[:] = np.array(Ux_Int_arr); del Ux_Int_arr

Iy[:] = np.array(Iy_arr); del Iy_arr
Iz[:] = np.array(Iz_arr); del Iz_arr

print(ncfile)
ncfile.close()