from netCDF4 import Dataset
import numpy as np
import time
from multiprocessing import Pool
from scipy import interpolate


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    mag_horz_vel = []; mag_fluc_horz_vel = []
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
        mag_fluc_vel_i = np.subtract(mag_fluc_vel_i,ux_mean)
        mag_horz_vel.extend(mag_horz_vel_i)
        mag_fluc_horz_vel.extend(mag_fluc_vel_i)
    mag_horz_vel = np.array(mag_horz_vel)
    mag_fluc_horz_vel = np.array(mag_fluc_horz_vel)
    return mag_horz_vel, mag_fluc_horz_vel


def Update(it):

    AH = 0; AL = 0; AI = 0
    IyH = 0; IyL = 0; IyI = 0; Iy = 0
    IzH = 0; IzL = 0; IzI = 0; Iz = 0
    UxH = []; UxL = []; UxI = []
    ijk = 0
    for j,k in zip(ys,zs):
        r = np.sqrt(j**2 + k**2)

        if r <= 63 and r > 1.5:
            Iy+=(u[it,ijk]*k*dA)
            Iz+=(u[it,ijk]*j*dA)

            u_pri_ijk = u_pri[it,ijk]
            print(u_pri_ijk)

            if u_pri_ijk >= 0.7:
                AH+=dA
                IyH+=(u[it,ijk]*k*dA)
                IzH+=(u[it,ijk]*j*dA)
                UxH.append(u[it,ijk])
            elif u_pri_ijk <= -0.7:
                AL+=dA
                IyL+=(u[it,ijk]*k*dA)
                IzL+=(u[it,ijk]*j*dA)
                UxL.append(u[it,ijk])
            else:
                AI+=dA
                IyI+=(u[it,ijk]*k*dA)
                IzI+=(u[it,ijk]*j*dA)
                UxI.append(u[it,ijk])
        print(AH)
        print(AL)
        print(AI)
        print(IyH)
        print(IyL)
        print(IyI)
        print(IzH)
        print(IzL)
        print(IzI)
        print(Iy)
        print(Iz)
        ijk+=1
        print(ijk)

    return AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy,Iz,np.average(UxH), np.average(UxL), np.average(UxI)



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
ux_mean_profile = u * np.cos(np.radians(29)) + v * np.sin(np.radians(29))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v


print("line 67", time.time()-start_time)

#directories
in_dir = "./"
out_dir = in_dir



a = Dataset("./sampling_r_-63.0.nc")

#time options
Time = np.array(a.variables["time"])
dt = Time[1] - Time[0]
tstart = 38000
tstart_idx = np.searchsorted(Time,tstart)
tend = 39200
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]


#rotor data
p = a.groups["p_r"]; del a

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points


normal = 29

#define plotting axes
coordinates = np.array(p.variables["coordinates"])


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

dy = ys[1] - ys[0]
dz = zs[1] - zs[0]
dA= dy*dz

#velocity field
u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
del p

u[u<0]=0; v[v<0] #remove negative velocities

u_hvel = []; u_pri = []
with Pool() as pool:
    for u_hvel_it,u_fluc_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        u_pri.append(u_fluc_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u = np.array(u_hvel); del u_hvel; del v
u_pri = np.array(u_pri)

print("line 139",time.time()-start_time)


# ncfile = Dataset(out_dir+"Asymmetry_Dataset.nc",mode="w",format='NETCDF4')
# ncfile.title = "Asymmetry data sampling output"

# #create global dimensions
# sampling_dim = ncfile.createDimension("sampling",None)

# #create variables
# Time_sampling = ncfile.createVariable("time", np.float64, ('sampling',),zlib=True)
# Time_sampling[:] = np.array(Time)

# Area_high = ncfile.createVariable("Area_high", np.float64, ('sampling',),zlib=True)
# Area_low = ncfile.createVariable("Area_low", np.float64, ('sampling',),zlib=True)
# Area_int = ncfile.createVariable("Area_int", np.float64, ('sampling',),zlib=True)

# Iy_high = ncfile.createVariable("Iy_high", np.float64, ('sampling',),zlib=True)
# Iy_low = ncfile.createVariable("Iy_low", np.float64, ('sampling',),zlib=True)
# Iy_int = ncfile.createVariable("Iy_int", np.float64, ('sampling',),zlib=True)

# Iz_high = ncfile.createVariable("Iz_high", np.float64, ('sampling',),zlib=True)
# Iz_low = ncfile.createVariable("Iz_low", np.float64, ('sampling',),zlib=True)
# Iz_int = ncfile.createVariable("Iz_int", np.float64, ('sampling',),zlib=True)

# Ux_high = ncfile.createVariable("Ux_high", np.float64, ('sampling',),zlib=True)
# Ux_low = ncfile.createVariable("Ux_low", np.float64, ('sampling',),zlib=True)
# Ux_int = ncfile.createVariable("Ux_int", np.float64, ('sampling',),zlib=True)

# Iy = ncfile.createVariable("Iy", np.float64, ('sampling',),zlib=True)
# Iz = ncfile.createVariable("Iz", np.float64, ('sampling',),zlib=True)

# #it = 0
# Time_steps = [0,1]
# A_High_arr = []; A_Low_arr = []; A_Int_arr = []
# Iy_High_arr = []; Iy_Low_arr = []; Iy_Int_arr = []
# Iz_High_arr = []; Iz_Low_arr = []; Iz_Int_arr = []
# Ux_High_arr = []; Ux_Low_arr = []; Ux_Int_arr = []
# Iy_arr = []; Iz_arr = []
# # with Pool() as pool:
# #     for AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy_it,Iz_it,UxH_it,UxL_it,UxI_it in pool.imap(Update,Time_steps):

# for it in Time_steps:
        

#     AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy_it,Iz_it,UxH_it,UxL_it,UxI_it = Update(it)
#     A_High_arr.append(AH); A_Low_arr.append(AL); A_Int_arr.append(AI)
#     Iy_High_arr.append(IyH); Iy_Low_arr.append(IyL); Iy_Int_arr.append(IyI)
#     Iz_High_arr.append(IyH); Iz_Low_arr.append(IzL); Iz_Int_arr.append(IzI)
#     Ux_High_arr.append(UxH_it); Ux_Low_arr.append(UxL_it); Ux_Int_arr.append(UxI_it)
#     Iy_arr.append(Iy_it); Iz_arr.append(Iz_it)

#     print(it)
#         #it+=1

# Area_high[:] = np.array(A_High_arr); del A_High_arr
# Area_low[:] = np.array(A_Low_arr); del A_Low_arr
# Area_int[:] = np.array(A_Int_arr); del A_Int_arr

# Iy_high[:] = np.array(Iy_High_arr); del Iy_High_arr
# Iy_low[:] = np.array(Iy_Low_arr); del Iy_Low_arr
# Iy_int[:] = np.array(Iy_Int_arr); del Iy_Int_arr

# Iz_high[:] = np.array(Iz_High_arr); del Iz_High_arr
# Iz_low[:] = np.array(Iz_Low_arr); del Iz_Low_arr
# Iz_int[:] = np.array(Iz_Int_arr); del Iz_Int_arr

# Ux_high[:] = np.array(Ux_High_arr); del Ux_High_arr
# Ux_low[:] = np.array(Ux_Low_arr); del Ux_Low_arr
# Ux_int[:] = np.array(Ux_Int_arr); del Ux_Int_arr

# Iy[:] = np.array(Iy_arr); del Iy_arr
# Iz[:] = np.array(Iz_arr); del Iz_arr

# print(ncfile)
# ncfile.close()