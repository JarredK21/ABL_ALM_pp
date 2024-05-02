from netCDF4 import Dataset
import numpy as np
import time
from multiprocessing import Pool
from scipy import interpolate
import math
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Circle

start_time = time.time()

x = 512; y = 512
U = np.ones((512*512))
xs = np.linspace(0,512,x)
ys = np.linspace(0,512,y)
xminidx = np.searchsorted(xs,245); xmaxidx = np.searchsorted(xs,252)
yminidx = np.searchsorted(ys,246); ymaxidx = np.searchsorted(ys,260)
Z = U.reshape(x,y)
X,Y = np.meshgrid(xs,ys)

fu = interpolate.interp2d(X[xminidx:xmaxidx,yminidx:ymaxidx],Y[xminidx:xmaxidx,yminidx:ymaxidx],Z[xminidx:xmaxidx,yminidx:ymaxidx])
xrotor = [246,247,248,249];yrotor = [246,247,248,249]
Zrotor = np.array(fu(xrotor,yrotor)[0])

print(Zrotor)

def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity():
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    mag_horz_vel = []
    mag_fluc_horz_vel = []
    for i in np.arange(0,len(ZS)):
        u_i = u[i*x:(i+1)*x]; v_i = v[i*x:(i+1)*x]
        if ZS[i] < h[0]:
            twist_h = f(h[0])
            ux_mean = f_ux(h[0])

        # elif ZS[i] > h[-1]:
        #     twist_h = f(h[-1])
        #     ux_mean = f_ux(h[-1])
        else:
            twist_h = f(ZS[i])
            ux_mean = f_ux(ZS[i])

        mag_horz_vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
        mag_fluc_horz_vel_i = np.subtract(mag_horz_vel_i,ux_mean)
        mag_horz_vel.extend(mag_horz_vel_i)
        mag_fluc_horz_vel.extend(mag_fluc_horz_vel_i)
    mag_horz_vel = np.array(mag_horz_vel)
    mag_fluc_horz_vel = np.array(mag_fluc_horz_vel)
    return mag_horz_vel,mag_fluc_horz_vel


def Update():

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)

    U = u
    U_pri = u_pri


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

                if U_pri_ijk > 0.7:
                    plt.plot(ys[ijk],zs[ijk],"ok")
                    AH+=dA
                    IyH+=(Uijk*k*dA)
                    IzH+=(Uijk*j*dA)
                    UxH.append(Uijk)
                elif U_pri_ijk < -0.7:
                    plt.plot(ys[ijk],zs[ijk],"+k")
                    AL+=dA
                    IyL+=(Uijk*k*dA)
                    IzL+=(Uijk*j*dA)
                    UxL.append(Uijk)
                else:
                    AI+=dA
                    IyI+=(Uijk*k*dA)
                    IzI+=(Uijk*j*dA)
                    UxI.append(Uijk)
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

    plt.xlim([ys[0],ys[-1]]);plt.ylim(zs[0],zs[-1])
    plt.tight_layout()
    plt.savefig("../../NREL_5MW_MCBL_R_CRPM_3/post_processing/test/test_H_{}.png".format(it))
    plt.close(fig)


    return AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy,Iz,UxH,UxL,UxI

start_time = time.time()

#defining twist angles with height from precursor
precursor = Dataset("../../ABL_precursor_2_restart/abl_statistics70000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38200)
t_end = np.searchsorted(precursor.variables["time"],39201)
u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v


print("line 67", time.time()-start_time)

#directories
in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
out_dir = in_dir



a = Dataset(in_dir+"sampling_r_-63.0_0.nc")

#time options
# Time = np.array(a.variables["time"])
# dt = Time[1] - Time[0]
# tstart = 38200
# tstart_idx = np.searchsorted(Time,tstart)
# tend = 38205
# tend_idx = np.searchsorted(Time,tend)
# Time_steps = np.arange(0, tend_idx-tstart_idx)
# Time = Time[tstart_idx:tend_idx]

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

ZS = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y)

dy = (max(ys) - min(ys))/x
dz = (max(zs) - min(zs))/y
dA = dy * dz

#velocity field
u = np.array(p.variables["velocityx"])
v = np.array(p.variables["velocityy"])
del p
u[u<0]=0
v[v<0]=0 #remove negative velocities

#fluctuating streamwise velocity
u_hvel,u_pri = Horizontal_velocity()
       

u_pri = np.array(u_pri)
u_hvel = np.array(u_hvel)


cmin = math.floor(np.min(u_pri))
cmax = math.ceil(np.max(u_pri))
print("line 244",cmin,cmax)


cmin = math.floor(np.min(u))
cmax = math.ceil(np.max(u))
print("line 249",cmin,cmax)

time.sleep(5)

it = 0
A_High_arr = []; A_Low_arr = []; A_Int_arr = []
Iy_High_arr = []; Iy_Low_arr = []; Iy_Int_arr = []
Iz_High_arr = []; Iz_Low_arr = []; Iz_Int_arr = []
Ux_High_arr = []; Ux_Low_arr = []; Ux_Int_arr = []
Iy_arr = []; Iz_arr = []      
    
print("time step",it)
AH,AL,AI,IyH,IyL,IyI,IzH,IzL,IzI,Iy_it,Iz_it,UxH_it,UxL_it,UxI_it = Update()
A_High_arr.append(AH); A_Low_arr.append(AL); A_Int_arr.append(AI)
Iy_High_arr.append(IyH); Iy_Low_arr.append(IyL); Iy_Int_arr.append(IyI)
Iz_High_arr.append(IzH); Iz_Low_arr.append(IzL); Iz_Int_arr.append(IzI)
Ux_High_arr.append(UxH_it); Ux_Low_arr.append(UxL_it); Ux_Int_arr.append(UxI_it)
Iy_arr.append(Iy_it); Iz_arr.append(Iz_it)

print("line 188",time.time()-start_time)
it+=1
