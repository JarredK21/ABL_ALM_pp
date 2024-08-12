from netCDF4 import Dataset
import numpy as np
from scipy import interpolate
from multiprocessing import Pool
import time
import matplotlib.pyplot as plt


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


def blade_vel_calc(it):

    U = u[it]

    u_plane = U.reshape(y,x)

    f = interpolate.interp2d(ys,zs,u_plane,kind="linear")

    Iy_inner_it = []; Iz_inner_it = []
    for it_inner in np.arange(it*100,(it+1)*100):

        R = np.linspace(0,63,300)

        Az = -Azimuth[it_inner]
        Y = []; Z = []

        Y = np.concatenate((Y,2560+R*np.sin(Az)))
        Z = np.concatenate((Z,90+R*np.cos(Az)))

        Az2 = Az-(2*np.pi)/3
        if Az2 < -2*np.pi:
            Az2 += (2*np.pi)

        Az3 = Az-(4*np.pi)/3
        if Az2 < -2*np.pi:
            Az2 += (2*np.pi)

        Y = np.concatenate((Y,2560+R*np.sin(Az2)))
        Z = np.concatenate((Z,90+R*np.cos(Az2)))

        Y = np.concatenate((Y,2560+R*np.sin(Az3)))
        Z = np.concatenate((Z,90+R*np.cos(Az3)))

        u_interp = []
        for i,j in zip(Y,Z):

            u_interp.append(f(i,j)[0])
        
        IyB1 = u_interp[0:300]*R*np.cos(Az)
        IyB1 = np.sum(IyB1)
        IzB1 = u_interp[0:300]*R*np.sin(Az)
        IzB1 = np.sum(IzB1)


        IyB2 = u_interp[300:600]*R*np.cos(Az2)
        IzB2 = u_interp[300:600]*R*np.sin(Az2)
        IyB2 = np.sum(IyB2)
        IzB2 = np.sum(IzB2)


        IyB3 = u_interp[600:900]*R*np.cos(Az3)
        IzB3 = u_interp[600:900]*R*np.sin(Az3)
        IyB3 = np.sum(IyB3)
        IzB3 = np.sum(IzB3)

        Iy_inner_it.append(IyB1+IyB2+IyB3)
        Iz_inner_it.append(IzB1+IzB2+IzB3)

    return Iy_inner_it, Iz_inner_it



start_time = time.time()

#create netcdf file
ncfile = Dataset("Dataset_Planar_asymmetry.nc",mode="w",format='NETCDF4')
ncfile.title = "Outputing blade asymmetry from planar data"

#create global dimensions
OF_dim = ncfile.createDimension("OF",None)

#create variables
time_OF = ncfile.createVariable("Time_OF", np.float64, ('OF',),zlib=True)

print("Outputting planar asymmetry variables",time.time()-start_time)

group = ncfile.createGroup("Planar_Asymmetry_Variables")

a = Dataset("Dataset.nc")

Time_OF = np.array(a.variables["Time_OF"])

OF_vars = a.groups["OpenFAST_Variables"]
Azimuth = np.radians(OF_vars.variables["Azimuth"])

print("line 15",time.time()-start_time)

#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics76000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38200)
u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 126", time.time()-start_time)

offsets = ["-5.5","-63.0"]
for offset in offsets:

    if offset == "-63.0":
        Azimuth = Azimuth+np.radians(334)
        
    a = Dataset("sampling_r_{}.nc".format(offset))

    Time_sampling = np.array(a.variables["time"])
    Time_sampling = Time_sampling - Time_sampling[0]
    Time_steps = np.arange(0,len(Time_sampling)-1)

    p = a.groups["p_r"]; del a

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
    v = np.array(p.variables["velocityy"])
    del p

    u[u<0]=0; v[v<0]=0 #remove negative velocities
    
    u_hvel = []
    ix=0
    with Pool() as pool:
        for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
            
            u_hvel.append(u_hvel_it)
            print(ix,time.time()-start_time)
            ix+=1
    u = np.array(u_hvel); del u_hvel; del v


    print("line 328",time.time()-start_time)

    Iy_array = []; Iz_array = []
    ix=0
    with Pool() as pool:
        for Iy_it, Iz_it in pool.imap(blade_vel_calc,Time_steps):
            
            Iy_array = np.concatenate((Iy_array,Iy_it))
            Iz_array = np.concatenate((Iz_array,Iz_it))
            print(ix)
            ix+=1


    
    group_inner = group.createGroup("{}".format(abs(offset)))

    Iy = group_inner.createVariable("Iy", np.float64, ('OF',),zlib=True)
    Iz = group_inner.createVariable("Iz", np.float64, ('OF',),zlib=True)

    Iy[:] = Iy_array; del Iy_array
    Iz[:] = Iz_array; del Iz_array

    del u

    print(ncfile.groups)

print(ncfile)
ncfile.close()