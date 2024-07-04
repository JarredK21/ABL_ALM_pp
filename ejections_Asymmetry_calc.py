from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import interpolate


def isInside(x, y):
     
    if ((x - 2560) * (x - 2560) +
        (y - 90) * (y - 90) < 63 * 63):
        return True
    else:
        return False
    

def Asymmetry_calc(it):

    H = Heights[it]
    y = []; z = []
    for j in np.arange(0,len(YS)):
        #is coordinate inside rotor disk
        cc = isInside(YS[j],H[j])

        if cc == True:

            z_i = np.min( np.roots([1,-180,(90**2-63**2+(YS[j]-2560)**2)]) )
            y.append(YS[j]); y.append(YS[j])
            z.append(z_i); z.append(H[j])


        #is coordinate above rotor disk so it is still covering it
        elif YS[j] > 2497 and YS[j] < 2623 and H[j] > 153:
            z_i = np.roots([1,-180,(90**2-63**2+(YS[j]-2560)**2)])
            y.append(YS[j]); y.append(YS[j])
            z.append(np.min(z_i)); z.append(np.max(z_i))

    if len(y) > 0:
        Iy,Iz = I_it_calc(it,y,z)
    else:
        Iy = 0; Iz = 0

    return Iy, Iz


def I_it_calc(it,y,z):

    Iy = 0
    Iz = 0
    ijk = 0
    for j,k in zip(ys,zs):
    
        if (ys[ijk]+rotor_coordiates[1]) in y:
            idx = y.index(ys[ijk]+rotor_coordiates[1])

            if z[idx] <= (zs[ijk]+rotor_coordiates[2]) <= z[idx+1]:
                Iy += u[it,ijk]*k*dA
                Iz += u[it,ijk]*j*dA
        ijk+=1

    return Iy,Iz



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

print("line 103")

a = Dataset("./sampling_r_-63.0.nc")

#time options
Time = np.array(a.variables["time"])
tstart = 38200
tstart_idx = np.searchsorted(Time,tstart)
tend = 39201
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

print("line 145")

#velocity field
u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

u[u<0]=0; v[v<0]=0 #remove negative velocities

with Pool() as pool:
    u_hvel = []
    for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        print(len(u_hvel))
u = np.array(u_hvel); del u_hvel; del v



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

dy = (max(Y) - min(Y))/x
dz = (max(Z) - min(Z))/y
dA = dy * dz

del p

print("line 189")

a = Dataset("Threshold_heights_Dataset.nc")

Time = np.array(a.variables["Time"])
Time_steps = np.arange(0,len(Time))
YS = np.array(a.variables["ys"])

Thresholds = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.4]

ncfile = Dataset("Threshold_Asymmetry_Dataset.nc",mode="w",format="NETCDF4")
ncfile.title = "Asymmeties at threshold data sampling output"

#create global dimensions
sampling_dim = ncfile.createDimension("sampling",None)

Time_sampling = ncfile.createVariable("Time", np.float64, ('sampling',),zlib=True)
Time_sampling[:] = Time

for threshold in Thresholds:

    group_c = ncfile.createGroup("{}".format(abs(threshold)))

    Iy_ejection = group_c.createVariable("Iy_ejection", np.float64, ('sampling'),zlib=True)
    Iz_ejection = group_c.createVariable("Iz_ejection", np.float64, ('sampling'),zlib=True)


    group = a.groups["{}".format(threshold)]

    Heights = np.array(group.variables["Height_ejection"])

    print(threshold)

    Iy_array = []
    Iz_array = []
    ix = 1
    with Pool() as pool:
        for Iy_it,Iz_it in pool.imap(Asymmetry_calc,Time_steps):

            Iy_array.append(Iy_it); Iz_array.append(Iz_it)

            print(ix)

            ix+=1
    
    Iy_ejection[:] = np.array(Iy_array); del Iy_array
    Iz_ejection[:] = np.array(Iz_array); del Iz_array

    print(ncfile.groups)


print(ncfile)
ncfile.close()


