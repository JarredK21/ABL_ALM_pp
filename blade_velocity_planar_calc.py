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

#blade asymmetry calc
df = Dataset("WTG01.nc")

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

a = Dataset("Dataset.nc")

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

offsets = ["-5.5"]
for offset in offsets:
    a = Dataset("sampling_r_{}.nc".format(offset))

    Time_sampling = np.array(a.variables["time"])
    Time_sampling = Time_sampling - Time_sampling[0]
    Time_steps = np.arange(0,len(Time_sampling)-1)

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
    v = np.array(p.variables["velocityy"])

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

    Iy = []; Iz = []
    ix=0
    with Pool() as pool:
        for Iy_it, Iz_it in pool.imap(blade_vel_calc,Time_steps):
            
            Iy = np.concatenate((Iy,Iy_it))
            Iz = np.concatenate((Iz,Iz_it))
            print(ix)
            ix+=1


    I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

    idx1 = np.searchsorted(Time_OF,200)
    T_end = np.searchsorted(Time,1199.6361)
    print(Time_OF,idx1)
    print(Time_OF[T_end])
    cc = round(correlation_coef(Iy[idx1+1:],IyB[:T_end]),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1+1:],Iy[idx1+1:],"-r",label="Blade asymmetry\nfrom planar data")
    plt.plot(Time_OF[idx1+1:T_end],IyB[:T_end],"-b",label="Blade asymmetry\nfrom actuator data")
    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry around y axis [$m^2/s$]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.grid()
    plt.tight_layout()
    plt.savefig("Iy_{}.png".format(offset))
    plt.close()

    cc = round(correlation_coef(Iz[idx1+1:],IzB[:T_end]),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1+1:],Iy[idx1+1:],"-r",label="Blade asymmetry\nfrom planar data")
    plt.plot(Time_OF[idx1+1:T_end],IzB[:T_end],"-b",label="Blade asymmetry\nfrom actuator data")
    plt.xlabel("Time [s]")
    plt.ylabel("Asymmetry around z axis [$m^2/s$]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.grid()
    plt.tight_layout()
    plt.savefig("Iz_{}.png".format(offset))
    plt.close()

    cc = round(correlation_coef(I[idx1+1:],IB[:T_end]),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF[idx1+1:],I[idx1+1:],"-r",label="Blade asymmetry\nfrom planar data")
    plt.plot(Time_OF[idx1+1:T_end],IB[:T_end],"-b",label="Blade asymmetry\nfrom actuator data")
    plt.xlabel("Time [s]")
    plt.ylabel("Magnitude Asymmetry [$m^2/s$]")
    plt.title("correlation coefficient = {}".format(cc))
    plt.grid()
    plt.tight_layout()
    plt.savefig("I_{}.png".format(offset))
    plt.close()



    


    