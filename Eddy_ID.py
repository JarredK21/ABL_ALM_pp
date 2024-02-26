from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
import time
from multiprocessing import Pool
from scipy import interpolate
from matplotlib.patches import Circle


def isInside(x, y):
     
    if ((x - 50) * (x - 50) +
        (y - 50) * (y - 50) <= 20 * 20):
        return True
    else:
        return False


class eddy:
    def __init__(self, Centroid, Area):
        self.number = np.nan
        self.Centroid = Centroid
        self.Area = Area


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
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 67", time.time()-start_time)

#directories
in_dir = "./"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)



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



#velocity field
u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
del p

u[u<0]=0; v[v<0] #remove negative velocities

u = np.subtract(u,np.mean(u))
v = np.subtract(v,np.mean(v))

with Pool() as pool:
    u_hvel = []
    for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_hvel.append(u_hvel_it)
        print(len(u_hvel),time.time()-start_time)
u = np.array(u_hvel); del u_hvel; del v

print("line 139",time.time()-start_time)

#find vmin and vmax for isocontour plots            
#min and max over data
cmin = math.floor(np.min(u))
cmax = math.ceil(np.max(u))


nlevs = int((cmax-cmin)/2)
if nlevs>abs(cmin) or nlevs>cmax:
    nlevs = min([abs(cmin),cmax])+1

levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
levels = np.concatenate((levs_min,levs_max[1:]))
print("line 153", levels)

#define thresholds with number of increments
levels_pos = np.linspace(0.7,cmax,4)
print("line 157", levels_pos)
levels_neg = np.linspace(cmin,-0.7,4)
print("line 159", levels_neg)


folder = out_dir+"Rotor_Plane_Fluctutating_horz_-63.0/"
isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)


def Update(it):

    U = u[it] #velocity time step it
    
    if it < 10:
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)



    u_plane = U.reshape(y,x)
    X,Y = np.meshgrid(ys,zs)

    Z = u_plane

    CS = plt.contour(X, Y, Z, levels=levels_pos)
    CZ = plt.contour(X,Y,Z, levels=levels_neg)

    T = round(Time[it],4)

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    cb = plt.colorbar(cs)

    #for +0.7m/s threshold
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []
    lines = CS.allsegs[0] #plot only threshold velocity
    for line in lines:
        X, Y = line[:,0], line[:,1]

        print(X,Y)
        #check if any point in line is inside circle
        cc = []
        for X_line, Y_line in zip(X,Y):
            cc.append(isInside(X_line,Y_line))

        print(cc)
        X_temp = np.copy(X); Y_temp = np.copy(Y); cc_temp = np.copy(cc)
        #if any point is inside cirlce plot #stop points outside of circle
        res = not any(cc)
        if res == False:     
            ix = 0
            for ic in np.arange(0,len(cc)-1):
                if cc[ic+1] != cc[ic]:

                    #equation of line intersecting circle
                    m = (Y[ic+1]-Y[ic])/(X[ic+1]-X[ic])
                    if m == np.inf or m ==-np.inf:
                        f = interpolate.interp1d([X[ic+1],X[ic]], [Y[ic+1],Y[ic]], fill_value='extrapolate')
                        c = float(f(0))
                        y_root = c
                    else:
                        f = interpolate.interp1d([X[ic+1],X[ic]], [Y[ic+1],Y[ic]], fill_value='extrapolate')
                        c = float(f(0))
                        y_roots = np.roots([(1+(m)**2), ((2*-2560)+(2*m*(c-90))), ((-2560)**2 + (c-90)**2 - 63**2)])
                        if y_roots[0] > np.min([X[ic], X[ic+1]]) and y_roots[0] < np.max([X[ic], X[ic+1]]):
                            y_root = y_roots[0]
                        else:
                            y_root = y_roots[1]
                        del y_roots

                    #z roots    
                    z_roots = np.roots([1, (2*-2560), (8100+(y_root-2560)**2 - 63**2)])
                    if z_roots[0] > np.min([Y[ic], Y[ic+1]]) and z_roots[0] < np.max([Y[ic], Y[ic+1]]):
                        z_root = z_roots[0]
                    else:
                        z_root = z_roots[1]
                    del z_roots

                    #insert x_root,y_root into X,Y and insert true at same index in cc
                    X_temp = np.insert(X_temp, ix+1, y_root); Y_temp = np.insert(Y_temp, ix+1, z_root); cc_temp = np.insert(cc_temp,ix+1,"True")

                    ix+=1 #add one for inserting
                ix+=1 #add one to increase index

        X = X_temp[cc_temp]; Y = Y_temp[cc_temp]; del X_temp; del Y_temp; del cc_temp
        print(X,Y)

        plt.plot(X, Y,"-k")

        if len(X) > 0:
            #calculate area and centroid of each contour
            np.append(X,X[-1]); np.append(Y,Y[-1])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)
            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]

            plt.plot(Centroid[0],Centroid[1],"ok",markersize=5)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

    Eddies_it_pos = {"Centroid_x_{}".format(it): Eddies_Cent_x, "Centroid_y_{}".format(it): Eddies_Cent_y, "Area_{}".format(it): Eddies_Area}

    #for -0.7m/s threshold
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []
    lines = CZ.allsegs[-1] #plot only threshold velocity
    for line in lines:
        X, Y = line[:,0], line[:,1]
        print(X,Y)

        #check if any point in line is inside circle
        cc = []
        for X_line, Y_line in zip(X,Y):
            cc.append(isInside(X_line,Y_line))
        print(cc)
        X_temp = np.copy(X); Y_temp = np.copy(Y); cc_temp = np.copy(cc)
        #if any point is inside cirlce plot #stop points outside of circle
        res = not any(cc)
        if res == False:     
            ix = 0
            for ic in np.arange(0,len(cc)-1):
                if cc[ic+1] != cc[ic]:

                    #equation of line intersecting circle
                    m = (Y[ic+1]-Y[ic])/(X[ic+1]-X[ic])
                    if m == np.inf or m ==-np.inf:
                        f = interpolate.interp1d([X[ic+1],X[ic]], [Y[ic+1],Y[ic]], fill_value='extrapolate')
                        c = float(f(0))
                        y_root = c
                    else:
                        f = interpolate.interp1d([X[ic+1],X[ic]], [Y[ic+1],Y[ic]], fill_value='extrapolate')
                        c = float(f(0))
                        y_roots = np.roots([(1+(m)**2), ((2*-2560)+(2*m*(c-90))), ((-2560)**2 + (c-90)**2 - 63**2)])
                        if y_roots[0] > np.min([X[ic], X[ic+1]]) and y_roots[0] < np.max([X[ic], X[ic+1]]):
                            y_root = y_roots[0]
                        else:
                            y_root = y_roots[1]
                        del y_roots

                    #z roots    
                    z_roots = np.roots([1, (2*-2560), (8100+(y_root-2560)**2 - 63**2)])
                    if z_roots[0] > np.min([Y[ic], Y[ic+1]]) and z_roots[0] < np.max([Y[ic], Y[ic+1]]):
                        z_root = z_roots[0]
                    else:
                        z_root = z_roots[1]
                    del z_roots

                    #insert x_root,y_root into X,Y and insert true at same index in cc
                    X_temp = np.insert(X_temp, ix+1, y_root); Y_temp = np.insert(Y_temp, ix+1, z_root); cc_temp = np.insert(cc_temp,ix+1,"True")

                    ix+=1 #add one for inserting
                ix+=1 #add one to increase index

        X = X_temp[cc_temp]; Y = Y_temp[cc_temp]; del X_temp; del Y_temp; del cc_temp
        print(X,Y)
        plt.plot(X, Y,"--k")

        if len(X) > 0:
            #calculate area and centroid of each contour
            np.append(X,X[-1]); np.append(Y,Y[-1])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)
            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]

            plt.plot(Centroid[0],Centroid[1],"ok",markersize=5)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

    Eddies_it_neg = {"Centroid_x_{}".format(it): Eddies_Cent_x, "Centroid_y_{}".format(it): Eddies_Cent_y, "Area_{}".format(it): Eddies_Area}


    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)


    plt.xlabel("y' axis (rotor frame of reference) [m]",fontsize=40)
    plt.ylabel("z' axis (rotor frame of reference) [m]",fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)


    #define titles and filenames for movie
    Title = "Rotor Plane. \nFluctuating horizontal velocity [m/s]: Offset = -63.0m, Time = {0}[s]".format(T)
    filename = "Rotor_Fluc_Horz_-63.0_{0}.png".format(Time_idx)
        

    plt.title(Title)
    plt.tight_layout()
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return Eddies_it_pos, Eddies_it_neg



it = 0
df_pos = pd.DataFrame(None)
df_neg = pd.DataFrame(None)
with Pool() as pool:
    for Eddies_pos, Eddies_neg in pool.imap(Update,Time_steps):

        df_0 = pd.DataFrame(Eddies_pos)
        df_pos = pd.concat([df_pos,df_0],axis=1); del df_0
        print(df_pos)

        df_0 = pd.DataFrame(Eddies_neg)
        df_neg = pd.concat([df_neg,df_0],axis=1); del df_0
        print(df_neg)

        it+=1
        print(it)

#saving data
df_pos.to_csv(in_dir+"Eddies_{}.csv".format(0.7))
df_neg.to_csv(in_dir+"Eddies_{}.csv".format(-0.7))
