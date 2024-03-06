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
from math import ceil


def isInside(x, y):
     
    if ((x - 2560) * (x - 2560) +
        (y - 90) * (y - 90) <= 63 * 63):
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


def openContour(cc,X,Y):
    crossing = []
    if any(cc) == False and all(cc) == False: #if all points are false skip loop in array
        return "skip", X, Y, cc, crossing
    elif any(cc) == True and all(cc) == True:
        return "closed", X, Y, cc, crossing
    else:
        #if there are points inside and outside remove points outside while interpolating between points to find points on edge of rotor
        X_temp = np.copy(X); Y_temp = np.copy(Y); cc_temp = np.copy(cc)
        ix = 0
        for i in np.arange(0,len(cc)-1):

            if cc[i+1] != cc[i]: #if next point is not the same as current (True -> False) or (False -> True) find point inbetween on rotor edge

                #equation of line intersecting circle
                m = (Y[i+1]-Y[i])/(X[i+1]-X[i])
                if m == np.inf or m ==-np.inf:
                    f = interpolate.interp1d([X[i+1],X[i]], [Y[i+1],Y[i]], fill_value='extrapolate')
                    c = float(f(0))
                    y_root = c
                else:
                    f = interpolate.interp1d([X[i+1],X[i]], [Y[i+1],Y[i]], fill_value='extrapolate')
                    c = float(f(0))
                    y_roots = np.roots([(1+(m)**2), ((2*-2560)+(2*m*(c-90))), ((-2560)**2 + (c-90)**2 - 63**2)])
                    if y_roots[0] > np.min([X[i], X[i+1]]) and y_roots[0] < np.max([X[i], X[i+1]]):
                        y_root = y_roots[0]
                    else:
                        y_root = y_roots[1]
                    del y_roots

                #z roots    
                z_roots = np.roots([1, (2*-90), (90**2+(y_root-2560)**2 - 63**2)])
                if z_roots[0] > np.min([Y[i], Y[i+1]]) and z_roots[0] < np.max([Y[i], Y[i+1]]):
                    z_root = z_roots[0]
                else:
                    z_root = z_roots[1]
                del z_roots

                #insert x_root,y_root into temporary X,Y
                X_temp = np.insert(X_temp,ix+1,y_root); Y_temp = np.insert(Y_temp, ix+1,z_root); cc_temp = np.insert(cc_temp, ix+1, True)
                if cc[i] == True and cc[i+1] == False:
                    crossing.append(["inOut", ix+1])
                elif cc[i] == False and cc[i+1] == True:
                    crossing.append(["outIn", ix+1])

                ix+=1 #add one for inserting new value
            ix+=1 #add one in for loop

        return "open", X_temp, Y_temp, cc_temp, crossing


def closeContour(X, Y, cc):

    X_contour = []; Y_contour = []
    ix = np.nan; iy = np.nan
    for i in np.arange(0,len(cc)-1):
        if cc[i] == True:
            X_contour.append(X[i]); Y_contour.append(Y[i])
        
        if i < len(cc)-1 and cc[i] != cc[i+1]:
            ix = i
            print(ix)

        if ix != np.nan and i != ix and cc[i] != cc[i+1]:
            iy = i
            print(iy)
            theta_0 = np.arctan2(Y[ix], X[ix])
            theta_2 = np.arctan2(Y[iy], X[iy])
            print(theta_0,theta_2)
            theta_arc = np.arange(theta_0,theta_2,5e-03)

            for theta in theta_arc:
                if theta < 0.0 and theta >= -np.pi/2: #bottom right quadrant
                    r = -63; theta = np.pi/2 - theta
                    x_i = 2560 - r*np.sin(theta)
                    y_i = 90 + r*np.cos(theta)
                elif theta < 0.0 and theta < -np.pi/2: #bottom left quadrant
                    r = -63; theta = theta - np.pi/2
                    x_i = 2560 + r*np.sin(theta)
                    y_i = 90 + r*np.cos(theta)
                elif theta >= 0.0 and theta <= np.pi/2: #top right quadrant
                    r = 63; theta = np.pi/2 - theta
                    x_i = 2560 + r*np.sin(theta)
                    y_i = 63 + r*np.cos(theta)
                elif theta >= 0.0 and theta >= np.pi/2: #top left quadrant
                    r = 63; theta = theta - np.pi/2
                    x_i = 2560 - r*np.sin(theta)
                    y_i = 2560 + r*np.cos(theta)

                X_contour.append(x_i); Y_contour.append(y_i)

            ix = np.nan; iy = np.nan

    return X_contour, Y_contour




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

csv_out_dir = in_dir + "csv_files/"
isExist = os.path.exists(csv_out_dir)
if isExist == False:
    os.makedirs(csv_out_dir)



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


folder = out_dir+"Rotor_Plane_Fluctutating_horz_-63.0_2/"
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

        #check if any point in line is inside circle
        cc = []
        for X_line, Y_line in zip(X, Y):
            cc.append(isInside(X_line,Y_line))

        #separate line into N contours if line is outside rotor disk
        C, X, Y,cc,crossings = openContour(cc,X,Y)
        #all points are outside of rotor disk
        if C == "skip":
            continue
        else:
            if C == "open":
                print(cc,crossings)
                for crossing in crossings[1:]:
                    if crossing[0] == "inOut":
                        Bx = X[:crossing[1]-1]
                        Ax = X[crossing[1]:]
                        By = Y[:crossing[1]-1]
                        Ay = Y[crossing[1]:]
                        Bcc = cc[:crossing[1]-1]
                        Acc = cc[crossing[1]:]

                        X = np.concatenate((Ax,Bx))
                        Y = np.concatenate((Ay,By))
                        cc = np.concatenate((Acc,Bcc))

                        break

                print(cc)
                print(len(X),len(Y))
                X,Y = closeContour(X,Y,cc)
                print(len(X),len(Y))

            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            X = np.append(X,X[0]); Y = np.append(Y,Y[0])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

            plt.plot(X,Y,"-k",linewidth=4)
            plt.plot(Centroid[0],Centroid[1],"ok",markersize=4)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

    Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area}

    #for -0.7m/s threshold
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []
    lines = CZ.allsegs[-1] #plot only threshold velocity
    for line in lines:
        X, Y = line[:,0], line[:,1]

        #check if any point in line is inside circle
        cc = []
        for X_line, Y_line in zip(X, Y):
            cc.append(isInside(X_line,Y_line))
        #separate line into N contours if line is outside rotor disk
        C, X, Y,cc,crossings = openContour(cc,X,Y)
        #all points are outside of rotor disk
        if C == "skip":
            continue
        else:
            if C == "open":
                print(cc,crossings)
                for crossing in crossings[1:]:
                    if crossing[0] == "inOut":
                        Bx = X[:crossing[1]-1]
                        Ax = X[crossing[1]:]
                        By = Y[:crossing[1]-1]
                        Ay = Y[crossing[1]:]
                        Bcc = cc[:crossing[1]-1]
                        Acc = cc[crossing[1]:]

                        X = np.concatenate((Ax,Bx))
                        Y = np.concatenate((Ay,By))
                        cc = np.concatenate((Acc,Bcc))

                        break

                print(cc)
                print(len(X),len(Y))
                X,Y = closeContour(X,Y,cc)
                print(len(X), len(Y))

            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            X = np.append(X,X[0]); Y = np.append(Y,Y[0])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)
            
            plt.plot(X,Y,"--k",linewidth=4)
            plt.plot(Centroid[0],Centroid[1],"ok",markersize=4)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

    Eddies_it_neg = {"Centroid_x_neg": Eddies_Cent_x, "Centroid_y_neg": Eddies_Cent_y, "Area_neg": Eddies_Area}


    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)


    plt.xlabel("y' axis (rotor frame of reference) [m]",fontsize=40)
    plt.ylabel("z' axis (rotor frame of reference) [m]",fontsize=40)
    plt.xticks(fontsize=40)
    plt.yticks(fontsize=40)


    #define titles and filenames for movie
    Title = "Rotor Plane. \nFluctuating horizontal velocity [m/s]: Offset = -63.0m, Time = {0}[s]".format(T)
    filename = "Rotor_Fluc_Horz_-63.0_{0}.png".format(Time_idx)
    
    x_c = [-1,-10]; y_c = [-1,-10]
    plt.plot(x_c,y_c,"-k",label="0.7 m/s")
    plt.plot(x_c,y_c,"--k",label="-0.7 m/s")

    plt.xlim([ys[0],ys[-1]]);plt.ylim(zs[0],zs[-1])
    plt.legend()
    plt.title(Title)
    plt.tight_layout()
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return Eddies_it_pos, Eddies_it_neg



it = 0
Time_steps = [0]
#with Pool() as pool:
    #for Eddies_pos, Eddies_neg in pool.imap(Update,Time_steps):

for it in Time_steps:
    Eddies_pos, Eddies_neg = Update(it)        

    df = pd.DataFrame(None)

    df_pos = pd.DataFrame(Eddies_pos)
    df = pd.concat([df,df_pos],axis=1); del df_pos

    df_neg = pd.DataFrame(Eddies_neg)
    df = pd.concat([df,df_neg],axis=1); del df_neg

    df.to_csv(csv_out_dir+"Eddies_0.7_{}.csv".format(it))
    print(df)
    del df

    it+=1
    print(it)

#saving data
df_pos.to_csv(in_dir+"Eddies_{}.csv".format(0.7))
df_neg.to_csv(in_dir+"Eddies_{}.csv".format(-0.7))
