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


def isInside(x, y):
     
    if ((x - 2560) * (x - 2560) +
        (y - 90) * (y - 90) < 63 * 63):
        return True
    else:
        return False


def openContour(cc,X,Y):
    crossings = []
    if any(cc) == False and all(cc) == False: #if all points are false skip loop in array
        return "skip", X, Y, cc, crossings
    elif any(cc) == True and all(cc) == True:
        return "closed", X, Y, cc, crossings
    else:
        #if there are points inside and outside remove points outside while interpolating between points to find points on edge of rotor
        X_temp = np.copy(X); Y_temp = np.copy(Y); cc_temp = np.copy(cc)
        ix = 0
        for i in np.arange(0,len(cc)-1):

            if cc[i+1] != cc[i]: #if next point is not the same as current (True -> False) or (False -> True) find point inbetween on rotor edge
                crossings.append(ix+1)
                #equation of line intersecting circle
                m = (Y[i+1]-Y[i])/(X[i+1]-X[i])
                if m == np.inf or m ==-np.inf:
                    f = interpolate.interp1d([Y[i+1],Y[i]],[X[i+1],X[i]], fill_value='extrapolate')
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

                ix+=1 #add one for inserting new value
            ix+=1 #add one in for loop

        return "open", X_temp, Y_temp, cc_temp, crossings


def ux_closed(Centroid,Xs,Ys,Z):
    #need better way
    xmin = Centroid[0]-1; xmax = Centroid[0]+1
    ymin = Centroid[1]-1; ymax = Centroid[1]+1

    xmin_idx = np.searchsorted(ys,xmin,side="left"); xmax_idx = np.searchsorted(ys,xmax,side="right")
    ymin_idx = np.searchsorted(zs,ymin,side="left"); ymax_idx = np.searchsorted(zs,ymax,side="right")

    f_ux = interpolate.interp2d(Xs[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Ys[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Z[ymin_idx:ymax_idx,xmin_idx:xmax_idx]) #make more efficient??

    ux_closed = f_ux(Centroid[0],Centroid[1])

    return ux_closed


def ux_interp(type,theta_loc,theta_order,Xs,Ys,Z,dtheta):

    theta_anti = theta_loc[1] + dtheta

    if theta_anti > 2*np.pi:
        theta_anti-=2*np.pi


    if round(theta_anti,2) >= round(theta_order[2],2):
        
        theta_anti = theta_loc[1] + abs(theta_order[2] - theta_order[1]) / 2

        print("anti dtheta",abs(theta_order[2] - theta_order[1]) / 2)

        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi

    print("anti dtheta",dtheta)

    if type == 2:

        theta_clock = theta_loc[1] - dtheta

        if theta_clock < 0:
            theta_clock +=2*np.pi

        if round(theta_clock,2) <= round(theta_order[0],2):
            
            theta_clock = theta_loc[1] - abs(theta_order[1] - theta_order[0]) / 2

            print("clock dtheta", abs(theta_order[1] - theta_order[0]) / 2)

            if theta_clock < 0:
                theta_clock +=2*np.pi
        
    else:
        theta_clock = theta_loc[1] - dtheta

        if theta_clock < 0:
            theta_clock +=2*np.pi


    print("clock dtheta", dtheta)

    r = 63
    x_anti = 2560 + r*np.cos(theta_anti)
    y_anti = 90 + r*np.sin(theta_anti)

    x_clock = 2560 + r*np.cos(theta_clock)
    y_clock = 90 + r*np.sin(theta_clock)

    xmin = np.min([x_anti, x_clock])-1; xmax = np.max([x_anti, x_clock])+1
    ymin = np.min([y_anti, y_clock])-1; ymax = np.max([y_anti, y_clock])+1

    xmin_idx = np.searchsorted(ys,xmin,side="left"); xmax_idx = np.searchsorted(ys,xmax,side="right")
    ymin_idx = np.searchsorted(zs,ymin,side="left"); ymax_idx = np.searchsorted(zs,ymax,side="right")

    f_ux = interpolate.interp2d(Xs[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Ys[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Z[ymin_idx:ymax_idx,xmin_idx:xmax_idx]) #make more efficient??

    ux_anti = f_ux(x_anti,y_anti)

    ux_clock = f_ux(x_clock,y_clock)

    print(theta_anti,theta_clock,ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock)

    return ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock



def isOutside(type,theta_loc,theta_order,Xs,Ys,Z,threshold):

    dtheta_arr = np.radians([2,4,6,8,10,12,14,16,18,20,24,26])

    for dtheta in dtheta_arr:
        ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock = ux_interp(type,theta_loc,theta_order,Xs,Ys,Z,dtheta)
        if threshold > 0.0:
            if ux_anti >= threshold and ux_clock >= threshold:
                continue
            if ux_anti >= threshold or ux_clock >= threshold:
                break
        elif threshold < 0.0:
            if ux_anti<=threshold and ux_clock<=threshold:
                continue
            if ux_anti <= threshold or ux_clock<= threshold:
                break


    if threshold > 0.0:
        plt.plot(x_anti,y_anti,"or",markersize=6)
        plt.plot(x_clock,y_clock,"or",markersize=6)

        if ux_anti > threshold:
            direction = "anticlockwise"
            Atheta = theta_order[2]
            
        elif ux_clock > threshold:
            direction = "clockwise"
            Atheta = theta_order[0]

        else:
            direction = "nan"
            Atheta = "skip"
            
    elif threshold < 0.0:
        plt.plot(x_anti,y_anti,"ob",markersize=6)
        plt.plot(x_clock,y_clock,"ob",markersize=6)

        if ux_clock < threshold:
            direction = "clockwise"
            Atheta = theta_order[0]
            
        elif ux_anti < threshold:
            direction = "anticlockwise"
            Atheta = theta_order[2]

        else:
            direction = "nan"
            Atheta = "skip"

    if Atheta < 0:
        Atheta+=2*np.pi
    elif Atheta >= 2*np.pi:
        Atheta-=2*np.pi

    print("direction",direction)
    return Atheta,direction


def closeContour(theta_180,theta_loc,theta_order,Xs,Ys,Z, X, Y,threshold):

    if len(theta_loc) > 3:
        type = 2
    else:
        type = 1

    Xcontours = []; Ycontours = []
    Xcontour = []; Ycontour = []   

    if type == 1:
        theta_order.append(theta_order[0]+2*np.pi)

        print("crossing", theta_loc[1])
        Atheta,direction = isOutside(type,theta_loc,theta_order,Xs,Ys,Z,threshold)
        print("Atheta",Atheta)

        Xcontour = np.concatenate((Xcontour,X[0])) #plot A->B
        Ycontour = np.concatenate((Ycontour,Y[0])) #plot A->B

        #check this part not working all the time
        if direction == "anticlockwise":
            if theta_loc[1] < theta_loc[0]:
            
                theta_AB = np.linspace(theta_loc[1],theta_loc[0],int(abs(theta_180[1]-theta_180[0])/5e-03))
            elif theta_loc[1] > theta_loc[0]:
                theta_AB1 = np.linspace(theta_180[1],0,int(abs(theta_180[1])/5e-03))
                theta_AB2 = np.linspace(0,theta_loc[0],int(theta_loc[0]/5e-03))
                theta_AB = np.concatenate((theta_AB1,theta_AB2))
        elif direction == "clockwise":
            if theta_loc[1] > theta_loc[0]:
                theta_AB = np.linspace(theta_loc[1],theta_loc[0],int(abs(theta_180[1]-theta_180[0])/5e-03))
            elif theta_loc[1] < theta_loc[0]:
                theta_AB1 = np.linspace(theta_loc[1],0,int(abs(theta_loc[1])/5e-03))
                theta_AB2 = np.linspace(0,theta_180[0],int(theta_180[0]/5e-03))
                theta_AB = np.concatenate((theta_AB1,theta_AB2))

        for i in np.arange(0,len(theta_AB)):
            if theta_AB[i] < 0:
                theta_AB[i]+=2*np.pi
            elif theta_AB[i] >= 2*np.pi:
                theta_AB[i]-=2*np.pi

        print("theta arc",theta_AB)

        r = 63
        Xarc = np.add(r*np.cos(theta_AB), 2560); Yarc = np.add(r*np.sin(theta_AB), 90)
        Xcontour = np.concatenate((Xcontour,Xarc))
        Ycontour = np.concatenate((Ycontour,Yarc))

        Xcontours.append(Xcontour); Ycontours.append(Ycontour)

    else:
        for i in np.arange(0,len(theta_loc)-1,2):
            theta = theta_loc[i+1] #theta B
            Bidx = theta_order.index(theta)
            print("crossing", theta)

            if Bidx == 0:
                theta_O = [theta_order[Bidx-1]-2*np.pi,theta_order[Bidx],theta_order[Bidx+1]]
            elif Bidx == len(theta_order)-1:
                theta_O = [theta_order[Bidx-1],theta_order[Bidx],theta_order[0]+2*np.pi]
            else:
                theta_O = theta_order[Bidx-1:Bidx+2]

            theta_L = theta_loc[i:i+3]
            print("theta_loc",theta_L)
            print("theta_order",theta_O)
            
            Atheta,direction = isOutside(type,theta_L,theta_O,Xs,Ys,Z,threshold)
            print("Atheta",Atheta)

            #this should be needed
            if Atheta == "skip":
                continue

            idx = int(i/2)
            Xcontour = np.concatenate((Xcontour,X[idx])) #plot A->B
            Ycontour = np.concatenate((Ycontour,Y[idx])) #plot A->B

            theta_AB = np.linspace(theta_loc[i+1],Atheta,int(abs(theta_loc[i+1]-Atheta)/5e-03))
            print("theta_arc",theta_AB)

            r = 63
            Xarc = np.add(r*np.cos(theta_AB), 2560); Yarc = np.add(r*np.sin(theta_AB), 90)
            Xcontour = np.concatenate((Xcontour,Xarc))
            Ycontour = np.concatenate((Ycontour,Yarc))

            if Atheta != theta_loc[i+2]:
                Xcontours.append(Xcontour); Ycontours.append(Ycontour)
                Xcontour = []; Ycontour = []

        if len(Xcontours) == 0:
            Xcontours.append(Xcontour); Ycontours.append(Ycontour)


    return Xcontours, Ycontours


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
    Xs,Ys = np.meshgrid(ys,zs)

    Z = u_plane

    CS = plt.contour(Xs, Ys, Z, levels=levels_pos)
    CZ = plt.contour(Xs,Ys,Z, levels=levels_neg)

    T = round(Time[it],4)

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = ax.contourf(Xs,Ys,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    cb = plt.colorbar(cs)

    print("positive contours")
    #for +0.7m/s threshold
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []
    lines = CS.allsegs[0] #plot only threshold velocity

    theta_180 = []
    theta_loc = []
    X_arr = []; Y_arr = []
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
        elif C == "closed":
            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            X = np.append(X,X[0]); Y = np.append(Y,Y[0])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

            ux_c = ux_closed(Centroid,Xs,Ys,Z)

            if ux_c >= 0.7:
                plt.plot(X,Y,"-k",linewidth=3)
            elif ux_c <= -0.7:
                plt.plot(X,Y,"--k",linewidth=3)
            else:
                plt.plot(X,Y,"-.k",linewidth=3)

            plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

        elif C == "open":

            #set up arrays
            for crossing in crossings:
                theta = np.arctan2((Y[crossing]-90), (X[crossing]-2560))
                theta_180.append(theta)
                if theta<0:
                    theta+=2*np.pi
                theta_loc.append(theta)

            if len(theta_loc) > 3:
                type = 2
            else:
                type = 1

            for i in np.arange(0,len(crossings),2):
                if type == 1:
                    Xline = X[cc]; Yline = Y[cc]
                else:
                    Xline = X[crossings[i]:crossings[i+1]]; Yline = Y[crossings[i]:crossings[i+1]]

                X_arr.append(Xline)
                Y_arr.append(Yline)

    theta_order = np.sort(theta_loc)
    theta_order = theta_order.tolist()

    theta_loc.append(theta_loc[0])
    theta_180.append(theta_180[0])

    print(theta_180)
    print(theta_loc)
    print(theta_order)

    X_contours,Y_contours = closeContour(theta_180,theta_loc,theta_order,Xs,Ys,Z, X_arr, Y_arr,threshold=0.7)

    for X,Y in zip(X_contours,Y_contours):
        Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
        X = np.append(X,X[0]); Y = np.append(Y,Y[0])
        Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

        plt.plot(X,Y,"-k",linewidth=3)
        plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

        Eddies_Cent_x.append(Centroid[0])
        Eddies_Cent_y.append(Centroid[1])
        Eddies_Area.append(Area)

    Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area}
    print(Eddies_it_pos)



    print("negative contours")
    #for -0.7m/s threshold
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []
    lines = CZ.allsegs[-1] #plot only threshold velocity

    theta_180 = []
    theta_loc = []
    X_arr = []; Y_arr = []
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
        elif C == "closed":
            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            X = np.append(X,X[0]); Y = np.append(Y,Y[0])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

            ux_c = ux_closed(Centroid,Xs,Ys,Z)

            if ux_c <= -0.7:
                plt.plot(X,Y,"--k",linewidth=3)
            elif ux_c >= 0.7:
                plt.plot(X,Y,"-k",linewidth=3)
            else:
                plt.plot(X,Y,"-.k",linewidth=3)

            plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

        elif C == "open":

            #set up arrays
            for crossing in crossings:
                theta = np.arctan2((Y[crossing]-90), (X[crossing]-2560))
                theta_180.append(theta)
                if theta<0:
                    theta+=2*np.pi
                theta_loc.append(theta)

            if len(theta_loc) > 3:
                type = 2
            else:
                type = 1

            for i in np.arange(0,len(crossings),2):
                if type == 1:
                    Xline = X[cc]; Yline = Y[cc]
                else:
                    Xline = X[crossings[i]:crossings[i+1]]; Yline = Y[crossings[i]:crossings[i+1]]
                    
                X_arr.append(Xline)
                Y_arr.append(Yline)

    theta_order = np.sort(theta_loc)
    theta_order = theta_order.tolist()

    theta_loc.append(theta_loc[0])
    theta_180.append(theta_180[0])
    print(theta_180)
    print(theta_loc)
    print(theta_order) 

    X_contours,Y_contours = closeContour(theta_180,theta_loc,theta_order,Xs,Ys,Z, X_arr, Y_arr,threshold=-0.7)

    for X,Y in zip(X_contours,Y_contours):
        Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
        X = np.append(X,X[0]); Y = np.append(Y,Y[0])
        Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

        plt.plot(X,Y,"--k",linewidth=3)
        plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

        Eddies_Cent_x.append(Centroid[0])
        Eddies_Cent_y.append(Centroid[1])
        Eddies_Area.append(Area)

    Eddies_it_neg = {"Centroid_x_neg": Eddies_Cent_x, "Centroid_y_neg": Eddies_Cent_y, "Area_neg": Eddies_Area}
    print(Eddies_it_neg)


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
    plt.plot(x_c,y_c,"-k",linewidth=5,label="$u_x' \geq 0.7 m/s$")
    plt.plot(x_c,y_c,"--k",linewidth=5,label="$u_x' \leq -0.7 m/s$")
    plt.plot(x_c,y_c,"-.k",linewidth=5,label="$-0.7 < u_x' < 0.7 m/s$")

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
#with Pool() as pool:
    #for Eddies_pos, Eddies_neg in pool.imap(Update,Time_steps):

for it in Time_steps:
    print("time step",it)
    Eddies_pos,Eddies_neg = Update(it)        

    df = pd.DataFrame(None)

    df_pos = pd.DataFrame(Eddies_pos)
    df = pd.concat([df,df_pos],axis=1); del df_pos

    df_neg = pd.DataFrame(Eddies_neg)
    df = pd.concat([df,df_neg],axis=1); del df_neg

    df.to_csv(csv_out_dir+"Eddies_0.7_{}.csv".format(it))
    print(df)
    del df

    it+=1

#saving data
# df_pos.to_csv(in_dir+"Eddies_{}.csv".format(0.7))
# df_neg.to_csv(in_dir+"Eddies_{}.csv".format(-0.7))
