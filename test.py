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
        (y - 90) * (y - 90) < 63 * 63):
        return True
    else:
        return False


def isOpen(cc):
    if any(cc) == False and all(cc) == False: #if all points are false skip loop in array
        return "skip"
    elif any(cc) == True and all(cc) == True:
        return "closed"
    else:
        return "open"


def openContour(cc,X,Y):

    #if there are points inside and outside remove points outside while interpolating between points to find points on edge of rotor
    X_contour = []; Y_contour = []; crossings = []
    X_temp = []; Y_temp = []
    for i in np.arange(0,len(cc)):
        
        if cc[i] == True:
            X_temp.append(X[i]); Y_temp.append(Y[i])

        if i < len(cc)-1 and cc[i+1] != cc[i]: #if next point is not the same as current (True -> False) or (False -> True) find point inbetween on rotor edge
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
            X_temp.append(y_root); Y_temp.append(z_root); crossings.append([y_root,z_root])

        if len(crossings) > 0 and len(crossings) % 2 == 0 and len(X_temp) > 0:

            X_contour.append(X_temp), Y_contour.append(Y_temp)
            X_temp = []; Y_temp = []

    return X_contour, Y_contour, crossings
            

def ux_closed(Centroid,Xs,Ys,Z):
    #need better way
    xmin = Centroid[0]-1; xmax = Centroid[0]+1
    ymin = Centroid[1]-1; ymax = Centroid[1]+1

    xmin_idx = np.searchsorted(ys,xmin,side="left"); xmax_idx = np.searchsorted(ys,xmax,side="right")
    ymin_idx = np.searchsorted(zs,ymin,side="left"); ymax_idx = np.searchsorted(zs,ymax,side="right")

    f_ux = interpolate.interp2d(Xs[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Ys[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Z[ymin_idx:ymax_idx,xmin_idx:xmax_idx]) #make more efficient??

    ux_closed = f_ux(Centroid[0],Centroid[1])

    return ux_closed


def UX_interp(x,y,Xs,Ys,Z):

    xmin = x - 4; xmax = x + 4; ymin = y - 4; ymax = y + 4

    xmin_idx = np.searchsorted(ys,xmin,side="left"); xmax_idx = np.searchsorted(ys,xmax,side="right")
    ymin_idx = np.searchsorted(zs,ymin,side="left"); ymax_idx = np.searchsorted(zs,ymax,side="right")

    f_ux = interpolate.interp2d(Xs[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Ys[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Z[ymin_idx:ymax_idx,xmin_idx:xmax_idx])

    ux = f_ux(x,y)

    return ux


def ux_interp(type,theta,theta_order,theta_180,Xs,Ys,Z,dtheta):

    theta_anti = theta + dtheta

    if theta_anti > 2*np.pi:
        theta_anti-=2*np.pi


    if round(theta_anti,2) >= round(theta_order[2],2):
        
        theta_anti = theta + abs(theta_180[2] - theta_180[1]) / 2

        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi

    if type == 2:

        theta_clock = theta - dtheta

        if theta_clock < 0:
            theta_clock +=2*np.pi

        if round(theta_clock,2) <= round(theta_order[0],2):
            
            theta_clock = theta - abs(theta_180[1] - theta_180[0]) / 2

            if theta_clock < 0:
                theta_clock +=2*np.pi
        
    else:
        theta_clock = theta - dtheta

        if theta_clock < 0:
            theta_clock +=2*np.pi

    r = 63
    x_anti = 2560 + r*np.cos(theta_anti)
    y_anti = 90 + r*np.sin(theta_anti)

    x_clock = 2560 + r*np.cos(theta_clock)
    y_clock = 90 + r*np.sin(theta_clock)

    ux_anti = UX_interp(x_anti,y_anti,Xs,Ys,Z)

    ux_clock = UX_interp(x_clock,y_clock,Xs,Ys,Z)

    print(ux_anti,ux_clock)

    f.write("{} {} {} {} {} {} {} {} \n".format(theta_anti,theta_clock,ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock))

    return ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock


def isOutside(type,theta,theta_order,theta_180,Xs,Ys,Z,threshold):

    dtheta_arr = np.radians([2,4,6,8,10,12,14,16,18,20,24,26])

    for dtheta in dtheta_arr:
        ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock = ux_interp(type,theta,theta_order,theta_180,Xs,Ys,Z,dtheta)
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

    return Atheta,direction


def isInlist(theta_not,theta):
    theta_in_list = False
    for theta_val in theta_not:
        if theta_val == theta:
            theta_in_list = True

    return theta_in_list


def closeContour(type,theta_180,theta_loc,theta_order,Xs,Ys,Z, X, Y,threshold):

    Xcontours = []; Ycontours = []
    Xcontour = []; Ycontour = []

    theta_not = []
    for i in np.arange(0,len(theta_loc)-1,2):

        theta = theta_loc[i+1]

        if type == 2:
            #check if theta is in theta not
            if isInlist(theta_not,theta) == True:
                theta_temp = theta_loc[i:i+2]
                idx_temp = theta_temp.index(theta)
                if idx_temp == 0:
                    theta = theta_temp[1]
                else:
                    theta = theta_temp[0]
                del theta_temp; del idx_temp

        Oidx = theta_order.index(theta)
        f.write("crossing {} \n".format(theta))

        if Oidx == 0:
            theta_O = [theta_order[-1]-2*np.pi,theta_order[Oidx],theta_order[Oidx+1]]
            theta_B = [theta_180[-1],theta_180[Oidx],theta_180[Oidx+1]]
        elif Oidx == len(theta_order)-1:
            theta_O = [theta_order[Oidx-1],theta_order[Oidx],theta_order[0]+2*np.pi]
            theta_B = [theta_180[Oidx-1],theta_180[Oidx],theta_180[0]]
        else:
            theta_O = theta_order[Oidx-1:Oidx+2]
            theta_B = theta_180[Oidx-1:Oidx+2]

        if type == 1:
            theta_B = theta_O

        f.write("theta 180 {} \n".format(str(theta_B)))
        f.write("theta order {} \n".format(str(theta_O)))
        
        Atheta,direction = isOutside(type,theta,theta_O,theta_B,Xs,Ys,Z,threshold)
        f.write("Atheta {}, direction {} \n".format(Atheta,direction))

        if type == 2:
            theta_not.append(theta);theta_not.append(Atheta)

        #this should not be needed
        if Atheta == "skip":
            continue
        
        idx = int(i/2)
        Xcontour = np.concatenate((Xcontour,X[idx])) #plot A->B
        Ycontour = np.concatenate((Ycontour,Y[idx])) #plot A->B


        if direction == "anticlockwise":
            if theta < Atheta:
            
                theta_AB = np.linspace(theta,Atheta,int(abs(theta_B[2]-theta_B[1])/5e-03))
            elif theta > Atheta:
                theta_AB1 = np.linspace(theta_B[1],0,int(abs(theta_B[1])/5e-03))
                theta_AB2 = np.linspace(0,Atheta,int(abs(Atheta)/5e-03))
                theta_AB = np.concatenate((theta_AB1,theta_AB2))
        elif direction == "clockwise":
            if theta > Atheta:
                theta_AB = np.linspace(theta,Atheta,int(abs(theta_B[1]-theta_B[0])/5e-03))
            elif theta < Atheta:
                theta_AB1 = np.linspace(theta,0,int(abs(theta_B[1])/5e-03))
                theta_AB2 = np.linspace(0,theta_B[0],int(abs(theta_B[0])/5e-03))
                theta_AB = np.concatenate((theta_AB1,theta_AB2))

        for j in np.arange(0,len(theta_AB)):
            if theta_AB[j] < 0:
                theta_AB[j]+=2*np.pi
            elif theta_AB[j] >= 2*np.pi:
                theta_AB[j]-=2*np.pi

        f.write("theta arc {} \n".format(theta_AB))
        print("theta arc",theta_AB)

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

#directories
in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)

a = Dataset(in_dir+"sampling_r_-63.0_0.nc")


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
u = np.array(p.variables["velocityx"])
v = np.array(p.variables["velocityy"])
del p

u[u<0]=0; v[v<0] #remove negative velocities

u = np.subtract(u,np.mean(u))
v = np.subtract(v,np.mean(v))

twist = np.radians(29)
mag_horz_vel = []
for i in np.arange(0,len(zs)):
    u_i = u[i*x:(i+1)*x]; v_i = v[i*x:(i+1)*x]

    mag_horz_vel_i = u_i*np.cos(twist) + v_i*np.sin(twist)
    mag_horz_vel.extend(mag_horz_vel_i)
u = np.array(mag_horz_vel); del mag_horz_vel; del v

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


folder = out_dir+"Rotor_Plane_Fluctutating_horz_-63.0_3/"
isExist = os.path.exists(folder)
if isExist == False:
    os.makedirs(folder)

#write log file
f = open(folder+"out.log", "w")

U = u #velocity time step it

Time_idx = "0000"

u_plane = U.reshape(y,x)
Xs,Ys = np.meshgrid(ys,zs)

Z = u_plane

CS = plt.contour(Xs, Ys, Z, levels=levels_pos)
CZ = plt.contour(Xs,Ys,Z, levels=levels_neg)

T = 0.0

fig,ax = plt.subplots(figsize=(50,30))
plt.rcParams['font.size'] = 40

cs = ax.contourf(Xs,Ys,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

cb = plt.colorbar(cs)


f.write("positive contours \n")
#for +0.7m/s threshold
Eddies_Cent_x = []
Eddies_Cent_y = []
Eddies_Area = []
lines = CS.allsegs[0] #plot only threshold velocity

X_contour = []; Y_contour = []; crossings = []
for line in lines:
    X, Y = line[:,0], line[:,1]

    #check if any point in line is inside circle
    cc = []
    for X_line, Y_line in zip(X, Y):
        cc.append(isInside(X_line,Y_line))

    C = isOpen(cc)

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

        plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

        Eddies_Cent_x.append(Centroid[0])
        Eddies_Cent_y.append(Centroid[1])
        Eddies_Area.append(Area)

    elif C == "open":

        #separate line into N contours if line is outside rotor disk
        X_temp,Y_temp,crossings_temp = openContour(cc,X,Y)

        for i in np. arange(0,len(X_temp)):
            X_contour.append(X_temp[i])
            Y_contour.append(Y_temp[i])
        for crossing in crossings_temp:
            crossings.append(crossing)
        del X_temp; del Y_temp; del crossings_temp

if len(X_contour) > 0:
    #set up arrays
    theta_loc = []
    for crossing in crossings:
        theta = np.arctan2((crossing[1]-90), (crossing[0]-2560))
        if theta<0:
            theta+=2*np.pi
        theta_loc.append(theta)

    theta_order = np.sort(theta_loc)
    theta_order = theta_order.tolist()

    theta_180 = []
    for theta in theta_order:
        if theta > np.pi:
            theta_180.append(theta-(2*np.pi))
        else:
            theta_180.append(theta)

    theta_loc.append(theta_loc[0])

    if len(theta_loc) > 3:
        type = 2
    else:
        type = 1

    X_contours,Y_contours = closeContour(type,theta_180,theta_loc,theta_order,Xs,Ys,Z,X_contour,Y_contour,threshold=0.7)


if len(X_contours) > 0:
    for X,Y in zip(X_contours,Y_contours):
        if len(X) > 0:
            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            X = np.append(X,X[0]); Y = np.append(Y,Y[0])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

            Ux_avg = 0

            plt.plot(X,Y,"-k",linewidth=3)
            plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area, "Ux_avg": Ux_avg}
f.write("{} \n".format(str(Eddies_it_pos)))


f.write("negative contours \n")
#for -0.7m/s threshold
Eddies_Cent_x = []
Eddies_Cent_y = []
Eddies_Area = []
lines = CZ.allsegs[-1] #plot only threshold velocity

X_contour = []; Y_contour = []; crossings = []
for line in lines:
    X, Y = line[:,0], line[:,1]

    #check if any point in line is inside circle
    cc = []
    for X_line, Y_line in zip(X, Y):
        cc.append(isInside(X_line,Y_line))

    C = isOpen(cc)

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

        plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

        Eddies_Cent_x.append(Centroid[0])
        Eddies_Cent_y.append(Centroid[1])
        Eddies_Area.append(Area)

    elif C == "open":

        #separate line into N contours if line is outside rotor disk
        X_temp,Y_temp,crossings_temp = openContour(cc,X,Y)

        for i in np. arange(0,len(X_temp)):
            X_contour.append(X_temp[i])
            Y_contour.append(Y_temp[i])
        for crossing in crossings_temp:
            crossings.append(crossing)
        del X_temp; del Y_temp; del crossings_temp

if len(X_contour) > 0:
    #set up arrays
    theta_loc = []
    for crossing in crossings:
        theta = np.arctan2((crossing[1]-90), (crossing[0]-2560))
        if theta<0:
            theta+=2*np.pi
        theta_loc.append(theta)

    theta_order = np.sort(theta_loc)
    theta_order = theta_order.tolist()

    theta_180 = []
    for theta in theta_order:
        if theta > np.pi:
            theta_180.append(theta-(2*np.pi))
        else:
            theta_180.append(theta)

    theta_loc.append(theta_loc[0])

    if len(theta_loc) > 3:
        type = 2
    else:
        type = 1

    X_contours,Y_contours = closeContour(type,theta_180,theta_loc,theta_order,Xs,Ys,Z,X_contour,Y_contour,threshold=-0.7)

if len(X_contours) > 0:
    for X,Y in zip(X_contours,Y_contours):
        if len(X) > 0:
            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            X = np.append(X,X[0]); Y = np.append(Y,Y[0])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

            Ux_avg = 0

            plt.plot(X,Y,"--k",linewidth=3)
            plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

Eddies_it_neg = {"Centroid_x_neg": Eddies_Cent_x, "Centroid_y_neg": Eddies_Cent_y, "Area_neg": Eddies_Area, "Ux_avg": Ux_avg}
f.write("{} \n".format(str(Eddies_it_neg)))

if len(Eddies_it_pos["Area_pos"]) == 0 and len(Eddies_it_neg["Area_neg"] == 0):
    ux_cent,ux_cl = UX_interp(2560,90,2560,90,Xs,Ys,Z)
    if ux_cent <= -0.7:
        Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=3,linestyle="--",edgecolor="k")
        Eddies_Cent_x = [2560]; Eddies_Cent_y = [90]; Eddies_Area = [(np.pi*63**2)]; Ux_avg = 0
        Eddies_it_neg = {"Centroid_x_neg": Eddies_Cent_x, "Centroid_y_neg": Eddies_Cent_y, "Area_neg": Eddies_Area, "Ux_avg": Ux_avg}
        f.write("{} \n".format(str(Eddies_it_neg)))
        plt.plot(2560,90,"+k",markersize=8)
        ax.add_artist(Drawing_uncolored_circle)
    elif ux_cent >= 0.7:
        Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=3,linestyle="-",edgecolor="k")
        Eddies_Cent_x = [2560]; Eddies_Cent_y = [90]; Eddies_Area = [(np.pi*63**2)]
        Ux_avg = 0
        Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area, "Ux_avg": Ux_avg}
        f.write("{} \n".format(str(Eddies_it_pos)))
        plt.plot(2560,90,"+k",markersize=8)
        ax.add_artist(Drawing_uncolored_circle)


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


print("end",time.time()-start_time)