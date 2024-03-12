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


def ux_interp(i,theta_loc,theta_180,Xs,Ys,Z,perc):

    if len(theta_loc) > 3:

        if theta_loc[i] < theta_loc[i+1] and theta_180[i+1] < theta_180[i]:
            
            #limit on minum angle change
            if abs((theta_loc[i]+2*np.pi) - theta_loc[i+1])*perc < np.radians(5):
                perc = 0.5
            
            xAB = abs((theta_loc[i]+2*np.pi) - theta_loc[i+1]) * perc

            #limit on maximum angle change
            if xAB > np.radians(25):
                xAB = np.radians(25)

            theta_anti = theta_loc[i+1] + xAB

        else:
            #limit on minum angle change
            if abs(theta_loc[i] - theta_loc[i+1])*perc < np.radians(5):
                perc = 0.5

            xAB = abs(theta_loc[i+1]-theta_loc[i]) *perc

            #limit on maximum angle change
            if xAB > np.radians(25):
                xAB = np.radians(25)

            theta_anti = theta_loc[i+1] + xAB 

        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi

        #limit on minum angle change
        if abs(theta_loc[i+1] - theta_loc[i+2])*perc < np.radians(5):
            perc = 0.5

        xBC = abs(theta_loc[i+1] - theta_loc[i+2]) * perc

        #limit on maximum angle change
        if xBC > np.radians(25):
                xBC = np.radians(25)

        theta_clock = theta_loc[i+1] - xBC

    elif len(theta_loc) < 4:

        if theta_loc[i] < theta_loc[i+1] and theta_180[i+1] < theta_180[i]:

            #limit on minum angle change
            if abs((theta_loc[i]+2*np.pi) - theta_loc[i+1])*perc < np.radians(5):
                perc = 0.5
            
            xAB = abs((theta_loc[i]+2*np.pi) - theta_loc[i+1]) * perc

            #limit on maximum angle change
            if xAB > np.radians(25):
                xAB = np.radians(25)
            theta_anti = theta_loc[i+1] + xAB

        else:

            #limit on minimum angle change
            if abs(theta_loc[i] - theta_loc[i+1])*perc < np.radians(5):
                perc = 0.5

            xAB = abs(theta_loc[i]-theta_loc[i+1]) *perc

            #limit on maximum angle change
            if xAB > np.radians(25):
                xAB = np.radians(25)

            theta_anti = theta_loc[i+1] + xAB 

        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi

        theta_clock = theta_loc[i+1] - xAB


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

    print(ux_anti,ux_clock,perc,x_anti,y_anti,x_clock,y_clock)

    return ux_anti, ux_clock,x_anti,y_anti,x_clock,y_clock



def isOutside(i,theta_loc,theta_order,theta_180,Xs,Ys,Z,threshold):

    theta = theta_loc[i+1] #theta B
    Bidx = theta_order.index(theta)

    for perc in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock = ux_interp(i,theta_loc,theta_180,Xs,Ys,Z,perc)
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
            Atheta = theta_order[Bidx+1]
            if theta > Atheta:
                Atheta+=2*np.pi
            
        elif ux_clock > threshold:
            Atheta = theta_order[Bidx-1]
            
    elif threshold < 0.0:
        plt.plot(x_anti,y_anti,"ob",markersize=6)
        plt.plot(x_clock,y_clock,"ob",markersize=6)

        if ux_clock < threshold:
            Atheta = theta_order[Bidx-1]
            
        elif ux_anti < threshold:
            Atheta = theta_order[Bidx+1]
            if theta > Atheta:
                Atheta+=2*np.pi

    return Atheta



def closeContour(Xs,Ys,Z,crossings,cc, X, Y,threshold):

    theta_loc = []; theta_180 = []
    for crossing in crossings:
        theta = np.arctan2((Y[crossing]-90), (X[crossing]-2560))
        theta_180.append(theta)
        if theta<0:
            theta+=2*np.pi
        theta_loc.append(theta)

    
    theta_order = np.sort(theta_loc)
    theta_order = theta_order.tolist()
    theta_order.append(theta_order[0])
    theta_loc.append(theta_loc[0])

    Xcontours = []; Ycontours = []
    Xcontour = []; Ycontour = []    

    if len(crossings) < 3:
        i = 0
        Xline = X[cc]; Yline = Y[cc]
        Xcontour = np.concatenate((Xcontour,Xline)) #plot A->B
        Ycontour = np.concatenate((Ycontour,Yline)) #plot A->B

        Atheta = isOutside(i,theta_loc,theta_order,theta_180,Xs,Ys,Z,threshold)

        theta_AB = np.linspace(theta_loc[i+1],Atheta,int(abs(theta_loc[i+1]-Atheta)/5e-03))

        r = 63
        Xarc = np.add(r*np.cos(theta_AB), 2560); Yarc = np.add(r*np.sin(theta_AB), 90)
        Xcontour = np.concatenate((Xcontour,Xarc))
        Ycontour = np.concatenate((Ycontour,Yarc))

        Xcontours.append(Xcontour); Ycontours.append(Ycontour)
    
    else:
        for i in np.arange(0,len(crossings),2):
            Xline = X[crossings[i]:crossings[i+1]]; Yline = Y[crossings[i]:crossings[i+1]]
            Xcontour = np.concatenate((Xcontour,Xline)) #plot A->B
            Ycontour = np.concatenate((Ycontour,Yline)) #plot A->B

            Atheta = isOutside(i,theta_loc,theta_order,theta_180,Xs,Ys,Z,threshold)

            theta_AB = np.linspace(theta_loc[i+1],Atheta,int(abs(theta_loc[i+1]-Atheta)/5e-03))

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


print("positive contours")
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
    elif C == "closed":
        Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
        X = np.append(X,X[0]); Y = np.append(Y,Y[0])
        Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

        plt.plot(X,Y,"-k",linewidth=3)
        plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

        Eddies_Cent_x.append(Centroid[0])
        Eddies_Cent_y.append(Centroid[1])
        Eddies_Area.append(Area)

    elif C == "open":

        X_contours,Y_contours = closeContour(Xs,Ys,Z,crossings,cc, X, Y,threshold=0.7)

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

        plt.plot(X,Y,"--k",linewidth=3)
        plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

        Eddies_Cent_x.append(Centroid[0])
        Eddies_Cent_y.append(Centroid[1])
        Eddies_Area.append(Area)

    elif C == "open":     

        X_contours,Y_contours = closeContour(Xs,Ys,Z,crossings,cc, X, Y,threshold=-0.7)

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


