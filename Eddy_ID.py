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
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    mag_fluc_horz_vel = []
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
        mag_fluc_horz_vel_i = np.subtract(mag_horz_vel_i,ux_mean)
        mag_fluc_horz_vel.extend(mag_fluc_horz_vel_i)
    mag_fluc_horz_vel = np.array(mag_fluc_horz_vel)
    return mag_fluc_horz_vel


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
            

def ux_interp(coordinates):

    x = coordinates[0]; y = coordinates[1]
    xmin = x - 4; xmax = x + 4; ymin = y - 4; ymax = y + 4

    xmin_idx = np.searchsorted(ys,xmin,side="left"); xmax_idx = np.searchsorted(ys,xmax,side="right")
    ymin_idx = np.searchsorted(zs,ymin,side="left"); ymax_idx = np.searchsorted(zs,ymax,side="right")

    f_ux = interpolate.interp2d(Xs[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Ys[ymin_idx:ymax_idx,xmin_idx:xmax_idx],Z[ymin_idx:ymax_idx,xmin_idx:xmax_idx])

    ux = f_ux(x,y)

    return ux


def ux_offset_perc(ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock,theta,theta_180,perc):
    r = 63

    if np.isnan(ux_anti) == True:
        theta_anti = theta + abs(theta_180[2] - theta_180[1]) / (1/perc)

        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi

        x_anti = 2560 + r*np.cos(theta_anti)
        y_anti = 90 + r*np.sin(theta_anti)

        ux_anti = ux_interp([x_anti,y_anti])

    if np.isnan(ux_clock) == True:
            
        theta_clock = theta - abs(theta_180[1] - theta_180[0]) / (1/perc)

        if theta_clock < 0:
            theta_clock +=2*np.pi

        x_clock = 2560 + r*np.cos(theta_clock)
        y_clock = 90 + r*np.sin(theta_clock)     
        ux_clock = ux_interp([x_clock,y_clock])

    f.write("perc: {} {} {} {} {} {} \n".format(ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock))
    print(ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock)
    return ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock


def ux_offset_deg(type,theta,theta_order,dtheta):
    r = 63

    theta_anti = theta + dtheta

    if type == 2:

        theta_clock = theta - dtheta      
    else:
        theta_clock = theta - dtheta

    if round(theta_clock,2) <= round(theta_order[0],2):
        ux_clock = np.nan
        x_clock = np.nan
        y_clock = np.nan
    else:
        if theta_clock < 0:
            theta_clock +=2*np.pi
        x_clock = 2560 + r*np.cos(theta_clock)
        y_clock = 90 + r*np.sin(theta_clock)
        ux_clock = ux_interp([x_clock,y_clock])
    
    if round(theta_anti,2) >= round(theta_order[2],2):
        ux_anti = np.nan
        x_anti = np.nan
        y_anti = np.nan
    else:
        if theta_anti > 2*np.pi:
            theta_anti-=2*np.pi
        x_anti = 2560 + r*np.cos(theta_anti)
        y_anti = 90 + r*np.sin(theta_anti)
        ux_anti = ux_interp([x_anti,y_anti])


    f.write("deg: {} {} {} {} {} {} {} {} \n".format(theta_anti,theta_clock,ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock))
    print(theta_anti,theta_clock,ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock)
    return ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock


def isOutside(type,theta,theta_order,theta_180):

    dtheta_arr = np.radians([2,4,6,8,10,12,14,16,18,20,24,26])
    percentage = [0.5,0.55,0.45,0.60,0.40,0.65,0.35,0.70,0.30,0.75,0.35,0.80]

    ip = 0
    for dtheta in dtheta_arr:
        ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock = ux_offset_deg(type,theta,theta_order,dtheta)

        if np.isnan(ux_anti) == True or np.isnan(ux_clock) == True:
            ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock = ux_offset_perc(ux_anti,ux_clock,x_anti,y_anti,x_clock,y_clock,theta,theta_180,percentage[ip])
            ip+=1

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
        if round(theta_val,4) == round(theta,4):
            theta_in_list = True

    return theta_in_list


def closeContour(type,theta_180,theta_loc,theta_order,X,Y):

    Xcontours = []; Ycontours = []
    Xcontour = []; Ycontour = []
    r = 63
    theta_not = []
    istart = 0
    for i in np.arange(0,len(theta_loc)-1,2):

        theta_start = theta_loc[istart]

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
        f.write("theta start {} \n".format(theta_start))
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
        
        Atheta,direction = isOutside(type,theta,theta_O,theta_B)
        f.write("Atheta {}, direction {} \n".format(Atheta,direction))

        if type == 2:
            theta_not.append(theta)
            if Atheta < 0:
                theta_not.append(Atheta+2*np.pi)
            elif Atheta >= 2*np.pi:
                theta_not.append(Atheta-2*np.pi)
            else:
                theta_not.append(Atheta)

        #this should not be needed
        if Atheta == "skip":
            continue

        idx = int(i/2)
        # f.write("first and last point {} {} \n".format([round(X[idx][0],2),round(r*np.cos(theta_loc[i])+2560,2)],[round(X[idx][-1],2),round(r*np.cos(theta)+2560,2)]))
        # print("first and last point {} {} \n".format([round(X[idx][0],2),round(r*np.cos(theta_loc[i])+2560,2)],[round(X[idx][-1],2),round(r*np.cos(theta)+2560,2)]))
        # if round(X[idx][-1],2) == round(r*np.cos(theta_loc[i])+2560,2) and round(X[idx][0],2) == round(r*np.cos(theta)+2560,2):
        #     Xline = X[idx][::-1]; Yline = Y[idx][::-1]
        # else:
        Xline = X[idx]; Yline = Y[idx]
        Xcontour = np.concatenate((Xcontour,Xline)) #plot A->B
        Ycontour = np.concatenate((Ycontour,Yline)) #plot A->B

        theta_AB = np.linspace(theta,Atheta,int(abs(theta-Atheta)/5e-03))

        for j in np.arange(0,len(theta_AB)):
            if theta_AB[j] < 0:
                theta_AB[j]+=2*np.pi
            elif theta_AB[j] >= 2*np.pi:
                theta_AB[j]-=2*np.pi

        f.write("theta arc {} \n".format(theta_AB))
        print("theta arc",theta_AB)

        Xarc = np.add(r*np.cos(theta_AB), 2560); Yarc = np.add(r*np.sin(theta_AB), 90)
        Xcontour = np.concatenate((Xcontour,Xarc))
        Ycontour = np.concatenate((Ycontour,Yarc))

        if Atheta < 0:
            Atheta+=2*np.pi
        elif Atheta >= 2*np.pi:
            Atheta-=2*np.pi

        if round(Atheta,4) == round(theta_start,4):
            Xcontours.append(Xcontour); Ycontours.append(Ycontour)
            Xcontour = []; Ycontour = []
        
            istart = i+2

    if len(Xcontours) == 0:
        Xcontours.append(Xcontour); Ycontours.append(Ycontour)


    return Xcontours, Ycontours
      

def ux_average_calc(X,Y,C):
    deltax = 1.25; deltay = 1.25
    xmin = np.min(X); xmax = np.max(X)
    f.write("xmin {} \n".format(xmin))
    f.write("xmax {} \n".format(xmax))
    print(xmin)
    print(xmax)
    xlist = np.arange(xmin+deltax,xmax-deltax,deltax)
    coordinates = []
    for xr in xlist:
        
        f.write("xr {}".format(xr))
        xidx = (X>(xr-0.15625))*(X<xr+0.15625)
        xidxlist = np.where(xidx)
        f.write("xidxlist {} \n".format(xidxlist[0]))
        print("xidxlist",xidxlist[0])
        print("len xidxlist",len(xidxlist[0]))
        if len(xidxlist[0]) == 0:
            continue

        ymin = np.min(Y[xidxlist[0]]); ymax = np.max(Y[xidxlist[0]])
        f.write("ymin {} \n".format(ymin))
        f.write("ymax {} \n".format(ymax))
        print("ymin",ymin);print("ymax",ymax)

        if ymin+deltay < ymax-deltay:
            ylist = np.arange(ymin+deltay,ymax-deltay,deltay)
            
            for yr in ylist:
                coordinates.append([xr,yr])
                print(xr,yr)

    Ux_avg = []
    for coordinate in coordinates:
        velx = ux_interp(coordinate)
        if C == "closed":
            if cmin<=velx<=cmax:
                Ux_avg.append(velx[0])
        else:
            if threshold > 0 and velx[0] >= threshold and velx[0] <= cmax:
                Ux_avg.append(velx[0])
            elif threshold < 0 and velx[0] <= threshold and velx[0] >= cmin:
                Ux_avg.append(velx[0])


    if len(Ux_avg) == 0:
        return 0
    else:
        return np.average(Ux_avg)



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
print(ux_mean_profile)

#directories
in_dir = "./"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)

csv_out_dir = in_dir + "csv_files_2/"
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

with Pool() as pool:
    u_pri = []
    for u_fluc_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
        
        u_pri.append(u_fluc_hvel_it)
        print(len(u_pri),time.time()-start_time)
u = np.array(u_pri); del u_pri; del v


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

#write log file
f = open(folder+"out.log", "w")
def Update(it):

    global Xs 
    global Ys 
    global Z
    global threshold

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

    f.write("positive contours \n")
    print("positive contours")
    #for +0.7m/s threshold
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []
    #Ux_avg = []
    threshold = 0.7
    lines = CS.allsegs[0] #plot only threshold velocity

    Xcontour = []; Ycontour = []; crossings = []
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
                
            #ux_c = ux_average_calc(X,Y,C)

            #Ux_avg.append(ux_c)

            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            X = np.append(X,X[0]); Y = np.append(Y,Y[0])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

            ux_c = ux_interp(Centroid)

            if ux_c > -0.7:
                plt.plot(X,Y,"-k",linewidth=3)
            elif ux_c <= -0.7:
                plt.plot(X,Y,"--k",linewidth=3)

            plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

        elif C == "open":

            #separate line into N contours if line is outside rotor disk
            Xtemp,Ytemp,crossings_temp = openContour(cc,X,Y)

            for i in np. arange(0,len(Xtemp)):
                Xcontour.append(Xtemp[i])
                Ycontour.append(Ytemp[i])
            for crossing in crossings_temp:
                crossings.append(crossing)
            del Xtemp; del Ytemp; del crossings_temp

    if len(Xcontour) > 0:
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

        f.write("theta_loc {} \n".format(theta_loc))
        print("theta_loc {}".format(theta_loc))

        if len(theta_loc) > 3:
            type = 2
        else:
            type = 1

        Xcontour,Ycontour = closeContour(type,theta_180,theta_loc,theta_order,Xcontour,Ycontour)


    if len(Xcontour) > 0:
        for X,Y in zip(Xcontour,Ycontour):
            if len(X) > 0:
                #Ux_avg.append(ux_average_calc(X,Y,C))
                Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
                X = np.append(X,X[0]); Y = np.append(Y,Y[0])
                Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

                plt.plot(X,Y,"-k",linewidth=3)
                plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

                Eddies_Cent_x.append(Centroid[0])
                Eddies_Cent_y.append(Centroid[1])
                Eddies_Area.append(Area)

    #Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area, "Ux_avg_pos": Ux_avg}
    Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area}
    print(Eddies_it_pos)

    f.write("negative contours \n")
    print("Negative contours")
    #for -0.7m/s threshold
    Eddies_Cent_x = []
    Eddies_Cent_y = []
    Eddies_Area = []
    #Ux_avg = []
    threshold = -0.7
    lines = CZ.allsegs[-1] #plot only threshold velocity

    Xcontour = []; Ycontour = []; crossings = []
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

            #ux_c = ux_average_calc(X,Y,C)

            #Ux_avg.append(ux_c)

            Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
            X = np.append(X,X[0]); Y = np.append(Y,Y[0])
            Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

            ux_c = ux_interp(Centroid)

            if ux_c >= 0.7:
                plt.plot(X,Y,"-k",linewidth=3)
            elif ux_c < 0.7:
                plt.plot(X,Y,"--k",linewidth=3)

            plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

            Eddies_Cent_x.append(Centroid[0])
            Eddies_Cent_y.append(Centroid[1])
            Eddies_Area.append(Area)

        elif C == "open":

            #separate line into N contours if line is outside rotor disk
            Xtemp,Ytemp,crossings_temp = openContour(cc,X,Y)

            for i in np. arange(0,len(Xtemp)):
                Xcontour.append(Xtemp[i])
                Ycontour.append(Ytemp[i])
            for crossing in crossings_temp:
                crossings.append(crossing)
            del Xtemp; del Ytemp; del crossings_temp

    if len(Xcontour) > 0:
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

        f.write("theta_loc {} \n".format(theta_loc))
        print("theta_loc {}".format(theta_loc))

        if len(theta_loc) > 3:
            type = 2
        else:
            type = 1

        Xcontour,Ycontour = closeContour(type,theta_180,theta_loc,theta_order,Xcontour,Ycontour)

    if len(Xcontour) > 0:
        for X,Y in zip(Xcontour,Ycontour):
            if len(X) > 0:
                #Ux_avg.append(ux_average_calc(X,Y,C))
                Centroid = [np.sum(X)/len(X), np.sum(Y)/len(Y)]
                X = np.append(X,X[0]); Y = np.append(Y,Y[0])
                Area = np.abs((np.sum(X[1:]*Y[:-1]) - np.sum(Y[1:]*X[:-1]))/2)

                plt.plot(X,Y,"--k",linewidth=3)
                plt.plot(Centroid[0],Centroid[1],"+k",markersize=8)

                Eddies_Cent_x.append(Centroid[0])
                Eddies_Cent_y.append(Centroid[1])
                Eddies_Area.append(Area)

    #Eddies_it_neg = {"Centroid_x_neg": Eddies_Cent_x, "Centroid_y_neg": Eddies_Cent_y, "Area_neg": Eddies_Area, "Ux_avg_neg": Ux_avg}
    Eddies_it_neg = {"Centroid_x_neg": Eddies_Cent_x, "Centroid_y_neg": Eddies_Cent_y, "Area_neg": Eddies_Area}
    print(Eddies_it_neg)

    if len(Eddies_it_pos["Area_pos"]) == 0 and len(Eddies_it_neg["Area_neg"]) == 0:
        ux_cent = ux_interp([2560,90])
        if ux_cent <= -0.7:
            theta = np.linspace(0,2*np.pi,360)
            X = 63*np.cos(theta) + 2560; Y = 63*np.sin(theta) + 90
            Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=3,linestyle="--",edgecolor="k")
            Eddies_Cent_x = [2560]; Eddies_Cent_y = [90]; Eddies_Area = [(np.pi*63**2)]
            #Ux_avg.append(ux_average_calc(X,Y,C="closed"))
            #Eddies_it_neg = {"Centroid_x_neg": Eddies_Cent_x, "Centroid_y_neg": Eddies_Cent_y, "Area_neg": Eddies_Area, "Ux_avg_neg": Ux_avg}
            Eddies_it_neg = {"Centroid_x_neg": Eddies_Cent_x, "Centroid_y_neg": Eddies_Cent_y, "Area_neg": Eddies_Area}
            plt.plot(2560,90,"+k",markersize=8)
            plt.plot(2560,90,"+k",markersize=8)
            ax.add_artist(Drawing_uncolored_circle)
        elif ux_cent >= 0.7:
            theta = np.linspace(0,2*np.pi,360)
            X = 63*np.cos(theta) + 2560; Y = 63*np.sin(theta) + 90
            Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=3,linestyle="-",edgecolor="k")
            Eddies_Cent_x = [2560]; Eddies_Cent_y = [90]; Eddies_Area = [(np.pi*63**2)]
            #Ux_avg.append(ux_average_calc(X,Y,C="closed"))
            #Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area, "Ux_avg_pos": Ux_avg}
            Eddies_it_pos = {"Centroid_x_pos": Eddies_Cent_x, "Centroid_y_pos": Eddies_Cent_y, "Area_pos": Eddies_Area}
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
    plt.plot(x_c,y_c,"-k",linewidth=5,label="$u_x' \geq 0.7 m/s$")
    plt.plot(x_c,y_c,"--k",linewidth=5,label="$u_x' \leq -0.7 m/s$")

    plt.xlim([ys[0],ys[-1]]);plt.ylim(zs[0],zs[-1])
    plt.legend()
    plt.title(Title)
    plt.tight_layout()
    plt.savefig(folder+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)

    return Eddies_it_pos, Eddies_it_neg




# it = 0
# with Pool() as pool:
#     for Eddies_pos, Eddies_neg in pool.imap(Update,Time_steps):

#         f.write("Time step {} \n".format(str(it)))
#         print("Time step = ",it)     

#         df = pd.DataFrame(None)

#         df_pos = pd.DataFrame(Eddies_pos)
#         df = pd.concat([df,df_pos],axis=1); del df_pos

#         df_neg = pd.DataFrame(Eddies_neg)
#         df = pd.concat([df,df_neg],axis=1); del df_neg

#         df.to_csv(csv_out_dir+"Eddies_0.7_{}.csv".format(it))
#         f.write("{} \n".format(str(df)))
#         del df

#         it+=1

# f.close()


for it in Time_steps:

    Eddies_pos,Eddies_neg = Update(it)

    f.write("Time step {} \n".format(str(it)))
    print("Time step = ",it)     

    df = pd.DataFrame(None)

    df_pos = pd.DataFrame(Eddies_pos)
    df = pd.concat([df,df_pos],axis=1); del df_pos

    df_neg = pd.DataFrame(Eddies_neg)
    df = pd.concat([df,df_neg],axis=1); del df_neg

    df.to_csv(csv_out_dir+"Eddies_0.7_{}.csv".format(it))
    f.write("{} \n".format(str(df)))
    del df

f.close()