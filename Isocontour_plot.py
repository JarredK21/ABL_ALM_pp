from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import operator
import sys
import time
from multiprocessing import Pool
import pyFAST.input_output as io
import math


#isocontourplot
def isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,dir):    
    
    if type(normal) == int and plane == "r" or type(normal) and plane == "tr": #rotor y',z planes
        u_plane = u.reshape(y,x)
        X,Y = np.meshgrid(ys,zs)
    elif type(normal) == int and plane == "t":
        u_plane = u.reshape(y,x)
        X,Y = np.meshgrid(xs,zs)
    elif normal == "z":
        u_plane = u.reshape(x,y)
        X,Y = np.meshgrid(xs,ys)
    elif normal == "x":
        u_plane = u.reshape(y,x)
        X,Y = np.meshgrid(ys,zs)


    fig = plt.figure()
    plt.rcParams['font.size'] = 12
    
    cs = plt.contourf(X,Y,u_plane,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)
    if normal == "x":
        plt.xlabel("Y axis [m]")
        plt.ylabel("Z axis [m]")
    elif normal == "y":
        plt.xlabel("X axis [m]")
        plt.ylabel("Z axis [m]")
    elif normal == "z":
        plt.xlabel("X axis [m]")
        plt.ylabel("Y axis [m]")
    elif type(normal) == int and plane == "r" or type(normal) and plane == "tr":
        plt.xlabel("Y' axis (rotor frame of reference) [m]")
        plt.ylabel("Z' axis (rotor frame of reference) [m]")
    elif type(normal) == int and plane == "t":
        plt.xlabel("X' axis (rotor frame of reference) [m]")
        plt.ylabel("Z' axis (rotor frame of reference) [m]")


    if plane == "r" and offset == -5.5:
        YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

        plt.plot(YB1,ZB1,color="k",linewidth = 0.5)
        plt.plot(YB2,ZB2,color="k",linewidth = 0.5)
        plt.plot(YB3,ZB3,color="k",linewidth = 0.5)  

    plt.title(Title,fontsize=12)
    cb = plt.colorbar(cs)
    plt.tight_layout()
    plt.savefig(dir+"{}".format(filename))

    cb.remove()
    plt.cla()
    plt.close(fig)


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(u,v,twist,x,normal,zs,h,height):
    if normal == "z":
        h_idx = np.searchsorted(h,height)
        mag_horz_vel = np.add( np.multiply(u,np.cos(twist[h_idx])) , np.multiply( v,np.sin(twist[h_idx])) )
    else:
        mag_horz_vel = []
        for i in np.arange(0,len(zs)):
            u_i = u[i*x:(i+1)*x]; v_i = v[i*x:(i+1)*x]
            height = zs[i]
            h_idx = np.searchsorted(h,height,side="left")
            if h_idx > 127:
                h_idx = 127
            mag_horz_vel_i = np.add( np.multiply(u_i,np.cos(twist[h_idx])) , np.multiply( v_i,np.sin(twist[h_idx])) )
            mag_horz_vel.extend(mag_horz_vel_i)
        mag_horz_vel = np.array(mag_horz_vel)
    return mag_horz_vel


def blade_positions(it):

    Time_it = Time[it]#find time from sampled data

    it_OF = np.searchsorted(Time_OF,Time_it)#find index from openfast

    R = 63
    Az = Azimuth[it_OF]
    Y = [2560]; Y2 = [2560]; Y3 = [2560]
    Z = [90]; Z2 = [90]; Z3 = [90]

    Y.append(Y[0]+R*np.sin(Az))
    Z.append(Z[0]+R*np.cos(Az))

    Az2 = Az+(2*np.pi)/3
    if Az2 > 2*np.pi:
        Az2 -= (2*np.pi)
    
    Az3 = Az-(2*np.pi)/3
    if Az2 < 0:
        Az2 += (2*np.pi)

    Y2.append(Y2[0]+R*np.sin(Az2))
    Z2.append(Z2[0]+R*np.cos(Az2))

    Y3.append(Y3[0]+R*np.sin(Az3))
    Z3.append(Z3[0]+R*np.cos(Az3))

    return Y, Z, Y2, Z2, Y3, Z3


start_time = time.time()

#defining twist angles with height from precursor
precursor = Dataset("./abl_statistics60000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],32500)
u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
del precursor

print("line 126", time.time()-start_time)

#Openfast data
df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()
#Openfast time
Time_OF = df["Time_[s]"]
#Azimuthal position for blade 1
Azimuth = np.array(np.radians(df["Azimuth_[deg]"]))
del df

print("line 144", time.time()-start_time)

#directories
in_dir = "./"
out_dir = in_dir + "ISOplots/"
isExist = os.path.exists(out_dir)
if isExist == False:
    os.makedirs(out_dir)
video_folder = in_dir + "videos/"
isExist = os.path.exists(video_folder)
if isExist == False:
    os.makedirs(video_folder)

planes = ["l","r", "tr","p_i","p_t"]
plane_labels = ["horizontal","rotor", "transverse rotor", "inflow", "longitudinal"]

ip = 0
for plane in planes:
    if plane == "l":
        offsets = [22,85,142.5]
    elif plane == "r":
        offsets = [-5.5,-63]
    elif plane == "tr":
        offsets = [0.0]
    elif plane == "i":
        offsets = [0.0]
    elif plane == "t":
        offsets = [0.0]

    ic = 0
    for offset in offsets:

        a = Dataset("./sampling_{0}_{1}.nc".format(plane,offset))

        p = a.groups["p_{0}".format(plane)]

        #time options
        Time = np.array(a.variables["time"])
        Time = Time - Time[0]


        #plotting option
        fluc_vel = False
        plot_specific_offsets = False
        plot_u = False; plot_v = False; plot_w = True; plot_hvelmag = True
        velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]
        plot_all_times = False
        custom_colorbar = False

        #time options
        if plot_all_times == True:
            tend_idx = np.searchsorted(Time,Time[-1])
            it_array = np.arange(0,tend_idx)
        else:
            #specify time steps to plot instantaneous isocontours at
            it_array = [0,10]

        #colorbar options
        if custom_colorbar == True:
            #specify color bar
            cmin = 0; cmax = 18


        x = p.ijk_dims[0] #no. data points
        y = p.ijk_dims[1] #no. data points

        #find normal
        if p.axis3[0] == 1:
            normal = "x"
        elif p.axis3[1] == 1:
            normal = "y"
        elif p.axis3[2] == 1:
            normal = "z"
        else:
            normal = 29

        #define plotting axes
        coordinates = np.array(p.variables["coordinates"])

        if type(normal) == int:
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
        elif normal == "x":
            xs = 0
            ys = np.linspace(p.origin[1],p.origin[1]+p.axis1[1],x)
            zs = np.linspace(p.origin[2],p.origin[2]+p.axis2[2],y)
        elif normal == "z":
            xs = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x)
            ys = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y)
            zs = 0

        #check if no velocity components selected
        if all(list(map(operator.not_, velocity_plot))) == True:
            sys.exit("error no velocity component selected")

        
        #loop over true velocity components
        velocity_comps = ["velocityx","velocityy","velocityz","Horizontal_velocity"]
        iv = 0
        for velocity_comp in velocity_comps:
            if velocity_plot[iv] == False:
                iv+=1
                continue
            
            print(plane_labels[ip],velocity_comps[iv],offset,time.time()-start_time)
                
            for it in it_array:

                #get velocity to plot for isocontour plots
                if velocity_comp == "Horizontal_velocity":
                    u = np.array(p.variables["velocityx"][it])
                    v = np.array(p.variables["velocityy"][it])
                    u = Horizontal_velocity(u,v,twist,x,normal,zs,h,height=90) #height only used for longitudinal planes
                else:
                    u = np.array(p.variables[velocity_comp][it])
                
                if fluc_vel == True:
                    def mean_velocity(u):
                        u_k = []
                        for u_j in u:
                            u_k.append(u_j - np.mean(u_j))
                        return u_k
                    
                    if fluc_vel == True:
                        u = mean_velocity(u)


                if custom_colorbar == False:                            
                    if fluc_vel == False:
                        if plane != "l" and velocity_comp == "velocityx" or plane != "l" and velocity_comp == "Horizontal_velocity":
                            cmin = 0
                        else:
                            cmin = math.floor(np.min(u))
                    else:
                        cmin = math.floor(np.min(u))
                    
                    cmax = math.ceil(np.max(u))
                    
                    #if custom_colorbar == True: specify cmain, cmax above
                    nlevs = (cmax-cmin)
                    levels = np.linspace(cmin,cmax,nlevs,dtype=int)
                    print("line 370",cmin,cmax)


                if it < 10:
                    Time_idx = "000{}".format(it)
                elif it >= 10 and it < 100:
                    Time_idx = "00{}".format(it)
                elif it >= 100 and it < 1000:
                    Time_idx = "0{}".format(it)
                elif it >= 1000 and it < 10000:
                    Time_idx = "{}".format(it)

                #define titles and filenames for isocontour plots
                if fluc_vel == True:
                    if velocity_comp == "Horizontal_velocity":
                        Title = "{0} Plane. \nFluctuating {1} [m/s]: Offset = {2}, Time = {3}s".format(plane_labels[ip], velocity_comp[:],float(offset),np.round(Time[it],2))
                        filename = "{0}_Fluc_{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[:],float(offset),Time_idx)
                    else:
                        Title = "{0} Plane. \nFluctuating velocity {1} [m/s]: Offset = {2}, Time = {3}s".format(plane_labels[ip],velocity_comp[-1],float(offsets),np.round(Time[it],2))
                        filename = "{0}_Fluc_vel{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[-1],float(offset),Time_idx)
                else:
                    u = np.array(u)
                    if velocity_comp == "Horizontal_velocity":
                        Title = "{0} Plane. \n{1} [m/s]: Offset = {2}, Time = {3}s".format(plane_labels[ip],velocity_comp[:],float(offset),np.round(Time[it],2))
                        filename = "{0}_{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[:],float(offset),Time_idx)
                    else:
                        Title = "{0} Plane. \nTotal velocity {1} [m/s]: Offset = {2}, time = {3}s".format(plane_labels[ip],velocity_comp[-1],float(offset),np.round(Time[it],2))
                        filename = "{0}_Tot_vel{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[-1],float(offset),Time_idx)
                    
                isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,out_dir)