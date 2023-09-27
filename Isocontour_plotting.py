from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob 
import os
from matplotlib import cm
from matplotlib.animation import PillowWriter
import operator
import math
import sys
import time
from multiprocessing import Pool
import cv2
import re
import pyFAST.input_output as io


#isocontourplot
def isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,dir):
    
    if type(normal) == int: #rotor plane
        u_plane = u.reshape(y,x)
        X,Y = np.meshgrid(ys,zs)
    elif normal == "z":
        u_plane = u.reshape(x,y)
        X,Y = np.meshgrid(xs,ys)
    elif normal == "x":
        u_plane = u.reshape(y,x)
        X,Y = np.meshgrid(ys,zs)


    fig = plt.figure()
    plt.rcParams['font.size'] = 12
    
    plt.contourf(X,Y,u_plane, cmap=cm.coolwarm)
    if normal == "x":
        plt.xlabel("Y axis [m]")
        plt.ylabel("Z axis [m]")
    elif normal == "y":
        plt.xlabel("X axis [m]")
        plt.ylabel("Z axis [m]")
    elif normal == "z":
        plt.xlabel("X axis [m]")
        plt.ylabel("Y axis [m]")
    else:
        plt.xlabel("Y' axis (rotor frame of reference) [m]")
        plt.ylabel("Z' axis (rotor frame of reference) [m]")

    plt.title(Title,fontsize=12)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(dir+"{}".format(filename))
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
t_start = np.searchsorted(precursor.variables["time"],32300)
t_end = np.searchsorted(precursor.variables["time"],33500)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
del precursor

print("line 126", time.time()-start_time)

#Openfast data
df = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()
#Openfast time
Time_OF = df["Time_[s]"]
#Azimuthal position for blade 1
Azimuth = np.array(np.radians(df["Azimuth_[deg]"])); del df

print("line 144", time.time()-start_time)

#directories
in_dir = "./"
out_dir = in_dir + "ISOplots/"
video_folder = in_dir + "videos/"
isExist = os.path.exists(video_folder)
if isExist == False:
    os.makedirs(video_folder)

planes = ["l","r", "t"]
plane_labels = ["longitudinal","rotor", "transverse"]

ip = 0
for plane in planes:
    if plane == "l":
        offsets = [85]
    elif plane == "r":
        offsets = [0.0, -63.0, -126, 126]
    elif plane == "t":
        offsets = [1280, 1930, 3190, 3820]

    ic = 0
    for offset in offsets:

        a = Dataset("./sampling_{0}_{1}.nc".format(plane,offset))

        p = a.groups["p_{0}".format(plane)]

        #time options
        Time = np.array(a.variables["time"])
        Time = Time - Time[0]

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
            normal = int(np.degrees(np.arccos(p.axis3[0])))

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

        
        plot_all_times = True
        if plot_all_times == False:
            tstart = 0
            tend = 1200
            tstart_idx = np.searchsorted(Time,tstart)
            tend_idx = np.searchsorted(Time,tend)
            time_steps = np.arange(tstart_idx,tend_idx)
        else:
            tend_idx = np.searchsorted(Time,Time[-1])
            time_steps = np.arange(0,tend_idx)


        #specify time steps to plot instantaneous isocontours at
        it_array = [0,10]


        #plotting option
        plot_isocontour = False
        fluc_vel = False
        movie_tot_vel_isocontour = True
        plot_specific_offsets = False
        plot_u = False; plot_v = False; plot_w = True; plot_hvelmag = True
        velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]

        #check if no velocity components selected
        if all(list(map(operator.not_, velocity_plot))) == True:
            sys.exit("error no velocity component selected")

        
        #loop over true velocity components
        velocity_comps = ["velocityx","velocityy","velocityz","Horizontal velocity"]
        iv = 0
        for velocity_comp in velocity_comps:
            if velocity_plot[iv] == False:
                iv+=1
                continue
            
            print(plane_labels[ip],velocity_comps[iv],offset,time.time()-start_time)

            #colorbar options
            custom_colorbar = False
            cmin = 0; cmax = 18
                
            for it in it_array:

                if plot_isocontour == True:
                    #get velocity to plot for isocontour plots
                    if velocity_comp == "Horizontal velocity":
                        u = np.array(p.variables["velocityx"][it])
                        v = np.array(p.variables["velocityy"][it])
                        u = Horizontal_velocity(u,v,twist,x,normal,zs,h,height=90) #height only used for longitudinal planes
                    else:
                        u = np.array(p.variables[velocity_comp][it])
                    
                    if fluc_vel == True:
                        u = u - np.mean(u) #get mean from precursor planes

                    #define titles and filenames for isocontour plots
                    if fluc_vel == True:
                        if velocity_comp == "Horizontal velocity":
                            Title = "{0} Plane. \nFluctuating {1} [m/s]: Offset = {2}, Time = {3}s".format(plane_labels[ip], velocity_comp[:],float(offset),np.round(Time[it],2))
                            filename = "{0}_Fluc_{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[:],float(offset),np.round(Time[it],2))
                        else:
                            Title = "{0} Plane. \nFluctuating velocity {1} [m/s]: Offset = {2}, Time = {3}s".format(plane_labels[ip],velocity_comp[-1],float(offsets),np.round(Time[it],2))
                            filename = "{0}_Fluc_vel{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[-1],float(offset),np.round(Time[it],2))
                    else:
                        u = np.array(u)
                        if velocity_comp == "Horizontal velocity":
                            Title = "{0} Plane. \n{1} [m/s]: Offset = {2}, Time = {3}s".format(plane_labels[ip],velocity_comp[:],float(offset),np.round(Time[it],2))
                            filename = "{0}_{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[:],float(offset),np.round(Time[it],2))
                        else:
                            Title = "{0} Plane. \nTotal velocity {1} [m/s]: Offset = {2}, time = {3}s".format(plane_labels[ip],velocity_comp[-1],float(offset),np.round(Time[it],2))
                            filename = "{0}_Tot_vel{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[-1],float(offset),np.round(Time[it],2))
                        
                    isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,out_dir)


############ isocontour movie script ################
            print("line 297", time.time()-start_time)
            if movie_tot_vel_isocontour == True:

                if fluc_vel == True:
                    folder = out_dir+"{0}_Plane_Fluctutating_{1}_{2}/".format(plane_labels[ip],velocity_comp,offset)
                else:
                    folder = out_dir+"{0}_Plane_Total_{1}_{2}/".format(plane_labels[ip],velocity_comp,offset)

                isExist = os.path.exists(folder)
                if isExist == False:
                    os.makedirs(folder)

                    #velocity field
                    if velocity_comp == "Horizontal velocity":
                        u = np.array(p.variables["velocityx"])
                        v = np.array(p.variables["velocityy"])
                        u = Horizontal_velocity(u,v,twist,x,normal,zs,h,height=90) #height only used for longitudinal planes
                    else:
                        u = np.array(p.variables[velocity_comp])

                    def mean_velocity(u):
                        u_k = []
                        for u_j in u:
                            u_k.append(u_j - np.mean(u_j))
                        return u_k
                    
                    if fluc_vel == True:
                        u = mean_velocity(u)
                    

                    print("line 328",time.time()-start_time)

                    #find vmin and vmax for isocontour plots            
                    #min and max over data
                    #for rotor and transverse planes always set cmin = 0
                    if custom_colorbar == False:                            
                        if fluc_vel == False:
                            if plane == "r" and velocity_comp != "velocityz" or plane == "t" and velocity_comp != "velocityz":
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


                    def Update(it):

                        U = u[it] #velocity time step it

                        if type(normal) == int: #rotor plane
                            u_plane = U.reshape(y,x)
                            X,Y = np.meshgrid(ys,zs)
                        elif normal == "z":
                            u_plane = U.reshape(x,y)
                            X,Y = np.meshgrid(xs,ys)
                        elif normal == "x":
                            u_plane = U.reshape(y,x)
                            X,Y = np.meshgrid(ys,zs)

                        Z = u_plane

                        T = Time[it]

                        fig = plt.figure(figsize=(50,30))
                        plt.rcParams['font.size'] = 40

                        cs = plt.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)
                        if normal == "x":
                            plt.xlabel("Y axis [m]")
                            plt.ylabel("Z axis [m]")
                        elif normal == "y":
                            plt.xlabel("X axis [m]")
                            plt.ylabel("Z axis [m]")
                        elif normal == "z":
                            plt.xlabel("X axis [m]")
                            plt.ylabel("Y axis [m]")
                        else:
                            plt.xlabel("Y' axis (rotor frame of reference) [m]")
                            plt.ylabel("Z' axis (rotor frame of reference) [m]")

                        cb = plt.colorbar(cs)

                        if plane == "r" and offset == 0.0 or plane == "r" and offset == "-63.0":
                            YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

                            plt.plot(YB1,ZB1,color="k",linewidth = 0.5)
                            plt.plot(YB2,ZB2,color="k",linewidth = 0.5)
                            plt.plot(YB3,ZB3,color="k",linewidth = 0.5)  

                        #define titles and filenames for movie
                        if fluc_vel == True:
                            if velocity_comp == "Horizontal velocity":
                                Title = "{0} Plane. \nFluctuating {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_labels[ip], velocity_comp[:],float(offset),round(T,4))
                                filename = "{0}_Fluc_{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[:],float(offset),round(T,4))
                            else:
                                Title = "{0} Plane. \nFluctuating velocity {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_labels[ip],velocity_comp[-1],float(offset),round(T,4))
                                filename = "{0}_Fluc_vel{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[-1],float(offset),round(T,4))
                        else:
                            if velocity_comp == "Horizontal velocity":
                                Title = "{0} Plane. \n{1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_labels[ip],velocity_comp[:],float(offset),round(T,4))
                                filename = "{0}_{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[:],float(offset),round(T,4))
                            else:
                                Title = "{0} Plane. \nTotal velocity {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_labels[ip],velocity_comp[-1],float(offset),round(T,4))
                                filename = "{0}_Tot_vel{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[-1],float(offset),round(T,4))

                        plt.title(Title)
                        plt.savefig(folder+filename)
                        plt.cla()
                        cb.remove()
                        plt.close(fig)

                        return T

                    with Pool() as pool:
                        for T in pool.imap(Update,time_steps):

                            print(T,time.time()-start_time)

                    time.sleep(60)

                #whether or not folder exists execute code
                #sort files
                def atof(text):
                    try:
                        retval = float(text)
                    except ValueError:
                        retval = text
                    return retval

                def natural_keys(text):
                    
                    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]
                
                print("line 464", time.time()-start_time)

                #define titles and filenames for movie
                if fluc_vel == True:
                    if velocity_comp == "Horizontal velocity":
                        filename = "{0}_Fluc_{1}_{2}.png".format(plane_labels[ip],velocity_comp[:],float(offset))
                    else:
                        filename = "{0}_Fluc_vel{1}_{2}.png".format(plane_labels[ip],velocity_comp[-1],float(offset))
                else:
                    if velocity_comp == "Horizontal velocity":
                        filename = "{0}_{1}_{2}.png".format(plane_labels[ip],velocity_comp[:],float(offset))
                    else:
                        filename = "{0}_Tot_vel{1}_{2}.png".format(plane_labels[ip],velocity_comp[-1],float(offset))

                    
                #sort files
                files = glob.glob(folder+"*.png")
                files.sort(key=natural_keys)

                no_files = math.floor((len(files)/2))
                #write to video
                img_array = []
                it_img = 0
                for file in files[:no_files]:
                    img = cv2.imread(file)
                    height, width, layers = img.shape
                    size = (width,height)
                    img_array.append(img)
                    print("line 454)", Time[time_steps[it_img]],time.time()-start_time)
                    it_img+=1
                
                #cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter(video_folder+filename+"_1"+'.avi',0, 4, size)
                it_vid = 0
                for im in range(len(img_array)):
                    out.write(img_array[im])
                    print("Line 462)",Time[time_steps[it_vid]],time.time()-start_time)
                    it_vid+=1
                out.release(); del img_array
                print("Line 465)",time.time()-start_time)

                
                #write to video
                img_array = []
                for file in files[no_files:]:
                    img = cv2.imread(file)
                    height, width, layers = img.shape
                    size = (width,height)
                    img_array.append(img)
                    print("line 475)", Time[time_steps[it_img]],time.time()-start_time)
                    it_img+=1
                
                #cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter(video_folder+filename+"_2"+'.avi',0, 4, size)
                for im in range(len(img_array)):
                    out.write(img_array[im])
                    print("Line 482)",Time[time_steps[it_vid]],time.time()-start_time)
                    it_vid+=1
                out.release(); del img_array
                print("Line 485)",time.time()-start_time)

            print(plane_labels[ip],velocity_comps[iv],offset,time.time()-start_time)

            iv+=1 #velocity index
        ic+=1 #offset index
    ip+=1 #planar index