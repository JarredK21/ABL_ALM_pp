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


def offset_data(p,velocity_comp, i, no_cells_offset,it):


    if velocity_comp == "coordinates":
        u = np.array(p.variables[velocity_comp]) #only time step
    else:
        u = np.array(p.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice

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
    Y = [2500]; Y2 = [2500]; Y3 = [2500]
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


def mean_velocity(it,Var):
    
    Time_it = Time[it]#find time from sampled data

    it_pre = np.searchsorted(Time_pre,Time_it)

    print(Time_pre[it_pre])

    if Var == "Horizontal velocity":
        u = mean_profiles["velocityx"][it_pre][:]
        v = mean_profiles["velocityy"][it_pre][:]
        u = np.add( np.multiply(u,np.cos(twist)), np.multiply(v,np.sin(twist)) )
    else:
        u = mean_profiles["{}".format(Var)][it_pre][:]

    return np.mean(u)



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

#openfast data
da = io.fast_output_file.FASTOutputFile("../NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()
db = io.fast_output_file.FASTOutputFile("../../NREL_5MW_MCBL_R_CRPM_100320/NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

#combine time
restart_time = 137.748
Time_a_OF = np.array(da["Time_[s]"]); Time_b_OF = np.array(db["Time_[s]"]); Time_b_OF = Time_b_OF+restart_time
restart_idx = np.searchsorted(Time_a_OF,restart_time); restart_idx-=1
Time_OF = np.concatenate((Time_a_OF[0:restart_idx],Time_b_OF))

#combine openFAST outputs
df = pd.concat((da[:][0:restart_idx],db[:])); del da; del db

#Azimuthal position for blade 1
Azimuth = np.array(np.radians(df["Azimuth_[deg]"]))


#directories
in_dir = "./"
out_dir = in_dir + "plots/"
video_folder = in_dir + "videos/"
isExist = os.path.exists(video_folder)
if isExist == False:
    os.makedirs(video_folder)

#initalize variables manual input
a = Dataset("./sampling.nc")

#l - longitudinal xy
#r - rotor 29deg yz
#t - tranverse yz
planes = ["l", "r", "t"]
plane_label = ["Longitudinal", "Rotor","Transverse"]


ip = 0
for plane in planes:

    p = a.groups["p_{0}".format(plane)]

    #time options
    Time = np.array(a.variables["time"])
    Time = Time - Time[0]

    del a

    no_cells = len(p.variables["coordinates"])
    if isinstance(p.offsets,np.float64) == True:
        offsets = [p.offsets]
    else:
        offsets = p.offsets
    no_offsets = len(offsets)
    no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

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


    it = 0; i = 0; velocity_comp="coordinates" #coordinates at the rotor plane
    coordinates = offset_data(p,velocity_comp, i, no_cells_offset,it)

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
        tstart = 50
        tend = 600
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


    #longitudinal offsets - 85m
    #rotor offsets - 0.0m, -63m, -126m
    #tranverse offsets - -10D, -5D, +5D, +10D
    if plot_specific_offsets == True:    
        spec_offsets = [[0],[0]] #longitudinal, rotor, transverse
        Offsets = []
        for offset_idx in spec_offsets[ip]:
            Offsets.append(offsets[offset_idx])
    else:
        Offsets = offsets



    start_time = time.time()
    #loop over true velocity components
    velocity_comps = ["velocityx","velocityy","velocityz","Horizontal velocity"]
    iv = 0
    for velocity_comp in velocity_comps:
        if velocity_plot[iv] == False:
            iv+=1
            continue

        #colorbar options
        custom_colorbar = False
        cmin = 0; cmax = 18

        #loop over offsets
        for i in np.arange(0,len(Offsets)):
            print(plane_label[ip],velocity_comps[iv],Offsets[i],time.time()-start_time)
            for it in it_array:

                if plot_isocontour == True:
                    #get velocity to plot for isocontour plots
                    if velocity_comp == "Horizontal velocity":
                        u = offset_data(p,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                        v = offset_data(p,velocity_comps[1], i, no_cells_offset,it)
                        u = Horizontal_velocity(u,v,twist,x,normal,zs,h,height=90) #height only used for longitudinal planes
                    else:
                        u = offset_data(p,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays
                    
                    if fluc_vel == True:
                        # mean_pre_velocity = mean_velocity(it,velocity_comp)
                        u = np.array(u) - np.mean(np.array(u)) #get mean from precursor planes

                    #define titles and filenames for isocontour plots
                    if fluc_vel == True:
                        if velocity_comp == "Horizontal velocity":
                            Title = "{0} Plane. \nFluctuating {1} [m/s]: Offset = {2}, Time = {3}s".format(plane_label[ip], velocity_comp[:],float(Offsets[i]),np.round(Time[it],2))
                            filename = "{0}_Fluc_{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[:],float(Offsets[i]),np.round(Time[it],2))
                        else:
                            Title = "{0} Plane. \nFluctuating velocity {1} [m/s]: Offset = {2}, Time = {3}s".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]),np.round(Time[it],2))
                            filename = "{0}_Fluc_vel{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]),np.round(Time[it],2))
                    else:
                        u = np.array(u)
                        if velocity_comp == "Horizontal velocity":
                            Title = "{0} Plane. \n{1} [m/s]: Offset = {2}, Time = {3}s".format(plane_label[ip],velocity_comp[:],float(Offsets[i]),np.round(Time[it],2))
                            filename = "{0}_{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[:],float(Offsets[i]),np.round(Time[it],2))
                        else:
                            Title = "{0} Plane. \nTotal velocity {1} [m/s]: Offset = {2}, time = {3}s".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]),np.round(Time[it],2))
                            filename = "{0}_Tot_vel{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]),np.round(Time[it],2))
                        
                    isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,out_dir)


############ isocontour movie script ################
            print("line 279", time.time()-start_time)
            if movie_tot_vel_isocontour == True:

                if fluc_vel == True:
                    folder = out_dir+"{0}_Plane_Fluctutating_{1}_{2}/".format(plane_label[ip],velocity_comp,Offsets[i])
                else:
                    folder = out_dir+"{0}_Plane_Total_{1}_{2}/".format(plane_label[ip],velocity_comp,Offsets[i])

                isExist = os.path.exists(folder)
                if isExist == False:
                    os.makedirs(folder)

                    def vmin_vmax(it):
                            
                        if velocity_comp == "Horizontal velocity":
                            u = offset_data(p,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                            v = offset_data(p,velocity_comps[1], i, no_cells_offset,it)
                            u = Horizontal_velocity(u,v,twist,x,normal,zs,h,height=90) #height only used for longitudinal planes
                        else:
                            u = offset_data(p,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays
                        
                        if fluc_vel == True:
                            # mean_pre_velocity = mean_velocity(it,velocity_comp)
                            u = np.array(u) - np.mean(np.array(u))

                        return np.min(u), np.max(u)

                    #find vmin and vmax for isocontour plots            
                    #min and max over data
                    #for rotor and transverse planes always set cmin = 0
                    if custom_colorbar == False:
                        vmin_arr = []; vmax_arr = []
                        with Pool() as pool:
                            for vmin,vmax in pool.imap(vmin_vmax,time_steps):
                                
                                vmin_arr.append(vmin); vmax_arr.append(vmax)
                                print("line 380", len(vmin))
                                
                        if fluc_vel == False:
                            if plane == "r" and velocity_comp != "velocityz" or plane == "t" and velocity_comp != "velocityz":
                                cmin = 0
                            else:
                                cmin = math.floor(np.min(vmin_arr))
                        
                        cmax = math.ceil(np.max(vmax_arr)); del vmin_arr; del vmax_arr
                    
                    #if custom_colorbar == True: specify cmain, cmax above
                    nlevs = (cmax-cmin)
                    levels = np.linspace(cmin,cmax,nlevs,dtype=int)



                    def Update(it):

                        if velocity_comp == "Horizontal velocity":
                            u = offset_data(p,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                            v = offset_data(p,velocity_comps[1], i, no_cells_offset,it)
                            u = Horizontal_velocity(u,v,twist,x,normal,zs,h,height=90) #height only used for longitudinal planes
                        else:
                            u = offset_data(p,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays

                        if fluc_vel == True:
                            # mean_pre_velocity = mean_velocity(it,velocity_comp)
                            u = np.array(u) - np.mean(np.array(u))

                        if type(normal) == int: #rotor plane
                            u_plane = u.reshape(y,x)
                            X,Y = np.meshgrid(ys,zs)
                        elif normal == "z":
                            u_plane = u.reshape(x,y)
                            X,Y = np.meshgrid(xs,ys)
                        elif normal == "x":
                            u_plane = u.reshape(y,x)
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

                        if plane == "r" and Offsets[i] == 0.0:
                            YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

                            plt.plot(YB1,ZB1,color="k",linewidth = 0.5)
                            plt.plot(YB2,ZB2,color="k",linewidth = 0.5)
                            plt.plot(YB3,ZB3,color="k",linewidth = 0.5)  

                        #define titles and filenames for movie
                        if fluc_vel == True:
                            if velocity_comp == "Horizontal velocity":
                                Title = "{0} Plane. \nFluctuating {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_label[ip], velocity_comp[:],float(Offsets[i]),round(T,4))
                                filename = "{0}_Fluc_{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[:],float(Offsets[i]),round(T,4))
                            else:
                                Title = "{0} Plane. \nFluctuating velocity {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]),round(T,4))
                                filename = "{0}_Fluc_vel{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]),round(T,4))
                        else:
                            if velocity_comp == "Horizontal velocity":
                                Title = "{0} Plane. \n{1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_label[ip],velocity_comp[:],float(Offsets[i]),round(T,4))
                                filename = "{0}_{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[:],float(Offsets[i]),round(T,4))
                            else:
                                Title = "{0} Plane. \nTotal velocity {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]),round(T,4))
                                filename = "{0}_Tot_vel{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]),round(T,4))

                        plt.title(Title)
                        plt.savefig(folder+filename)
                        plt.cla()
                        cb.remove()
                        plt.close(fig)

                        return T

                    with Pool() as pool:
                        for T in pool.imap(Update,time_steps):

                            print(T,time.time()-start_time)


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
                
                print("line 372", time.time()-start_time)
                if fluc_vel == True:
                    if velocity_comp == "Horizontal velocity":
                        filename = "{0}_{1}_{2}.png".format(plane_label[ip],velocity_comp[:],float(Offsets[i]))
                    else:
                        filename = "{0}_Fluc_vel{1}_{2}.png".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]))
                else:
                    if velocity_comp == "Horizontal velocity":
                        filename = "{0}_{1}_{2}.png".format(plane_label[ip],velocity_comp[:],float(Offsets[i]))
                    else:
                        filename = "{0}_Tot_vel{1}_{2}.png".format(plane_label[ip],velocity_comp[-1],float(Offsets[i]))

                    
                #sort files
                files = glob.glob(folder+filename[0:-4]+"*.png")
                files.sort(key=natural_keys)

                #write to video
                img_array = []
                it = 0
                for file in files:
                    img = cv2.imread(file)
                    height, width, layers = img.shape
                    size = (width,height)
                    img_array.append(img)
                    print("line 441)", Time[time_steps[it]],time.time()-start_time)
                    it+=1
                
                #cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter(video_folder+filename+'.avi',0, 1, size)
                it = 0
                for im in range(len(img_array)):
                    out.write(img_array[im])
                    print("Line 449)",Time[time_steps[it]],time.time()-start_time)
                    it+=1
                out.release(); del img_array
                print("Line 452)",time.time()-start_time)

            print(plane_label[ip],velocity_comps[iv],Offsets[i],time.time()-start_time)

        iv+=1 #velocity index
    ip+=1 #planar index