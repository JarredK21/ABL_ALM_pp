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
        u_plane = u.reshape(y,x) #needs fixing x and y lengths and number of points aren't consistent.
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





#directories
in_dir = "../../NREL_5MW_MCBL_R_CRPM/test_post_processing/"
out_dir = in_dir + "plots/"

#initalize variables
sampling = glob.glob(in_dir + "sampling*")
a = Dataset("./{}".format(sampling[0]))

#l - longitudinal xy
#r - rotor 29deg yz
#t - tranverse yz
# planes = ["l", "r", "t"]
# plane_label = ["Longitudinal", "Rotor","Transverse"]

planes = ["r"]
plane_label = ["Rotor"]

ip = 0
for plane in planes:

    p = a.groups["p_{0}".format(plane)]
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


    #time options
    CFD_dt = 0.0039 #manual input
    Time = np.array(a.variables["time"])
    Time = Time - Time[0]
    dt = round(a.variables["time"][1] - a.variables["time"][0],4)
    frequency = dt/CFD_dt
 
    plot_all_times = True
    if plot_all_times == False:
        tstart = 50
        tend = 350
        tstart_idx = np.searchsorted(Time,tstart)
        tend_idx = np.searchsorted(Time,tend)
        time_steps = np.arange(tstart_idx,tend_idx)
    else:
        tend_idx = np.searchsorted(Time,Time[-1])
        time_steps = np.arange(0,tend_idx)


    #plotting option
    plot_isocontour = False
    plot_u = False; plot_v = False; plot_w = False; plot_hvelmag = True
    velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]

    #check if no velocity components selected
    if all(list(map(operator.not_, velocity_plot))) == True:
        sys.exit("error no velocity component selected")


    fluc_vel = False
    movie_tot_vel_isocontour = True
    plot_specific_offsets = True

    #longitudinal offsets - 85m
    #rotor offsets - 0.0m, -63m, 126m
    #tranverse offsets - -10D, -5D, +5D, +10D
    if plot_specific_offsets == True:    
        spec_offsets = [0]
        Offsets = []
        for offset_idx in spec_offsets:
            Offsets.append(offsets[offset_idx])
    else:
        spec_offsets = offsets


    #specify time steps to plot instantaneous isocontours at
    it_array = [0,10]

    #colorbar options
    custom_colorbar = False
    cmin = 3; cmax = 18 #custom range


    start_time = time.time()
    #loop over true velocity components
    velocity_comps = ["velocityx","velocityy","velocityz","Magnitude horizontal velocity"]
    iv = 0
    for velocity_comp in velocity_comps:
        if velocity_plot[iv] == False:
            iv+=1
            continue

        #loop over offsets
        for i in np.arange(0,len(spec_offsets)):
            for it in it_array: #parallel
                if velocity_comp == "Magnitude horizontal velocity":
                    u = offset_data(p,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                    v = offset_data(p,velocity_comps[1], i, no_cells_offset,it)
                    u = np.add( np.multiply(u,np.cos(np.radians(29))) , np.multiply( v,np.sin(np.radians(29))) )

                else:
                    u = offset_data(p,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays

                #plots it = time specified above
                if plot_isocontour == True:
                    if fluc_vel == True:
                        u = np.array(u) - np.mean(np.array(u))
                        if velocity_comp == "Magnitude horizontal velocity":
                            Title = "{0} Plane. \nFluctuating {1} [m/s]: Offset = {2}, Time = {3}s".format(plane_label[ip], velocity_comp[:],float(spec_offsets[i]),np.round(Time[it],2))
                            filename = "{0}_Fluc_{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[:],float(spec_offsets[i]),np.round(Time[it],2))
                        else:
                            Title = "{0} Plane. \nFluctuating velocity {1} [m/s]: Offset = {2}, Time = {3}s".format(plane_label[ip],velocity_comp[-1],float(spec_offsets[i]),np.round(Time[it],2))
                            filename = "{0}_Fluc_vel{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[-1],float(spec_offsets[i]),np.round(Time[it],2))
                    else:
                        u = np.array(u)
                        if velocity_comp == "Magnitude horizontal velocity":
                            Title = "{0} Plane. \n{1} [m/s]: Offset = {2}, Time = {3}s".format(plane_label[ip],velocity_comp[:],float(spec_offsets[i]),np.round(Time[it],2))
                            filename = "{0}_{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[:],float(spec_offsets[i]),np.round(Time[it],2))
                        else:
                            Title = "{0} Plane. \nTotal velocity {1} [m/s]: Offset = {2}, time = {3}s".format(plane_label[ip],velocity_comp[-1],float(spec_offsets[i]),np.round(Time[it],2))
                            filename = "{0}_Tot_vel{1}_{2}_{3}.png".format(plane_label[ip],velocity_comp[-1],float(spec_offsets[i]),np.round(Time[it],2))
                        
                    isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,out_dir)


############ isocontour movie script ################
            print("line 239", time.time()-start_time)
            #generate movie for specific plane
            if movie_tot_vel_isocontour == True:

                folder = out_dir+"{0}_{1}/".format(velocity_comp,offsets[i])
                #need to delete or change name of folder if folder exists
                os.makedirs(folder)

                if fluc_vel == True:
                    if velocity_comp == "Magnitude horizontal velocity":
                        Title = "{0} Plane. \nFluctuating {1} [m/s]: Offset = {2}".format(plane_label[ip], velocity_comp[:],float(spec_offsets[i]))
                        filename = "{0}_Fluc_{1}_{2}.png".format(plane_label[ip],velocity_comp[:],float(spec_offsets[i]))
                    else:
                        Title = "{0} Plane. \nFluctuating velocity {1} [m/s]: Offset = {2}".format(plane_label[ip],velocity_comp[-1],float(spec_offsets[i]))
                        filename = "{0}_Fluc_vel{1}_{2}.png".format(plane_label[ip],velocity_comp[-1],float(spec_offsets[i]))
                else:
                    if velocity_comp == "Magnitude horizontal velocity":
                        Title = "{0} Plane. \n{1} [m/s]: Offset = {2}".format(plane_label[ip],velocity_comp[:],float(spec_offsets[i]))
                        filename = "{0}_{1}_{2}.png".format(plane_label[ip],velocity_comp[:],float(spec_offsets[i]))
                    else:
                        Title = "{0} Plane. \nTotal velocity {1} [m/s]: Offset = {2}".format(plane_label[ip],velocity_comp[-1],float(spec_offsets[i]))
                        filename = "{0}_Tot_vel{1}_{2}.png".format(plane_label[ip],velocity_comp[-1],float(spec_offsets[i]))
                    
                print(filename)


                def vmin_vmax(it):
                        
                    if velocity_comp == "Magnitude horizontal velocity":
                        u = offset_data(p,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                        v = offset_data(p,velocity_comps[1], i, no_cells_offset,it)
                        u = np.add( np.multiply(u,np.cos(np.radians(29))) , np.multiply( v,np.sin(np.radians(29))) )

                    else:
                        u = offset_data(p,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays
                    
                    if fluc_vel == True:
                        u = u - np.mean(u)

                    return np.min(u), np.max(u)

                #find vmin and vmax for isocontour plots            
                #min and max over data
                if custom_colorbar == False:
                    vmin_arr = []; vmax_arr = []
                    with Pool() as pool:
                        for vmin,vmax in pool.imap(vmin_vmax,time_steps):
                            
                            vmin_arr.append(vmin); vmax_arr.append(vmax)

                    cmin = math.floor(np.min(vmin_arr)); cmax = math.ceil(np.max(vmax_arr))
                
                #if custom_colorbar == True: specify cmain, cmax above
                print("line 292",time.time()-start_time)
                nlevs = (cmax-cmin)
                levels = np.linspace(cmin,cmax,nlevs,dtype=int)



                def Update(it):

                    if velocity_comp == "Magnitude horizontal velocity":
                        u = offset_data(p,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                        v = offset_data(p,velocity_comps[1], i, no_cells_offset,it)
                        u = np.add( np.multiply(u,np.cos(np.radians(29))) , np.multiply( v,np.sin(np.radians(29))) )

                    else:
                        u = offset_data(p,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays

                    if fluc_vel == True:
                        u = u - np.mean(u)

                    if type(normal) == int: #rotor plane
                        u_plane = u.reshape(y,x) #needs fixing x and y lengths and number of points aren't consistent.
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

                    Title = Title + ", Time = {0}[s]".format(round(T,4))
                    plt.title(Title)

                    plt.savefig(folder+"{0}_{1}.png".format(filename,round(T,4)))
                    plt.cla()
                    cb.remove()
                    plt.close(fig)

                    return T

                with Pool() as pool:
                    for T in pool.imap(Update,time_steps):

                        print(T,time.time()-start_time)


                #sort files
                def atof(text):
                    try:
                        retval = float(text)
                    except ValueError:
                        retval = text
                    return retval

                def natural_keys(text):
                    
                    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]
                
                #sort files
                files = glob.glob(folder+filename+"*.png")
                files.sort(key=natural_keys)

                #write to video
                img_array = []
                it = 0
                for file in files:
                    img = cv2.imread(file)
                    height, width, layers = img.shape
                    size = (width,height)
                    img_array.append(img)
                    print(Time[time_steps[it]],time.time()-start_time)
                    it+=1
                
                #cv2.VideoWriter_fourcc(*'DIVX')
                out = cv2.VideoWriter(folder+filename+'.avi',0, 15, size)
                it = 0
                for im in range(len(img_array)):
                    out.write(img_array[im])
                    print(Time[time_steps[it]],time.time()-start_time)
                    it+=1
                out.release()
                print("Line 264",time.time()-start_time)

    iv+=1 #velocity index
    print(velocity_comp,time.time()-start_time)

    ip+=1