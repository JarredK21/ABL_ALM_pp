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
planes = ["l", "r", "t"]
plane_label = ["Longitudinal", "Rotor","Transverse"]

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
        ys = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],x)
        zs = np.linspace(p.origin[2],p.origin[2]+p.axis1[2],y)
    elif normal == "z":
        xs = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x)
        ys = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y)
        zs = 0


    # tstart = 50
    # tend = 350
    #CFD_dt = 0.0039 #manual input
    Time = np.array(a.variables["time"])
    Time = Time - Time[0]
    # tstart_idx = np.searchsorted(Time,tstart)
    # tend_idx = np.searchsorted(Time,tend)
    # time_steps = np.arange(tstart_idx,tend_idx)
    # dt = round(a.variables["time"][1] - a.variables["time"][0],4)
    # frequency = dt/CFD_dt




    #plotting option
    plot_isocontour = True
    plot_u = False; plot_v = False; plot_w = False; plot_hvelmag = True
    velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]

    #check if no velocity components selected
    if all(list(map(operator.not_, velocity_plot))) == True:
        sys.exit("error no velocity component selected")


    fluc_vel = False
    movie_tot_vel_isocontour = False
    plot_specific_offsets = False

    #longitudinal offsets - 85m
    #rotor offsets - 0.0m, -63m, 126m
    #tranverse offsets - -10D, -5D, +5D, +10D
    if plot_specific_offsets == True:    
        spec_offsets = [1]
        offsets = []
        for offset_idx in spec_offsets:
            offsets.append(offsets[offset_idx])
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

    ip+=1