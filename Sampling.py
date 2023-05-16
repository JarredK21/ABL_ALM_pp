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

#this was updated 16/05

def offset_data(p_h,velocity_comp, i, no_cells_offset,it):


    if velocity_comp == "coordinates":
        u = np.array(p_h.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_h.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice


#isocontourplot
def isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,dir):
    
    if normal != 1: #rotor plane
        u_plane = u.reshape(y,x) #needs fixing x and y lengths and number of points aren't consistent.
        X,Y = np.meshgrid(ys,zs)
    else:
        u_plane = u.reshape(x,y)
        X,Y = np.meshgrid(xs,ys)


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



#initalize variables
sampling = glob.glob("../post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))

p_h = a.groups["p_sw1"] #modify to select correct sampling file
#p_h = a.groups["p_h"]

no_cells = len(p_h.variables["coordinates"])
no_offsets = len(p_h.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

x = p_h.ijk_dims[0] #no. data points
y = p_h.ijk_dims[1] #no. data points

#find normal
if p_h.axis3[0] == 1:
    normal = "x"
elif p_h.axis3[1] == 1:
    normal = "y"
elif p_h.axis3[2] == 1:
    normal = "z"
else:
    normal = int(np.degrees(np.arccos(p_h.axis3[0])))


it = 0; i = 2; velocity_comp="coordinates" #coordinates at the rotor plane
coordinates = offset_data(p_h,velocity_comp, i, no_cells_offset,it)

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
    zs = np.linspace(p_h.origin[2],p_h.origin[2]+p_h.axis2[2],y)
else:
    xs = np.linspace(p_h.origin[0],p_h.origin[0]+p_h.axis1[2],x)
    ys = np.linspace(p_h.origin[1],p_h.origin[1]+p_h.axis2[1],y)
    zs = 0


tstart = 50
tend = 55
CFD_dt = 0.0039 #manual input
Time = np.array(a.variables["time"])
Time = Time - Time[0]
tstart_idx = np.searchsorted(Time,tstart)
tend_idx = np.searchsorted(Time,tend)
time_steps = np.arange(tstart_idx,tend_idx)
dt = round(a.variables["time"][1] - a.variables["time"][0],4)
frequency = dt/CFD_dt


dir = "../post_processing/plots/"

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

if plot_specific_offsets == True:    
    spec_offsets = [2] #rotor plane
else:
    spec_offsets = np.arange(0,no_offsets, 1, dtype=int)

offsets = []
for offset in spec_offsets:
    offsets.append(p_h.offsets[offset])


col_names = []
for col in offsets:
    col_names.append(str(col))
plane_data =  pd.DataFrame(data=None, columns=col_names)


#specify time steps to plot instantaneous isocontours at
it_array = [0]


start_time = time.time()
#loop over true velocity components
velocity_comps = ["velocityx","velocityy","velocityz","Magnitude horizontal velocity"]
iv = 0
for velocity_comp in velocity_comps:
    if velocity_plot[iv] == False:
        iv+=1
        continue

    #loop over offsets
    for i in spec_offsets:
        for it in it_array: #parallel
            if velocity_comp == "Magnitude horizontal velocity":
                u = offset_data(p_h,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                v = offset_data(p_h,velocity_comps[1], i, no_cells_offset,it)
                u = np.add( np.multiply(u,np.cos(np.radians(29))) , np.multiply( v,np.sin(np.radians(29))) )

            else:
                u = offset_data(p_h,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays

            plane_data[str(p_h.offsets[i])] = u

            #plots it = time specified above
            if plot_isocontour == True:
                offset = str(p_h.offsets[i])
                if fluc_vel == True:
                    u = np.array(plane_data[:][offset]) - np.mean(np.array(plane_data[:][offset]))
                    Title = "Fluctuating velocity {0} [m/s]: Offset = {1}, Time = {2}s".format(velocity_comp[-1],float(offset),np.round(Time[it],2))
                    filename = "Fluc_vel{0}_{1}_{2}.png".format(velocity_comp[-1],float(offset),np.round(Time[it],2))
                elif velocity_comp == "Magnitude horizontal velocity":
                    u = np.array(plane_data[:][offset])
                    Title = "{0} [m/s]: Offset = {1}, Time = {2}s".format(velocity_comp[:],float(offset),np.round(Time[it],2))
                    filename = "{0}_{1}_{2}.png".format(velocity_comp[:],float(offset),np.round(time[it],2))
                else:
                    u = np.array(plane_data[:][offset])
                    Title = "Total velocity {0} [m/s]: Offset = {1}, time = {2}s".format(velocity_comp[-1],float(offset),np.round(Time[it],2))
                    filename = "Tot_vel{0}_{1}.png".format(velocity_comp[-1],float(offset),np.round(Time[it],2))
                    
                isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,dir)


        print("line 227", time.time()-start_time)
        #generate movie for specific plane
        if movie_tot_vel_isocontour == True:


            #metadata = dict(title="Movie",artist="Jarred")
            #writer = PillowWriter(fps=25,metadata=metadata)

            if fluc_vel == True:
                f = "Fluctuating"
            else:
                f = "Total"
            if velocity_comp == "Magnitude horizontal velocity":
                ft = f + " Velocity $<Ux'>$ [m/s]"
                fn = f + "_velHz"
            else:
                ft = f + " {} [m/s]".format(velocity_comp)
                fn = f + "_{}".format(velocity_comp)

            filename = "{0}_Offset={1}".format(fn,p_h.offsets[i])
            print(ft,fn)

            def vmin_vmax(it):
                    
                if velocity_comp == "Magnitude horizontal velocity":
                    u = offset_data(p_h,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                    v = offset_data(p_h,velocity_comps[1], i, no_cells_offset,it)
                    u = np.add( np.multiply(u,np.cos(np.radians(29))) , np.multiply( v,np.sin(np.radians(29))) )

                else:
                    u = offset_data(p_h,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays
                
                if fluc_vel == True:
                    u = u - np.mean(u)

                return np.min(u), np.max(u)

            #find vmin and vmax for isocontour plots            
            #min and max over data
            vmin_arr = []; vmax_arr = []
            with Pool() as pool:
                for vmin,vmax in pool.imap(vmin_vmax,time_steps):
                    
                    vmin_arr.append(vmin); vmax_arr.append(vmax)

            cmin = math.floor(np.min(vmin_arr)); cmax = math.ceil(np.max(vmax_arr))
            print("line 268",time.time()-start_time)
            nlevs = (cmax-cmin)
            levels = np.linspace(cmin,cmax,nlevs,dtype=int)



            def Update(it):

                if velocity_comp == "Magnitude horizontal velocity":
                    u = offset_data(p_h,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                    v = offset_data(p_h,velocity_comps[1], i, no_cells_offset,it)
                    u = np.add( np.multiply(u,np.cos(np.radians(29))) , np.multiply( v,np.sin(np.radians(29))) )

                else:
                    u = offset_data(p_h,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays

                if fluc_vel == True:
                    u = u - np.mean(u)

                if normal != 1: #rotor plane
                    u_plane = u.reshape(y,x)
                    X,Y = np.meshgrid(ys,zs)
                else:
                    u_plane = u.reshape(x,y)
                    X,Y = np.meshgrid(xs,ys)

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

                Title = "{0}, Offset = {1}, Time = {2}[s]".format(ft,p_h.offsets[i],round(T,4))
                plt.title(Title)
                
                #writer.grab_frame()
                plt.savefig(dir+"{0}_{1}.png".format(filename,round(T,4)))
                plt.cla()
                cb.remove()
                plt.close(fig)

                return T

            #with writer.saving(fig,dir+"{0}".format(filename),len(time_steps)):
            with Pool() as pool:
                for T in pool.imap(Update,time_steps):

                    print(T,time.time()-start_time)


            #sort files
            def atoi(text):
                return int(text) if text.isdigit() else text

            def natural_keys(text):

                return [ atoi(c) for c in re.split(r'(\d+)', text) ]

            files = glob.glob(dir+filename+"*")

            files.sort(key=natural_keys)

            #write to video
            img_array = []
            for file in files:
                print(file)
                img = cv2.imread(file)
                height, width, layers = img.shape
                size = (width,height)
                img_array.append(img)
            
            
            out = cv2.VideoWriter(dir+filename+'.gif',cv2.VideoWriter_fourcc(*'DIVX'), 15, size)
            
            for i in range(len(img_array)):
                out.write(img_array[i])
            out.release()

    iv+=1 #velocity index
    print(velocity_comp,time.time()-start_time)