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


def offset_data(p_h,velocity_comp, i, no_cells_offset,it):


    if velocity_comp == "coordinates":
        u = np.array(p_h.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_h.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice


def vmin_vmax(p_h,velocity_comp,i,no_cells_offset):
            
        #min and max over data
        vmin_arr = []; vmax_arr = []
        with Pool() as pool:
            for u in pool.imap(offset_data,time_steps): #missing inputs

                if fluc_vel == True:
                    u = u - np.mean(u)
                
                vmin_arr.append(np.min(u)); vmax_arr.append(np.max(u))
                print(time.time()-start_time)

        vmin = math.floor(np.min(vmin_arr)); vmax = math.ceil(np.max(vmax_arr))

        return vmin, vmax


#isocontourplot
def isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,dir):
    
    # if normal != 1: #rotor plane
    #     u_plane = u.reshape(y,x) #needs fixing x and y lengths and number of points aren't consistent.
    #     X,Y = np.meshgrid(ys,zs)
    # else:
    u_plane = u.reshape(y,x)
    X,Y = np.meshgrid(xs,ys)


    fig = plt.figure(figsize=(12,6))
    plt.rcParams['font.size'] = 16
    
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

    plt.title(Title,fontsize=16)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(dir+"{}".format(filename))
    plt.close(fig)


dir = "../../ABL_precursor/post_processing/plots/"
#initalize variables
sampling = glob.glob("../../ABL_precursor/post_processing/horz_sampling*")
a = Dataset("./{}".format(sampling[0]))

#p_h = a.groups["p_sw1"] #modify to select correct sampling file
p_h = a.groups["p_h"]

no_cells = len(p_h.variables["coordinates"])
#no_offsets = len(p_h.offsets)
no_offsets = 1
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


it = 0; i = 0; velocity_comp="coordinates" #coordinates at the rotor plane
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
    xs = np.linspace(p_h.origin[0],p_h.origin[0]+p_h.axis1[0],x)
    ys = np.linspace(p_h.origin[2],p_h.origin[2]+p_h.axis2[2],y)
    zs = 0


tstart = 50
tend = 350
CFD_dt = 0.0039 #manual input
Time = np.array(a.variables["time"])
#Time = Time - Time[0]
tstart_idx = np.searchsorted(Time,tstart)
tend_idx = np.searchsorted(Time,tend)
time_steps = np.arange(tstart_idx,tend_idx)
dt = round(a.variables["time"][1] - a.variables["time"][0],4)
frequency = dt/CFD_dt




#plotting option
plot_isocontour = True
plot_u = False; plot_v = False; plot_w = False; plot_hvelmag = False; plot_temp = True
velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag, plot_temp]

#check if no velocity components selected
if all(list(map(operator.not_, velocity_plot))) == True:
    sys.exit("error no velocity component selected")


fluc_vel = False
movie_tot_vel_isocontour = False
plot_specific_offsets = False

if plot_specific_offsets == True:    
    spec_offsets = [2] #rotor plane
else:
    spec_offsets = np.arange(0,no_offsets, 1, dtype=int)

# offsets = []
# for offset in spec_offsets:
#     offsets.append(p_h.offsets[offset])

offsets = [0.0]


col_names = []
for col in offsets:
    col_names.append(str(col))
plane_data =  pd.DataFrame(data=None, columns=col_names)


#specify time steps to plot instantaneous isocontours at
it_array = [0]


#coriolis twist calc
def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


b = Dataset("../../ABL_precursor/post_processing/abl_statistics60000.nc")

mean_profiles = b.groups["mean_profiles"] #create variable to hold mean profiles

t_start = np.searchsorted(b.variables["time"],32300)
t_end = np.searchsorted(b.variables["time"],33500)

u = np.average(mean_profiles.variables["u"][t_start:t_end][:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end][:],axis=0)

twist = coriolis_twist(u=u,v=v)
z = mean_profiles["h"][:]

zi = np.average(b.variables["zi"][t_start:t_end])


start_time = time.time()
#loop over true velocity components
velocity_comps = ["velocityx","velocityy","velocityz","Magnitude horizontal velocity","temperature"]
iv = 0
for velocity_comp in velocity_comps:
    if velocity_plot[iv] == False:
        iv+=1
        continue

    #loop over offsets
    for i in spec_offsets:
        for it in it_array: #parallel
            if velocity_comp == "Magnitude horizontal velocity":
                
                if normal == "z":
                    offset_height = p_h.offsets[i]
                    offset_height_idx = np.searchsorted(z,offset_height)
                    twist_angle = twist[offset_height_idx]
                    u = offset_data(p_h,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                    v = offset_data(p_h,velocity_comps[1], i, no_cells_offset,it)
                    u = np.add( np.multiply(u,np.cos(twist_angle)) , np.multiply( v,np.sin(twist_angle)) )
                else:
                    u = offset_data(p_h,velocity_comps[0], i, no_cells_offset,it) #slicing data into offset arrays
                    v = offset_data(p_h,velocity_comps[1], i, no_cells_offset,it)
                    u = np.sqrt(np.add(np.square(u),np.square(v)))
            else:
                u = offset_data(p_h,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays

            plane_data[str(offsets[i])] = u

            #plots it = time specified above
            if plot_isocontour == True:
                offset = str(offsets[i])
                if fluc_vel == True and velocity_comp != "Magnitude horizontal velocity":
                    u = np.array(plane_data[:][offset]) - np.mean(np.array(plane_data[:][offset]))
                    #Title = "Fluctuating velocity {0} [m/s]: $z/z_i$ = {1}, Time = {2}s".format(velocity_comp[-1],np.round(float(offset)/zi,2),np.round(Time[it],2))
                    #filename = "Fluc_vel{0}_{1}_{2}.png".format(velocity_comp[-1],float(offset),np.round(Time[it],2))
                    Title = "Fluctuating velocity {0} [m/s]: x-z plane {1}m, Time = {2}s".format(velocity_comp[-1],p_h.origin[1],np.round(Time[it],2))
                    filename = "Fluc_vel{0}_x_z_{2}.png".format(velocity_comp[-1],float(offset),np.round(Time[it],2))
                elif fluc_vel != True and velocity_comp == "Magnitude horizontal velocity":
                    u = np.array(plane_data[:][offset])
                    #Title = "{0} [m/s]: $z/z_i$ = {1}, Time = {2}s".format(velocity_comp[:],np.round(float(offset)/zi,2),np.round(Time[it],2))
                    #filename = "{0}_{1}_{2}.png".format(velocity_comp[:],float(offset),np.round(time[it],2))
                    Title = "{0} [m/s]: x-z plane {1}m, Time = {2}s".format(velocity_comp[:],p_h.origin[1],np.round(Time[it],2))
                    filename = "{0}_x_z_{2}.png".format(velocity_comp[:],float(offset),np.round(Time[it],2))
                elif velocity_comp == "Magnitude horizontal velocity" and fluc_vel == True:
                    u = np.array(plane_data[:][offset]) - np.mean(np.array(plane_data[:][offset]))
                    #Title = "Fluctuating horizontal velocity [m/s]: $z/z_i$ = {0}, Time = {1}s".format(np.round(float(offset)/zi,2),np.round(Time[it],2))
                    #filename = "Fluc_hvel{0}_{1}.png".format(float(offset),np.round(Time[it],2))
                    Title = "Fluctuating horizintal velocity [m/s]: x-z plane {0}m, Time = {1}s".format(p_h.origin[1],np.round(Time[it],2))
                    filename = "Fluc_hvel_x_z_{0}.png".format(np.round(Time[it],2))
                elif velocity_comp == "temperature":
                    u = np.array(plane_data[:][offset])
                    Title = "{0} [K]: x-z plane {1}m, time = {2}s".format(velocity_comp[:],p_h.origin[1],np.round(Time[it],2))
                    filename = "{0}_x_z_{1}.png".format(velocity_comp[:],np.round(Time[it],2))
                else:
                    u = np.array(plane_data[:][offset])
                    Title = "Total velocity {0} [m/s]: $z/z_i$ = {1}, time = {2}s".format(velocity_comp[-1],np.round(float(offset)/zi,2),np.round(Time[it],2))
                    filename = "Tot_vel{0}_{1}.png".format(velocity_comp[-1],float(offset),np.round(Time[it],2))
                    
                isocontourplot(u,x,y,normal,xs,ys,zs,Title,filename,dir)


        print("line 227", time.time()-start_time)
        #generate movie for specific plane
        if movie_tot_vel_isocontour == True:

            fig = plt.figure(figsize=(50,30))
            plt.rcParams['font.size'] = 40


            def Update(it):
                u = offset_data(p_h,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays

                if fluc_vel == True:
                    u = u - np.mean(u)

                if normal != 1: #rotor plane
                    u_plane = u.reshape(y,x) #needs fixing x and y lengths and number of points aren't consistent.
                    X,Y = np.meshgrid(ys,zs)
                else:
                    u_plane = u.reshape(x,y)
                    X,Y = np.meshgrid(xs,ys)

                Z = u_plane

                return X,Y,Z


            metadata = dict(title="Movie",artist="Jarred")
            writer = PillowWriter(fps=5,metadata=metadata)

            if fluc_vel == True:
                f = "Fluctuating"
            else:
                f = "Total"

            ft = f + " Velocity {} [m/s]".format(velocity_comp[-1])
            fn = f + "_vel{}".format(velocity_comp[-1])

            filename = "{0}_Offset={1}.gif".format(fn,p_h.offsets[i])

            #find vmin and vmax for isocontour plots
            cmin, cmax = vmin_vmax(p_h,velocity_comp,i,no_cells_offset)
            levels = np.linspace(cmin,cmax,10,dtype=int)
            print("line 262",time.time()-start_time)
            with writer.saving(fig,dir+"{0}".format(filename),time_steps):
                for it in time_steps:
                    
                    X,Y,Z = Update(it)
                    print("line 267",time.time()-start_time)
                    T = Time[it]

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

                    cb = plt.colorbar(cs)
                    
                    Title = "{0}, Offset = {1}, Time = {2}[s]".format(ft,p_h.offsets[i],T)
                    
                    plt.title(Title)

                    writer.grab_frame()

                    plt.cla()
                    cb.remove()
                    print(it,start_time-time.time())

    iv+=1 #velocity index
    print(velocity_comp,time.time()-start_time)