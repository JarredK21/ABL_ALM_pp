from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob 
import os
from matplotlib import cm
import operator
import math
import sys
import time
from multiprocessing import Pool
import pyFAST.input_output as io
from scipy import interpolate


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    if normal == "z":
        height = offset
        twist_h = f(height)
        mag_horz_vel = u[it]*np.cos(twist_h) + v[it]*np.sin(twist_h)
    else:
        twist_hs = f(zs)
        mag_horz_vel = []
        for i in np.arange(0,len(height)):
            u_i = u[it,i*x:(i+1)*x]; v_i = v[it,i*x:(i+1)*x]
            twist_h = twist_hs[i]
            mag_horz_vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
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
del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v

print("line 126", time.time()-start_time)

#plotting precursor planes?
precursor = True

if precursor == False:
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


plot_l = True; plot_r = True; plot_tr = True; plot_i = True; plot_t = True
planes_plot = [plot_l,plot_r,plot_tr,plot_i,plot_t]

#check if no velocity components selected
if all(list(map(operator.not_, planes_plot))) == True:
    sys.exit("error no velocity component selected")


#loop over true planes
planes = ["l","r", "tr","p_i","p_t"]
plane_labels = ["horizontal","rotor", "transverse rotor", "inflow", "longitudinal"]
ip = 0
for plane in planes:
    if planes_plot[ip] == False:
        ip+=1
        continue

    if plane == "l":
        offsets = [22.5,85,142.5]
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
        plot_u = False; plot_v = False; plot_w = True; plot_hvelmag = True
        velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]

        #check if no velocity components selected
        if all(list(map(operator.not_, velocity_plot))) == True:
            sys.exit("error no velocity component selected")
        
        plot_all_times = False
        custom_colorbar = False

        #time options
        if plot_all_times == True:
            tend_idx = np.searchsorted(Time,Time[-1])
            Time_steps = np.arange(0,tend_idx)
        else:
            #specify time steps to plot instantaneous isocontours at
            Time_steps = [0,10]

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

        
        #loop over true velocity components
        velocity_comps = ["velocityx","velocityy","velocityz","Horizontal_velocity"]
        iv = 0
        for velocity_comp in velocity_comps:
            if velocity_plot[iv] == False:
                iv+=1
                continue
            
            print(plane_labels[ip],velocity_comps[iv],offset,time.time()-start_time)

            if fluc_vel == True:
                folder = out_dir+"{0}_Plane_Fluctutating_{1}_{2}/".format(plane_labels[ip],velocity_comp,offset)
            else:
                folder = out_dir+"{0}_Plane_Total_{1}_{2}/".format(plane_labels[ip],velocity_comp,offset)

            isExist = os.path.exists(folder)
            if isExist == False:
                os.makedirs(folder)

                #velocity field
                if velocity_comp == "Horizontal_velocity":
                    u = np.array(p.variables["velocityx"])
                    v = np.array(p.variables["velocityy"])
                    with Pool() as pool:
                        u = []
                        for u_it in pool.imap(Horizontal_velocity,Time_steps):
                            
                            u.append(u_it)
                            print(len(u),time.time()-start_time)
                    u = np.array(u)
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


                def Update(it):

                    U = u[it] #velocity time step it
                    
                    if it < 10:
                        Time_idx = "000{}".format(it)
                    elif it >= 10 and it < 100:
                        Time_idx = "00{}".format(it)
                    elif it >= 100 and it < 1000:
                        Time_idx = "0{}".format(it)
                    elif it >= 1000 and it < 10000:
                        Time_idx = "{}".format(it)


                    if type(normal) == int and plane == "r" or type(normal) and plane == "tr": #rotor planes
                        u_plane = U.reshape(y,x)
                        X,Y = np.meshgrid(ys,zs)
                    elif type(normal) == int and plane == "t":
                        u_plane = U.reshape(y,x)
                        X,Y = np.meshgrid(xs,zs)
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

                    if precursor == False and plane == "r" and offset == -5.5:
                        YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

                        plt.plot(YB1,ZB1,color="k",linewidth = 0.5)
                        plt.plot(YB2,ZB2,color="k",linewidth = 0.5)
                        plt.plot(YB3,ZB3,color="k",linewidth = 0.5)  

                    #define titles and filenames for movie
                    if fluc_vel == True:
                        if velocity_comp == "Horizontal_velocity":
                            Title = "{0} Plane. \nFluctuating {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_labels[ip], velocity_comp[:],float(offset),round(T,4))
                            filename = "{0}_Fluc_{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[:],float(offset),Time_idx)
                        else:
                            Title = "{0} Plane. \nFluctuating velocity {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_labels[ip],velocity_comp[-1],float(offset),round(T,4))
                            filename = "{0}_Fluc_vel{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[-1],float(offset),Time_idx)
                    else:
                        if velocity_comp == "Horizontal_velocity":
                            Title = "{0} Plane. \n{1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_labels[ip],velocity_comp[:],float(offset),round(T,4))
                            filename = "{0}_{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[:],float(offset),Time_idx)
                        else:
                            Title = "{0} Plane. \nTotal velocity {1} [m/s]: Offset = {2}, Time = {3}[s]".format(plane_labels[ip],velocity_comp[-1],float(offset),round(T,4))
                            filename = "{0}_Tot_vel{1}_{2}_{3}.png".format(plane_labels[ip],velocity_comp[-1],float(offset),Time_idx)

                    plt.title(Title)
                    plt.savefig(folder+filename)
                    plt.cla()
                    cb.remove()
                    plt.close(fig)

                    return T

                with Pool() as pool:
                    for T in pool.imap(Update,Time_steps):

                        print(T,time.time()-start_time)

                time.sleep(30)


                print(plane_labels[ip],velocity_comps[iv],offset,time.time()-start_time)

            iv+=1 #velocity index
        ic+=1 #offset index
    ip+=1 #planar index