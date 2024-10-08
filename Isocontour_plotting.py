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
    f_u = interpolate.interp1d(h,u_mean_profile)
    vel = []
    fluct_vel = []
    for i in np.arange(0,len(zs)):
        if velocity_comp == "Horizontal_velocity":
            u_i = u[it,i*x:(i+1)*x]; v_i = v[it,i*x:(i+1)*x]
        else:
            u_i = u[it,i*x:(i+1)*x]

        if zs[i] < h[0]:
            twist_h = f(h[0])
            u_mean = f_u(h[0])

        elif zs[i] > h[-1]:
            twist_h = f(h[-1])
            u_mean = f_u(h[-1])
        else:
            twist_h = f(zs[i])
            u_mean = f_u(zs[i])

        if velocity_comp == "Horizontal_velocity":
            vel_i = u_i*np.cos(twist_h) + v_i*np.sin(twist_h)
        else:
            vel_i = u_i

        fluc_vel_i = np.subtract(vel_i,u_mean)
        vel.extend(vel_i)
        fluct_vel.extend(fluc_vel_i)
    vel = np.array(vel)
    fluct_vel = np.array(fluct_vel)
    return vel,fluct_vel


def blade_positions(it):

    R = 63
    Az = -Azimuth[it]
    Y = [2560]; Y2 = [2560]; Y3 = [2560]
    Z = [90]; Z2 = [90]; Z3 = [90]

    Y.append(Y[0]+R*np.sin(Az))
    Z.append(Z[0]+R*np.cos(Az))

    Az2 = Az-(2*np.pi)/3
    if Az2 < -2*np.pi:
        Az2 += (2*np.pi)
    
    Az3 = Az-(4*np.pi)/3
    if Az2 < -2*np.pi:
        Az2 += (2*np.pi)

    Y2.append(Y2[0]+R*np.sin(Az2))
    Z2.append(Z2[0]+R*np.cos(Az2))

    Y3.append(Y3[0]+R*np.sin(Az3))
    Z3.append(Z3[0]+R*np.cos(Az3))

    return Y, Z, Y2, Z2, Y3, Z3


start_time = time.time()

#plotting precursor planes?
precursor = False

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


plot_l = False; plot_r = True; plot_tr = False; plot_i = False; plot_t = False
planes_plot = [plot_l,plot_r,plot_tr,plot_i,plot_t]

#check if no velocity components selected
if all(list(map(operator.not_, planes_plot))) == True:
    sys.exit("error no velocity component selected")


#loop over true planes
planes = ["l","r", "tr","i","t"]
plane_labels = ["horizontal","rotor", "transverse_rotor", "inflow", "longitudinal"]
ip = 0
for plane in planes:
    if planes_plot[ip] == False:
        ip+=1
        continue

    if plane == "l":
        offsets = [85]
    elif plane == "r":
        offsets = [-63.0]
    elif plane == "tr":
        offsets = [0.0]
    elif plane == "i":
        offsets = [0.0]
    elif plane == "t":
        offsets = [0.0]

    ic = 0
    for offset in offsets:

        if offset == -63.0:
            print("offset = ",offset)
            Azimuth = Azimuth+np.radians(334)

        a = Dataset("./sampling_{0}_{1}.nc".format(plane,offset))

        p = a.groups["p_{0}".format(plane)]

        #time options
        Time = np.array(a.variables["time"])
        Time = Time - Time[0]
        tstart = 200
        tstart_idx = np.searchsorted(Time,tstart)
        tend = 1201
        tend_idx = np.searchsorted(Time,tend)
        Time_steps = np.arange(0, tend_idx-tstart_idx)
        Time = Time[tstart_idx:tend_idx]

        f = interpolate.interp1d(Time_OF,Azimuth)
        Azimuth = f(Time)
        print(len(Azimuth))
        print(len(Time))

        #plotting option
        fluc_vel = True
        plot_contours = False
        plot_u = False; plot_v = False; plot_w = False; plot_hvelmag = True
        velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]

        #check if no velocity components selected
        if all(list(map(operator.not_, velocity_plot))) == True:
            sys.exit("error no velocity component selected")
        
        custom_colorbar = False

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
                folder = out_dir+"{0}_Plane_Fluctutating_{1}_{2}_16_08/".format(plane_labels[ip],velocity_comp,offset)
            else:
                folder = out_dir+"{0}_Plane_Total_{1}_{2}/".format(plane_labels[ip],velocity_comp,offset)

            isExist = os.path.exists(folder)
            if isExist == False:
                os.makedirs(folder)

                if fluc_vel == True:
                    #defining twist angles with height from precursor
                    precursor_df = Dataset("./abl_statistics76000.nc")
                    Time_pre = np.array(precursor_df.variables["time"])
                    mean_profiles = precursor_df.groups["mean_profiles"] #create variable to hold mean profiles
                    t_start = np.searchsorted(precursor_df.variables["time"],38200)
                    u = np.average(mean_profiles.variables["u"][t_start:],axis=0)
                    v = np.average(mean_profiles.variables["v"][t_start:],axis=0)
                    w = np.average(mean_profiles.variables["w"][t_start:],axis=0)
                    h = mean_profiles["h"][:]
                    twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
                    if velocity_comp == "Horizontal_velocity":
                        u_mean_profile = []
                        for i in np.arange(0,len(twist)):
                            u_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
                    elif velocity_comp == "velocityx":
                        u_mean_profile = u
                    elif velocity_comp == "velocityy":
                        u_mean_profile = v
                    elif velocity_comp == "velocityz":
                        u_mean_profile = w

                    del precursor_df; del Time_pre; del mean_profiles; del t_start; del u; del v; del w

                print("line 126", time.time()-start_time)

                #velocity field
                if velocity_comp == "Horizontal_velocity":
                    u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
                    v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

                    u[u<0]=0; v[v<0]=0 #remove negative velocities
                    
                    u_hvel = []; u_pri = []
                    ix=0
                    with Pool() as pool:
                        for u_hvel_it, u_hvel_pri_it in pool.imap(Horizontal_velocity,Time_steps):
                            
                            u_hvel.append(u_hvel_it)
                            u_pri.append(u_hvel_pri_it)
                            print(ix)
                            ix+=1
                    if fluc_vel == True:
                        u = np.array(u_pri); del u_hvel; del u_pri; del v
                    else:
                        u = np.array(u_hvel); del u_pri; del u_hvel; del v
                
                else:
                    u = np.array(p.variables[velocity_comp][tstart_idx:tend_idx])

                    if fluc_vel == True:
                        with Pool() as pool:
                            for u_hvel_it, u_hvel_pri_it in pool.imap(Horizontal_velocity,Time_steps):
                                
                                u_hvel.append(u_hvel_it)
                                u_pri.append(u_hvel_pri_it)
                                print(ix)
                                ix+=1
                        u = np.array(u_pri); del u_hvel; del u_pri

                print("line 328",time.time()-start_time)


                #find vmin and vmax for isocontour plots            
                #min and max over data
                if custom_colorbar == False:                            

                    cmin = math.floor(np.min(u))
                    cmax = math.ceil(np.max(u))
                
                if fluc_vel == True or velocity_comp == "velocityz":
                    nlevs = int((cmax-cmin)/2)
                    if nlevs>abs(cmin) or nlevs>cmax:
                        nlevs = min([abs(cmin),cmax])+1

                    levs_min = np.linspace(cmin,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax,nlevs,dtype=int)
                    levels = np.concatenate((levs_min,levs_max[1:]))
                else:
                    nlevs = (cmax-cmin)
                    levels = np.linspace(cmin,cmax,nlevs,dtype=int)
                    
                print("line 370",levels)


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

                    if plot_contours == True:
                        CS = plt.contour(X, Y, Z, levels=levels, colors='k')
                        plt.clabel(CS, fontsize=18, inline=True)


                    #show where rotor is and reduce plot area
                    if plane == "t":
                        plt.xlim([2000,3000]); plt.ylim([0,300])
                    elif plane == "l":
                        plt.xlim([2000,3000]); plt.ylim([2000,3000])

                    if plane == "t":
                        x_lims = [2555,2555]; y_lims = [27,153]
                        plt.plot(x_lims,y_lims,linewidth=1.0,color="k")
                    elif plane == "l":
                        x_lims = [2524.5,2585.5]; y_lims = [2615.1,2504.9]
                        plt.plot(x_lims,y_lims,linewidth=1.0,color="k")

                    if normal == "x":
                        plt.xlabel("y axis [m]")
                        plt.ylabel("z axis [m]")
                    elif normal == "y":
                        plt.xlabel("x axis [m]")
                        plt.ylabel("y axis [m]")
                    elif normal == "z":
                        plt.xlabel("x axis [m]")
                        plt.ylabel("y axis [m]")
                    elif type(normal) == int and plane == "r" or type(normal) and plane == "tr": #rotor planes
                        plt.xlabel("y' axis (rotor frame of reference) [m]")
                        plt.ylabel("z' axis (rotor frame of reference) [m]")
                    elif type(normal) == int and plane == "t":
                        plt.xlabel("x' axis (rotor frame of reference) [m]")
                        plt.ylabel("z' axis (rotor frame of reference) [m]")

                    cb = plt.colorbar(cs)

                    if precursor == False and plane == "r":
                        YB1,ZB1,YB2,ZB2,YB3,ZB3 = blade_positions(it)

                        plt.plot(YB1,ZB1,color="k",linewidth = 1,label="blade 1")
                        plt.plot(YB2,ZB2,color="r",linewidth = 1,label="blade 2")
                        plt.plot(YB3,ZB3,color="b",linewidth = 1,label="blade 3")  

                    plt.legend()

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