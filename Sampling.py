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



def offset_data(p_h,velocity_comp, i, no_cells_offset,it):

    u = np.array(p_h.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice


def vmin_vmax(p_h,velocity_comp,i,no_cells_offset):
            
        #min and max over data
        vmin_arr = []; vmax_arr = []
        for it in np.arange(0,time_steps,1):

            u = offset_data(p_h,velocity_comp,i,no_cells_offset,it)
            if fluc_vel == True:
                u = u - np.mean(u)
            
            vmin_arr.append(np.min(u)); vmax_arr.append(np.max(u))

        vmin = math.floor(np.min(vmin_arr)); vmax = math.ceil(np.max(vmax_arr))

        return vmin, vmax



#lineplot
def lineplot(x,y, plane_data_u,offset, xlabel, filename,case,dir):
    u = np.array(plane_data_u[:][offset])
    u_plane = u.reshape(x,y) #needs fixing
    u_line = np.average(u_plane, axis=0)

    stats = glob.glob("{0}/post_processing/abl_statistics*".format(case))
    b = Dataset("./{}".format(stats[0]))
    mg = b.groups["mean_profiles"]
    height = mg.variables["h"][:]

    fig = plt.figure()
    plt.rcParams['font.size'] = 12
    plt.plot(u_line, height)
    plt.xlabel(xlabel)
    plt.ylabel("height from surface [m]")
    plt.title("Offset = "+offset)
    plt.savefig(dir+"{}".format(filename))
    plt.close(fig)


#isocontourplot
def isocontourplot(u,p_h,x,y,Title,filename,dir,normal):
    
    u_plane = u.reshape(x,y) #needs fixing x and y lengths and number of points aren't consistent.

    if normal == "x":
        l = 1; m = 2
    elif normal == "y":
        l = 0; m = 2
    elif normal == "z":
        l = 0; m = 1

    x_array = np.linspace(p_h.origin[l],(p_h.origin[l]+p_h.axis1[l]),x)
    y_array = np.linspace(p_h.origin[m],(p_h.origin[m]+p_h.axis2[m]),y)
    X,Y = np.meshgrid(x_array,y_array)

    fig = plt.figure()
    plt.rcParams['font.size'] = 12
    #plt.contourf(X,Y,np.transpose(u_plane), cmap=cm.coolwarm)
    
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
    plt.title(Title)
    plt.colorbar()
    plt.savefig(dir+"{}".format(filename))
    plt.close(fig)



dir = "Ex1.3/post_processing/plots/"
cases = ["Ex1.3"]

for case in cases:

    sampling = glob.glob("{0}/post_processing/sampling*".format(case))
    a = Dataset("./{}".format(sampling[0]))
    p_h = a.groups["p_h"]


    no_cells = len(p_h.variables["coordinates"])
    no_offsets = len(p_h.offsets)
    no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

    x = p_h.ijk_dims[0] #no. data points
    y = p_h.ijk_dims[1] #no. data points


    time_steps = len(p_h.variables["velocityx"])
    frequency = 1 #manual
    dt = 0.001
    time_per_time_step = frequency*dt



    #plotting option
    plot_line_plots = False
    plot_isocontour = True
    plot_u = False; plot_v = False; plot_w = False; plot_hvelmag = True
    velocity_plot = [plot_u,plot_v,plot_w,plot_hvelmag]
    
    #check if no velocity components selected
    if all(list(map(operator.not_, velocity_plot))) == True:
        print("error no velocity component selected")
        break

    fluc_vel = False
    movie_tot_vel_isocontour = False
    plot_specific_offsets = False

    if plot_specific_offsets == True:    
        spec_offsets = [3]
    else:
        spec_offsets = np.arange(0,no_offsets, 1, dtype=int)


    #find normal
    if p_h.axis3[0] == 1:
        normal = "x"
    elif p_h.axis3[1] == 1:
        normal = "y"
    elif p_h.axis3[2] == 1:
        normal = "z"


    col_names = []
    for col in p_h.offsets:
        col_names.append(str(col))
    plane_data =  pd.DataFrame(data=None, columns=col_names)

    #loop over true velocity components
    velocity_comps = ["velocityx","velocityy","velocityz","Magnitude horizontal velocity"]
    iv = 0
    for velocity_comp in velocity_comps:
        if velocity_plot[iv] == False:
            continue

        #loop over offsets
        for i in spec_offsets:

            if velocity_comp == "Magnitude horizontal velocity":
                u = offset_data(p_h,velocity_comps[0], i, no_cells_offset,it=0) #slicing data into offset arrays
                v = offset_data(p_h,velocity_comps[1], i, no_cells_offset,it=0)
                u = np.sqrt(u**2 + v**2)

            else:
                u = offset_data(p_h,velocity_comp, i, no_cells_offset,it=0) #slicing data into offset arrays

            plane_data[str(p_h.offsets[i])] = u


            if plot_line_plots == True:
                offset = str(p_h.offsets[i])
                
                if velocity_comp == "Magnitude horizontal velocity":
                    lineplot(x,y, plane_data,offset, "{} [m/s]".format(velocity_comp[:]), "{0}_{1}.png".format(float(velocity_comp[:],offset)),case,dir)
                else:
                    lineplot(x,y, plane_data,offset, "velocity {} [m/s]".format(velocity_comp[-1]), "vel{0}_{1}.png".format(float(velocity_comp[-1],offset)),case,dir)

            #plots it = time specified above
            if plot_isocontour == True:
                offset = str(p_h.offsets[i])
                if fluc_vel == True:
                    u = np.array(plane_data[:][offset]) - np.mean(np.array(plane_data[:][offset]))
                    Title = "Fluctuating velocity {0} [m/s]: Offset = {1}".format(velocity_comp[-1],float(offset))
                    filename = "Fluc_vel{0}_{1}.png".format(velocity_comp[-1],float(offset))
                elif velocity_comp == "Magnitude horizontal velocity":
                    u = np.array(plane_data[:][offset])
                    Title = "{0} [m/s]: Offset = {1}".format(velocity_comp[:],float(offset))
                    filename = "{0}_{1}.png".format(velocity_comp[:],float(offset))
                else:
                    u = np.array(plane_data[:][offset])
                    Title = "Total velocity {0} [m/s]: Offset = {1}".format(velocity_comp[-1],float(offset))
                    filename = "Tot_vel{0}_{1}.png".format(velocity_comp[-1],float(offset))
                    
                isocontourplot(u,p_h,x,y,Title,filename,dir, normal)


            #generate movie for specific plane
            if movie_tot_vel_isocontour == True:

                fig = plt.figure(figsize=(50,30))
                plt.rcParams['font.size'] = 40


                def Update(it):
                    u = offset_data(p_h,velocity_comp, i, no_cells_offset,it) #slicing data into offset arrays

                    if fluc_vel == True:
                        u = u - np.mean(u)
                    
                    u_plane = u.reshape(x,y)

                    if normal == "x":
                        l = 1; m = 2
                    elif normal == "y":
                        l = 0; m = 2
                    elif normal == "z":
                        l = 0; m = 1

                    x_array = np.linspace(p_h.origin[l],(p_h.origin[l]+p_h.axis1[l]),x)
                    y_array = np.linspace(p_h.origin[m],(p_h.origin[m]+p_h.axis2[m]),y)
                    X,Y = np.meshgrid(x_array,y_array)

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

                with writer.saving(fig,dir+"{0}".format(filename),time_steps):
                    for it in np.arange(0,time_steps,1):
                        
                        X,Y,Z = Update(it)

                        T = it*time_per_time_step

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

        iv+=1 #velocity index