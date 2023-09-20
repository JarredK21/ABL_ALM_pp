import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from scipy import interpolate
from multiprocessing import Pool
import time

def average_velocity(it):
    
    avg_velx_it = []
    for idx in height_idx:
        if plane == "t":
            avg_velx_it.append(np.average(velocityx[it,idx:(idx+z)]))
        elif plane == "r":
            hvelmag = np.add(np.multiply(velocityx[it,idx:(idx+z)], np.cos(np.radians(29))) ,np.multiply(velocityy[it,idx:(idx+z)], np.sin(np.radians(29))))
            avg_velx_it.append(np.average(hvelmag))
    return avg_velx_it

start_time = time.time()

in_dir = "./"
#in_dir = "../../NREL_5MW_MCBL_R_CRPM_2/post_processing/"
out_dir = in_dir+"Quasi-stationarity/"

heights = [10,40,90,500,1000,1200]

planes = ["t", "t", "r", "r","r","r"]
offsets = [1280, 3820,0.0,126,-63.0,-126]

plot_average_y = False
plot_hub_height = True

for offset,plane in zip(offsets,planes):

    a = Dataset(in_dir+"sampling_{}_{}.nc".format(plane,offset))
    Time_sample = np.array(a.variables["time"])
    Time_sample = Time_sample - Time_sample[0]
    time_idx = len(Time_sample)

    p = a.groups["p_{}".format(plane)]

    coordinates = np.array(p.variables["coordinates"])

    if plane == "r":
        y = p.ijk_dims[0] #no. data points
        z = p.ijk_dims[1] #no. data points

        xo = coordinates[:,0]
        yo = coordinates[:,1]
        zo = coordinates[:,2]

        rotor_coordinates = [2560,2560,90]

        x_trans = xo - rotor_coordinates[0]
        y_trans = yo - rotor_coordinates[1]

        phi = np.radians(-29)
        xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
        ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
        zs = zo - rotor_coordinates[2]

        Y = np.linspace(round(np.min(ys),0), round(np.max(ys),0),y )
        Z = np.linspace(round(np.min(zs),0), round(np.max(zs),0),z )

        height_idx = []
        for height in heights:
            height_idx.append(np.searchsorted(zs,height))
    else:
        y = p.ijk_dims[0]; z = p.ijk_dims[1]
        ys = coordinates[:,1]
        zs = coordinates[:,2]
        Y = np.linspace(round(np.min(ys),0), round(np.max(ys),0),y )
        Z = np.linspace(round(np.min(zs),0), round(np.max(zs),0),z )

        height_idx = []
        for height in heights:
            height_idx.append(np.searchsorted(zs,height))

    if plot_average_y == True:

        velocityx = np.array(p.variables["velocityx"])
        if plane == "r":
            velocityy = np.array(p.variables["velocityy"])

        avg_velx = []
        with Pool() as pool:
            ic = 1
            for avg_velx_it in pool.imap(average_velocity,np.arange(0,time_idx)):
                
                avg_velx.append(avg_velx_it)
                print(ic,time.time()-start_time)
                ic+=1
        

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time_sample,avg_velx)
        plt.xlabel("Time [s]",fontsize=16)
        if plane == "t":
            plt.ylabel("Ux averaged in the y direction [m/s]",fontsize=16)
            plt.title("Ux averaged in the y direction at {}m from inlet".format(offset),fontsize=18)
        elif plane == "r":
            plt.ylabel("Ux' averaged in the y' direction [m/s]",fontsize=16)
            plt.title("Ux' averaged in the y' direction at {}m from tower centerline".format(offset),fontsize=18)
        plt.legend(heights)
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"avg_velocityx_{}_{}.png".format(plane,offset))
        plt.close(fig)

    if plot_hub_height == True:
        height = 90
        trans = 2560
        hub_height_vel = []
        for it in np.arange(0,time_idx):

            velx = np.reshape(velocityx[it],(z,y))
            fx = interpolate.interp2d(Y,Z,velx,kind="linear")
            Ux = fx(rotor_coordinates[1],rotor_coordinates[2])

            if plane == "r":
                vely = np.reshape(velocityy[it],(z,y))
                fy = interpolate.interp2d(Y,Z,vely,kind="linear")
                Uy = fy(rotor_coordinates[1],rotor_coordinates[2])
                hub_height_vel.append(Ux*np.cos(np.radians(29))+Uy*np.sin(np.radians(29)))
            else:
                hub_height_vel.append(Ux)
            print(it,time.time()-start_time)

        fig = plt.figure(figsize=(14,8))
        plt.plot(Time_sample,hub_height_vel)
        plt.xlabel("Time [s]",fontsize=16)
        if plane == "t":
            plt.ylabel("Ux [m/s]",fontsize=16)
            plt.title("Ux at hub height {}m from inlet".format(offset),fontsize=18)
        elif plane == "r":
            plt.ylabel("Ux' [m/s]",fontsize=16)
            plt.title("Ux' at hub height {}m from tower centerline".format(offset),fontsize=18)
        plt.legend(heights)
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"hub_height_velocityx_{}_{}.png".format(plane,offset))
        plt.close(fig)