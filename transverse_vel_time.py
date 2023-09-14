import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from multiprocessing import Pool
import time

def average_velocity(it):
    
    avg_velx_it = []
    for idx in height_idx:
        if plane == "t":
            avg_velx_it.append(np.average(velocityx[it,idx:(idx+ys)]))
        elif plane == "r":
            hvelmag = np.add(np.multiply(velocityx[it,idx:(idx+ys)], np.cos(np.radians(29))) ,np.multiply(velocityy[it,idx:(idx+ys)], np.sin(np.radians(29))))
            avg_velx_it.append(np.average(hvelmag))
    return avg_velx_it

start_time = time.time()

in_dir = "./"
out_dir = in_dir+"Quasi-stationarity/"

heights = [10,40,90,500,1000,1200]

planes = ["t", "t", "r", "r","r","r"]
offsets = [1280, 3820,0.0,126,-63,-126]

for offset,plane in zip(offsets,planes):

    a = Dataset(in_dir+"sampling_{}_{}.nc".format(plane,offset))
    Time_sample = np.array(a.variables["time"])
    Time_sample = Time_sample - Time_sample[0]
    time_idx = len(Time_sample)

    p = a.groups["p_{}".format(plane)]

    ys = p.ijk_dims[0]; zs = p.ijk_dims[1]

    z = np.array(p.variables["coordinates"][:,2])
    
    height_idx = []
    for height in heights:
        height_idx.append(np.searchsorted(z,height))

    velocityx = np.array(p.variables["velocityx"])
    if plane == "r":
        velocityy = np.array(p.varaibles["velocityy"])

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