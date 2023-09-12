import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from multiprocessing import Pool
import time

def average_velocity(it):
    
    avg_velx_it = []
    for i in heights:
        avg_velx_it.append(np.average(velocityx[it,(i*ys):((i+1)*ys)]))
    return avg_velx_it

start_time = time.time()

heights = [1,4,9,50,100,150]

offsets = [1280, 3820]

for offset in offsets:

    a = Dataset("../../NREL_5MW_MCBL_R_CRPM_2/post_processing/sampling_t_{}.nc".format(offset))
    Time_sample = np.array(a.variables["time"])
    Time_sample = Time_sample - Time_sample[0]
    time_idx = len(Time_sample)

    p_t = a.groups["p_t"]

    ys = p_t.ijk_dims[0]; zs = p_t.ijk_dims[1]

    coordinates = np.array(p_t.variables["coordinates"])

    velocityx = np.array(p_t.variables["velocityx"])

    avg_velx = []
    with Pool() as pool:
        ic = 1
        for avg_velx_it in pool.imap(average_velocity,np.arange(0,time_idx)):
            
            avg_velx.append(avg_velx_it)
            print(ic,time.time()-start_time)
            ic+=1
    
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_sample,avg_velx)
    plt.xlabel("Time [s]")
    plt.ylabel("Average velocity x direction [m/s]")
    plt.tight_layout()
    plt.savefig("../../NREL_5MW_MCBL_R_CRPM_2/post_processing/avg_velocityx_{}.png".format(offset))
    plt.close(fig)