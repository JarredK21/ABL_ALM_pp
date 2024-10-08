from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def coordinate_rotation(it):

    xo = np.array(WT_E.variables["xyz"][it,1:,0])
    yo = np.array(WT_E.variables["xyz"][it,1:,1])
    zs_E = np.array(WT_E.variables["xyz"][it,1:,2])


    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
    ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))

    xs_E = xs + Rotor_coordinates[0]
    ys_E = ys + Rotor_coordinates[1]

    xo = np.array(WT_R.variables["xyz"][it,1:,0])
    yo = np.array(WT_R.variables["xyz"][it,1:,1])
    zs_R = np.array(WT_R.variables["xyz"][it,1:,2])

    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
    ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))

    xs_R = xs + Rotor_coordinates[0]
    ys_R = ys + Rotor_coordinates[1]

    return xs_E,ys_E,zs_E, xs_R,ys_R,zs_R



def update(it):

    if it < 10:
        Time_idx = "00000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "0000{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "000{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "00{}".format(it)
    elif it >= 10000 and it < 100000:
        Time_idx = "0{}".format(it)
    elif it >= 100000 and it < 10000000:
        Time_idx = "{}".format(it)


    xco_E,yco_E,zco_E, xco_R,yco_R,zco_R = coordinate_rotation(it)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(32,16),sharey=True)
    ax1.plot(xco_R[:300],zco_R[:300],"-r",label="Rigid")
    ax1.plot(xco_E[:300],zco_E[:300],"-b",label="Elastic")
    ax1.set_xlabel("x' coordinate [m]")
    ax1.grid()
    ax1.legend(loc="upper right")
    ax1.set_xlim([Rotor_coordinates[0]-10,Rotor_coordinates[0]+10]); ax1.set_ylim([20,160])

    ax2.plot(yco_R[:300],zco_R[:300],"-r",label="Rigid")
    ax2.plot(yco_E[:300],zco_E[:300],"-b",label="Elastic")
    ax2.set_xlabel("y' coordinate [m]")
    ax2.grid()
    ax2.legend(loc="upper right")
    ax2.set_xlim([2480,2630]); ax2.set_ylim([20,160])

    fig.supylabel("z coordinate [m]")
    fig.suptitle("Time: {}s".format(Time[it]))

    plt.savefig(out_dir+"{}.png".format(Time_idx))
    plt.close(fig)

    return Time_idx



in_dir="actuator76000/"

df_E = Dataset(in_dir+"WTG01.nc")

WT_E = df_E.groups["WTG01"]

Time = np.array(WT_E.variables["time"])
dt = Time[1] - Time[0]

Start_time_idx = np.searchsorted(Time,Time[0]+200)
#Time_steps = np.arange(Start_time_idx,len(Time))
Time_steps = np.arange(0,Start_time_idx)


Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]


in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/actuator76000/"

df_R = Dataset(in_dir+"WTG01.nc")

WT_R = df_R.groups["WTG01"]

out_dir="deforming_blade_2/"
plt.rcParams['font.size'] = 30
with Pool() as pool:
    for it in pool.imap(update,Time_steps):
        print(it)