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

def tranform_fixed_frame(Y_pri,Z_pri,it):

    Y = ((Y_pri-Rotor_coordinates[1])*np.cos(Azimuth[it]) - (Z_pri-Rotor_coordinates[2])*np.sin(Azimuth[it])) + Rotor_coordinates[1]
    Z = ((Y_pri-Rotor_coordinates[1])*np.sin(Azimuth[it]) + (Z_pri-Rotor_coordinates[2])*np.cos(Azimuth[it])) + Rotor_coordinates[2]

    return Y,Z

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

    yE_fixed,zE_fixed = tranform_fixed_frame(yco_E,zco_E,it)
    yR_fixed,zR_fixed = tranform_fixed_frame(yco_R,zco_R,it)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(32,16),sharey=True)
    ax1.plot(xco_R[:300],zR_fixed[:300],"-r",label="Rigid")
    ax1.plot(xco_E[:300],zE_fixed[:300],"-b",label="Elastic")
    ax1.set_xlabel("x' coordinate rotating frame of reference [m]")
    ax1.grid()
    ax1.legend(loc="upper right")
    ax1.set_xlim([Rotor_coordinates[0]-5,Rotor_coordinates[0]+10]); ax1.set_ylim([80,160])

    ax2.plot(yR_fixed[:300],zR_fixed[:300],"-r",label="Rigid")
    ax2.plot(yE_fixed[:300],zE_fixed[:300],"-b",label="Elastic")
    ax2.set_xlabel("y' coordinate rotating frame of reference [m]")
    ax2.grid()
    ax2.legend(loc="upper right")
    ax2.set_xlim([Rotor_coordinates[1]-5,Rotor_coordinates[1]+5]); ax2.set_ylim([80,160])

    fig.supylabel("z coordinate rotating frame of reference [m]")
    fig.suptitle("Time: {}s".format(Time[it]))

    plt.savefig(out_dir+"{}.png".format(Time_idx))
    plt.close(fig)

    return xco_E,yE_fixed,zE_fixed



in_dir="actuator76000/"

df = Dataset("Dataset.nc")
OF_vars = df.groups["OpenFAST_Variables"]
Azimuth = np.array(OF_vars.variables["Azimuth"])
Azimuth = 360 - Azimuth[1:]

df_E = Dataset(in_dir+"WTG01.nc")

WT_E = df_E.groups["WTG01"]

Time = np.array(WT_E.variables["time"])
dt = Time[1] - Time[0]

Time_steps = np.arange(0,len(Time))
#Time_steps = np.arange(0,Start_time_idx)


Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]


in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/actuator76000/"

df_R = Dataset(in_dir+"WTG01.nc")

WT_R = df_R.groups["WTG01"]
Time_steps = [0,1]
ix = 0
x = []; y = []; z = []
out_dir="deforming_blade_3/"
plt.rcParams['font.size'] = 30
with Pool() as pool:
    for xit,yit,zit in pool.imap(update,Time_steps):
        x.append(xit); y.append(yit); z.append(zit)
        print(np.shape(x))
        print(ix)
        ix+=1

x = np.mean(x,axis=0); y = np.mean(y,axis=0); z = np.mean(z,axis=0)
print(x); print(y); print(z)