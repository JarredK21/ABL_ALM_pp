from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
import pandas as pd


def coordinate_rotation(it):

    xo = np.array(WT_E.variables["xyz"][it,1:301,0])
    yo = np.array(WT_E.variables["xyz"][it,1:301,1])
    zs_E = np.array(WT_E.variables["xyz"][it,1:301,2])


    x_trans = xo - Rotor_coordinates[0]
    y_trans = yo - Rotor_coordinates[1]

    phi = np.radians(-29.29)
    xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
    ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))

    xs_E = xs + Rotor_coordinates[0]
    ys_E = ys + Rotor_coordinates[1]

    xo = np.array(WT_R.variables["xyz"][it,1:301,0])
    yo = np.array(WT_R.variables["xyz"][it,1:301,1])
    zs_R = np.array(WT_R.variables["xyz"][it,1:301,2])

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


    xco_E,yE_fixed,zE_fixed, xco_R, yR_fixed, zR_fixed = rotating_frame_coordinates(it)

    fig,(ax1,ax2) = plt.subplots(1,2,figsize=(32,16),sharey=True)
    ax1.plot(xco_R,zR_fixed,"-r",label="Rigid")
    ax1.plot(xco_E,zE_fixed,"-b",label="Elastic")
    ax1.set_xlabel("x' coordinate rotating frame of reference [m]")
    ax1.grid()
    ax1.legend(loc="upper right")
    ax1.set_xlim([Rotor_coordinates[0]-5,Rotor_coordinates[0]+10]); ax1.set_ylim([80,160])

    ax2.plot(yR_fixed,zR_fixed,"-r",label="Rigid")
    ax2.plot(yE_fixed,zE_fixed,"-b",label="Elastic")
    ax2.set_xlabel("y' coordinate rotating frame of reference [m]")
    ax2.grid()
    ax2.legend(loc="upper right")
    ax2.set_xlim([Rotor_coordinates[1]-5,Rotor_coordinates[1]+5]); ax2.set_ylim([80,160])

    fig.supylabel("z coordinate rotating frame of reference [m]")
    fig.suptitle("Time: {}s".format(Time[it]))

    plt.savefig(out_dir+"{}.png".format(Time_idx))
    plt.close(fig)

    return Time_idx


def rotating_frame_coordinates(it):



    xco_E,yco_E,zco_E, xco_R,yco_R,zco_R = coordinate_rotation(it)

    yE_fixed,zE_fixed = tranform_fixed_frame(yco_E,zco_E,it)
    yR_fixed,zR_fixed = tranform_fixed_frame(yco_R,zco_R,it)

    return xco_E,yE_fixed,zE_fixed, xco_R, yR_fixed, zR_fixed


in_dir="actuator76000/"

#in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

df = Dataset(in_dir+"Dataset.nc")
OF_vars = df.groups["OpenFAST_Variables"]
Azimuth = np.array(OF_vars.variables["Azimuth"])
Rt = np.array(OF_vars.variables["RtAeroFxh"])

Azimuth_new = Azimuth
for it in np.arange(0,len(Azimuth_new)-1):
    if Azimuth_new[it+1] < Azimuth_new[it]:
        Azimuth_new[it+1:]+=360

Azimuth = 360 - Azimuth[1:]
Azimuth = np.radians(Azimuth)

Azimuth_new = 360 - np.array(Azimuth_new[1:])

df_E = Dataset(in_dir+"WTG01b.nc")

WT_E = df_E.groups["WTG01"]

Time = np.array(WT_E.variables["time"])
dt = Time[1] - Time[0]

Tstart_idx = np.searchsorted(Time,Time[0]+200)
Time_steps = np.arange(Tstart_idx,len(Time))


Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]


in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/actuator76000/"
#in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_R = Dataset(in_dir+"WTG01b.nc")

WT_R = df_R.groups["WTG01"]


out_dir="deforming_blade_3/"
plt.rcParams['font.size'] = 30
with Pool() as pool:
    for it in pool.imap(update,Time_steps):
        
        print(it)



# ix = 0
# xE = []
# with Pool() as pool:
#     for xs_E,ys_E,zs_E, xs_R,ys_R,zs_R in pool.imap(coordinate_rotation,Time_steps):
#         xE.append(xs_E[-1])
#         print(ix)
#         ix+=1

# out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"
# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(Azimuth_new[Tstart_idx:],xE)
# plt.xlabel("Azimuth position [deg]")
# plt.ylabel("x coordinate [m]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Elastic_deformations_analysis/Azimuth_xco.png")
# plt.close(fig)


# ix = 0
# xE = []; yE = []; zE = []
# xR = []; yR = []; zR = []
# #out_dir="deforming_blade_3/"

# #plt.rcParams['font.size'] = 30
# with Pool() as pool:
#     for xEit,yEit,zEit,xRit,yRit,zRit in pool.imap(rotating_frame_coordinates,Time_steps):
#         xE.append(xEit); yE.append(yEit); zE.append(zEit)
#         xR.append(xRit); yR.append(yRit); zR.append(zRit)
#         #print(np.shape(x))
#         print(ix)
#         ix+=1

# xE = np.mean(xE,axis=0); yE = np.mean(yE,axis=0); zE = np.mean(zE,axis=0)
# xR = np.mean(xR,axis=0); yR = np.mean(yR,axis=0); zR = np.mean(zR,axis=0)

# plt.rcParams['font.size'] = 30
# fig,(ax1,ax2) = plt.subplots(1,2,figsize=(32,16),sharey=True)
# ax1.plot(xE,zE,"-b",label="Deformed")
# ax1.plot(xR,zR,"-r",label="Rigid")
# ax1.set_xlabel("x' coordinate rotating frame of reference [m]")
# ax1.grid()
# ax1.legend()
# ax1.set_xlim([Rotor_coordinates[0]-5,Rotor_coordinates[0]+10]); ax1.set_ylim([80,160])

# ax2.plot(yE,zE,"-b",label="Deformed")
# ax2.plot(yR,zR,"-r",label="Rigid")
# ax2.set_xlabel("y' coordinate rotating frame of reference [m]")
# ax2.grid()
# ax2.legend()
# ax2.set_xlim([Rotor_coordinates[1]-5,Rotor_coordinates[1]+5]); ax2.set_ylim([80,160])

# fig.supylabel("z coordinate rotating frame of reference [m]")
# fig.suptitle("Mean deflected blade position")

# plt.savefig(out_dir+"Elastic_deformations_analysis/mean_deflected_blade.png")
# plt.close(fig)

# BlFract = np.linspace(0,1,len(xE))
# BlFract_new = []
# Theta = []
# for i in np.arange(0,len(xE)-1):
#     delx = xE[i+1] - xE[i]
#     delz = zE[i+1] - zE[i]
#     Theta.append(np.degrees(np.arctan(delx/delz)))
#     BlFract_new.append(np.average([BlFract[i+1],BlFract[i]]))

# plt.rcParams['font.size'] = 16
# fig = plt.figure(figsize=(14,8))
# plt.plot(BlFract_new,Theta)
# plt.xlabel("Blade fraction [-]")
# plt.ylabel("Mean Deformation angle [deg]")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Elastic_deformations_analysis/deformation_angle.png")
# plt.close(fig)


# d = {"x": x, "y": y, "z": z}
# df = pd.DataFrame(data=d)

# df.to_csv(out_dir+"mean_coordinates.csv",index=False)