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


def energy_contents_check(Var,e_fft,signal,dt):

    E = (1/dt)*np.sum(e_fft)

    q = np.sum(np.square(signal))

    E2 = q

    print(Var, E, E2, abs(E2/E))    


def temporal_spectra(signal,dt,Var):

    fs =1/dt
    n = len(signal) 
    if n%2==0:
        nhalf = int(n/2+1)
    else:
        nhalf = int((n+1)/2)
    frq = np.arange(nhalf)*fs/n
    Y   = np.fft.fft(signal)
    PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
    PSD[1:-1] = PSD[1:-1]*2


    energy_contents_check(Var,PSD,signal,dt)

    return frq, PSD



#in_dir="actuator76000/"

in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

df = Dataset(in_dir+"Dataset.nc")
OF_vars = df.groups["OpenFAST_Variables"]
Azimuth = np.array(OF_vars.variables["Azimuth"])

Azimuth = 360 - Azimuth[1:]
Azimuth = np.radians(Azimuth)

df_E = Dataset(in_dir+"WTG01b.nc")

WT_E = df_E.groups["WTG01"]

Time = np.array(WT_E.variables["time"])
dt = Time[1] - Time[0]

Tstart_idx = np.searchsorted(Time,Time[0]+200)
Time_steps = np.arange(Tstart_idx,len(Time))


Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]


#in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/actuator76000/"
in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_R = Dataset(in_dir+"WTG01b.nc")

WT_R = df_R.groups["WTG01"]


# out_dir="deforming_blade_3/"
# plt.rcParams['font.size'] = 30
# with Pool() as pool:
#     for it in pool.imap(update,Time_steps):
        
#         print(it)



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


ix = 0
xE = []; yE = []; zE = []
xR = []; yR = []; zR = []
out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"

#plt.rcParams['font.size'] = 30
with Pool() as pool:
    for xEit,yEit,zEit,xRit,yRit,zRit in pool.imap(rotating_frame_coordinates,Time_steps):
        xE.append(xEit[-1]); yE.append(yEit[-1]); zE.append(zEit[-1])
        xR.append(xRit[-1]); yR.append(yRit[-1]); zR.append(zRit[-1])
        #print(np.shape(x))
        print(ix)
        ix+=1

xD = np.subtract(xE,xR); yD = np.subtract(yE,yR); zD = np.subtract(zE,zR)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(xD,dt,"xD")
# plt.loglog(frq,PSD)
# plt.ylabel("PSD: Displacement relative to rigid blade in x direction [m]")
# plt.xlabel("Frequency [Hz]")
# plt.title("Measured at the tip 63m")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_x_disp.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(yD,dt,"zD")
# plt.loglog(frq,PSD)
# plt.ylabel("PSD: Displacement relative to rigid blade in y direction [m]")
# plt.xlabel("Frequency [Hz]")
# plt.title("Measured at the tip 63m")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_y_disp.png")
# plt.close(fig)

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(zD,dt,"zD")
# plt.loglog(frq,PSD)
# plt.ylabel("PSD: Displacement relative to rigid blade in z direction [m]")
# plt.xlabel("Frequency [Hz]")
# plt.title("Measured at the tip 63m")
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_z_disp.png")
# plt.close(fig)

xE = np.mean(xE,axis=0); yE = np.mean(yE,axis=0); zE = np.mean(zE,axis=0)
xR = np.mean(xR,axis=0); yR = np.mean(yR,axis=0); zR = np.mean(zR,axis=0)


BMassDen = [6.7893500E+02, 6.7893500E+02, 7.7336300E+02, 7.4055000E+02, 7.4004200E+02, 5.9249600E+02, 4.5027500E+02, 4.2405400E+02, 4.0063800E+02, 3.8206200E+02, 3.9965500E+02,
            4.2632100E+02, 4.1682000E+02, 4.0618600E+02, 3.8142000E+02, 3.5282200E+02, 3.4947700E+02, 3.4653800E+02, 3.3933300E+02, 3.3000400E+02, 3.2199000E+02, 3.1382000E+02,
            2.9473400E+02, 2.8712000E+02, 2.6334300E+02, 2.5320700E+02, 2.4166600E+02, 2.2063800E+02, 2.0029300E+02, 1.7940400E+02, 1.6509400E+02, 1.5441100E+02, 1.3893500E+02, 
            1.2955500E+02, 1.0726400E+02, 9.8776000E+01, 9.0248000E+01, 8.3001000E+01, 7.2906000E+01, 6.8772000E+01, 6.6264000E+01, 5.9340000E+01, 5.5914000E+01, 5.2484000E+01,
            4.9114000E+01, 4.5818000E+01, 4.1669000E+01, 1.1453000E+01, 1.0319000E+01]

plt.rcParams['font.size'] = 30
fig,(ax1,ax2) = plt.subplots(1,2,figsize=(32,16),sharey=True)
ax1.plot(xE,zE,"-b",label="Deformed")
ax1.plot(xR,zR,"-r",label="Rigid")
ax1.set_xlabel("x' coordinate rotating frame of reference [m]")
ax1.grid()
ax1.legend()
ax1.set_xlim([Rotor_coordinates[0]-5,Rotor_coordinates[0]+10]); ax1.set_ylim([80,160])

# ax2.plot(yE,zE,"-b",label="Deformed")
# ax2.plot(yR,zR,"-r",label="Rigid")
# ax2.set_xlabel("y' coordinate rotating frame of reference [m]")
# ax2.grid()
# ax2.legend()
# ax2.set_xlim([Rotor_coordinates[1]-5,Rotor_coordinates[1]+5]); ax2.set_ylim([80,160])

fig.supylabel("z coordinate rotating frame of reference [m]")
fig.suptitle("Mean deflected blade position")

plt.savefig(out_dir+"Elastic_deformations_analysis/mean_deflected_blade.png")
plt.close(fig)

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