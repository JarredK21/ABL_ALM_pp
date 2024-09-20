from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


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


def coordinate_displacement(it):


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

    xs = np.subtract(xs_E,xs_R); ys = np.subtract(ys_E,ys_R); zs = np.subtract(zs_E,zs_R)

    return xs,ys,zs


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
        Time_idx = "000{}".format(it)
    elif it >= 10 and it < 100:
        Time_idx = "00{}".format(it)
    elif it >= 100 and it < 1000:
        Time_idx = "0{}".format(it)
    elif it >= 1000 and it < 10000:
        Time_idx = "{}".format(it)
    elif it >= 10000 and it < 100000:
        Time_idx = "{}".format(it)
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
    plt.tight_layout()

    plt.savefig(out_dir+"{}.png".format(Time_idx))
    plt.close()

    return Time_idx



in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

df = Dataset(in_dir+"Dataset.nc")

OF_Vars = df.groups["OpenFAST_Variables"]



df_E = Dataset(in_dir+"WTG01b.nc")

WT_E = df_E.groups["WTG01"]

Time = np.array(WT_E.variables["time"])
dt = Time[1] - Time[0]

Start_time_idx = np.searchsorted(Time,Time[0]+200)
Time_steps = np.arange(Start_time_idx,len(Time))


Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]

# xco_75 = np.array(WT_E.variables["xyz"][Start_time_idx:,75,0]); xco_225 = np.array(WT_E.variables["xyz"][Start_time_idx:,225,0]); xco_300 = np.array(WT_E.variables["xyz"][Start_time_idx:,299,0])
# yco_75 = np.array(WT_E.variables["xyz"][Start_time_idx:,75,1]); yco_225 = np.array(WT_E.variables["xyz"][Start_time_idx:,225,1]); yco_300 = np.array(WT_E.variables["xyz"][Start_time_idx:,299,1])
# zco_75 = np.array(WT_E.variables["xyz"][Start_time_idx:,75,2]); zco_225 = np.array(WT_E.variables["xyz"][Start_time_idx:,225,2]); zco_300 = np.array(WT_E.variables["xyz"][Start_time_idx:,299,2])

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(xco_75,dt,Var="xco_75")
# plt.loglog(frq,PSD,"-g",label="15.75m")
# frq,PSD = temporal_spectra(xco_225,dt,Var="xco_225")
# plt.loglog(frq,PSD,"-r",label="47.25")
# frq,PSD = temporal_spectra(xco_300,dt,Var="xco_300")
# plt.loglog(frq,PSD,"-b",label="62.58m")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("x coordinate [m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_x.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(yco_75,dt,Var="yco_75")
# plt.loglog(frq,PSD,"-g",label="15.75m")
# frq,PSD = temporal_spectra(yco_225,dt,Var="yco_225")
# plt.loglog(frq,PSD,"-r",label="47.25")
# frq,PSD = temporal_spectra(yco_300,dt,Var="yco_300")
# plt.loglog(frq,PSD,"-b",label="62.58m")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("y coordinate [m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_y.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(zco_75,dt,Var="zco_75")
# plt.loglog(frq,PSD,"-g",label="15.75m")
# frq,PSD = temporal_spectra(zco_225,dt,Var="zco_225")
# plt.loglog(frq,PSD,"-r",label="47.25")
# frq,PSD = temporal_spectra(zco_300,dt,Var="zco_300")
# plt.loglog(frq,PSD,"-b",label="62.58m")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("z coordinate [m]")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"Spectra_z.png")
# plt.close()



in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_R = Dataset(in_dir+"WTG01b.nc")

WT_R = df_R.groups["WTG01"]

out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"
plt.rcParams['font.size'] = 30
update(51283)





ix=0
xco_75 = []; yco_75 = []; zco_75 = []
xco_225 = []; yco_225 = []; zco_225 = []
xco_300 = []; yco_300 = []; zco_300 = []
with Pool() as pool:
    for xco,yco,zco in pool.imap(coordinate_displacement,Time_steps):
        xco_75.append(xco[75]); yco_75.append(yco[75]); zco_75.append(zco[75])
        xco_225.append(xco[225]); yco_225.append(yco[225]); zco_225.append(zco[225])
        xco_300.append(xco[299]); yco_300.append(yco[299]); zco_300.append(zco[299])
        ix+=1
        print(ix)


out_dir=in_dir+"Elastic_deformations_analysis/"
plt.rcParams['font.size'] = 16
fig = plt.figure(figsize=(14,8))
plt.plot(Time,xco_75,"-g",label="15.75m")
plt.plot(Time,xco_225,"-r",label="47.25m")
plt.plot(Time,xco_300,"-b",label="62.58m")
plt.xlabel("Time [s]")
plt.ylabel("Fluctuating displacement x' direction [m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"Fluc_x.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(xco_75[Start_time_idx:],dt,Var="xco_75")
plt.loglog(frq,PSD,"-g",label="15.75m")
frq,PSD = temporal_spectra(xco_225[Start_time_idx:],dt,Var="xco_225")
plt.loglog(frq,PSD,"-r",label="47.25m")
frq,PSD = temporal_spectra(xco_300[Start_time_idx:],dt,Var="xco_300")
plt.loglog(frq,PSD,"-b",label="62.58m")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD - Fluctuating displacement x' direction [m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Fluc_x.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.plot(Time,yco_75,"-g",label="15.75m")
plt.plot(Time,yco_225,"-r",label="47.25m")
plt.plot(Time,yco_300,"-b",label="62.58m")
plt.xlabel("Time [s]")
plt.ylabel("Fluctuating displacement y' direction [m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"Fluc_y.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(yco_75[Start_time_idx:],dt,Var="yco_75")
plt.loglog(frq,PSD,"-g",label="15.75m")
frq,PSD = temporal_spectra(yco_225[Start_time_idx:],dt,Var="yco_225")
plt.loglog(frq,PSD,"-r",label="47.25m")
frq,PSD = temporal_spectra(yco_300[Start_time_idx:],dt,Var="yco_300")
plt.loglog(frq,PSD,"-b",label="62.58m")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD - Fluctuating displacement y' direction [m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Fluc_y.png")
plt.close()

fig = plt.figure(figsize=(14,8))
plt.plot(Time,zco_75,"-g",label="15.75m")
plt.plot(Time,zco_225,"-r",label="47.25m")
plt.plot(Time,zco_300,"-b",label="62.58m")
plt.xlabel("Time [s]")
plt.ylabel("Fluctuating displacement z direction [m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"Fluc_z.png")
plt.close()

fig = plt.figure(figsize=(14,8))
frq,PSD = temporal_spectra(zco_75[Start_time_idx:],dt,Var="zco_75")
plt.loglog(frq,PSD,"-g",label="15.75m")
frq,PSD = temporal_spectra(zco_225[Start_time_idx:],dt,Var="zco_225")
plt.loglog(frq,PSD,"-r",label="47.25m")
frq,PSD = temporal_spectra(zco_300[Start_time_idx:],dt,Var="zco_300")
plt.loglog(frq,PSD,"-b",label="62.58m")
plt.xlabel("Frequency [Hz]")
plt.ylabel("PSD - Fluctuating displacement z direction [m]")
plt.grid()
plt.legend()
plt.tight_layout()
plt.savefig(out_dir+"Spectra_Fluc_z.png")
plt.close()