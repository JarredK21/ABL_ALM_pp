from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pyFAST.input_output as io
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


def moments(y):
    mu = round(np.mean(y),2)
    std = round(np.std(y),2)
    N = len(y)

    skewness = round((np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3),2)
    kurotsis = round((np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4),2)

    return mu, std, skewness,kurotsis



#plotting options
plot_relative_displacement = False
plot_3d_rotor = False
plot_AoA = True
plot_radial_vars = False

in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"


df_E = Dataset(in_dir+"WTG01b.nc")

WT_E = df_E.groups["WTG01"]

# xo = np.array(WT_E.variables["xyz"][0,1:,0])
# yo = np.array(WT_E.variables["xyz"][0,1:,1])
# zs_E = np.array(WT_E.variables["xyz"][0,1:,2])

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')

# ax.scatter(xo, yo, zs_E)

# xo = np.array(WT_E.variables["xyz"][0,601:901,0])
# yo = np.array(WT_E.variables["xyz"][0,601:901,1])
# zs_E = np.array(WT_E.variables["xyz"][0,601:901,2])

# ax.scatter(xo, yo, zs_E,color="r")

# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')

# plt.show()


Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]

in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_R = Dataset(in_dir+"WTG01b.nc")

WT_R = df_R.groups["WTG01"]




if plot_AoA == True:

    in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

    out_dir=in_dir+"Elastic_deformations_analysis/"

    df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    Time_OF = np.array(df["Time_[s]"])

    dt_OF = Time_OF[1] - Time_OF[0]

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    AoA_75_E = np.array(df["AB1N075Alpha_[deg]"][Start_time_idx:])
    AoA_225_E = np.array(df["AB1N225Alpha_[deg]"][Start_time_idx:])
    AoA_300_E = np.array(df["AB1N299Alpha_[deg]"][Start_time_idx:])

    
    in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
    
    df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    Time_OF = np.array(df["Time_[s]"])

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    Time_OF = Time_OF[Start_time_idx:]

    AoA_75_R = np.array(df["AB1N075Alpha_[deg]"][Start_time_idx:])
    AoA_225_R = np.array(df["AB1N225Alpha_[deg]"][Start_time_idx:])
    AoA_300_R = np.array(df["AB1N299Alpha_[deg]"][Start_time_idx:])

    
    plt.rcParams['font.size'] = 16
    cc = round(correlation_coef(AoA_75_E,AoA_75_R),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.subtract(AoA_75_R,15),"-r",label="15.75m, Rigid -15deg")
    plt.plot(Time_OF,AoA_75_E,"-b",label="15.75m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("AoA [deg]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(AoA_75_E),moments(AoA_75_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"AoA_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(AoA_225_E,AoA_225_R),2)
    plt.plot(Time_OF,np.add(AoA_225_R,10),"-r",label="47.25m, Rigid +10deg")
    plt.plot(Time_OF,AoA_225_E,"-b",label="47.25m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("AoA [deg]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(AoA_225_E),moments(AoA_225_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"AoA_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(AoA_300_E,AoA_300_R),2)
    plt.plot(Time_OF,np.add(AoA_300_R,10),"-r",label="62.58m, Rigid +10deg")
    plt.plot(Time_OF,AoA_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("AoA [deg]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(AoA_300_E),moments(AoA_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"AoA_300.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(AoA_75_R,dt_OF,Var="AoA 75 R")
    plt.loglog(frq,PSD,"-r",label="15.75m, Rigid")
    frq,PSD = temporal_spectra(AoA_75_E,dt_OF,Var="AoA 75 R")
    plt.loglog(frq,PSD,"-b",label="15.75m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD AoA [deg]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_AoA_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(AoA_225_R,dt_OF,Var="AoA 225 R")
    plt.loglog(frq,PSD,"-r",label="47.25m, Rigid")
    frq,PSD = temporal_spectra(AoA_225_E,dt_OF,Var="AoA 225 R")
    plt.loglog(frq,PSD,"-b",label="47.25m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD AoA [deg]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_AoA_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(AoA_300_R,dt_OF,Var="AoA 300 R")
    plt.loglog(frq,PSD,"-r",label="63m, Rigid")
    frq,PSD = temporal_spectra(AoA_300_E,dt_OF,Var="AoA 300 R")
    plt.loglog(frq,PSD,"-b",label="63m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD AoA [deg]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_AoA_300.png")
    plt.close(fig)



if plot_radial_vars == True:

    in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

    Time = np.array(WT_E.variables["time"])
    dt = Time[1] - Time[0]

    Start_time_idx = np.searchsorted(Time,Time[0]+200)
    Time_steps = np.arange(Start_time_idx,len(Time))

    Time = Time[Start_time_idx:]

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
    df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    Time_OF = np.array(df["Time_[s]"])

    dt_OF = Time_OF[1] - Time_OF[0]

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    FLx = np.array(df["B1N021FLx_[kN]"][Start_time_idx:])
    FLy = np.array(df["B1N021FLy_[kN]"][Start_time_idx:])

    plt.rcParams['font.size'] = 16
    fig,ax = plt.subplots(figsize=(14,8))
    frq,PSD = temporal_spectra(FLx,dt_OF,Var="FLx")
    ax.loglog(frq,PSD,"-r")
    ax.set_ylabel("Tip Force flapwise direction [kN]")
    ax2=ax.twinx()
    frq,PSD = temporal_spectra(xco_300[Start_time_idx:],dt,Var="xco_300")
    ax2.loglog(frq,PSD,"-b")
    ax2.set_ylabel("Tip Displacement x' direction [m]")
    fig.supxlabel("Frequency [Hz]")
    ax.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FLx.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(FLx,dt_OF,Var="FLx")
    plt.loglog(frq,PSD,"-r",label="Tip force flapwise direction [kN]")
    frq,PSD = temporal_spectra(xco_300[Start_time_idx:],dt,Var="xco_300")
    plt.loglog(frq,PSD,"-b",label="Tip displacement x' direction [m]")
    plt.ylabel("PSD")
    plt.xlabel("Frequency [Hz]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FLx_2.png")
    plt.close(fig)


    fig,ax = plt.subplots(figsize=(14,8))
    frq,PSD = temporal_spectra(FLy,dt_OF,Var="FLy")
    ax.loglog(frq,PSD,"-r")
    ax.set_ylabel("Tip Force edgewise direction [kN]")
    ax2=ax.twinx()
    frq,PSD = temporal_spectra(yco_300[Start_time_idx:],dt,Var="yco_300")
    ax2.loglog(frq,PSD,"-b")
    ax2.set_ylabel("Tip Displacement y' direction [m]")
    fig.supxlabel("Frequency [Hz]")
    ax.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FLy.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(FLy,dt_OF,Var="FLy")
    plt.loglog(frq,PSD,"-r",label="Tip force edgewise direction [kN]")
    frq,PSD = temporal_spectra(yco_300[Start_time_idx:],dt,Var="yco_300")
    plt.loglog(frq,PSD,"-b",label="Tip displacement y' direction [m]")
    plt.ylabel("PSD")
    plt.xlabel("Frequency [Hz]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"FLy_2.png")
    plt.close(fig)



if plot_3d_rotor == True:

    Time = np.array(WT_E.variables["time"])
    dt = Time[1] - Time[0]

    Start_time_idx = np.searchsorted(Time,Time[0]+200)
    Time_steps = np.arange(Start_time_idx,len(Time))

    Time = Time[Start_time_idx:]

    xco_75 = np.array(WT_E.variables["xyz"][Start_time_idx:,75,0]); xco_225 = np.array(WT_E.variables["xyz"][Start_time_idx:,225,0]); xco_300 = np.array(WT_E.variables["xyz"][Start_time_idx:,299,0])
    yco_75 = np.array(WT_E.variables["xyz"][Start_time_idx:,75,1]); yco_225 = np.array(WT_E.variables["xyz"][Start_time_idx:,225,1]); yco_300 = np.array(WT_E.variables["xyz"][Start_time_idx:,299,1])
    zco_75 = np.array(WT_E.variables["xyz"][Start_time_idx:,75,2]); zco_225 = np.array(WT_E.variables["xyz"][Start_time_idx:,225,2]); zco_300 = np.array(WT_E.variables["xyz"][Start_time_idx:,299,2])


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(xco_75,dt,Var="xco_75")
    plt.loglog(frq,PSD,"-g",label="15.75m")
    frq,PSD = temporal_spectra(xco_225,dt,Var="xco_225")
    plt.loglog(frq,PSD,"-r",label="47.25")
    frq,PSD = temporal_spectra(xco_300,dt,Var="xco_300")
    plt.loglog(frq,PSD,"-b",label="62.58m")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("x coordinate [m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_x.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(yco_75,dt,Var="yco_75")
    plt.loglog(frq,PSD,"-g",label="15.75m")
    frq,PSD = temporal_spectra(yco_225,dt,Var="yco_225")
    plt.loglog(frq,PSD,"-r",label="47.25")
    frq,PSD = temporal_spectra(yco_300,dt,Var="yco_300")
    plt.loglog(frq,PSD,"-b",label="62.58m")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("y coordinate [m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_y.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(zco_75,dt,Var="zco_75")
    plt.loglog(frq,PSD,"-g",label="15.75m")
    frq,PSD = temporal_spectra(zco_225,dt,Var="zco_225")
    plt.loglog(frq,PSD,"-r",label="47.25")
    frq,PSD = temporal_spectra(zco_300,dt,Var="zco_300")
    plt.loglog(frq,PSD,"-b",label="62.58m")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("z coordinate [m]")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_z.png")
    plt.close()


if plot_relative_displacement == True:
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


    out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time,xco_75,"-g",label="15.75m")
    plt.plot(Time,xco_225,"-r",label="47.25m")
    plt.plot(Time,xco_300,"-b",label="63m")
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
    plt.loglog(frq,PSD,"-b",label="63m")
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
    plt.plot(Time,yco_300,"-b",label="63m")
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
    plt.loglog(frq,PSD,"-b",label="63m")
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
    plt.plot(Time,zco_300,"-b",label="63m")
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
    plt.loglog(frq,PSD,"-b",label="63m")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD - Fluctuating displacement z direction [m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Fluc_z.png")
    plt.close()