from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import pyFAST.input_output as io
from scipy.fft import fft, fftfreq, fftshift,ifft
from multiprocessing import Pool
import time


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


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def hard_filter(signal,cutoff,dt,filter_type):

    N = len(signal)
    spectrum = fft(signal)
    F = fftfreq(N,dt)
    if filter_type=="lowpass":
        spectrum_filter = spectrum*(np.absolute(F)<cutoff)
    elif filter_type=="highpass":
        spectrum_filter = spectrum*(np.absolute(F)>cutoff)
    elif filter_type=="bandpass":
        spectrum_filter = spectrum*(np.absolute(F)>cutoff[0])
        spectrum_filter = spectrum_filter*(np.absolute(F)<cutoff[1])
        

    spectrum_filter = ifft(spectrum_filter)

    return np.real(spectrum_filter)


def transform_rotating_frame(it):
    xs_E,ys_E,zs_E, xs_R,ys_R,zs_R = coordinate_rotation(it)
    z_fixed = ((ys_E[it] - Rotor_coordinates[1]) * np.sin(Azimuth[it]) + (zs_E[it] - Rotor_coordinates[2])*np.cos(Azimuth[it]))+Rotor_coordinates[2]

    return z_fixed

#plotting options
plot_relative_displacement = False
plot_3d_rotor = False
plot_AoA = False
plot_radial_vars = False
plot_acceleration = False
plot_OF_vars = False
plot_radial_cc = False
plot_bar_pairs = False
plot_radial_pairs = False
plot_Aero_radial = True


in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"


df_E = Dataset(in_dir+"WTG01b.nc")

WT_E = df_E.groups["WTG01"]

ds_E = Dataset(in_dir+"Dataset.nc")

Rotor_coordinates = [np.float64(WT_E.variables["xyz"][0,0,0]),np.float64(WT_E.variables["xyz"][0,0,1]),np.float64(WT_E.variables["xyz"][0,0,2])]

OF_vars = ds_E.groups["OpenFAST_Variables"]
Azimuth = np.array(OF_vars.variables["Azimuth"])
Azimuth = 360 - Azimuth[1:]

# xo = np.array(WT_E.variables["xyz"][0,1:,0])
# yo = np.array(WT_E.variables["xyz"][0,1:,1])
# zs_E = np.array(WT_E.variables["xyz"][0,1:,2])



#fig = plt.figure()
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



in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

df_R = Dataset(in_dir+"WTG01b.nc")

WT_R = df_R.groups["WTG01"]

ds_R = Dataset(in_dir+"Dataset.nc")

xs_E,ys_E,zs_E, xs_R,ys_R,zs_R = coordinate_rotation(it=200)
print(Azimuth[200])
print(ys_R[299],zs_R[299])
stop =1 

if plot_acceleration == True:
    in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

    out_dir=in_dir+"Elastic_deformations_analysis/"

    df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    Time_OF = np.array(df["Time_[s]"])

    dt_OF = Time_OF[1] - Time_OF[0]

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    Time_OF = Time_OF[Start_time_idx:]

    FLx_Elasto = np.array(df["B1N021FLx_[kN]"][Start_time_idx:])
    FLx_Aero = (np.array(df["AB1N239Fx_[N/m]"][Start_time_idx:]))/1000

    ax = np.subtract(FLx_Elasto,FLx_Aero)

    FLy_Elasto = np.array(df["B1N021FLy_[kN]"][Start_time_idx:])
    FLz_Elasto = np.array(df["B1N021FLz_[kN]"][Start_time_idx:])
    Azimuth = np.radians(np.array(df["Azimuth_[deg]"][Start_time_idx:]))
    FLy_fixed_Elasto, FLz_fixed_Elasto = tranform_fixed_frame(FLy_Elasto,FLz_Elasto,Azimuth)
    FLy_Aero = (np.array(df["AB1N239Ft_[N/m]"][Start_time_idx:]))/1000

    fig = plt.figure()
    plt.plot(Time_OF,FLy_Elasto)

    fig = plt.figure()
    plt.plot(Time_OF,FLz_Elasto)

    fig = plt.figure()
    frq,PSD = temporal_spectra(FLy_Elasto,dt_OF,Var="FLy elasto")
    plt.loglog(frq,PSD)
    
    fig = plt.figure()
    frq,PSD = temporal_spectra(FLz_Elasto,dt_OF,Var="FLz elasto")
    plt.loglog(frq,PSD)

    # fig = plt.figure()
    # plt.plot(Time_OF,FLy_fixed_Elasto)

    # fig = plt.figure()
    # plt.plot(Time_OF,FLz_fixed_Elasto)

    fig = plt.figure()
    plt.plot(Time_OF,FLy_Aero)

    fig = plt.figure()
    frq,PSD = temporal_spectra(FLy_Aero,dt_OF,Var="FLy Aero")
    plt.loglog(frq,PSD)

    plt.show()


    ay = np.subtract(FLy_Elasto,FLy_Aero)

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
    ax1.plot(Time_OF,FLx_Elasto,"-r",label="Elastodyn")
    ax1.plot(Time_OF,FLx_Aero,"-b",label="Aerodyn")
    ax1.set_ylabel("Actuator force in x' direction ~50.4m [kN]")
    ax1.legend()
    ax1.grid()

    ax2.plot(Time_OF,ax)
    ax2.set_ylabel("Force due to acceleration in x' direction ~50.4m [kN]")
    ax2.grid()
    fig.supxlabel("Time [s]")
    plt.tight_layout()

    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
    ax1.plot(Time_OF,FLy_Elasto,"-r",label="Elastodyn")
    ax1.plot(Time_OF,FLy_Aero,"-b",label="Aerodyn")
    ax1.set_ylabel("Actuator force in y' direction ~50.4m [kN]")
    ax1.legend()
    ax1.grid()

    ax2.plot(Time_OF,ay)
    ax2.set_ylabel("Force due to acceleration in y' direction ~50.4m [kN]")
    ax2.grid()
    fig.supxlabel("Time [s]")
    plt.tight_layout()


    plt.show()


if plot_radial_cc == True:

    in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

    out_dir=in_dir+"Elastic_deformations_analysis/"

    df_E = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    Time_OF = np.array(df_E["Time_[s]"])

    dt_OF = Time_OF[1] - Time_OF[0]

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
    
    df_R = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    Variables = ["Alpha_[deg]", "Vrel_[m/s]", "Fn_[N/m]", "Ft_[N/m]"]
    cc_array = []
    #mean_array = []; std_array = []; skew_array = []; flat_array = []
    mean = []; std = []; skew = []; flat = []
    for Var in Variables:
        print(Var)

        cc = []
        for i in np.arange(1,299):
            print(i)
            if i < 10:
                num = "00{}".format(i)
            elif i < 100:
                num = "0{}".format(i)
            elif i < 1000:
                num  = "{}".format(i)

            txt = "AB1N"+num+Var

            var_E = np.array(df_E[txt][Start_time_idx:])
            var_R = np.array(df_R[txt][Start_time_idx:])

            
            cc.append(correlation_coef(var_E,var_R))
            if Var == "Alpha_[deg]":
                moms_E = moments(var_E); moms_R =  moments(var_R)
                mean.append(moms_E[0]-moms_R[0]); std.append(moms_E[1]-moms_R[1]); skew.append(moms_E[2]-moms_R[2]); flat.append(moms_E[3]-moms_R[3])
        #mean = np.true_divide(mean,np.max(mean)); std = np.true_divide(std,np.max(std))
        cc_array.append(cc)
        #mean_array.append(mean); std_array.append(std); skew_array.append(skew); flat_array.append(flat)


    plt.rcParams['font.size'] = 16
    R = np.linspace(0,63,len(np.arange(1,299)))
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(0,len(Variables)):
        plt.plot(R,cc_array[i],label=Variables[i])
    plt.xlabel("Distance along Span [m]")
    plt.ylabel("Correlation coefficient between rigid and deformable rotor [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"cc_R.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    #for i in np.arange(0,len(Variables)):
    plt.plot(R,mean)
    plt.xlabel("Distance along Span [m]")
    plt.ylabel("Difference in mean AoA T=1000s\n between deformable and rigid rotor [deg]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"mean_R.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    #for i in np.arange(0,len(Variables)):
    plt.plot(R,std)
    plt.xlabel("Distance along Span [m]")
    plt.ylabel("Difference in standard deviation AoA T=1000s\nbetween deformable and rigid rotor [deg]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"std_R.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    #for i in np.arange(0,len(Variables)):
    plt.plot(R,skew)
    plt.xlabel("Distance along Span [m]")
    plt.ylabel("Difference in skewness AoA T=1000s\nbetween deformable and rigid rotor [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"skew_R.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    #for i in np.arange(0,len(Variables)):
    plt.plot(R,flat)
    plt.xlabel("Distance along Span [m]")
    plt.ylabel("Difference in flatness AoA T=1000s\nbetween deformable and rigid rotor [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"flat_R.png")
    plt.close(fig)


if plot_radial_vars == True:

    in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

    out_dir=in_dir+"Elastic_deformations_analysis/"

    df_E = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()    

    Time_OF = np.array(df_E["Time_[s]"])

    dt_OF = Time_OF[1] - Time_OF[0]

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    Time_OF = Time_OF[Start_time_idx:]

    in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
    
    df_R = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N016FLx_[kN]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N016FLx_[kN]"][Start_time_idx:]),"-r")
    plt.ylabel("FLx")
    plt.grid()

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N002FLy_[kN]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N002FLy_[kN]"][Start_time_idx:]),"-r")
    plt.ylabel("FLy 2nd")
    plt.grid()

    print("elastic FLy 002\n{}".format(moments(np.array(df_E["B1N002FLy_[kN]"][Start_time_idx:]))))
    print("Rigid FLy 002\n{}".format(moments(np.array(df_R["B1N002FLy_[kN]"][Start_time_idx:]))))

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N020FLy_[kN]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N020FLy_[kN]"][Start_time_idx:]),"-r")
    plt.ylabel("FLy 20th")
    plt.grid()

    print("elastic FLy 020\n{}".format(moments(np.array(df_E["B1N020FLy_[kN]"][Start_time_idx:]))))
    print("Rigid FLy 020\n{}".format(moments(np.array(df_R["B1N020FLy_[kN]"][Start_time_idx:]))))

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N009MLx_[kN-m]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N009MLx_[kN-m]"][Start_time_idx:]),"-r")
    plt.ylabel("MLx 9th")
    plt.grid()

    print("elastic MLx 009\n{}".format(moments(np.array(df_E["B1N009MLx_[kN-m]"][Start_time_idx:]))))
    print("Rigid MLx 009\n{}".format(moments(np.array(df_R["B1N009MLx_[kN-m]"][Start_time_idx:]))))

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N010MLx_[kN-m]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N010MLx_[kN-m]"][Start_time_idx:]),"-r")
    plt.ylabel("MLx 10th")
    plt.grid()
    
    print("elastic MLx 010\n{}".format(moments(np.array(df_E["B1N010MLx_[kN-m]"][Start_time_idx:]))))
    print("Rigid MLx 010\n{}".format(moments(np.array(df_R["B1N010MLx_[kN-m]"][Start_time_idx:]))))

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N007MLy_[kN-m]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N007MLy_[kN-m]"][Start_time_idx:]),"-r")
    plt.ylabel("MLy 7th")
    plt.grid()

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N016MLy_[kN-m]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N016MLy_[kN-m]"][Start_time_idx:]),"-r")
    plt.ylabel("MLy 16th")
    plt.grid()

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N002MLz_[kN-m]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N002MLz_[kN-m]"][Start_time_idx:]),"-r")
    plt.ylabel("MLz 2nd")
    plt.grid()

    print("elastic MLz 002\n{}".format(moments(np.array(df_E["B1N002MLz_[kN-m]"][Start_time_idx:]))))
    print("Rigid MLz 002\n{}".format(moments(np.array(df_R["B1N002MLz_[kN-m]"][Start_time_idx:]))))

    fig = plt.figure()
    plt.plot(Time_OF,np.array(df_E["B1N020MLz_[kN-m]"][Start_time_idx:]),"-b")
    plt.plot(Time_OF,np.array(df_R["B1N020MLz_[kN-m]"][Start_time_idx:]),"-r")
    plt.ylabel("MLz 20th")
    plt.grid()

    print("elastic MLz 020\n{}".format(moments(np.array(df_E["B1N020MLz_[kN-m]"][Start_time_idx:]))))
    print("Rigid MLz 020\n{}".format(moments(np.array(df_R["B1N020MLz_[kN-m]"][Start_time_idx:]))))


    plt.show()


    Variables = ["FLx_[kN]","FLy_[kN]","FLz_[kN]","MLx_[kN-m]","MLy_[kN-m]","MLz_[kN-m]"]
    cc_array = []
    mean_array = []; std_array = []; skew_array = []; flat_array = []
    mean_2_array = []; std_2_array = []; skew_2_array = []; flat_2_array = []
    for Var in Variables:
        print(Var)

        cc = []; mean = []; std = []; skew = []; flat = []
        mean_2 = []; std_2 = []; skew_2 = []; flat_2 = []
        for i in np.arange(1,22):
            print(i)
            if i < 10:
                num = "00{}".format(i)
            elif i < 100:
                num = "0{}".format(i)
            elif i < 1000:
                num  = "{}".format(i)

            txt = "B1N"+num+Var

            var_E = np.array(df_E[txt][Start_time_idx:])
            var_R = np.array(df_R[txt][Start_time_idx:])

            
            cc.append(correlation_coef(var_E,var_R))
            moms_E = moments(var_E); moms_R =  moments(var_R)
            mean.append(((moms_E[0]-moms_R[0])/moms_R[0])*100); std.append(((moms_E[1]-moms_R[1])/moms_R[1])*100)
            skew.append(((moms_E[2]-moms_R[2])/moms_R[2])*100); flat.append(((moms_E[3]-moms_R[3])/moms_R[3])*100)

            mean_2.append(moms_E[0]-moms_R[0]); std_2.append(moms_E[1]-moms_R[1])
            skew_2.append(moms_E[2]-moms_R[2]); flat_2.append(moms_E[3]-moms_R[3])

        cc_array.append(cc)
        mean_array.append(mean); std_array.append(std); skew_array.append(skew); flat_array.append(flat)

        mean_2_array.append(mean_2); std_2_array.append(std_2); skew_2_array.append(skew_2); flat_2_array.append(flat_2)


    ylabels = ["FLx [kN]","FLy [kN]","FLz [kN]","MLx [kN-m]","MLy [kN-m]","MLz [kN-m]"]
    x = np.linspace(0,63,21)

    for i in np.arange(0,len(Variables)):
        fig = plt.figure(figsize=(14,8))
        plt.plot(x,cc_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("Correlation coefficient Rigid cc Elastic [-]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_cc_R_{}.png".format(Variables[i]))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        plt.plot(x,mean_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("relative change in mean [%]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_mean_R_{}.png".format(Variables[i]))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        plt.plot(x,std_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("relative change in standard deviation [%]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_std_R_{}.png".format(Variables[i]))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        plt.plot(x,skew_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("relative change in skewness [%]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_skew_R_{}.png".format(Variables[i]))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        plt.plot(x,flat_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("relative change in flatness [%]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_flat_R_{}.png".format(Variables[i]))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        plt.plot(x,mean_2_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("change in mean")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_2_mean_R_{}.png".format(Variables[i]))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        plt.plot(x,std_2_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("change in standard deviation")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_2_std_R_{}.png".format(Variables[i]))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        plt.plot(x,skew_2_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("change in skewness")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_2_skew_R_{}.png".format(Variables[i]))
        plt.close(fig)

        fig = plt.figure(figsize=(14,8))
        plt.plot(x,flat_2_array[i],"-o")
        plt.xlabel("Spanwise distance from root [m]")
        plt.title(ylabels[i])
        plt.ylabel("change in flatness")
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_2_flat_R_{}.png".format(Variables[i]))
        plt.close(fig)


if plot_Aero_radial == True:

    in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

    out_dir=in_dir+"Elastic_deformations_analysis/"

    df_E = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()    

    Time_OF = np.array(df_E["Time_[s]"])

    dt_OF = Time_OF[1] - Time_OF[0]

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    Time_OF = Time_OF[Start_time_idx:]

    in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
    
    df_R = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    Variables = ["Vrel_[m/s]", "Alpha_[deg]", "Cl_[-]", "Cd_[-]", "Fn_[N/m]", "Ft_[N/m]"]
    label = ["Vrel", "AoA", "Cl", "Cd", "Fn", "Ft"]
    cc_array = []
    mean_array = []; std_array = []; skew_array = []; flat_array = []
    mean_2_array = []; std_2_array = []; skew_2_array = []; flat_2_array = []
    for j in np.arange(0,len(Variables)):
        print(Variables[j])

        cc = []; mean = []; std = []; skew = []; flat = []
        mean_2 = []; std_2 = []; skew_2 = []; flat_2 = []
        mean_R = []; mean_E = []; std_R = []; std_E = []
        for i in np.arange(1,300):
            print(i)
            if i < 10:
                num = "00{}".format(i)
            elif i < 100:
                num = "0{}".format(i)
            elif i < 1000:
                num  = "{}".format(i)

            txt = "AB1N"+num+Variables[j]

            var_E = np.array(df_E[txt][Start_time_idx:])
            var_R = np.array(df_R[txt][Start_time_idx:])

            
            cc.append(correlation_coef(var_E,var_R))
            moms_E = moments(var_E); moms_R =  moments(var_R)
            mean.append(((moms_E[0]-moms_R[0])/moms_R[0])*100); std.append(((moms_E[1]-moms_R[1])/moms_R[1])*100)
            skew.append(((moms_E[2]-moms_R[2])/moms_R[2])*100); flat.append(((moms_E[3]-moms_R[3])/moms_R[3])*100)

            mean_2.append(moms_E[0]-moms_R[0]); std_2.append(moms_E[1]-moms_R[1])
            skew_2.append(moms_E[2]-moms_R[2]); flat_2.append(moms_E[3]-moms_R[3])

            mean_E.append(np.mean(var_E)); mean_R.append(np.mean(var_R))
            std_E.append(np.std(var_E)); std_R.append(np.std(var_R))

        mean_E = np.array(mean_E); mean_R = np.array(mean_R)
        std_E = np.array(std_E); std_R = np.array(std_R)

        x = np.linspace(0,63,299)
        fig = plt.figure(figsize=(14,8))
        plt.plot(x,mean_E,"-b",label="mean deform")
        plt.fill_between(x,mean_E-std_E,mean_E+std_E,color="b",alpha=0.3,label="std deform")
        plt.plot(x,mean_R,"-r",label="mean Rigid")
        plt.fill_between(x,mean_R-std_R,mean_R+std_R,color="r",alpha=0.3,label="std rigid")
        plt.xlabel("Time [s]")
        plt.ylabel(Variables[j])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_R_{}.png".format(label[j]))
        plt.close(fig)

        cc_array.append(cc)
        mean_array.append(mean); std_array.append(std); skew_array.append(skew); flat_array.append(flat)
        mean_2_array.append(mean_2); std_2_array.append(std_2); skew_2_array.append(skew_2); flat_2_array.append(flat_2)


    x = np.linspace(0,63,299)
    fig = plt.figure(figsize=(14,8))
    for i in np.arange(0,len(Variables)):      
        plt.plot(x,cc_array[i],label="{}".format(Variables[i]))
    plt.xlabel("Spanwise distance from root [m]")
    plt.ylabel("Correlation coefficient Rigid cc Elastic [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"summary_cc_R_Aerodyn.png".format(label[i]))
    plt.close(fig)

    for i in np.arange(0,len(Variables)):
        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(x,mean_array[i])
        fig.supxlabel("Spanwise distance from root [m]")
        fig.suptitle(Variables[i])
        ax1.set_ylabel("relative change in mean [%]")
        ax1.grid()
        ax2.plot(x,mean_2_array[i])
        ax2.set_ylabel("Change in mean")
        ax2.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_mean_R_{}.png".format(label[i]))
        plt.close(fig)

        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(x,std_array[i])
        fig.supxlabel("Spanwise distance from root [m]")
        fig.suptitle(Variables[i])
        ax1.set_ylabel("relative change in standard deviation [%]")
        ax1.grid()
        ax2.plot(x,std_2_array[i])
        ax2.set_ylabel("change in standard deviation")
        ax2.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_std_R_{}.png".format(label[i]))
        plt.close(fig)

        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(x,skew_array[i])
        fig.supxlabel("Spanwise distance from root [m]")
        fig.suptitle(Variables[i])
        ax1.set_ylabel("relative change in skewness [%]")
        ax1.grid()
        ax2.plot(x,skew_2_array[i])
        ax2.set_ylabel("change in skewness")
        ax2.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_skew_R_{}.png".format(label[i]))
        plt.close(fig)

        fig,(ax1,ax2) = plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax1.plot(x,flat_array[i],"-")
        fig.supxlabel("Spanwise distance from root [m]")
        fig.suptitle(Variables[i])
        ax1.set_ylabel("relative change in flatness [%]")
        ax1.grid()
        ax2.plot(x,flat_2_array[i])
        ax2.set_ylabel("change in flatness")
        ax2.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_flat_R_{}.png".format(label[i]))
        plt.close(fig)



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
    LPF_AoA_300_E = hard_filter(AoA_300_E,0.3,dt_OF,"lowpass")
    BPF_AoA_300_E = hard_filter(AoA_300_E,[0.3,0.9],dt_OF,"bandpass")
    HPF_AoA_300_E = hard_filter(AoA_300_E,[1.5,40],dt_OF,"bandpass")

    Vrel_75_E = np.array(df["AB1N075Vrel_[m/s]"][Start_time_idx:])
    Vrel_225_E = np.array(df["AB1N225Vrel_[m/s]"][Start_time_idx:])
    Vrel_300_E = np.array(df["AB1N299Vrel_[m/s]"][Start_time_idx:])

    Cl_75_E = np.array(df["AB1N075Cl_[-]"][Start_time_idx:])
    Cl_225_E = np.array(df["AB1N225Cl_[-]"][Start_time_idx:])
    Cl_300_E = np.array(df["AB1N299Cl_[-]"][Start_time_idx:])

    Cd_75_E = np.array(df["AB1N075Cd_[-]"][Start_time_idx:])
    Cd_225_E = np.array(df["AB1N225Cd_[-]"][Start_time_idx:])
    Cd_300_E = np.array(df["AB1N299Cd_[-]"][Start_time_idx:])

    Fn_75_E = np.array(df["AB1N075Fn_[N/m]"][Start_time_idx:])
    Fn_225_E = np.array(df["AB1N225Fn_[N/m]"][Start_time_idx:])
    Fn_300_E = np.array(df["AB1N299Fn_[N/m]"][Start_time_idx:])

    Ft_75_E = np.array(df["AB1N075Ft_[N/m]"][Start_time_idx:])
    Ft_225_E = np.array(df["AB1N225Ft_[N/m]"][Start_time_idx:])
    Ft_300_E = np.array(df["AB1N299Ft_[N/m]"][Start_time_idx:])

    
    in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
    
    df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    Time_OF = np.array(df["Time_[s]"])

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    Time_OF = Time_OF[Start_time_idx:]

    AoA_75_R = np.array(df["AB1N075Alpha_[deg]"][Start_time_idx:])
    AoA_225_R = np.array(df["AB1N225Alpha_[deg]"][Start_time_idx:])
    AoA_300_R = np.array(df["AB1N299Alpha_[deg]"][Start_time_idx:])
    LPF_AoA_300_R = hard_filter(AoA_300_R,0.3,dt_OF,"lowpass")
    BPF_AoA_300_R = hard_filter(AoA_300_R,[0.3,0.9],dt_OF,"bandpass")
    HPF_AoA_300_R = hard_filter(AoA_300_R,[1.5,40],dt_OF,"bandpass")

    Vrel_75_R = np.array(df["AB1N075Vrel_[m/s]"][Start_time_idx:])
    Vrel_225_R = np.array(df["AB1N225Vrel_[m/s]"][Start_time_idx:])
    Vrel_300_R = np.array(df["AB1N299Vrel_[m/s]"][Start_time_idx:])

    Cl_75_R = np.array(df["AB1N075Cl_[-]"][Start_time_idx:])
    Cl_225_R = np.array(df["AB1N225Cl_[-]"][Start_time_idx:])
    Cl_300_R = np.array(df["AB1N299Cl_[-]"][Start_time_idx:])

    Cd_75_R = np.array(df["AB1N075Cd_[-]"][Start_time_idx:])
    Cd_225_R = np.array(df["AB1N225Cd_[-]"][Start_time_idx:])
    Cd_300_R = np.array(df["AB1N299Cd_[-]"][Start_time_idx:])

    Fn_75_R = np.array(df["AB1N075Fn_[N/m]"][Start_time_idx:])
    Fn_225_R = np.array(df["AB1N225Fn_[N/m]"][Start_time_idx:])
    Fn_300_R = np.array(df["AB1N299Fn_[N/m]"][Start_time_idx:])

    Ft_75_R = np.array(df["AB1N075Ft_[N/m]"][Start_time_idx:])
    Ft_225_R = np.array(df["AB1N225Ft_[N/m]"][Start_time_idx:])
    Ft_300_R = np.array(df["AB1N299Ft_[N/m]"][Start_time_idx:])

    
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
    cc = round(correlation_coef(LPF_AoA_300_E,LPF_AoA_300_R),2)
    plt.plot(Time_OF,LPF_AoA_300_R,"-r",label="62.58m, Rigid +10deg")
    plt.plot(Time_OF,LPF_AoA_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("LPF AoA [deg]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(LPF_AoA_300_E),moments(LPF_AoA_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"LPF_AoA_300.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(BPF_AoA_300_E,BPF_AoA_300_R),2)
    plt.plot(Time_OF,BPF_AoA_300_R,"-r",label="62.58m, Rigid +10deg")
    plt.plot(Time_OF,BPF_AoA_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("BPF AoA [deg]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(BPF_AoA_300_E),moments(BPF_AoA_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"BPF_AoA_300.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(HPF_AoA_300_E,HPF_AoA_300_R),2)
    plt.plot(Time_OF,HPF_AoA_300_R,"-r",label="62.58m, Rigid +10deg")
    plt.plot(Time_OF,HPF_AoA_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("HPF AoA [deg]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(HPF_AoA_300_E),moments(HPF_AoA_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"HPF_AoA_300.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(AoA_75_R,dt_OF,Var="AoA 75 R")
    plt.loglog(frq,PSD,"-r",label="15.75m, Rigid")
    frq,PSD = temporal_spectra(AoA_75_E,dt_OF,Var="AoA 75 E")
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
    frq,PSD = temporal_spectra(AoA_225_E,dt_OF,Var="AoA 225 E")
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
    frq,PSD = temporal_spectra(AoA_300_E,dt_OF,Var="AoA 300 E")
    plt.loglog(frq,PSD,"-b",label="63m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD AoA [deg]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_AoA_300.png")
    plt.close(fig)


    cc = round(correlation_coef(Vrel_75_E,Vrel_75_R),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.subtract(Vrel_75_R,0),"-r",label="15.75m, Rigid")
    plt.plot(Time_OF,Vrel_75_E,"-b",label="15.75m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Relative velocity [m/s]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Vrel_75_E),moments(Vrel_75_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Vrel_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Vrel_225_E,Vrel_225_R),2)
    plt.plot(Time_OF,np.add(Vrel_225_R,0),"-r",label="47.25m, Rigid")
    plt.plot(Time_OF,Vrel_225_E,"-b",label="47.25m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Relative velocity [m/s]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Vrel_225_E),moments(Vrel_225_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Vrel_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Vrel_300_E,Vrel_300_R),2)
    plt.plot(Time_OF,np.add(Vrel_300_R,0),"-r",label="62.58m, Rigid")
    plt.plot(Time_OF,Vrel_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Relative velocity [m/s]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(AoA_300_E),moments(AoA_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Vrel_300.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Vrel_75_R,dt_OF,Var="Vrel 75 R")
    plt.loglog(frq,PSD,"-r",label="15.75m, Rigid")
    frq,PSD = temporal_spectra(Vrel_75_E,dt_OF,Var="Vrel 75 E")
    plt.loglog(frq,PSD,"-b",label="15.75m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Relative velocity [m/s]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Vrel_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Vrel_225_R,dt_OF,Var="Vrel 225 R")
    plt.loglog(frq,PSD,"-r",label="47.25m, Rigid")
    frq,PSD = temporal_spectra(Vrel_225_E,dt_OF,Var="Vrel 225 E")
    plt.loglog(frq,PSD,"-b",label="47.25m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Relative velocity [m/s]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Vrel_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Vrel_300_R,dt_OF,Var="Vrel 300 R")
    plt.loglog(frq,PSD,"-r",label="63m, Rigid")
    frq,PSD = temporal_spectra(Vrel_300_E,dt_OF,Var="Vrel 300 E")
    plt.loglog(frq,PSD,"-b",label="63m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD AoA [deg]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Vrel_300.png")
    plt.close(fig)


    cc = round(correlation_coef(Cl_75_E,Cl_75_R),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.subtract(Cl_75_R,0),"-r",label="15.75m, Rigid")
    plt.plot(Time_OF,Cl_75_E,"-b",label="15.75m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Cl [-]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Cl_75_E),moments(Cl_75_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Cl_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Cl_225_E,Cl_225_R),2)
    plt.plot(Time_OF,np.add(Cl_225_R,0),"-r",label="47.25m, Rigid")
    plt.plot(Time_OF,Cl_225_E,"-b",label="47.25m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Cl [-]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Cl_225_E),moments(Cl_225_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Cl_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Cl_300_E,Cl_300_R),2)
    plt.plot(Time_OF,np.add(Cl_300_R,0),"-r",label="62.58m, Rigid")
    plt.plot(Time_OF,Cl_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Cl [-]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Cl_300_E),moments(Cl_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Cl_300.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Cl_75_R,dt_OF,Var="Cl 75 R")
    plt.loglog(frq,PSD,"-r",label="15.75m, Rigid")
    frq,PSD = temporal_spectra(Cl_75_E,dt_OF,Var="Cl 75 E")
    plt.loglog(frq,PSD,"-b",label="15.75m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Cl [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Cl_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Cl_225_R,dt_OF,Var="Cl 225 R")
    plt.loglog(frq,PSD,"-r",label="47.25m, Rigid")
    frq,PSD = temporal_spectra(Cl_225_E,dt_OF,Var="Cl 225 E")
    plt.loglog(frq,PSD,"-b",label="47.25m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Cl [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Cl_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Cl_300_R,dt_OF,Var="Cl 300 R")
    plt.loglog(frq,PSD,"-r",label="63m, Rigid")
    frq,PSD = temporal_spectra(Cl_300_E,dt_OF,Var="Cl 300 E")
    plt.loglog(frq,PSD,"-b",label="63m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Cl [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Cl_300.png")
    plt.close(fig)


    cc = round(correlation_coef(Cd_75_E,Cd_75_R),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.subtract(Cd_75_R,0),"-r",label="15.75m, Rigid")
    plt.plot(Time_OF,Cd_75_E,"-b",label="15.75m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Cd [-]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Cd_75_E),moments(Cd_75_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Cd_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Cd_225_E,Cd_225_R),2)
    plt.plot(Time_OF,np.add(Cd_225_R,0),"-r",label="47.25m, Rigid")
    plt.plot(Time_OF,Cd_225_E,"-b",label="47.25m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Cd [-]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Cd_225_E),moments(Cd_225_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Cd_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Cd_300_E,Cd_300_R),2)
    plt.plot(Time_OF,np.add(Cd_300_R,0),"-r",label="62.58m, Rigid")
    plt.plot(Time_OF,Cd_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Cd [-]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Cd_300_E),moments(Cd_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Cd_300.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Cd_75_R,dt_OF,Var="Cd 75 R")
    plt.loglog(frq,PSD,"-r",label="15.75m, Rigid")
    frq,PSD = temporal_spectra(Cd_75_E,dt_OF,Var="Cd 75 E")
    plt.loglog(frq,PSD,"-b",label="15.75m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Cd [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Cd_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Cd_225_R,dt_OF,Var="Cd 225 R")
    plt.loglog(frq,PSD,"-r",label="47.25m, Rigid")
    frq,PSD = temporal_spectra(Cd_225_E,dt_OF,Var="Cd 225 E")
    plt.loglog(frq,PSD,"-b",label="47.25m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Cd [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Cd_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Cd_300_R,dt_OF,Var="Cd 300 R")
    plt.loglog(frq,PSD,"-r",label="63m, Rigid")
    frq,PSD = temporal_spectra(Cd_300_E,dt_OF,Var="Cd 300 E")
    plt.loglog(frq,PSD,"-b",label="63m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Cd [-]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Cd_300.png")
    plt.close(fig)

    cc = round(correlation_coef(Fn_75_E,Fn_75_R),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.subtract(Fn_75_R,0),"-r",label="15.75m, Rigid")
    plt.plot(Time_OF,Fn_75_E,"-b",label="15.75m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Fn [N/m]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Fn_75_E),moments(Fn_75_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Fn_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Fn_225_E,Fn_225_R),2)
    plt.plot(Time_OF,np.add(Fn_225_R,0),"-r",label="47.25m, Rigid")
    plt.plot(Time_OF,Fn_225_E,"-b",label="47.25m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Fn [N/m]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Fn_225_E),moments(Fn_225_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Fn_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Fn_300_E,Fn_300_R),2)
    plt.plot(Time_OF,np.add(Fn_300_R,0),"-r",label="62.58m, Rigid")
    plt.plot(Time_OF,Fn_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Fn [N/m]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Fn_300_E),moments(Fn_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Fn_300.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Fn_75_R,dt_OF,Var="Fn 75 R")
    plt.loglog(frq,PSD,"-r",label="15.75m, Rigid")
    frq,PSD = temporal_spectra(Fn_75_E,dt_OF,Var="Fn 75 E")
    plt.loglog(frq,PSD,"-b",label="15.75m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Fn [N/m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Fn_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Fn_225_R,dt_OF,Var="Fn 225 R")
    plt.loglog(frq,PSD,"-r",label="47.25m, Rigid")
    frq,PSD = temporal_spectra(Fn_225_E,dt_OF,Var="Fn 225 E")
    plt.loglog(frq,PSD,"-b",label="47.25m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Fn [N/m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Fn_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Fn_300_R,dt_OF,Var="Fn 300 R")
    plt.loglog(frq,PSD,"-r",label="63m, Rigid")
    frq,PSD = temporal_spectra(Fn_300_E,dt_OF,Var="Fn 300 E")
    plt.loglog(frq,PSD,"-b",label="63m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Fn [N/m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Fn_300.png")
    plt.close(fig)


    cc = round(correlation_coef(Ft_75_E,Ft_75_R),2)
    fig = plt.figure(figsize=(14,8))
    plt.plot(Time_OF,np.subtract(Ft_75_R,0),"-r",label="15.75m, Rigid")
    plt.plot(Time_OF,Ft_75_E,"-b",label="15.75m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Ft [N/m]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Ft_75_E),moments(Ft_75_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Ft_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Ft_225_E,Ft_225_R),2)
    plt.plot(Time_OF,np.add(Ft_225_R,0),"-r",label="47.25m, Rigid")
    plt.plot(Time_OF,Ft_225_E,"-b",label="47.25m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Ft [N/m]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Ft_225_E),moments(Ft_225_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Ft_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    cc = round(correlation_coef(Ft_300_E,Ft_300_R),2)
    plt.plot(Time_OF,np.add(Ft_300_R,0),"-r",label="62.58m, Rigid")
    plt.plot(Time_OF,Ft_300_E,"-b",label="63m, Elastic")
    plt.xlabel("Time [s]")
    plt.ylabel("Ft [N/m]")
    plt.title("correlation coefficient = {}\nElastic: {}\nRigid: {}".format(cc,moments(Ft_300_E),moments(Ft_300_R)))
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Ft_300.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Ft_75_R,dt_OF,Var="Ft 75 R")
    plt.loglog(frq,PSD,"-r",label="15.75m, Rigid")
    frq,PSD = temporal_spectra(Ft_75_E,dt_OF,Var="Ft 75 E")
    plt.loglog(frq,PSD,"-b",label="15.75m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Ft [N/m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Ft_75.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Ft_225_R,dt_OF,Var="Ft 225 R")
    plt.loglog(frq,PSD,"-r",label="47.25m, Rigid")
    frq,PSD = temporal_spectra(Ft_225_E,dt_OF,Var="Ft 225 E")
    plt.loglog(frq,PSD,"-b",label="47.25m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Ft [N/m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Ft_225.png")
    plt.close(fig)


    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(Ft_300_R,dt_OF,Var="Ft 300 R")
    plt.loglog(frq,PSD,"-r",label="63m, Rigid")
    frq,PSD = temporal_spectra(Ft_300_E,dt_OF,Var="Ft 300 E")
    plt.loglog(frq,PSD,"-b",label="63m, Elastic")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD Ft [N/m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"Spectra_Ft_300.png")
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
    # plt.savefig(out_dir+"Fluc_x.png")
    # plt.close()

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
    # plt.savefig(out_dir+"Spectra_Fluc_x.png")
    # plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time,yco_75,"-g",label="15.75m")
    plt.plot(Time,yco_225,"-r",label="47.25m")
    plt.plot(Time,yco_300,"-b",label="63m")
    plt.xlabel("Time [s]")
    plt.ylabel("Fluctuating displacement y' direction [m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # plt.savefig(out_dir+"Fluc_y.png")
    # plt.close()

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
    # plt.savefig(out_dir+"Spectra_Fluc_y.png")
    # plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(Time,zco_75,"-g",label="15.75m")
    plt.plot(Time,zco_225,"-r",label="47.25m")
    plt.plot(Time,zco_300,"-b",label="63m")
    plt.xlabel("Time [s]")
    plt.ylabel("Fluctuating displacement z direction [m]")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    # plt.savefig(out_dir+"Fluc_z.png")
    # plt.close()

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
    # plt.savefig(out_dir+"Spectra_Fluc_z.png")
    # plt.close()

    plt.show()



if plot_OF_vars == True:

    Time_OF = np.array(ds_E.variables["Time_OF"])
    dt_OF = Time_OF[1] - Time_OF[0]
    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    Time_OF = Time_OF[Start_time_idx:]

    OF_Vars = ds_E.groups["OpenFAST_Variables"]

    print(OF_Vars)

    RtAeroFxa_E = np.array(OF_Vars.variables["LSShftFxa"][Start_time_idx:])

    RtAeroFys_E = np.array(OF_Vars.variables["LSShftFys"][Start_time_idx:])
    RtAeroFzs_E = np.array(OF_Vars.variables["LSShftFzs"][Start_time_idx:])

    RtAeroMxa_E = np.array(OF_Vars.variables["LSShftMxa"][Start_time_idx:])

    RtAeroMys_E = np.array(OF_Vars.variables["LSSTipMys"][Start_time_idx:])
    RtAeroMzs_E = np.array(OF_Vars.variables["LSSTipMzs"][Start_time_idx:])

    MR_E = np.sqrt(np.add(np.square(RtAeroMys_E),np.square(RtAeroMzs_E)))

    #Total radial aerodynamic bearing force aeroFBR
    L1 = 1.912; L2 = 2.09

    FBMy_E = RtAeroMzs_E/L2; FBFy_E = -RtAeroFys_E*((L1+L2)/L2)
    FBMz_E = -RtAeroMys_E/L2; FBFz_E = -RtAeroFzs_E*((L1+L2)/L2)

    FBy_E = -(FBMy_E + FBFy_E); FBz_E = -(FBMz_E + FBFz_E)

    FBR_E = np.sqrt(np.add(np.square(FBy_E),np.square(FBz_E)))


    #rigid variables

    OF_Vars = ds_R.groups["OpenFAST_Variables"]

    RtAeroFxa_R = np.array(OF_Vars.variables["LSShftFxa"][Start_time_idx:])

    RtAeroFys_R = np.array(OF_Vars.variables["LSShftFys"][Start_time_idx:])
    RtAeroFzs_R = np.array(OF_Vars.variables["LSShftFzs"][Start_time_idx:])

    RtAeroMxa_R = np.array(OF_Vars.variables["LSShftMxa"][Start_time_idx:])

    RtAeroMys_R = np.array(OF_Vars.variables["LSSTipMys"][Start_time_idx:])
    RtAeroMzs_R = np.array(OF_Vars.variables["LSSTipMzs"][Start_time_idx:])

    MR_R = np.sqrt(np.add(np.square(RtAeroMys_R),np.square(RtAeroMzs_R)))

    #Total radial aerodynamic bearing force aeroFBR
    L1 = 1.912; L2 = 2.09

    FBMy_R = RtAeroMzs_R/L2; FBFy_R = -RtAeroFys_R*((L1+L2)/L2)
    FBMz_R = -RtAeroMys_R/L2; FBFz_R = -RtAeroFzs_R*((L1+L2)/L2)

    FBy_R = -(FBMy_R + FBFy_R); FBz_R = -(FBMz_R + FBFz_R)

    FBR_R = np.sqrt(np.add(np.square(FBy_R),np.square(FBz_R)))   


    cc_array = []; mean_array = []; std_array = []; skew_array = []; flat_array = []

    cc_array.append(correlation_coef(FBR_E,FBR_R)); cc_array.append(correlation_coef(FBy_E,FBy_R)); cc_array.append(correlation_coef(FBz_E,FBz_R))
    cc_array.append(correlation_coef(MR_E,MR_R)); cc_array.append(correlation_coef(RtAeroMys_E,RtAeroMys_R)); cc_array.append(correlation_coef(RtAeroMzs_E,RtAeroMzs_R))
    cc_array.append(correlation_coef(RtAeroFys_E,RtAeroFys_R)); cc_array.append(correlation_coef(RtAeroFzs_E,RtAeroFzs_R))
    cc_array.append(correlation_coef(RtAeroFxa_E,RtAeroFxa_R)); cc_array.append(correlation_coef(RtAeroMxa_E,RtAeroMxa_R))

    #FBR
    Moms_E = moments(FBR_E); Moms_R = moments(FBR_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #FBy
    Moms_E = moments(FBy_E); Moms_R = moments(FBy_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #FBz
    Moms_E = moments(FBz_E); Moms_R = moments(FBz_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #MR
    Moms_E = moments(MR_E); Moms_R = moments(MR_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #My
    Moms_E = moments(RtAeroMys_E); Moms_R = moments(RtAeroMys_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #Mz
    Moms_E = moments(RtAeroMzs_E); Moms_R = moments(RtAeroMzs_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #Fy
    Moms_E = moments(RtAeroFys_E); Moms_R = moments(RtAeroFys_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #Fz
    Moms_E = moments(RtAeroFzs_E); Moms_R = moments(RtAeroFzs_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #Fx
    Moms_E = moments(RtAeroFxa_E); Moms_R = moments(RtAeroFxa_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)

    #Mx
    Moms_E = moments(RtAeroMxa_E); Moms_R = moments(RtAeroMxa_R); mean_array.append(((Moms_E[0]-Moms_R[0])/Moms_R[0])*100); std_array.append(((Moms_E[1]-Moms_R[1])/Moms_R[1])*100)
    skew_array.append(((Moms_E[2]-Moms_R[2])/Moms_R[2])*100); flat_array.append(((Moms_E[3]-Moms_R[3])/Moms_R[3])*100)


    #bar charts
    out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
    plt.rcParams['font.size'] = 16

    # fig = plt.figure(figsize=(14,8))
    # plt.plot(Time_OF,RtAeroMys_R,"-r",label="$M_H,y$ Rigid: {}".format(moments(RtAeroMys_R)))
    # plt.plot(Time_OF,RtAeroMys_E,"-b",label="$M_H,y$ Elastic: {}".format(moments(RtAeroMys_E)))
    # plt.xlabel("Time [s]")
    # plt.ylabel("Rotor Moment y component [kN-m]")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(out_dir+"My.png")
    # plt.close(fig)

    # fig = plt.figure(figsize=(14,8))
    # plt.plot(Time_OF,RtAeroMzs_R,"-r",label="$M_H,z$ Rigid: {}".format(moments(RtAeroMzs_R)))
    # plt.plot(Time_OF,RtAeroMzs_E,"-b",label="$M_H,z$ Elastic: {}".format(moments(RtAeroMzs_E)))
    # plt.xlabel("Time [s]")
    # plt.ylabel("Rotor Moment z component [kN-m]")
    # plt.legend()
    # plt.grid()
    # plt.tight_layout()
    # plt.savefig(out_dir+"Mz.png")
    # plt.close(fig)



    xlabel = np.array(["$|F_B|$", "$F_{B_y}$", "$F_{B_z}$", "$|M_H|$", "$M_{H_y}$", "$M_{H_z}$", "$F_{H_y}$", "$F_{H_z}$", "$F_{H_x}$", "$M_{H_x}$"])

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,cc_array)
    plt.ylabel("Correlation coefficient")
    plt.title("Rigid cc Elastic")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_cc.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,mean_array)
    plt.axhline(y=0.0,color="k")
    plt.axhline(y=10,linestyle="--",color="k")
    plt.axhline(y=-10,linestyle="--",color="k")
    plt.ylabel("Relative change in mean [%]")
    plt.title("(Elastic-Rigid)/Rigid *100")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_mean.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,std_array)
    plt.axhline(y=0.0,color="k")
    plt.axhline(y=10,linestyle="--",color="k")
    plt.axhline(y=-10,linestyle="--",color="k")
    plt.ylabel("Relative change in standard deviation [%]")
    plt.title("(Elastic-Rigid)/Rigid *100")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_std.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,skew_array)
    plt.axhline(y=0.0,color="k")
    plt.axhline(y=10,linestyle="--",color="k")
    plt.axhline(y=-10,linestyle="--",color="k")
    plt.ylabel("Relative change in skewness [%]")
    plt.title("(Elastic-Rigid)/Rigid *100")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_skew.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,flat_array)
    plt.axhline(y=0.0,color="k")
    plt.axhline(y=10,linestyle="--",color="k")
    plt.axhline(y=-10,linestyle="--",color="k")
    plt.ylabel("Relative change in flatness [%]")
    plt.title("(Elastic-Rigid)/Rigid *100")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_flat.png")
    plt.close(fig)


    cc_array = []; mean_array = []; std_array = []; skew_array = []; flat_array = []
    #FBR
    Moms_E = moments(FBR_E); Moms_R = moments(FBR_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #FBy
    Moms_E = moments(FBy_E); Moms_R = moments(FBy_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #FBz
    Moms_E = moments(FBz_E); Moms_R = moments(FBz_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #MR
    Moms_E = moments(MR_E); Moms_R = moments(MR_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #My
    Moms_E = moments(RtAeroMys_E); Moms_R = moments(RtAeroMys_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #Mz
    Moms_E = moments(RtAeroMzs_E); Moms_R = moments(RtAeroMzs_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #Fy
    Moms_E = moments(RtAeroFys_E); Moms_R = moments(RtAeroFys_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #Fz
    Moms_E = moments(RtAeroFzs_E); Moms_R = moments(RtAeroFzs_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #Fx
    Moms_E = moments(RtAeroFxa_E); Moms_R = moments(RtAeroFxa_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])

    #Mx
    Moms_E = moments(RtAeroMxa_E); Moms_R = moments(RtAeroMxa_R); mean_array.append(Moms_E[0]-Moms_R[0]); std_array.append(Moms_E[1]-Moms_R[1])
    skew_array.append(Moms_E[2]-Moms_R[2]); flat_array.append(Moms_E[3]-Moms_R[3])


    #bar charts
    out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
    plt.rcParams['font.size'] = 16
    xlabel = np.array(["$|F_B|$", "$F_{B_y}$", "$F_{B_z}$", "$|M_H|$", "$M_{H_y}$", "$M_{H_z}$", "$F_{H_y}$", "$F_{H_z}$", "$F_{H_x}$", "$M_{H_x}$"])

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,mean_array)
    plt.axhline(y=0.0,color="k")
    plt.ylabel("change in mean")
    plt.title("Elastic-Rigid")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_2_mean.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,std_array)
    plt.axhline(y=0.0,color="k")
    plt.ylabel("change in standard deviation")
    plt.title("Elastic-Rigid")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_2_std.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,skew_array)
    plt.axhline(y=0.0,color="k")
    plt.ylabel("change in skewness")
    plt.title("Elastic-Rigid")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_2_skew.png")
    plt.close(fig)

    fig = plt.figure(figsize=(14,8))
    plt.bar(xlabel,flat_array)
    plt.axhline(y=0.0,color="k")
    plt.ylabel("change in flatness")
    plt.title("Elastic-Rigid")
    plt.tight_layout()
    plt.savefig(out_dir+"summary_2_flat.png")
    plt.close(fig)




if plot_bar_pairs == True:

    Time_OF = np.array(ds_E.variables["Time_OF"])

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    Time_OF = Time_OF[Start_time_idx:]

    OF_Vars = ds_E.groups["OpenFAST_Variables"]

    print(OF_Vars)

    RtAeroFxa_E = np.array(OF_Vars.variables["LSShftFxa"][Start_time_idx:])

    RtAeroFys_E = np.array(OF_Vars.variables["LSShftFys"][Start_time_idx:])
    RtAeroFzs_E = np.array(OF_Vars.variables["LSShftFzs"][Start_time_idx:])

    RtAeroMxa_E = np.array(OF_Vars.variables["LSShftMxa"][Start_time_idx:])

    RtAeroMys_E = np.array(OF_Vars.variables["LSSTipMys"][Start_time_idx:])
    RtAeroMzs_E = np.array(OF_Vars.variables["LSSTipMzs"][Start_time_idx:])

    MR_E = np.sqrt(np.add(np.square(RtAeroMys_E),np.square(RtAeroMzs_E)))

    #Total radial aerodynamic bearing force aeroFBR
    L1 = 1.912; L2 = 2.09

    FBMy_E = RtAeroMzs_E/L2; FBFy_E = -RtAeroFys_E*((L1+L2)/L2)
    FBMz_E = -RtAeroMys_E/L2; FBFz_E = -RtAeroFzs_E*((L1+L2)/L2)

    FBy_E = -(FBMy_E + FBFy_E); FBz_E = -(FBMz_E + FBFz_E)

    FBR_E = np.sqrt(np.add(np.square(FBy_E),np.square(FBz_E)))


    #rigid variables

    OF_Vars = ds_R.groups["OpenFAST_Variables"]

    RtAeroFxa_R = np.array(OF_Vars.variables["LSShftFxa"][Start_time_idx:])

    RtAeroFys_R = np.array(OF_Vars.variables["LSShftFys"][Start_time_idx:])
    RtAeroFzs_R = np.array(OF_Vars.variables["LSShftFzs"][Start_time_idx:])

    RtAeroMxa_R = np.array(OF_Vars.variables["LSShftMxa"][Start_time_idx:])

    RtAeroMys_R = np.array(OF_Vars.variables["LSSTipMys"][Start_time_idx:])
    RtAeroMzs_R = np.array(OF_Vars.variables["LSSTipMzs"][Start_time_idx:])

    MR_R = np.sqrt(np.add(np.square(RtAeroMys_R),np.square(RtAeroMzs_R)))

    #Total radial aerodynamic bearing force aeroFBR
    L1 = 1.912; L2 = 2.09

    FBMy_R = RtAeroMzs_R/L2; FBFy_R = -RtAeroFys_R*((L1+L2)/L2)
    FBMz_R = -RtAeroMys_R/L2; FBFz_R = -RtAeroFzs_R*((L1+L2)/L2)

    FBy_R = -(FBMy_R + FBFy_R); FBz_R = -(FBMz_R + FBFz_R)

    FBR_R = np.sqrt(np.add(np.square(FBy_R),np.square(FBz_R)))


    mean_array = []; std_array = []

    mean_array.append(np.mean(FBR_E)); mean_array.append(np.mean(FBR_R))
    mean_array.append(np.mean(FBy_E)); mean_array.append(np.mean(FBy_R))
    mean_array.append(np.mean(FBz_E)); mean_array.append(np.mean(FBz_R))
    mean_array.append(np.mean(MR_E)); mean_array.append(np.mean(MR_R))
    mean_array.append(np.mean(RtAeroMys_E)); mean_array.append(np.mean(RtAeroMys_R))
    mean_array.append(np.mean(RtAeroMzs_E)); mean_array.append(np.mean(RtAeroMzs_R))
    mean_array.append(np.mean(RtAeroFys_E)); mean_array.append(np.mean(RtAeroFys_R))
    mean_array.append(np.mean(RtAeroFzs_E)); mean_array.append(np.mean(RtAeroFzs_R))
    mean_array.append(np.mean(RtAeroFxa_E)); mean_array.append(np.mean(RtAeroFxa_R))
    mean_array.append(np.mean(RtAeroMxa_E)); mean_array.append(np.mean(RtAeroMxa_R))

  
    std_array.append(np.std(FBR_E)); std_array.append(np.std(FBR_R))
    std_array.append(np.std(FBy_E)); std_array.append(np.std(FBy_R))
    std_array.append(np.std(FBz_E)); std_array.append(np.std(FBz_R))
    std_array.append(np.std(MR_E)); std_array.append(np.std(MR_R))
    std_array.append(np.std(RtAeroMys_E)); std_array.append(np.std(RtAeroMys_R))
    std_array.append(np.std(RtAeroMzs_E)); std_array.append(np.std(RtAeroMzs_R))
    std_array.append(np.std(RtAeroFys_E)); std_array.append(np.std(RtAeroFys_R))
    std_array.append(np.std(RtAeroFzs_E)); std_array.append(np.std(RtAeroFzs_R))
    std_array.append(np.std(RtAeroFxa_E)); std_array.append(np.std(RtAeroFxa_R))
    std_array.append(np.std(RtAeroMxa_E)); std_array.append(np.std(RtAeroMxa_R))

    xlabel = np.array(["$|F_B|_E$","$|F_B|_R$", "$F_{B,y,E}$","$F_{B,y,R}$", "$F_{B,z,E}$","$F_{B,z,R}$", "$|M_H|_E$","$|M_H|_R$", "$M_{H,y,E}$","$M_{H,y,R}$", 
                       "$M_{H,z,E}$","$M_{H,z,R}$", "$F_{H,y,E}$","$F_{H,y,R}$", "$F_{H,z,E}$","$F_{H,z,R}$", "$F_{H,x,E}$","$F_{H,x,R}$", "$M_{H,x,E}$","$M_{H,x,R}$"])
    colors = ["b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r","b","r"]
    out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(18,8))
    plt.bar(xlabel,mean_array,color=colors)
    plt.axhline(y=0.0,color="k")
    plt.errorbar(xlabel,mean_array,yerr=std_array,fmt = "o",color="k",capsize=10)
    plt.tight_layout()
    plt.savefig(out_dir+"summary_bar_pairs.png")
    plt.close(fig)



if plot_radial_pairs == True:

    in_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/"

    out_dir=in_dir+"Elastic_deformations_analysis/"

    df_E = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()    

    Time_OF = np.array(df_E["Time_[s]"])

    Start_time_idx = np.searchsorted(Time_OF,Time_OF[0]+200)

    Time_OF = Time_OF[Start_time_idx:]

    in_dir="../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
    
    df_R = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

    out_dir="../../NREL_5MW_MCBL_E_CRPM/post_processing/Elastic_deformations_analysis/"
    plt.rcParams['font.size'] = 16
    Variables = ["FLx_[kN]","FLy_[kN]","FLz_[kN]","MLx_[kN-m]","MLy_[kN-m]","MLz_[kN-m]"]
    for Var in Variables:
        print(Var)

        cc = []; mean_E = []; std_E = []; mean_R = []; std_R = []
        for i in np.arange(1,22):
            print(i)
            if i < 10:
                num = "00{}".format(i)
            elif i < 100:
                num = "0{}".format(i)
            elif i < 1000:
                num  = "{}".format(i)

            txt = "B1N"+num+Var

            var_E = np.array(df_E[txt][Start_time_idx:])
            var_R = np.array(df_R[txt][Start_time_idx:])

            mean_E.append(np.mean(var_E)); mean_R.append(np.mean(var_R))
            std_E.append(np.std(var_E)); std_R.append(np.std(var_R))

        mean_E = np.array(mean_E); mean_R = np.array(mean_R)
        std_E = np.array(std_E); std_R = np.array(std_R)

        x = np.linspace(0,63,21)
        fig = plt.figure(figsize=(14,8))
        plt.plot(x,mean_E,"-b",label="mean deform")
        plt.fill_between(x,mean_E-std_E,mean_E+std_E,color="b",alpha=0.3,label="std deform")
        plt.plot(x,mean_R,"-r",label="mean Rigid")
        plt.fill_between(x,mean_R-std_R,mean_R+std_R,color="r",alpha=0.3,label="std rigid")
        plt.xlabel("Time [s]")
        plt.ylabel(Var)
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"summary_R_{}.png".format(Var))
        plt.close(fig)