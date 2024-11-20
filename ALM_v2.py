import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
import pyFAST.postpro as postpro
from scipy.fft import fft, fftfreq, fftshift
import pandas as pd
from scipy import interpolate
import math
from scipy.signal import butter,filtfilt


def hard_filter(signal,cutoff,dt,filter_type):

    N = len(signal)
    spectrum = np.fft.fft(signal)
    F = np.fft.fftfreq(N,dt)
    #F = (1/(dt*N)) * np.arange(N)
    if filter_type=="lowpass":
        spectrum_filter = spectrum*(np.abs(F)<cutoff)
    elif filter_type=="highpass":
        spectrum_filter = spectrum*(np.abs(F)>cutoff)
    elif filter_type=="bandpass":
        spectrum_filter = spectrum*(np.abs(F)>cutoff[0])
        spectrum_filter = spectrum_filter*(np.abs(F)<cutoff[1])
        

    spectrum_filter = np.fft.ifft(spectrum_filter)

    return np.real(spectrum_filter)



# in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/ALM sensitivity Analysis/"

# int_variables = ["RtAeroFxh_[N]","RtAeroFyh_[N]","RtAeroFzh_[N]","RtAeroMxh_[N-m]","RtAeroMyh_[N-m]","RtAeroMzh_[N-m]"]

# iv = 0
# for var in int_variables:
#     ix = 0


#     dfa = io.fast_output_file.FASTOutputFile(in_dir+"eps_dx_1/NREL_5MW_Main.out").toDataFrame()
#     dfb = io.fast_output_file.FASTOutputFile(in_dir+"dblade_1.0/NREL_5MW_Main.out").toDataFrame()
#     dfc = io.fast_output_file.FASTOutputFile(in_dir+"eps_dx_4/NREL_5MW_Main.out").toDataFrame()
    
#     Timea = np.array(dfa["Time_[s]"])
#     dta = Timea[1] - Timea[0]
#     Time_start_idx = np.searchsorted(Timea,Timea[0]+5)
#     Timea = np.array(Timea[Time_start_idx:])

#     ya = np.array(dfa[var][Time_start_idx:])

#     if np.min(ya) < 0:
#         ya = ya-(1.5*np.min(ya))

#     ya = hard_filter(ya,40,dta,"lowpass")

#     Timeb = np.array(dfb["Time_[s]"])
#     dtb = Timeb[1] - Timeb[0]
#     Time_start_idx = np.searchsorted(Timeb,Timeb[0]+5)
#     Timeb = np.array(Timeb[Time_start_idx:])

#     yb = np.array(dfb[var][Time_start_idx:])

#     if np.min(yb) < 0:
#         yb = yb-(1.5*np.min(yb))

#     yb = hard_filter(yb,40,dtb,"lowpass")

#     #yb = interpolate.interp1d(Timeb, yb,kind="linear")(Timea)


#     Timec = np.array(dfc["Time_[s]"])
#     dtc = Timec[1] - Timec[0]
#     Time_start_idx = np.searchsorted(Timec,Timec[0]+5)
#     Timec = np.array(Timec[Time_start_idx:])

#     yc = np.array(dfc[var][Time_start_idx:])

#     if np.min(yc) < 0:
#         yc = yc-(1.5*np.min(yc))

#     yc = hard_filter(yc,40,dtc,"lowpass")

#     #yc = interpolate.interp1d(Timec, yc,kind="linear")(Timea)
    
#     if var=="RtAeroMyh_[N-m]":
#         plt.rcParams['font.size'] = 16
#         fig = plt.figure(figsize=(14,8))
#         plt.plot(Timea,ya,"-r",label="3a")
#         plt.plot(Timeb,yb,"-b",label="3b")
#         plt.plot(Timec,yc,"-g",label="3c")
#         plt.xlabel("Time [s]")
#         plt.ylabel("Aerodynamic hub moment $\widetilde{M}_{\widehat{H},y}$ [N-m]")
#         plt.grid()
#         plt.legend()
#         plt.tight_layout()
#         plt.savefig("../../Thesis/Figures/ALM_sensitivity_eps_dx_My.png")
#         plt.close(fig)


#     diffab = round((np.average(abs((ya - yb)/yb)))*100,2)
#     diffbc = round((np.average(abs((yb - yc)/yc)))*100,2)

#     print("diff a-b: {} {}".format(var,diffab))
#     print("diff b-c: {} {}".format(var,diffbc))


# cases = ["eps_dx_1","dblade_1.0","eps_dx_4"]

# act_stations_cases = [54,54,54]
# dt_cases = [0.0039,0.0039, 0.0039]
# labels = ["2a","2b","2c"]
# rad_variables = ["Alpha_[deg]","Cd_[-]","Cl_[-]","Fn_[N/m]","Ft_[N/m]","Vx_[m/s]"]
# rad_label = ["Alpha","Cd","Cl","Fn","Ft","Vx"]

# #x_min = np.linspace(0,1,47)
# iv = 0
# for var in rad_variables:
#     data = []
#     ix=0
#     for case in cases:
#         df = io.fast_output_file.FASTOutputFile(in_dir+case+"/NREL_5MW_Main.out").toDataFrame()

#         act_stations = act_stations_cases[ix]
#         x = np.linspace(0,1,act_stations)

#         Var_list = []
#         for i in np.arange(1,act_stations+1):
#             if i < 10:
#                 txt = "AB1N00{}{}".format(i,var)
#             elif i >= 10 and i < 100:
#                 txt = "AB1N0{}{}".format(i,var)
#             elif i >= 100:
#                 txt = "AB1N{}{}".format(i,var)


#             tstart_idx = len((df["Time_[s]"])) - int((3 * 5)/dt_cases[ix])
            
#             y = np.array(df[txt][tstart_idx:])
#             y = hard_filter(y,40,dt_cases[ix],"lowpass")

#             Var_list.append(np.average(y))

#         data.append(Var_list)

#         #data.append(interpolate.interp1d(x, Var_list,kind="linear")(x_min))

#         ix+=1


#     data = np.array(data)

#     perc_diffab = abs((data[0] - data[1])/data[1])
#     perc_diffbc = abs((data[1] - data[2])/data[2])

#     perc_diffab[np.isnan(perc_diffab)] = 0
#     perc_diffab[np.isinf(perc_diffab)] = 0

#     perc_diffbc[np.isnan(perc_diffbc)] = 0
#     perc_diffbc[np.isinf(perc_diffbc)] = 0


#     # plt.rcParams['font.size'] = 16
#     # fig = plt.figure(figsize=(14,8))
#     # plt.plot(x_min,perc_diffab*100,"-r",label="(2a-2b)/2b")
#     # plt.plot(x_min,perc_diffbc*100,"-b",label="(2b-2c)/2c")
#     # plt.legend()
#     # plt.axvline(x=4.1/61.5,color="k",linestyle="--")
#     # plt.axvline(x=9.35/61.5,color="k",linestyle="--")
#     # plt.axvline(x=14.35/61.5,color="k",linestyle="--")
#     # plt.axvline(x=22.55/61.5,color="k",linestyle="--")
#     # plt.axvline(x=26.65/61.5,color="k",linestyle="--")
#     # plt.axvline(x=34.85/61.5,color="k",linestyle="--")
#     # plt.axvline(x=43.05/61.5,color="k",linestyle="--")
#     # plt.xlabel("Normalized blade span [-]")
#     # plt.ylabel("Absolute percentage difference [%]")
#     # plt.title("Local {} averaged over last 3 blade rotations".format(rad_label[iv]))
#     # plt.grid()
#     # plt.tight_layout()
#     # plt.savefig("../../Thesis/Figures/ALM_sensitivity_eps_dr_radial_spurious_spikes_{}.png".format(rad_label[iv]))
#     # plt.close(fig)

#     # R = [4.1/61.5,10.2/61.5,14.35/61.5,22.55/61.5,26.65/61.5,34.85/61.5,43.05/61.5]
#     # for r in R:
#     #     idx = np.searchsorted(x_min,r)
#     #     perc_diffab[idx] = 0
#     #     perc_diffbc[idx] = 0

#     # perc_diffab[7] = 0
#     # perc_diffbc[7] = 0

#     diffab = round((np.average(perc_diffab))*100,2)
#     diffbc = round((np.average(perc_diffbc))*100,2)


#     print("diff a-b: {} {}".format(var,diffab))
#     print("diff b-c: {} {}".format(var,diffbc))

#     iv+=1



in_dir="../../../../media/jarred/EXTERNAL_US/Jarred PhD Data/ALM sensitivity Analysis/"

int_variables = ["RtAeroFxh_[N]","RtAeroFyh_[N]","RtAeroFzh_[N]","RtAeroMxh_[N-m]","RtAeroMyh_[N-m]","RtAeroMzh_[N-m]"]

iv = 0
for var in int_variables:
    ix = 0


    dfa = io.fast_output_file.FASTOutputFile(in_dir+"fllc_Ex5/NREL_5MW_Main.out").toDataFrame()
    dfb = io.fast_output_file.FASTOutputFile(in_dir+"fllc_Ex3/NREL_5MW_Main.out").toDataFrame()
    
    Timea = np.array(dfa["Time_[s]"])
    dta = Timea[1] - Timea[0]
    Time_start_idx = np.searchsorted(Timea,Timea[0]+5)
    Timea = np.array(Timea[Time_start_idx:])

    ya = np.array(dfa[var][Time_start_idx:])

    if np.min(ya) < 0:
        ya = ya-(1.5*np.min(ya))

    ya = hard_filter(ya,40,dta,"lowpass")

    Timeb = np.array(dfb["Time_[s]"])
    dtb = Timeb[1] - Timeb[0]
    Time_start_idx = np.searchsorted(Timeb,Timeb[0]+5)
    Timeb = np.array(Timeb[Time_start_idx:])

    yb = np.array(dfb[var][Time_start_idx:])

    if np.min(yb) < 0:
        yb = yb-(1.5*np.min(yb))

    yb = hard_filter(yb,40,dtb,"lowpass")

    #yb = interpolate.interp1d(Timeb, yb,kind="linear")(Timea)

    
    # if var=="RtAeroMxh_[N-m]":
    #     plt.rcParams['font.size'] = 16
    #     fig = plt.figure(figsize=(14,8))
    #     plt.plot(Timea,ya,"-r",label="Classical")
    #     plt.plot(Timeb,yb,"-b",label="FLLC")
    #     plt.xlabel("Time [s]")
    #     plt.ylabel("Aerodynamic hub moment $\widetilde{M}_{\widehat{H},x}$ [N-m]")
    #     plt.grid()
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.savefig("../../Thesis/Figures/ALM_sensitivity_eps_c_FLLC_Mx.png")
    #     plt.close(fig)


    diffab = round((np.average(abs((ya - yb)/yb)))*100,2)

    print("diff a-b: {} {}".format(var,diffab))


cases = ["fllc_Ex5","fllc_Ex3"]

act_stations_cases = [300,300]
dt_cases = [0.0039,0.0039]
labels = ["Classical $\epsilon /c = 0.5$","FLLC"]
rad_variables = ["Alpha_[deg]","Cd_[-]","Cl_[-]","Fn_[N/m]","Ft_[N/m]","Vx_[m/s]"]
rad_label = ["Alpha","Cd","Cl","Fn","Ft","Vx"]

#x_min = np.linspace(0,1,74)
iv = 0
for var in rad_variables:
    data = []
    ix=0
    for case in cases:
        df = io.fast_output_file.FASTOutputFile(in_dir+case+"/NREL_5MW_Main.out").toDataFrame()

        act_stations = act_stations_cases[ix]
        x = np.linspace(0,1,act_stations)

        Var_list = []
        for i in np.arange(1,act_stations+1):
            if i < 10:
                txt = "AB1N00{}{}".format(i,var)
            elif i >= 10 and i < 100:
                txt = "AB1N0{}{}".format(i,var)
            elif i >= 100:
                txt = "AB1N{}{}".format(i,var)


            tstart_idx = len((df["Time_[s]"])) - int((3 * 5)/dt_cases[ix])
            
            y = np.array(df[txt][tstart_idx:])
            y = hard_filter(y,40,dt_cases[ix],"lowpass")

            Var_list.append(np.average(y))

        data.append(Var_list)
        #data.append(interpolate.interp1d(x, Var_list,kind="linear")(x_min))

        ix+=1


    data = np.array(data)

    perc_diffab = abs((data[0] - data[1])/data[1])

    perc_diffab[np.isnan(perc_diffab)] = 0
    perc_diffab[np.isinf(perc_diffab)] = 0


    if var=="Vx_[m/s]":
        plt.rcParams['font.size'] = 16
        fig = plt.figure(figsize=(14,8))
        plt.plot(x,data[0],"-r",label="4a")
        plt.plot(x,data[1],"-b",label="4b")
        plt.xlabel("Time [s]")
        plt.ylabel("Local axial velocity [m/s]")
        plt.title("Averaged over last 3 blade rotations")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("../../Thesis/Figures/ALM_sensitivity_FLLC_eps_Vx.png")
        plt.close(fig)

    diffab = round((np.average(perc_diffab))*100,2)


    print("diff a-b: {} {}".format(var,diffab))

    iv+=1



