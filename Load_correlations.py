import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
import pyFAST.input_output as io
from matplotlib.backends.backend_pdf import PdfPages

def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def low_pass_filter(signal, cutoff):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z



in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

out_dir = in_dir + "correlations/"

Time_OF = np.array(a.variables["time_OF"])
Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

Time_start = 200
Time_end = Time_sampling[-1]

dt = Time_OF[1] - Time_OF[0]

Time_start_idx = np.searchsorted(Time_OF,Time_start)
Time_end_idx = np.searchsorted(Time_OF,Time_end)

Time_OF = Time_OF[Time_start_idx:Time_end_idx]

Azimuth = np.radians(np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx]))

RtAeroFxh = np.array(a.variables["RtAeroFxh"][Time_start_idx:Time_end_idx])
RtAeroFyh = np.array(a.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
RtAeroFzh = np.array(a.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)

RtAeroFR = np.sqrt( np.add( np.square(RtAeroFys), np.square(RtAeroFzs) ) )

RtAeroMxh = np.array(a.variables["RtAeroMxh"][Time_start_idx:Time_end_idx])
RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 

LSShftMxa = np.array(a.variables["LSShftMxa"][Time_start_idx:Time_end_idx])
LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

LSShftFxa = np.array(a.variables["LSShftFxa"][Time_start_idx:Time_end_idx])
LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
LSShftFR = np.sqrt( np.add(np.square(LSShftFys), np.square(LSShftFzs)) )

LSSGagMys = np.array(a.variables["LSSGagMys"][Time_start_idx:Time_end_idx])
LSSGagMzs = np.array(a.variables["LSSGagMzs"][Time_start_idx:Time_end_idx])
LSSGagMR = np.sqrt( np.add(np.square(LSSGagMys), np.square(LSSGagMzs)) )

L1 = 1.912; L2 = 2.09

FBMy = LSSTipMzs/L2; FBFy = -LSShftFys*((L1+L2)/L2)
FBMz = -LSSTipMys/L2; FBFz = -LSShftFzs*((L1+L2)/L2)

FBy = FBMy + FBFy; FBz = FBMz + FBFz
FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
Theta_FB = np.degrees(np.arctan2(FBz,FBy))


Aero_FBMy = RtAeroMzs/L2; Aero_FBFy = -RtAeroFys*((L1+L2)/L2)
Aero_FBMz = -RtAeroMys/L2; Aero_FBFz = -RtAeroFzs*((L1+L2)/L2)

Aero_FBy = Aero_FBMy + Aero_FBFy; Aero_FBz = Aero_FBMz + Aero_FBFz

Aero_FBR = np.sqrt(np.add(np.square(Aero_FBy),np.square(Aero_FBz)))
Rel_Aero_FBy = np.true_divide(np.square(Aero_FBy),np.square(Aero_FBR))
Rel_Aero_FBz = np.true_divide(np.square(Aero_FBz),np.square(Aero_FBR))
add_Aero_RelFB = np.add(Rel_Aero_FBy,Rel_Aero_FBz)
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))

offset = "5.5"
group = a.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Uz = np.array(group.variables["Uz"])
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])



f = interpolate.interp1d(Time_sampling,Ux)
Ux = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Uz)
Uz = f(Time_OF)

f = interpolate.interp1d(Time_sampling,IA)
IA = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iy)
Iy = f(Time_OF)

f = interpolate.interp1d(Time_sampling,Iz)
Iz = f(Time_OF)


Variables_arr = [["Ux", "RtAeroFxh"],["Ux", "RtAeroMxh"], ["Ux", "RtAeroMR"], ["IA", "RtAeroMR"], ["IA", "AeroFBR"],
                 ["AeroFBy", "AeroFBz"], ["RtAeroMys", "RtAeroMzs"], ["Iy", "RtAeroMys"], ["Iz", "RtAeroMzs"],
                 ["Iy", "AeroFBz"], ["Iz", "AeroFBy"], ["Iy", "Iz"], ["RtAeroMys", "AeroFBz"], ["RtAeroFzs", "AeroFBz"],
                 ["RtAeroMzs", "AeroFBy"], ["RtAeroFys", "AeroFBy"]]
units_arr = [["[m/s]", "[kN]"], ["[m/s]", "[kN-m]"], ["[m/s]", "[kN-m]"], ["[$m^4/s$]", "[kN-m]"], ["$m^4/s$", "[kN]"],
               ["[kN]", "[kN]"], ["[kN-m]", "[kN-m]"], ["[$m^4/s$]", "[kN-m]"], ["[$m^4/s$]", "[kN-m]"],
               ["[$m^4/s$]", "[kN]"], ["[$m^4/s$]", "[kN]"], ["[$m^4/s$]", "[$m^4/s$]"], ["[kN-m]", "[kN]"], ["[kN]", "[kN]"],
               ["[kN-m]", "[kN]"], ["[kN]", "[kN]"]]
Ylabels_arr = [["Rotor averaged horizontal velocity", "Thrust"],["Rotor averaged horizonta velocity", "Torque"], 
               ["Rotor averaged horizontal velocity", "Aerodynamic Rotor out-of-plane bending moment"], 
               ["Asymmtry parameter", "Aerodynamic Rotor out-of-plane bending moment"], 
               ["Asymmetry parameter", "Aerodynamic Bearing Force"],
                 ["Aerodynamic Bearing force y component", "Aerodynamic Bearing force z component"], 
                 ["Aerodynamic Rotor moment y component", "Aerodynamic Rotor moment z component"],
                  ["Asymmetry around y axis", "Aerodynamic Rotor moment y component"], 
                  ["Asymmetry around z axis", "Aerodynamic Rotor moment z component"],
                 ["Asymmetry around y axis", "Aerodynamic Bearing force z component"],
                  ["Asymmetry around z axis", "Aerodynamic Bearing force y component"],
                   ["Asymmetry around y axis", "Asymmetry around z axis"], 
                   ["Aerodynamic Rotor moment y component", "Aerodynamic Bearing force z component"],
                    ["Aerodynamic Rotor force z component", "Aerodynamic Bearing force z component"],
                 ["Aerodynamic Rotor moment z component", "Aerodynamic Bearing force y component"],
                  ["Aerodynamic Rotor force y component", "Aerodynamic Bearing force y component"]]
h_vars_arr = [[Ux, RtAeroFxh],[Ux, RtAeroMxh], [Ux, RtAeroMR], [IA, RtAeroMR], [IA, Aero_FBR],
                 [Aero_FBy, Aero_FBz], [RtAeroMys, RtAeroMzs], [Iy, RtAeroMys], [Iz, RtAeroMzs],
                 [Iy, Aero_FBz], [Iz, Aero_FBy], [Iy, Iz], [RtAeroMys, Aero_FBz], [RtAeroFzs, Aero_FBz],
                 [RtAeroMzs, Aero_FBy], [RtAeroFys, Aero_FBy]]

cutoff = 0.3

with PdfPages(out_dir+'precursor_plots_{}.pdf'.format(cutoff)) as pdf:
    for i in np.arange(0,len(Variables_arr)):
        Variables = Variables_arr[i]
        Ylabels = Ylabels_arr[i]
        units = units_arr[i]
        h_vars = h_vars_arr[i]

        signal_LP_0 = low_pass_filter(h_vars[0], cutoff)
        signal_LP_1 = low_pass_filter(h_vars[1], cutoff)

        fig,ax = plt.subplots(figsize=(14,8))
        
        corr = correlation_coef(signal_LP_0,signal_LP_1)
        corr = round(corr,2)

        ax.plot(Time_OF,signal_LP_0,'-b')
        ax.set_ylabel("{0} {1}".format(Ylabels[0],units[0]),fontsize=14)

        ax2=ax.twinx()
        ax2.plot(Time_OF,signal_LP_1,"-r")
        ax2.set_ylabel("{0} {1}".format(Ylabels[1],units[1]),fontsize=14)

        plt.title("Low passs filtered at {0}Hz.\nCorrelation = {1}".format(cutoff,corr),fontsize=16)
        ax.set_xlabel("Time [s]",fontsize=16)
        plt.tight_layout()
        pdf.savefig()
        plt.close()