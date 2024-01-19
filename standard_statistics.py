import numpy as np
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def correlation_coef_LPF(x,y):
    
    x = low_pass_filter(x,0.3); y = low_pass_filter(y,0.3)

    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def correlation_coef_trend(x,y):
    
    x_tilde = low_pass_filter(x,0.3); y_tilde = low_pass_filter(y,0.3)

    r = (np.sum(((np.subtract(x,x_tilde))*(np.subtract(y,y_tilde)))))/(np.sqrt(np.sum(np.square(np.subtract(x,x_tilde)))*np.sum(np.square(np.subtract(y,y_tilde)))))

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


def standard_statistics(phi,Variable):
    phi_tilde = low_pass_filter(phi,0.3)
    mu = np.mean(phi)
    mu_tilde = np.mean(phi_tilde)
    phi_pri = phi - mu
    phi_pri_tilde = np.subtract(phi,phi_tilde)
    var = np.average(np.square(phi_pri))
    var_tilde = np.average(np.square(phi_pri_tilde))
    var_phi_tilde = np.average(np.square((phi_tilde-mu_tilde)))
    I = np.sqrt(var)/mu
    I_tilde = np.sqrt(var_tilde)/mu_tilde

    df_stats.loc["{}".format(Variable)] = [mu,var,var_tilde,var_phi_tilde,I,I_tilde]


def write_dataframe_to_xlsx():

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(out_dir+"statistics.xlsx") as writer:

        # Write the dataframe data to XlsxWriter. Turn off the default header and
        # index and skip one row to allow us to insert a user defined header.
        df_stats.to_excel(writer, sheet_name="Statistics")
        df_corr.to_excel(writer, sheet_name="Correlations")
        df_corr_LPF.to_excel(writer, sheet_name="Low pass filter correlations")
        df_corr_trend.to_excel(writer, sheet_name="Trend correlations")

    # Close the Pandas Excel writer and output the Excel file.
    writer.close()

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

out_dir = in_dir

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
Theta_Aero_FB = np.degrees(np.arctan2(Aero_FBz,Aero_FBy))

offset = "5.5"
group = a.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Uz = np.array(group.variables["Uz"])
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])


statistics_catelog = ["Mean","Variance", "Trend variance", "Variance of the trend", "Intensity", "Trend intensity"]
Variables = ["RtAeroFx","RtAeroFy","RtAeroFz","RtAeroMx","RtAeroMy","RtAeroMz","RtAeroMR","AeroFBy","AeroFBz","AeroFBR","AeroTheta",
             "RtElastFy","RtElastFz","RtElastMy","RtElastMz","RtElastMR","ElastFBy","ElastFBz","ElastFBR","ElastTheta",
             "Ux","Uz","IA","Iy","Iz"]
PHI = [RtAeroFxh/1000,RtAeroFys/1000,RtAeroFzs/1000,RtAeroMxh/1000,RtAeroMys/1000,RtAeroMzs/1000,RtAeroMR/1000,
       Aero_FBy/1000,Aero_FBz/1000,Aero_FBR/1000,Theta_Aero_FB,
             LSShftFys,LSShftFzs,LSSTipMys,LSSTipMzs,LSSTipMR,FBy,FBz,FBR,Theta_FB,Ux,Uz,IA,Iy,Iz]
# Create a Pandas dataframe from some data.
df_stats = pd.DataFrame(columns=statistics_catelog, index=Variables)
df_corr = pd.DataFrame(columns=Variables, index=Variables)
df_corr_LPF = pd.DataFrame(columns=Variables, index=Variables)
df_corr_trend = pd.DataFrame(columns=Variables, index=Variables)

for Variable,phi in zip(Variables,PHI):
    standard_statistics(phi,Variable)


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

PHI = [RtAeroFxh/1000,RtAeroFys/1000,RtAeroFzs/1000,RtAeroMxh/1000,RtAeroMys/1000,RtAeroMzs/1000,RtAeroMR/1000,
       Aero_FBy/1000,Aero_FBz/1000,Aero_FBR/1000,Theta_Aero_FB,
             LSShftFys,LSShftFzs,LSSTipMys,LSSTipMzs,LSSTipMR,FBy,FBz,FBR,Theta_FB,Ux,Uz,IA,Iy,Iz]

for Variable_x,x in zip(Variables,PHI):
    for Variable_y,y in zip(Variables,PHI):
        corr = correlation_coef(x,y)
        df_corr.loc[Variable_x,Variable_y] = corr

for Variable_x,x in zip(Variables,PHI):
    for Variable_y,y in zip(Variables,PHI):
        corr = correlation_coef_LPF(x,y)
        df_corr_LPF.loc[Variable_x,Variable_y] = corr

for Variable_x,x in zip(Variables,PHI):
    for Variable_y,y in zip(Variables,PHI):
        corr = correlation_coef_trend(x,y)
        df_corr_trend.loc[Variable_x,Variable_y] = corr


write_dataframe_to_xlsx()