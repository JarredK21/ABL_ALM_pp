import numpy as np
import pyFAST.input_output as io
import matplotlib.pyplot as plt
from netCDF4 import Dataset

def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(y)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt





in_dir="../../NREL_5MW_3.4.1/Steady_Rigid_blades_shear_0.2/"

df = io.fast_output_file.FASTOutputFile(in_dir+"NREL_5MW_Main.out").toDataFrame()

Time_OF = np.array(df["Time_[s]"])
dt = Time_OF[1] - Time_OF[0]

Time_start = 200
Time_start_idx = np.searchsorted(Time_OF,Time_start)

Time_OF = Time_OF[Time_start_idx:]


Azimuth = np.radians(np.array(df["Azimuth_[deg]"][Time_start_idx:]))

RtAeroFyh = np.array(df["RtAeroFyh_[N]"][Time_start_idx:])
RtAeroFzh = np.array(df["RtAeroFzh_[N]"][Time_start_idx:])

RtAeroFys = []; RtAeroFzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
    RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
RtAeroFys = np.array(RtAeroFys)/1000; RtAeroFzs = np.array(RtAeroFzs)/1000


RtAeroMyh = np.array(df["RtAeroMyh_[N-m]"][Time_start_idx:])
RtAeroMzh = np.array(df["RtAeroMzh_[N-m]"][Time_start_idx:])

RtAeroMys = []; RtAeroMzs = []
for i in np.arange(0,len(Time_OF)):
    RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
    RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
RtAeroMys = np.array(RtAeroMys)/1000; RtAeroMzs = np.array(RtAeroMzs)/1000


#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))



dFBR = np.array(dt_calc(FBR,dt))
zero_crossings_index = np.where(np.diff(np.sign(dFBR)))[0]

dF = []
dt = []
for i in np.arange(0,len(zero_crossings_index)-1):
    idx1 = zero_crossings_index[i]
    idx2 = zero_crossings_index[i+1]

    dF.append(abs(FBR[idx2]-FBR[idx1]))
    dt.append(Time_OF[idx2] - Time_OF[idx1])

print(np.average(dF))
print(np.average(dt))


