import pyFAST.input_output as io
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy import interpolate
from scipy.signal import butter,filtfilt
from scipy.stats import pearsonr
import glob 
from scipy import interpolate
from netCDF4 import Dataset
import pandas as pd
from multiprocessing import Pool


#openfast data
df = io.fast_output_file.FASTOutputFile("./NREL_5MW_3.4.1/Steady_Rigid_blades/NREL_5MW_Main.out").toDataFrame()

#sampling data
sampling = glob.glob("./post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]

offsets = p_rotor.offsets

Variables = ["Time_OF","Time_sample","Ux_{}".format(offsets[0]),"Ux_{}".format(offsets[1]),"Ux_{}".format(offsets[2]),
             "IA_{}".format(offsets[0]),"IA_{}".format(offsets[1]),"IA_{}".format(offsets[2]),
             "RtAeroFxh","RtAeroMxh","MR","Theta"]
units = ["[s]","[s]", "[m/s]", "[m/s]", "[m/s]","[$m^4/s$]","[$m^4/s$]","[$m^4/s$]","[N]","[N-m]","[N-m]","[rads]"]


dq = pd.DataFrame(data=None,columns=Variables)
print(dq)

time_OF = np.array(df["Time_[s]"])
time_sample = np.array(a.variables["time"])
time_sample = time_sample - time_sample[0]

tstart = 50
tend = 60
tstart_OF_idx = np.searchsorted(time_OF,tstart)
tend_OF_idx = np.searchsorted(time_OF,tend)
tstart_sample_idx = np.searchsorted(time_sample,tstart)
tend_sample_idx = np.searchsorted(time_sample,tend)


dq["Time_OF"] = time_OF[tstart_OF_idx:tend_OF_idx]
dq["Time_sample"] = time_sample[tstart_sample_idx:tend_sample_idx]

no_cells = len(p_rotor.variables["coordinates"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

y = p_rotor.ijk_dims[0] #no. data points
z = p_rotor.ijk_dims[1] #no. data points

rotor_coordinates = np.array([2560,2560,90])
ly = 400
Oy = 2560 - ly/2

Oz = p_rotor.origin[2]
lz = p_rotor.axis2[2]

ys = np.linspace(Oy,Oy+ly,y) - rotor_coordinates[1]
zs = np.linspace(Oz,Oz+lz,z) - rotor_coordinates[2]

#create R,theta space over rotor
R = np.linspace(1.5,63,500)
Theta = np.arange(0,2*np.pi,(2*np.pi)/729)



def offset_data(p_rotor,no_cells_offset,i,it,velocity_comp):

    if velocity_comp =="coordinates":
        u = np.array(p_rotor.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_rotor.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice


def Ux(r,theta,f):
    Y = r*np.cos(theta)
    Z = r*np.sin(theta)

    Ux =  f(Y,Z)

    return Ux


def Ux_it_offset(i,it):
        
    velocityx = offset_data(p_rotor, no_cells_offset,i,it,velocity_comp="velocityx")
    velocityy = offset_data(p_rotor, no_cells_offset,i,it,velocity_comp="velocityy")

    hvelmag = np.add( np.multiply(velocityx,np.cos(np.radians(29))) , np.multiply( velocityy,np.sin(np.radians(29))) )
    hvelmag = hvelmag.reshape((z,y))

    f = interpolate.interp2d(ys,zs,hvelmag,kind="linear")

    items = [(r,theta,f) for r in R for theta in Theta]
    Ux_rotor = 0
    with Pool() as pool:
        for Ux_i in pool.starmap(Ux,items):              

            Ux_rotor+=Ux_i

    return Ux_rotor/len(Ux_rotor)



def asymmetry_parameter_it():
    for i in np.arange(0,no_offsets,1):
        IA_it = []
        for it in np.arange(tstart_sample_idx,tend_sample_idx):

            velocityx = offset_data(p_rotor,no_cells_offset,i,it,velocity_comp="velocityx")
            velocityy = offset_data(p_rotor,no_cells_offset,i,it,velocity_comp="velocityy")

            hvelmag = np.add( np.multiply(velocityx,np.cos(np.radians(29))) , np.multiply( velocityy,np.sin(np.radians(29))) )

            hvelmag = hvelmag.reshape((z,y))

            f = interpolate.interp2d(ys,zs,hvelmag,kind="linear")

            #create R,theta space over rotor
            R = np.linspace(1.5,63,500)
            Theta = np.arange(0,2*np.pi,(2*np.pi)/729)

            dR = R[1]-R[0]
            dTheta = Theta[1] - Theta[0]
            dA = (dTheta/2)*(dR**2)

            def asymmetry_parameter(r,theta):
                Y_0 = r*np.cos(theta)
                Z_0 = r*np.sin(theta)

                if theta + ((2*np.pi)/3) > (2*np.pi):
                    theta_1 = theta +(2*np.pi)/3 - (2*np.pi)
                else:
                    theta_1 = theta + (2*np.pi)/3

                Y_1 = r*np.cos(theta_1)
                Z_1 = r*np.sin(theta_1)


                if theta - ((2*np.pi)/3) < 0:
                    theta_2 = theta - ((2*np.pi)/3) + (2*np.pi)
                else:
                    theta_2 = theta - ((2*np.pi)/3)

                Y_2 = r*np.cos(theta_2)
                Z_2 = r*np.sin(theta_2)

                Ux_0 =  f(Y_0,Z_0)
                Ux_1 =  f(Y_1,Z_1)
                Ux_2 =  f(Y_2,Z_2)

                delta_Ux =  np.max( [abs( Ux_0 - Ux_1 ), abs( Ux_0 - Ux_2 )] )

                IA = r * delta_Ux * dA

                return IA


            items = [(r,theta) for r in R for theta in Theta]
            IA = 0
            with Pool() as pool:
                for IA_i in pool.starmap(asymmetry_parameter,items):              

                    IA += IA_i
            
            IA_it.append(IA)

        return IA_it


for iv in np.arange(2,len(Variables)):
    Variable = Variables[iv]
    if Variable[0:1] == "Ux":
        items2 = [(i,it) for i in np.arange(0,no_offsets) for it in np.arange(tstart_sample_idx,tend_sample_idx)]
        Ux_it = []
        with Pool() as pool:
            for Ux_j in pool.starmap(Ux_it_offset,items2): 
                Ux_it.append(Ux_j)

        if Variable == "Ux_{}".format(offsets[0]):
            signal = Ux_it[0:(1/3)*len(Ux_it)]
        elif Variable == "Ux_{}".format(offsets[1]):
            signal = Ux_it[(1/3)*len(Ux_it):(2/3)*len(Ux_it)]
        elif Variable == "Ux_{}".format(offsets[2]):
            signal = Ux_it[(2/3)*len(Ux_it):]

    elif Variable[0:1] == "IA":
        IA_it_offset = []
        IA_it_offset.append(asymmetry_parameter_it())

        if Variable == "IA_{}".format(offsets[0]):
            signal = IA_it_offset[0:(1/3)*len(IA_it_offset)]
        elif Variable == "IA_{}".format(offsets[1]):
            signal = IA_it_offset[(1/3)*len(Ux_it):(2/3)*len(IA_it_offset)]
        elif Variable == "IA_{}".format(offsets[2]):
            signal = IA_it_offset[(2/3)*len(IA_it_offset):]

    elif Variable == "MR" or Variable == "Theta":
        signaly = df["RtAeroMyh_[N-m]"][tstart_OF_idx:tend_OF_idx]
        signalz = df["RtAeroMzh_[N-m]"][tstart_OF_idx:tend_OF_idx]
        
        if Variable == "MR":
            signal = np.sqrt( np.square(signaly) + np.square(signalz) ) 
        elif Variable == "Theta": 
            signal = np.arctan(np.true_divide(signalz,signaly))   

    else:
        txt = "{0}_{1}".format(Variable,units[iv])
        signal = df[txt][tstart_OF_idx:tend_OF_idx]



    dq[Variable] = signal

print(dq[:])
dq.to_csv("./post_processing/out.csv")