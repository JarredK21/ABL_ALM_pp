from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob 
import os
from matplotlib import cm
from matplotlib.animation import PillowWriter
import operator
import math


init_path = "../../../jarred/ALM_sensitivity_analysis/"

cases = ["Ex1","Ex1_dblade_1.0","Ex1_dblade_2.0"]


def offset_data(p_h,no_cells_offset,i,velocity_comp,it):

    if velocity_comp =="coordinates":
        u = np.array(p_h.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_h.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice


avg_rotor_field = []
for case in cases:

    case_path = init_path + case
    
    sampling = glob.glob("{0}/post_processing/sampling*".format(case_path))
    a = Dataset("./{}".format(sampling[0]))
    p_h = a.groups["p_sw1"]

    no_cells = len(p_h.variables["coordinates"])
    no_offsets = len(p_h.offsets)
    no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

    coordinates = offset_data(p_h,no_cells_offset,i=0,velocity_comp="coordinates",it=0)

    rotor_coordinates = np.array([2560,2560,90])

    coordinates = np.subtract(np.array(coordinates),rotor_coordinates)

    ys = coordinates[:,1]
    zs = coordinates[:,2]

    arc = np.sqrt( np.square(zs) + np.square(ys) )

    avg_rotor_field_offset = []
    for j in np.arange(0,no_offsets):

        velocityx = offset_data(p_h, no_cells_offset,j,velocity_comp="velocityx",it=0)
        velocityy = offset_data(p_h, no_cells_offset,j,velocity_comp="velocityy",it=0)

        hvelmag = np.sqrt( np.square( np.array(velocityx) ) + np.square( np.array(velocityy) ) )

        rotor_field = []
        for i in np.arange(0,len(arc)):
            if arc[i] <= 63.0:
                rotor_field.append(hvelmag[i])
        
        avg_rotor_field_offset.append(np.average(rotor_field))

    avg_rotor_field.append(avg_rotor_field_offset)
avg_rotor_field =  np.array(avg_rotor_field)
for i in np.arange(0,no_offsets):
    vel = avg_rotor_field[:,i]
    fig = plt.figure()
    plt.plot(cases,vel,"-ok")
    plt.xlabel("case")
    plt.ylabel("Average axial velocity at {}m offset to rotor plane".format(p_h.offsets[i]))
    plt.show()