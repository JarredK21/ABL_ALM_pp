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
import time


#sampling data
sampling = glob.glob("../../../jarred/ALM_sensitivity_analysis/test10/post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]

no_cells = len(p_rotor.variables["coordinates"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

def offset_data(p_rotor,no_cells_offset,it,i,velocity_comp):

    if velocity_comp == "coordinates":
        u = np.array(p_rotor.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_rotor.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice

coordinates = offset_data(p_rotor,no_cells_offset,it=0,i=2,velocity_comp="coordinates")

x = coordinates[:,0]
y = coordinates[:,1]
z = coordinates[:,2]

rotor_coordiates = [2560,2560,90]

x_trans = x - rotor_coordiates[0]
y_trans = y - rotor_coordiates[1]

phi = np.radians(-29)
x_pri = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
y_pri = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))