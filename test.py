from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import cm
import math
import time
from multiprocessing import Pool
from scipy import interpolate
from matplotlib.patches import Circle
from math import ceil

in_dir = "../../ABL_precursor_2_restart/"

#defining twist angles with height from precursor
precursor = Dataset(in_dir+"abl_statistics70000.nc")
Time_pre = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start = np.searchsorted(precursor.variables["time"],38200)
t_end = np.searchsorted(precursor.variables["time"],39200)
u = np.average(mean_profiles.variables["u"][t_start:t_end],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end],axis=0)
h = mean_profiles["h"][:]
f_u = interpolate.interp1d(h,u); f_v = interpolate.interp1d(h,v)
u_90 = f_u(90); v_90 = f_v(90)
print(u_90*np.cos(np.radians(29))+v_90*np.sin(np.radians(29)))

del precursor; del Time_pre; del mean_profiles; del t_start; del u; del v