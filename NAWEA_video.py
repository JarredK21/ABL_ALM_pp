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
import sys
import time
from multiprocessing import Pool
import cv2
import re
import pyFAST.input_output as io

start_time = time.time()

#whether or not folder exists execute code
#sort files
def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

print("line 464", time.time()-start_time)


in_dir = "./"
out_dir = in_dir + "ISOplots/"
video_folder = in_dir + "NAWEA_videos/"

plane_label = "longitudinal"
velocity_comp = "Horizontal velocity"
offset = 85

folder = out_dir+"{0}_Plane_Total_{1}_{2}/".format(plane_label,velocity_comp,offset)

#define titles and filenames for movie
filename = "{0}_Tot_vel{1}_{2}.png".format(plane_label,velocity_comp,float(offset))

#sort files
files = glob.glob(folder+"*.png")
files.sort(key=natural_keys)

no_files = math.floor((len(files)/2))
#write to video
img_array = []
for file in files[no_files:(no_files+514)]:
    img = cv2.imread(file)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    print("line 475)",time.time()-start_time)

#cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter(video_folder+filename+'.avi',0, 12, size)
for im in range(len(img_array)):
    out.write(img_array[im])
    print("Line 482)",time.time()-start_time)
out.release(); del img_array
print("Line 485)",time.time()-start_time)