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


def theta_360(Theta):
    Theta_360 = []
    for theta in Theta:
        if theta < 0:
            Theta_360.append(theta+360)
        else:
            Theta_360.append(theta)
    return Theta_360


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z


Start_time = time.time()

#in_dir = "./"
in_dir = "../../NREL_5MW_MCBL_R_CRPM/post_processing/"

out_dir = in_dir + "polar_plots/"

video_folder = in_dir + "polar_videos/"
isExist = os.path.exists(video_folder)
if isExist == False:
    os.makedirs(video_folder)

a = Dataset(in_dir+"OF_Dataset.nc")

Times = [100,400,700,1000,1300,1600,1990]

for ic in np.arange(0,len(Times)-1):

    Time_OF = np.array(a.variables["time_OF"])

    Time_start = Times[ic]
    Time_end = Times[ic+1]

    Time_start_idx = np.searchsorted(Time_OF,Time_start)
    Time_end_idx = np.searchsorted(Time_OF,Time_end)

    time_steps = np.arange(0,Time_end_idx-Time_start_idx,25)

    Time_OF = Time_OF[Time_start_idx:Time_end_idx]

    Azimuth = np.array(a.variables["Azimuth"][Time_start_idx:Time_end_idx])
    Azimuth = np.radians(Azimuth)

    RtAeroFyh = np.array(a.variables["RtAeroFyh"][Time_start_idx:Time_end_idx])
    RtAeroFzh = np.array(a.variables["RtAeroFzh"][Time_start_idx:Time_end_idx])

    RtAeroFys = []; RtAeroFzs = []
    for i in np.arange(0,len(Time_OF)):
        RtAeroFys_i, RtAeroFzs_i = tranform_fixed_frame(RtAeroFyh[i],RtAeroFzh[i],Azimuth[i])
        RtAeroFys.append(RtAeroFys_i); RtAeroFzs.append(RtAeroFzs_i)
    RtAeroFys = np.array(RtAeroFys); RtAeroFzs = np.array(RtAeroFzs)

    RtAeroFR = np.sqrt( np.add( np.square(RtAeroFys), np.square(RtAeroFzs) ) )

    RtAeroMyh = np.array(a.variables["RtAeroMyh"][Time_start_idx:Time_end_idx])
    RtAeroMzh = np.array(a.variables["RtAeroMzh"][Time_start_idx:Time_end_idx])

    RtAeroMys = []; RtAeroMzs = []
    for i in np.arange(0,len(Time_OF)):
        RtAeroMys_i, RtAeroMzs_i = tranform_fixed_frame(RtAeroMyh[i],RtAeroMzh[i],Azimuth[i])
        RtAeroMys.append(RtAeroMys_i); RtAeroMzs.append(RtAeroMzs_i)
    RtAeroMys = np.array(RtAeroMys); RtAeroMzs = np.array(RtAeroMzs)

    RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 

    LSSGagMys = np.array(a.variables["LSSGagMys"][Time_start_idx:Time_end_idx])
    LSSGagMzs = np.array(a.variables["LSSGagMzs"][Time_start_idx:Time_end_idx])
    LSSGagMR = np.sqrt( np.add(np.square(LSSGagMys), np.square(LSSGagMzs)) )

    LSSTipMys = np.array(a.variables["LSSTipMys"][Time_start_idx:Time_end_idx])
    LSSTipMzs = np.array(a.variables["LSSTipMzs"][Time_start_idx:Time_end_idx])
    LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

    LSShftFys = np.array(a.variables["LSShftFys"][Time_start_idx:Time_end_idx])
    LSShftFzs = np.array(a.variables["LSShftFzs"][Time_start_idx:Time_end_idx])
    LSShftFR = np.sqrt( np.add(np.square(LSShftFys), np.square(LSShftFzs)) )

    Theta_AeroF = np.degrees(np.arctan2(RtAeroFzs,RtAeroFys))
    Theta_AeroF = theta_360(Theta_AeroF)
    Theta_AeroF = np.radians(np.array(Theta_AeroF))
    Theta_AeroM = np.degrees(np.arctan2(RtAeroMzs,RtAeroMys))
    Theta_AeroM = theta_360(Theta_AeroM)
    Theta_AeroM = np.radians(np.array(Theta_AeroM))
    Theta_LSSTipF = np.degrees(np.arctan2(LSShftFzs,LSShftFys))
    Theta_LSSTipF = theta_360(Theta_LSSTipF)
    Theta_LSSTipF = np.radians(np.array(Theta_LSSTipF))
    Theta_LSSTipM = np.degrees(np.arctan2(LSSTipMzs,LSSTipMys))
    Theta_LSSTipM = theta_360(Theta_LSSTipM)
    Theta_LSSTipM = np.radians(np.array(Theta_LSSTipM))

    print("line 106", time.time()-Start_time)

    def Update(it):

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(projection='polar')
        c = ax.scatter(x_var[it], y_var[it], c="k", s=20)
        ax.arrow(0, 0, x_var[it], y_var[it], length_includes_head=True)
        print(x_var[it])
        ax.set_ylim(0,np.max(y_var))
        ax.set_title("{} {}\nTime = {}s".format(Ylabels[j],units[j],Time_OF[it]), va='bottom')
        T = Time_OF[it]
        filename = "{}.png".format(round(T,4))
        plt.savefig(folder+filename)
        plt.close(fig)

        return T



    # Variables = ["AeroF", "AeroM", "LSSTipF", "LSSTipM"]
    # units = ["[kN]","[kN-m]","[kN]","[kN-m]"]
    # Ylabels = ["Rotor Aerodynamic Force", "Rotor Aerodynamic Moment", "Rotor Aeroelastic Force", "Rotor Aeroelastic Moment"]
    # x_vars = [Theta_AeroF, Theta_AeroM, Theta_LSSTipF, Theta_LSSTipM]
    # y_vars = [RtAeroFR/1000, RtAeroMR/1000, LSShftFR, LSSTipMR]
    Variables = ["LSSTipF"]
    units = ["[kN]"]
    Ylabels = ["Rotor Aeroelastic Force"]
    x_vars = [Theta_LSSTipF]
    y_vars = [LSShftFR]
    for j in np.arange(0,len(x_vars)):

        x_var = x_vars[j]; y_var = y_vars[j]

        folder = out_dir+"{}_{}/".format(Variables[j],Times[ic])

        isExist = os.path.exists(folder)
        if isExist == False:
            os.makedirs(folder)

            with Pool() as pool:
                for T in pool.imap(Update,time_steps):

                    print(T,time.time()-Start_time)

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
        
        print("line 163", time.time()-Start_time)
            
        #sort files
        files = glob.glob(folder+"*.png")
        files.sort(key=natural_keys)

        print("line 173", time.time()-Start_time)

        #write to video
        img_array = []
        it_img = 0
        for file in files:
            img = cv2.imread(file)
            height, width, layers = img.shape
            size = (width,height)
            img_array.append(img)
            print("line 177)", Time_OF[time_steps[it_img]],time.time()-Start_time)
            it_img+=1
        
        #cv2.VideoWriter_fourcc(*'DIVX')
        filename = "{}_{}".format(Variables[j], Times[ic])
        out = cv2.VideoWriter(video_folder+filename+'.avi',0, 20, size)
        it_vid = 0
        for im in range(len(img_array)):
            out.write(img_array[im])
            print("Line 190)",Time_OF[time_steps[it_vid]],time.time()-Start_time)
            it_vid+=1
        out.release(); del img_array

        print(Variables[j],Times[ic], time.time()-Start_time)