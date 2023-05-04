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
import pyFAST.input_output as io


path = "../../../jarred/ALM_sensitivity_analysis/Ex1_dblade_1.0/post_processing/WTG01.nc"

a = Dataset(path)
WT = a.groups["WTG01"]

xyz_vel = np.array(WT.variables["vel_xyz"][-1])
xyz_act = np.array(WT.variables["xyz"][-2])

x_vel = xyz_vel[:,0]
y_vel = xyz_vel[:,1]
z_vel = xyz_vel[:,2]


x_act = xyz_act[:,0]
y_act = xyz_act[:,1]
z_act = xyz_act[:,2]

print(x_vel[:-1]-x_act)
print(y_vel[:-1]-y_act)
print(z_vel[:-1]-z_act)


fig = plt.figure()
ax = fig.add_subplot(projection='3d')


ax.scatter(x_vel, y_vel, z_vel, marker="o")
ax.scatter(x_act, y_act, z_act, marker="D")

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()

# #from pyFAST.input_output import FASTOutputFile
# #import pyFAST.input_output as io
# import pyFAST.input_output as io
# import matplotlib.pyplot as plt
# import numpy as np
# import os
# import pyFAST.postpro as postpro
# from scipy.fft import fft, fftfreq, fftshift
# import pandas as pd
# from scipy import interpolate
# import math


# dir = "../../../jarred/ALM_sensitivity_analysis/joint_plots/dt_study/"


# # cases = ["Ex1","Ex2","Ex3"]
# # act_stations_cases = [54,47,59]
# # dt_cases = [0.001,0.001,0.001]

# # colors = ["red","blue","green"]
# # markers = ["o","D","s"]
# # trans = [1,0.5,0.25]

# cases = ["Ex1","Ex1_dblade_2.0","Ex1_dblade_1.0"]
# act_stations_cases = [54,54,54]
# dt_cases = [0.001,0.0078,0.0039]
# tstart = [24,24,24]

# colors = ["red","blue","green"]
# markers = ["o","D","s"]
# trans = [1,0.5,0.25]

# # legends = []
# # for i in np.arange(0,len(act_stations_cases)):
# #     legends.append("{0}: {1} actuator points".format(cases[i],act_stations_cases[i]))

# legends = []
# for i in np.arange(0,len(dt_cases)):
#     legends.append("{0}: {1}s dt".format(cases[i],dt_cases[i]))


# rad_variables = ["Vrel","Alpha", "Cl","Cd","Fn","Ft","Vx"]
# rad_YLabel = ["Local Relative Velocity", "Local Angle of Attack", "Local Coeffcient of Lift", "Local Coefficient of Drag",
#                 "Local Aerofoil Normal Force", "Local Aerofoil Tangential Force", "Local Axial Velocity"]
# rad_units = ["[m/s]","[deg]","[-]","[-]","[N/m]","[N/m]","[m/s]"]
# number_rotor_rotations = 3



# time_start = [10,10,10] #time in seconds to remove from start of data - insert 0 if plot all time
# time_end = [24,24,24] #time in seconds to plot upto - insert False if plot all time
# int_variables = ["Wind1VelX","RotTorq","RtAeroFxh","RtAeroFyh","RtAeroFzh","RtAeroMxh","RtAeroMyh","RtAeroMzh"]
# int_YLabel = ["Hub height Velocity", "Rotor Torque", "Rotor Force in X direction", "Rotor Force in Y direction", 
#                 "Rotor Force in Z direction", "Rotor Moment in X direction", "Rotor Moment in Y direction", 
#                 "Rotor Moment in Z direction"]
# int_units = ["[m/s]","[kN-m]","[N]","[N]","[N]","[N-m]","[N-m]","[N-m]"]


# #plotting options
# plot_ints = True
# plot_spectra = True
# plot_radial = True



# def Root_mean_squared(data_set):

    
#     RMSE = []
#     for l in np.arange(1,len(cases)):

#         MSE = np.square(np.subtract(data_set[0],data_set[l])).mean() 
 
#         RMSE.append( math.sqrt(MSE)/np.average(data_set[0]) )

#     return RMSE



# def difference(data_set):

#     norm_diff = []
#     mean_diff = []
#     perc_diff = []
#     for l in np.arange(1,len(cases)):

#         diff = np.subtract(data_set[l],data_set[0])

#         perc_diff_i = np.true_divide(abs(diff),data_set[0])

#         perc_diff_i = [x for x in perc_diff_i if math.isnan(x) == False]

#         perc_diff.append( np.average(perc_diff_i) * 100 )
 
#         norm_diff.append( diff/np.average(data_set[0]) )

#         mean_diff.append( np.sum(abs(norm_diff[l-1]))/len(norm_diff[l-1]) )

#     return norm_diff, mean_diff, perc_diff





# if plot_radial == True:
#     #radial plots
#     for i in np.arange(0,len(rad_variables),1):

#         # legends = []
#         # for j in np.arange(0,len(act_stations_cases)):
#         #     legends.append("{0}: {1} actuator points".format(cases[j],act_stations_cases[j]))
#         legends = []
#         for j in np.arange(0,len(dt_cases)):
#             legends.append("{0}: {1}s dt".format(cases[j],dt_cases[j]))

#         Var = rad_variables[i]
#         unit = rad_units[i]
#         YLabel = rad_YLabel[i]
#         no_rots = number_rotor_rotations

#         data_set =  pd.DataFrame(data=None, columns=cases)

#         fig = plt.figure()

#         ix = 0 #case counter
#         for case in cases:

#             df = io.fast_output_file.FASTOutputFile("../../../jarred/ALM_sensitivity_analysis/{0}/post_processing/NREL_5MW_Main.out".format(case)).toDataFrame()

#             # for col in df.columns:
#             #     print(col)

#             time = df["Time_[s]"]
#             time = np.array(time)
            
#             Az = np.array(df["Azimuth_[deg]"])

#             act_stations = act_stations_cases[ix]
#             x = np.linspace(0,1,act_stations)
#             x_max = np.linspace(0,1,act_stations_cases[0])

#             Var_list = []
#             for i in np.arange(1,act_stations+1):
#                 if i < 10:
#                     txt = "AB1N00{0}{1}_{2}".format(i,Var,unit)
#                 elif i >= 10:
#                     txt = "AB1N0{0}{1}_{2}".format(i,Var,unit)

#                 tstart_idx = np.searchsorted(time,tstart[ix])
#                 Var_dist = df[txt][tstart_idx]
                
#                 Var_list.append(Var_dist)

#             data_set[ix] = interpolate.interp1d(x, Var_list,kind="linear")(x_max)            

#             plt.plot(x,Var_list,color=colors[ix],marker=markers[ix],markersize=4,alpha=trans[ix])

#             ix+=1


#         RMSE = Root_mean_squared(data_set)

#         norm_diff, mean_diff, percent_diff = difference(data_set)

#         for k in np.arange(1,len(cases)):
#             legends[k] = legends[k] + "\nRMS = {0} \nAverage Percentage difference = {1}%".format(round(RMSE[k-1],6),round(percent_diff[k-1],6))
#         plt.ylabel("{0} {1}".format(YLabel,unit),fontsize=16)
#         plt.xlabel("Normalized blade radius [-]",fontsize=16)
#         plt.legend(legends)
#         plt.title("plotted at 25s, 54 actuator points".format(no_rots),fontsize=12)
#         plt.tight_layout()
#         plt.savefig(dir+"{0}_2.png".format(Var))
#         plt.close(fig)
