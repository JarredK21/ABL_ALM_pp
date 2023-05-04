from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob 
from scipy import interpolate


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

    y = p_h.ijk_dims[0] #no. data points
    z = p_h.ijk_dims[1] #no. data points

    rotor_coordinates = np.array([2560,2560,90])
    ly = 400
    Oy = 2560 - ly/2

    Oz = p_h.origin[2]
    lz = p_h.axis2[2]

    ys = np.linspace(Oy,Oy+ly,y) - rotor_coordinates[1]
    zs = np.linspace(Oz,Oz+lz,z) - rotor_coordinates[2]

    #create R,theta space over rotor
    R = np.linspace(0,63,500)
    Theta = np.arange(0,2*np.pi,(2*np.pi)/729)


    avg_rotor_field_offset = []
    for j in np.arange(0,no_offsets):

        velocityx = offset_data(p_h, no_cells_offset,j,velocity_comp="velocityx",it=0)
        velocityy = offset_data(p_h, no_cells_offset,j,velocity_comp="velocityy",it=0)

        hvelmag = np.add( np.multiply(velocityx,np.cos(np.radians(29))) , np.multiply( velocityy,np.sin(np.radians(29))) )
        hvelmag = hvelmag.reshape((z,y))

        f = interpolate.interp2d(ys,zs,hvelmag,kind="linear")

        Ux_rotor = []
        for r in R:
            for theta in Theta:

                Y = r*np.cos(theta)
                Z = r*np.sin(theta)

                Ux =  f(Y,Z)

                Ux_rotor.append(Ux)
        
        avg_rotor_field_offset.append(np.average(Ux_rotor))

    avg_rotor_field.append(avg_rotor_field_offset)
avg_rotor_field =  np.array(avg_rotor_field)
for i in np.arange(0,no_offsets):
    vel = avg_rotor_field[:,i]
    fig = plt.figure()
    plt.plot(cases,vel,"-ok")
    plt.xlabel("case")
    plt.ylabel("Average axial velocity at {}m offset to rotor plane".format(p_h.offsets[i]))
plt.show()


print(avg_rotor_field)