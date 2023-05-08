from netCDF4 import Dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob 
from scipy import interpolate


# init_path = "../../../jarred/ALM_sensitivity_analysis/"

# cases = ["Ex1","Ex1_dblade_1.0","Ex1_dblade_2.0"]

dir = "./post_processing/plots/"

cases = [1]

sampling = glob.glob("./post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]

time = p_rotor.variables["time"]
tstart = 50
tend = 350
tstart_idx = np.searchsorted(time,tstart)
tend_idx = np.searchsorted(time,tend)

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


for i in np.arange(0,no_offsets):
    avg_rotor_it = []
    for it in np.arange(tstart_idx,tend_idx):

        velocityx = offset_data(p_rotor, no_cells_offset,i,it,velocity_comp="velocityx")
        velocityy = offset_data(p_rotor, no_cells_offset,i,it,velocity_comp="velocityy")

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
        
        avg_rotor_it.append(np.average(Ux_rotor))


    fig = plt.figure(figsize=(14,8))
    plt.plot(time[tstart_idx:tend_idx], avg_rotor_it)
    plt.xlabel('time [s]',fontsize=16)
    plt.ylabel("Ux' - Rotor normal Velocity [m/s]",fontsize=16)
    plt.title('Rotor Normal velocity averaged over rotor plane: {0}m'.format(p_rotor.offsets[i]))
    plt.grid()
    plt.tight_layout()
    plt.savefig(dir+"rotor_velocity_{0}.png".format(p_rotor.offsets[i]))
    plt.close(fig)