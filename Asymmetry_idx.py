import matplotlib.pyplot as plt
import numpy as np
import glob 
from netCDF4 import Dataset
from scipy import interpolate

init_path = "../../../jarred/ALM_sensitivity_analysis/"
case = "Ex1"

sampling = glob.glob("{0}/post_processing/sampling*".format(init_path+case))
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]

time = a.variables["time"]
no_cells = len(p_rotor.variables["coordinates"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

y = p_rotor.ijk_dims[0] #no. data points
z = p_rotor.ijk_dims[1] #no. data points


def offset_data(p_h,no_cells_offset,i,velocity_comp,it):

    if velocity_comp =="coordinates":
        u = np.array(p_h.variables[velocity_comp]) #only time step
    else:
        u = np.array(p_h.variables[velocity_comp][it]) #only time step

    u_slice = u[i*no_cells_offset:((i+1)*no_cells_offset)]

    return u_slice

rotor_coordinates = [2560,2560,90]


ly = 400
Oy = 2560 - ly/2

Oz = p_rotor.origin[2]
lz = p_rotor.axis2[2]

ys = np.linspace(Oy,Oy+ly,y) - rotor_coordinates[1]
zs = np.linspace(Oz,Oz+lz,z) - rotor_coordinates[2]


velocityx = offset_data(p_rotor,no_cells_offset,i=0,velocity_comp="velocityx",it=0)
velocityy = offset_data(p_rotor,no_cells_offset,i=0,velocity_comp="velocityx",it=0)


hvelmag = np.add( np.multiply(velocityx,np.cos(np.radians(29))) , np.multiply( velocityy,np.sin(np.radians(29))) )

hvelmag = hvelmag.reshape((z,y))

f = interpolate.interp2d(ys,zs,hvelmag,kind="linear")

#create R,theta space over rotor
R = np.linspace(0,63,500)
Theta = np.arange(0,2*np.pi,(2*np.pi)/729)

dR = R[1]-R[0]
dTheta = Theta[1] - Theta[0]
dA = (dTheta/2)*(dR**2)

IA = 0
ir = 0
for r in R:
    itheta = 0
    for theta in Theta:

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

        IA += r * delta_Ux * dA

        itheta+=1
    ir+=1

print(IA)