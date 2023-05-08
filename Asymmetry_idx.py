import matplotlib.pyplot as plt
import numpy as np
import glob 
from netCDF4 import Dataset
from scipy import interpolate
from multiprocessing import Pool


dir = "./post_processing/plots/"

sampling = glob.glob("./post_processing/sampling*")
a = Dataset("./{}".format(sampling[0]))
p_rotor = a.groups["p_sw1"]

time = a.variables["time"]
time = time - time[0]
no_cells = len(p_rotor.variables["coordinates"])
no_offsets = len(p_rotor.offsets)
no_cells_offset = int(no_cells/no_offsets) #Number of points per offset

tstart = 50
tend = 350
tstart_idx = np.searchsorted(time,tstart)
tend_idx = np.searchsorted(time,tend)

Title = "54 Actuator points, no. levels of refinement = 5, dt = 0.0039s"

y = p_rotor.ijk_dims[0] #no. data points
z = p_rotor.ijk_dims[1] #no. data points


def offset_data(p_h,no_cells_offset,i,it,velocity_comp):

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


for i in np.arange(0,no_offsets,1):
    IA_it = []
    for it in np.arange(tstart_idx,tend_idx):

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

    fig = plt.figure(figsize=(14,8))
    plt.plot(time[tstart_idx:tend_idx], IA_it)
    plt.xlabel('time [s]',fontsize=16)
    plt.ylabel('IA - Asymmetry Parameter [$m^4/s$]',fontsize=16)
    plt.title('Asymmetry Parameter: {0}m, {1}'.format(p_rotor.offsets[i],Title))
    plt.grid()
    plt.tight_layout()
    plt.savefig(dir+"Asymmetry_parameter_{0}.png".format(p_rotor.offsets[i]))
    plt.close(fig)