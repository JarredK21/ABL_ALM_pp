import numpy as np
from netCDF4 import Dataset

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"
a = Dataset(in_dir+"sampling_r_-63.0_0.nc")

#rotor data
p = a.groups["p_r"]; del a

x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points

normal = 29

#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xo = coordinates[0:x,0]
yo = coordinates[0:x,1]

rotor_coordiates = [2560,2560,90]

x_trans = xo - rotor_coordiates[0]
y_trans = yo - rotor_coordiates[1]

phi = np.radians(-normal)
xs = np.subtract(x_trans*np.cos(phi), y_trans*np.sin(phi))
ys = np.add(y_trans*np.cos(phi), x_trans*np.sin(phi))
xs = xs + rotor_coordiates[0]
ys = ys + rotor_coordiates[1]

test = "end"