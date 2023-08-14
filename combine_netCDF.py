from netCDF4 import Dataset
import numpy as np

ncfile = Dataset("../../NREL_5MW_MCBL_R_CRPM/test/new.nc",mode="w",format='NETCDF4') #change name

ncfile.title = "AMR-Wind data sampling output combined"

#create global dimensions
time_dim = ncfile.createDimension("num_time_steps",None)
dim_dim = ncfile.createDimension("ndims",3)

#create time variable
time = ncfile.createVariable("time", np.float64, ('num_time_steps',))

#open files to be combined
restart_idx = 100320
a = Dataset("./sampling65000.nc") #check
b = Dataset("./sampling65000.nc") #check

#combine time
Time_a = np.array(a.variables["time"][0:restart_idx-1]); Time_b = np.array(b.variables["time"][restart_idx:]); Time = np.concatenate((Time_a,Time_b))
time[:] = Time; del Time; del Time_a; del Time_b


planes = ["l", "r", "t"]
groups = ["group_l", "group_r", "group_t"]

for plane in planes:
    #open group
    p_a = a.groups["p_{0}".format(plane)]
    p_b = b.groups["p_{0}".format(plane)]

    ijk_dims = np.array(p_a.ijk_dims)
    origin = np.array(p_a.origin)
    axis1 = np.array(p_a.axis1)
    axis2 = np.array(p_a.axis2)
    axis3 = np.array(p_a.axis3)
    offsets = np.array(p_a.offsets)

    if plane == "l":
        group = ncfile.createGroup("p_l")
    elif plane == "r":
        group = ncfile.createGroup("p_r")
    elif plane == "t":
        group = ncfile.createGroup("p_t")


    points_dim = group.createDimension("num_points",None)

    group.sampling_type = "PlaneSampler"
    group.ijk_dims = ijk_dims; del ijk_dims
    group.origin = origin; del origin
    group.axis1 = axis1; del axis1
    group.axis2 = axis2; del axis2
    group.axis3 = axis3; del axis3
    group.offsets = offsets; del offsets


    coordinates = group.createVariable("coordinates",np.float64,("num_points","ndims"),zlib=True)
    velocityx = group.createVariable("velocityx",np.float64,("num_time_steps","num_points"),zlib=True)
    velocityy = group.createVariable("velocityy",np.float64,("num_time_steps","num_points"),zlib=True)
    velocityz = group.createVariable("velocityz",np.float64,("num_time_steps","num_points"),zlib=True)

    coord = np.array(p_a.variables["coordinates"]); coordinates[:] = coord; del coord

    velx_a = np.array(p_a.variables["velocityx"][0:restart_idx-1]); velx_b = np.array(p_b.variables["velocityx"][restart_idx:]); velx = np.concatenate((velx_a,velx_b))
    velocityx[:] = velx; del velx; del velx_a; del velx_b
    vely_a = np.array(p_a.variables["velocityy"][0:restart_idx-1]); vely_b = np.array(p_b.variables["velocityy"][restart_idx:]); vely = np.concatenate((vely_a,vely_b))
    velocityy[:] = vely; del vely; del vely_a; del vely_b
    velz_a = np.array(p_a.variables["velocityz"][0:restart_idx-1]); velz_b = np.array(p_b.variables["velocityz"][restart_idx:]); velz = np.concatenate((velz_a,velz_b))
    velocityz[:] = velz; del velz; del velz_a; del velz_b

print(ncfile)
print(ncfile.groups)
ncfile.close()