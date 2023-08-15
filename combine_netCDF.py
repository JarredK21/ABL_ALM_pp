from netCDF4 import Dataset
import numpy as np

ncfile = Dataset("./sampling.nc",mode="w",format='NETCDF4') #change name

ncfile.title = "AMR-Wind data sampling output combined"

#create global dimensions
time_dim = ncfile.createDimension("num_time_steps",None)
dim_dim = ncfile.createDimension("ndims",3)

#create time variable
time = ncfile.createVariable("time", np.float64, ('num_time_steps',))

#open files to be combined
a = Dataset("./sampling65000.nc") #check
b = Dataset("./sampling100320.nc") #check

#determine restart index
Time_a = np.array(a.variables["time"]); Time_b = np.array(b.variables["time"])
restart_time = Time_b[0]
restart_idx = np.searchsorted(Time_a, restart_time); restart_idx-=1


#combine time
Time = np.concatenate((np.array(Time_a[0:restart_idx]),np.array(Time_b)))
time[:] = Time; del Time; del Time_a; del Time_b
print("line 28")


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
    print("line 63")


    coordinates = group.createVariable("coordinates",np.float64,("num_points","ndims"),zlib=True)
    velocityx = group.createVariable("velocityx",np.float64,("num_time_steps","num_points"),zlib=True)
    velocityy = group.createVariable("velocityy",np.float64,("num_time_steps","num_points"),zlib=True)
    velocityz = group.createVariable("velocityz",np.float64,("num_time_steps","num_points"),zlib=True)

    coord = np.array(p_a.variables["coordinates"]); coordinates[:] = coord; del coord
    print("line 72")

    velx = np.concatenate((np.array(p_a.variables["velocityx"][0:restart_idx]), np.array(p_b.variables["velocityx"])))
    velocityx[:] = velx; del velx
    print("line 76")
    vely = np.concatenate((np.array(p_a.variables["velocityy"][0:restart_idx]), np.array(p_b.variables["velocityy"])))
    velocityy[:] = vely; del vely
    print("line 79")
    velz = np.concatenate((np.array(p_a.variables["velocityz"][0:restart_idx]), np.array(p_b.variables["velocityz"])))
    velocityz[:] = velz; del velz
    print("line 82")

print(ncfile)
print(ncfile.groups)
ncfile.close()