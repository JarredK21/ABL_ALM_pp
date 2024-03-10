from netCDF4 import Dataset
import numpy as np


def offset_data(i, no_cells_offset,var):

    print("shape {}".format(var)); print(i), print(no_cells_offset)
    if var == "coordinates":
        u_slice = np.array(p.variables["coordinates"][(i*no_cells_offset):((i+1)*no_cells_offset)])
    else:
        u_slice = np.array(p.variables["{}".format(var)][0,(i*no_cells_offset):((i+1)*no_cells_offset)])
    print("shape u_slice",np.shape(u_slice))

    return u_slice

planes = ["r"]
plane_labels = ["rotor"]
groups = ["group_r"]

ip = 0
for plane in planes:
    offsets = [-63.0]
    no_cells_offset = 1228800
    
    io = 0
    for offset in offsets:

        ncfile = Dataset("./sampling_{0}_{1}_0.nc".format(plane,offset),mode="w",format='NETCDF4') #change name

        ncfile.title = "AMR-Wind data sampling output {0} at {1}".format(plane_labels[ip],offset)

        #create global dimensions
        time_dim = ncfile.createDimension("num_time_steps",None)
        dim_dim = ncfile.createDimension("ndims",3)


        #open files to be combined
        a = Dataset("./sampling70000.nc")

        #open group
        p = a.groups["p_{0}".format(plane)]


        ijk_dims = np.array(p.ijk_dims)
        origin = np.array(p.origin)
        axis1 = np.array(p.axis1)
        axis2 = np.array(p.axis2)
        axis3 = np.array(p.axis3)

        group = ncfile.createGroup("p_{}".format(plane))


        points_dim = group.createDimension("num_points",None)

        group.sampling_type = "PlaneSampler"
        group.ijk_dims = ijk_dims; del ijk_dims
        group.origin = origin; del origin
        group.axis1 = axis1; del axis1
        group.axis2 = axis2; del axis2
        group.axis3 = axis3; del axis3
        group.offsets = offset; del offset
        print("line 92")


        coordinates = group.createVariable("coordinates",np.float64,("num_points","ndims"),zlib=True)
        velocityx = group.createVariable("velocityx",np.float64,("num_time_steps","num_points"),zlib=True)
        velocityy = group.createVariable("velocityy",np.float64,("num_time_steps","num_points"),zlib=True)

        coord = offset_data(io, no_cells_offset,var="coordinates"); print("shape coord", np.shape(coord))
        coordinates[:] = coord; del coord
        print("line 103")

        velx = offset_data(io,no_cells_offset,var="velocityx"); print("shape velx", np.shape(velx))
        velocityx[:] = velx; del velx
        print("line 109")
        vely = offset_data(io,no_cells_offset,var="velocityy"); print("shape vely", np.shape(vely))
        velocityy[:] = vely; del vely
        print("line 113")

        print(ncfile)
        print(ncfile.groups)
        ncfile.close()

        io+=1
    ip+=1