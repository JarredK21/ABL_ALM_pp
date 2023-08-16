from netCDF4 import Dataset
import numpy as np


def offset_data(x, i, no_cells_offset):

    print("shape x",np.shape(x)); print(i), print(no_cells_offset)
    u_slice = x[:,0:1228800]
    print("shape u_slice",np.shape(u_slice))

    return u_slice

planes = ["r", "t"]
plane_labels = ["rotor", "transverse"]
groups = ["group_r", "group_t"]

ip = 0
for plane in planes:
    if plane == "l":
        offsets = [85]
        no_cells_offset = 262144
    elif plane == "r":
        offsets = [0.0, -63.0, -126, 126]
        no_cells_offset = 1228800
    elif plane == "t":
        offsets = [1280, 1930, 3190, 3820]
        no_cells_offset = 65536
    
    io = 0
    for offset in offsets:

        ncfile = Dataset("./sampling_{0}_{1}.nc".format(plane,offset),mode="w",format='NETCDF4') #change name

        ncfile.title = "AMR-Wind data sampling output {0} at {1}".format(plane_labels[ip],offset)

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
        Time = np.concatenate((Time_a[0:restart_idx],Time_b))
        time[:] = Time; del Time; del Time_a; del Time_b
        print("line 52")


        #open group
        p_a = a.groups["p_{0}".format(plane)]
        p_b = b.groups["p_{0}".format(plane)]


        ijk_dims = np.array(p_a.ijk_dims)
        origin = np.array(p_a.origin)
        axis1 = np.array(p_a.axis1)
        axis2 = np.array(p_a.axis2)
        axis3 = np.array(p_a.axis3)

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
        group.offsets = offset; del offset
        print("line 92")


        coordinates = group.createVariable("coordinates",np.float64,("num_points","ndims"),zlib=True)
        velocityx = group.createVariable("velocityx",np.float64,("num_time_steps","num_points"),zlib=True)
        velocityy = group.createVariable("velocityy",np.float64,("num_time_steps","num_points"),zlib=True)
        velocityz = group.createVariable("velocityz",np.float64,("num_time_steps","num_points"),zlib=True)

        coord = np.array(p_a.variables["coordinates"])
        coord = offset_data(coord, io, no_cells_offset)
        coordinates[:] = coord; del coord
        print("line 103")

        velx = np.concatenate((np.array(p_a.variables["velocityx"][0:restart_idx]), np.array(p_b.variables["velocityx"])))
        velx = offset_data(velx,io,no_cells_offset); print("shape velx", np.shape(velx))
        velocityx[:] = velx; del velx
        print("line 109")
        vely = np.concatenate((np.array(p_a.variables["velocityy"][0:restart_idx]), np.array(p_b.variables["velocityy"])))
        vely = offset_data(vely, io,no_cells_offset)
        velocityy[:] = vely; del vely
        print("line 113")
        velz = np.concatenate((np.array(p_a.variables["velocityz"][0:restart_idx]), np.array(p_b.variables["velocityz"])))
        velz = offset_data(velz,io,no_cells_offset)
        velocityz[:] = velz; del velz
        print("line 117")

        print(ncfile)
        print(ncfile.groups)
        ncfile.close()

        io+=1
    ip+=1