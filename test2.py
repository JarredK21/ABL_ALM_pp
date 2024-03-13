# from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

direction = "clockwise"
theta_180  = [-2.35, -1.0, -2.35]
theta_loc = [3.933, 5.28, 3.933]
theta_order = [3.933, 5.28, 3.933]
i = 0
theta_anti = 0.9
theta_clock = 6.0

if direction == "anticlockwise" and theta_loc[i+1] < theta_anti:
    thetaAB = np.linspace(theta_180[i+1],theta_loc[i+2],int(abs(theta_180[i+1]-theta_180[i+2])/5e-03))
            
elif direction == "clockwise" and theta_loc[i+1] > theta_clock:
    thetaAB = np.linspace(theta_180[i+1],theta_180[i+2],int(abs(theta_180[i+1]-theta_loc[i+2])/5e-03))

for j in np. arange(0,len(thetaAB)):
    if thetaAB[j] < 0:
        thetaAB[j]+=2*np.pi

theta_loc = [2.8560959215491546, 1.4511820527916064, 2.8560959215491546]
theta_order = [1.4511820527916064, 2.8560959215491546, 1.4511820527916064]
fig,ax = plt.subplots(figsize=(50,30))
plt.rcParams['font.size'] = 40
crossings = [1,2]

for i in np.arange(0,len(crossings),2):


    theta = theta_loc[i+1] #theta B
    Bidx = theta_order.index(theta)

    Drawing_uncolored_circle = Circle( (2560, 90),radius=63 ,fill = False, linewidth=1)
    ax.add_artist(Drawing_uncolored_circle)
    plt.xlim([2400,2700])
    plt.ylim([0,300])

    r = 63
    x = 2560 + r*np.cos(theta)
    y = 90 + r*np.sin(theta)
    plt.plot(x,y,"^b",markersize=6)

    if len(theta_loc) > 3:
        
        print(theta_loc[i+1],theta_loc[i])
        if theta_loc[i] < theta_loc[i+1]:

            theta_anti = (theta_loc[i+1]+theta_loc[i]+2*np.pi)/2
            theta_anti-=2*np.pi
        else:
            theta_anti = (theta_loc[i+1]+theta_loc[i])/2

        x = 2560 + r*np.cos(theta_anti)
        y = 90 + r*np.sin(theta_anti)
        plt.plot(x,y,"sr",markersize=6)

        if theta_loc[i] < theta_loc[i+1]:

            dtheta_anti = theta_anti+2*np.pi - theta
        else:
            dtheta_anti = theta_anti - theta

        print(dtheta_anti)

        if dtheta_anti > np.radians(15):
            theta_anti = theta + np.radians(15)

        print(theta_anti)
        
        print(theta_loc[i+1],theta_loc[i+2])

        theta_clock = (theta_loc[i+1]+theta_loc[i+2])/2

        x = 2560 + r*np.cos(theta_clock)
        y = 90 + r*np.sin(theta_clock)
        plt.plot(x,y,"sg",markersize=6)

        dtheta_clock = theta - theta_clock

        if dtheta_clock > np.radians(15):
            theta_clock = theta - np.radians(15)

        print(theta_clock)

    elif len(theta_loc) < 4:

        print(theta_loc[i+1],theta_loc[i])
        if theta_loc[i] < theta_loc[i+1]:

            theta_anti = (theta_loc[i+1]+theta_loc[i]+2*np.pi)/2
            theta_anti-=2*np.pi
        else:
            theta_anti = (theta_loc[i+1]+theta_loc[i])/2

        x = 2560 + r*np.cos(theta_anti)
        y = 90 + r*np.sin(theta_anti)
        plt.plot(x,y,"sr",markersize=6)

        if theta_loc[i] < theta_loc[i+1]:

            dtheta_anti = theta_anti+2*np.pi - theta
        else:
            dtheta_anti = theta_anti - theta

        print(dtheta_anti)

        if dtheta_anti > np.radians(15):
            theta_anti = theta + np.radians(15)

        print(theta_anti)

        theta_clock = theta - np.radians(15)

        print(theta_clock)



    x = 2560 + r*np.cos(theta_anti)
    y = 90 + r*np.sin(theta_anti)
    plt.plot(x,y,"or",markersize=6)

    x = 2560 + r*np.cos(theta_clock)
    y = 90 + r*np.sin(theta_clock)
    plt.plot(x,y,"og",markersize=6)
    print(theta_anti,theta_clock)
    plt.show()

theta_AB = np.linspace(np.pi,0,int(abs(theta_loc[i]-theta_loc[i+1])/5e-03))

# def offset_data(i, no_cells_offset,var):

#     print("shape {}".format(var)); print(i), print(no_cells_offset)
#     if var == "coordinates":
#         u_slice = np.array(p.variables["coordinates"][(i*no_cells_offset):((i+1)*no_cells_offset)])
#     else:
#         u_slice = np.array(p.variables["{}".format(var)][0,(i*no_cells_offset):((i+1)*no_cells_offset)])
#     print("shape u_slice",np.shape(u_slice))

#     return u_slice

# planes = ["r"]
# plane_labels = ["rotor"]
# groups = ["group_r"]

# ip = 0
# for plane in planes:
#     offsets = [-63.0]
#     no_cells_offset = 1228800
    
#     io = 0
#     for offset in offsets:

#         ncfile = Dataset("./sampling_{0}_{1}_0.nc".format(plane,offset),mode="w",format='NETCDF4') #change name

#         ncfile.title = "AMR-Wind data sampling output {0} at {1}".format(plane_labels[ip],offset)

#         #create global dimensions
#         time_dim = ncfile.createDimension("num_time_steps",None)
#         dim_dim = ncfile.createDimension("ndims",3)


#         #open files to be combined
#         a = Dataset("./sampling76000.nc")

#         #open group
#         p = a.groups["p_{0}".format(plane)]


#         ijk_dims = np.array(p.ijk_dims)
#         origin = np.array(p.origin)
#         axis1 = np.array(p.axis1)
#         axis2 = np.array(p.axis2)
#         axis3 = np.array(p.axis3)

#         group = ncfile.createGroup("p_{}".format(plane))


#         points_dim = group.createDimension("num_points",None)

#         group.sampling_type = "PlaneSampler"
#         group.ijk_dims = ijk_dims; del ijk_dims
#         group.origin = origin; del origin
#         group.axis1 = axis1; del axis1
#         group.axis2 = axis2; del axis2
#         group.axis3 = axis3; del axis3
#         group.offsets = offset; del offset
#         print("line 92")


#         coordinates = group.createVariable("coordinates",np.float64,("num_points","ndims"),zlib=True)
#         velocityx = group.createVariable("velocityx",np.float64,("num_points"),zlib=True)
#         velocityy = group.createVariable("velocityy",np.float64,("num_points"),zlib=True)

#         coord = offset_data(io, no_cells_offset,var="coordinates"); print("shape coord", np.shape(coord))
#         coordinates[:] = coord; del coord
#         print("line 103")

#         velx = offset_data(io,no_cells_offset,var="velocityx"); print("shape velx", np.shape(velx))
#         velocityx[:] = velx; del velx
#         print("line 109")
#         vely = offset_data(io,no_cells_offset,var="velocityy"); print("shape vely", np.shape(vely))
#         velocityy[:] = vely; del vely
#         print("line 113")

#         print(ncfile)
#         print(ncfile.groups)
#         ncfile.close()

#         io+=1
#     ip+=1