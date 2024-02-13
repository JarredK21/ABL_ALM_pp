import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy import interpolate
from multiprocessing import Pool
from netCDF4 import Dataset


def isInside(x, y):
     
    if ((x - 2) * (x - 2) +
        (y - 2) * (y - 2) <= 1 * 1):
        return True
    else:
        return False


def Update(it):

    y = 5
    x = 5

    ys = np.linspace(0,4,y)
    zs = np.linspace(0,4,x)
    X,Y = np.meshgrid(ys,zs)

    u_plane = u[it]

    f = interpolate.interp2d(ys,zs,u_plane)

    y = 1000
    x = 1000

    ys = np.linspace(0,4,y)
    zs = np.linspace(0,4,x)
    X,Y = np.meshgrid(ys,zs)

    u_plane = f(ys,zs)

    T = Time[it]

    fig,ax = plt.subplots(figsize=(50,30))
    plt.rcParams['font.size'] = 40
    cs = ax.contourf(X,Y,u_plane)
    Drawing_uncolored_circle = Circle( (2, 2),radius=1 ,fill = False, linewidth=0.5)
    ax.add_artist(Drawing_uncolored_circle)

    plt.colorbar(cs)

    for t in np.arange(0,len(thresholds)):

        storage = np.zeros(len(ys))
        for j in np.arange(0,len(ys)):
            for k in np.arange(0,len(zs)-1):
                
                if u_plane[j,k+1] > thresholds[t]:
                    
                    storage[j] = zs[k]

                    break

        plt.plot(ys,storage,linewidth=4,label="{}m/s".format(thresholds[t]))



    plt.xlabel("y axis [m]")
    plt.ylabel("z axis [m]")
    ax.legend(loc="upper right")
    plt.title("Time {}".format(it))
    plt.savefig(out_dir+"test_{}.png".format(it))
    plt.close(fig)

    return T
    

def Update_data(it):

    y = 5
    x = 5

    ys = np.linspace(0,4,y)
    zs = np.linspace(0,4,x)
    X,Y = np.meshgrid(ys,zs)

    u_plane = u[it]

    f = interpolate.interp2d(ys,zs,u_plane)

    y = 1000
    x = 1000

    ys = np.linspace(0,4,y)
    zs = np.linspace(0,4,x)
    X,Y = np.meshgrid(ys,zs)

    u_plane = f(ys,zs)


    h = []
    for j in np.arange(0,len(ys)):
        for k in np.arange(0,len(zs)-1):
            
            if u_plane[j,k+1] > thresholds[t]:

                break
        
        cc = isInside(ys[j],zs[k])
        if cc == True:
            z = np.min( np.roots([1,-4,(4-1+(ys[j]-2)**2)]) )
            h.append(zs[k]-z)
        elif ys[j] > 1 and ys[j] < 3 and zs[k] > 2:
            cc = True
            z = np.roots([1,-4,(4-1+(ys[j]-2)**2)])
            h.append(z[0]-z[1])

    A = 0
    delta_y = ys[1] - ys[0]
    for i in np.arange(0,len(h)-1):
        A+=((h[i+1] + h[i])/2)*delta_y

    if A == 0:
        prop = 0.0
    else:
        prop = A/(np.pi*1**2)

    return it,prop


out_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/test/"

Time = np.array([0,1,2,3,4,5,6,7,8,9])
u = np.array( [ [[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]],
[[-4,-2,0,1,1],[-5,-4,-1,0,1],[-6,-4,-2,-0.7,1],[-6,-5,-2,-2,1],[-9,-5,-3,-0.7,1]] ] )

Iy = np.array([200000,300000,400000,500000,100000,200000,300000,-100000,-200000,300000])
Iz = np.array([-200000,-300000,-400000,-500000,-100000,-200000,-300000,100000,200000,-300000])



thresholds = np.arange(-12.0,-0.0,2)
thresholds = np.append(thresholds,-0.7)


with Pool() as pool:
    for T in pool.imap(Update,Time):
        print(T)
            


#create netcdf file
ncfile = Dataset(out_dir+"Thresholding_Dataset.nc",mode="w",format='NETCDF4')
ncfile.title = "Threshold data sampling output"

#create global dimensions
sampling_dim = ncfile.createDimension("sampling",None)

time = ncfile.createVariable("Time", np.float64, ('sampling',),zlib=True)
time[:] = Time

threshold_label = np.arange(12.0,0.0,-2)
threshold_label = np.append(threshold_label,0.7)


for t in np.arange(0,len(thresholds)):

    group = ncfile.createGroup("{}".format(threshold_label[t]))

    Iy_data = group.createVariable("Iy", np.float64, ('sampling'),zlib=True)
    Iz_data = group.createVariable("Iz", np.float64, ('sampling'),zlib=True)
    P_data = group.createVariable("P", np.float64, ('sampling'),zlib=True)

    Iy_it = []
    Iz_it = []
    P_it = []

    ic = 0
    with Pool() as pool:
        for it,P_i in pool.imap(Update_data,Time):

            if P_i != 0:
                Iy_it.append(Iy[it])
                Iz_it.append(Iz[it])
                P_it.append(P_i)
            else:
                Iy_it.append(np.nan)
                Iz_it.append(np.nan)
                P_it.append(P_i)

            
            print(ic)
            ic+=1

        Iy_data[:] = np.array(Iy_it); del Iy_it
        Iz_data[:] = np.array(Iz_it); del Iz_it 
        P_data[:] = np.array(P_it); del P_it

    print(ncfile.groups)


print(ncfile)
ncfile.close()

