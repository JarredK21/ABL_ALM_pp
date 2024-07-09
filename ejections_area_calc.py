from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
import time
from multiprocessing import Pool

def correlation_coef(x,y):

    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def isInside(x, y):
     
    if ((x - 2560) * (x - 2560) +
        (y - 90) * (y - 90) < 63 * 63):
        return True
    else:
        return False
    

def Area_calc(it):

    H = Heights[it]
    A = 0
    h = []
    for j in np.arange(0,len(ys)):
        #is coordinate inside rotor disk
        cc = isInside(ys[j],H[j])

        if cc == True:

            z = np.min( np.roots([1,-180,(90**2-63**2+(ys[j]-2560)**2)]) )
            h.append(H[j]-z) #height from coordinate zs to coordinate z on rotor disk

        #is coordinate above rotor disk so it is still covering it
        elif ys[j] > 2497 and ys[j] < 2623 and H[j] > 153:
            z = np.roots([1,-180,(90**2-63**2+(ys[j]-2560)**2)])
            h.append(z[0]-z[1]) #height

        if len(h) > 1 and isInside(ys[j+1],H[j+1]) == False:

            #integrate over sub area covering rotor disk
            for i in np.arange(0,len(h)-1):
                A+=((h[i+1] + h[i])/2)*dy

            h = [] #reset h array


    return A
    

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Threshold_Asymmetry_Dataset.nc")

Time_a = np.array(a.variables["Time"])


b = Dataset(in_dir+"Threshold_heights_Dataset.nc")

Time_B = np.array(b.variables["Time"])
Time_steps = np.arange(0,len(Time_B))
ys = np.array(b.variables["ys"])
dy = ys[1]-ys[0]

Thresholds = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.4]

out_dir=in_dir+"Asymmetry_analysis/Threshold_Asymmetry/"

for threshold in Thresholds:

    group_a = a.groups["{}".format(threshold)]

    print(group_a)

    Iy = np.array(group_a.variables["Iy_ejection"])
    Iz = -np.array(group_a.variables["Iz_ejection"])
    I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

    plt.rcParams.update({'font.size': 18})
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(21,12),sharex=True)
    ax1.plot(Time_a,I,"-b")
    ax1.set_title("Magnitude Asymmetry vector [$m^4/s$]")
    ax1.grid()
    fig.supxlabel("Time [s]")


    group_b = b.groups["{}".format(threshold)]

    Heights = np.array(group_b.variables["Height_ejection"])

    Area_array = []
    ix = 1
    with Pool() as pool:
        for A_it in pool.imap(Area_calc,Time_steps):

            Area_array.append(A_it)

            print(ix)

            ix+=1

    Area_array = np.array(Area_array)

    ax2.plot(Time_B,Area_array,"-k")
    ax2.set_title("Area of {}m/s surges inside rotor disk [$m^2$]".format(threshold))
    ax2.grid()

    cc = round(correlation_coef(I,Area_array),2)
    fig.suptitle("correlation coefficient = {}".format(cc))
    plt.tight_layout()
    plt.savefig(out_dir+"I_{}_threshold_areas.png".format(threshold))
    plt.close()


