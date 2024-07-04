from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool


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

a = Dataset(in_dir+"Asymmetry_Dataset.nc")

Time = np.array(a.variables["time"])
t_start_idx = np.searchsorted(Time,200+Time[0])
Time = Time[t_start_idx:]
A_high = np.array(a.variables["Area_high"][t_start_idx:])
A_low = np.array(a.variables["Area_low"][t_start_idx:])
A_int = np.array(a.variables["Area_int"][t_start_idx:])

a = Dataset(in_dir+"Threshold_heights_Dataset.nc")

Time_B = np.array(a.variables["Time"])
Time_steps = np.arange(0,len(Time_B))
ys = np.array(a.variables["ys"])
dy = ys[1]-ys[0]

Thresholds = [5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.4]

out_dir=in_dir+"Asymmetry_analysis/"

for threshold in Thresholds:

    plt.rcParams.update({'font.size': 18})
    fig,(ax1,ax2) = plt.subplots(2,1,figsize=(21,12),sharex=True)
    ax1.plot(Time,A_low,"-b")
    ax1.set_title("Area low [$m^2$]")
    ax1.grid()
    fig.supxlabel("Time [s]")


    group = a.groups["{}".format(threshold)]

    Heights = np.array(group.variables["Height_ejection"])

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

    plt.tight_layout()
    plt.savefig(out_dir+"{}_threshold_areas.png".format(threshold))
    plt.close()


