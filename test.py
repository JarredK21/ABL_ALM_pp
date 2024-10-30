from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import netCDF4 as nc
from scipy.fft import fft2, fftfreq, fftshift
import numpy as np
from multiprocessing import Pool
from math import floor, ceil
import pandas as pd


def Horizontal_velocity(it):
    mag_horz_vel = u[it]*np.cos(np.radians(29)) + v[it]*np.sin(np.radians(29))
    return mag_horz_vel



def probability_dist(it):
    
    if filtered_data == True:
        y = data[str(Time[it])]
    else:
        y = u[it]

    std = np.std(y)
    bin_width = std/20
    x = np.arange(np.min(y),np.max(y)+bin_width,bin_width)
    dx = x[1]-x[0]
    P = []
    X = []
    for i in np.arange(0,len(x)-1):
        p = 0
        for yi in y:
            if yi >= x[i] and yi <= x[i+1]:
                p+=1
        P.append(p/(dx*len(y)))
        X.append((x[i+1]+x[i])/2)

    print(np.sum(P)*dx)

    return P,X



a = Dataset("sampling_l_85.nc")

p = a.groups["p_l"]

#time options
Time = np.array(a.variables["time"])
tstart = 38000
tstart_idx = np.searchsorted(Time,tstart)
tend = 39200
tend_idx = np.searchsorted(Time,tend)
Time_steps = np.arange(0, tend_idx-tstart_idx)
Time = Time[tstart_idx:tend_idx]

col_names = []
for it in Time_steps:
    col_names.append(str(Time[it]))

PDF_data_uu =  pd.DataFrame(data=None, columns=col_names)
PDF_data_ww = pd.DataFrame(data=None, columns=col_names)


x = p.ijk_dims[0] #no. data points
y = p.ijk_dims[1] #no. data points


#define plotting axes
coordinates = np.array(p.variables["coordinates"])

xs = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x)
ys = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y)
zs = 0


velocities = ["u", "w"]
for velocity in velocities:


    if velocity == "u":
        u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
        v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])
        u = np.subtract(u,np.mean(u))
        v = np.subtract(v,np.mean(v))

        with Pool() as pool:
            u_hvel = []
            for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
                
                u_hvel.append(u_hvel_it)
                print(len(u_hvel))
        u = np.array(u_hvel); del u_hvel; del v; del a


        #PDF of unfiltered data
        data_max = np.max(u)
        data_min = np.min(u)

        filtered_data = False
        ix = 0
        unfilted_data = []
        with Pool() as pool:
            for P,X in pool.imap(probability_dist, Time_steps):
                unfilted_data.append(P)
                ix+=1
                print(ix)

        X_unfilt = X
        PDF_unfilt_mean = np.mean(unfilted_data,axis=0)


        #PDF of filtered data
        data = pd.read_csv('LPF_data_uu.csv')

        data_max = data.to_numpy().max()
        data_min = data.to_numpy().min()

        filtered_data = True
        ix = 0
        with Pool() as pool:
            for P,X in pool.imap(probability_dist, Time_steps):
                PDF_data_uu["{}".format(Time[ix])] = P
                ix+=1
                print(ix)


        PDF_uu_mean = PDF_data_uu.mean(axis=1)
        PDF_data_uu["mean"] = PDF_uu_mean

        PDF_data_uu['X'] = X

        PDF_data_uu.to_csv('PDF_data_uu.csv',index=False)

        plt.rcParams['font.size'] = 12

        PDF_uu_mean = np.array(PDF_uu_mean)

        CDF_i = 0
        CDF = []
        dx = X[1]-X[0]
        for f in PDF_uu_mean:
            CDF_i+=f*dx
            CDF.append(CDF_i)

        LSS_idx = np.searchsorted(CDF,0.3); HSS_idx = np.searchsorted(CDF,0.7)
        print("LSS = ", X[LSS_idx], "HSS = ", X[HSS_idx], "mean = ", np.mean(u))

        fig = plt.figure(figsize=(14,8))
        plt.plot(X,CDF)
        plt.xlabel("Fluctuating Horizontal velocity [m/s]",fontsize=16)
        plt.ylabel("Probability filtered [-]",fontsize=16)
        plt.xticks(np.arange(floor(np.min(X)),ceil(np.max(X)),1))
        plt.title("CDF averaged over final 1000s",fontsize=18)
        plt.tight_layout()
        plt.savefig("CDF_Horizontal_velocity.png")
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(X,PDF_uu_mean,"b-")
        plt.plot(X_unfilt,PDF_unfilt_mean,"r--")
        plt.xticks(np.arange(floor(np.min(X)),ceil(np.max(X)),1))
        plt.axvline(X[LSS_idx],color="k",linestyle="--"); plt.axvline(X[HSS_idx],color="k",linestyle="--")
        plt.axvline(np.mean(u),color="k",linestyle="--")
        plt.xlabel("Fluctuating Horizontal velocity [m/s]",fontsize=16)
        plt.ylabel("Probability [-]",fontsize=16)
        plt.title("PDF averaged over final 1000s",fontsize=18)
        plt.legend(["filtered", "unfiltered"],fontsize=12)
        plt.tight_layout()
        plt.savefig("PDF_Horizontal_velocity.png")
        plt.close()


    if velocity == "w":
        u = np.array(p.variables["velocityz"][tstart_idx:tend_idx])


        #PDF of unfiltered data
        data_max = np.max(u)
        data_min = np.min(u)

        filtered_data = False
        ix = 0
        unfilted_data = []
        with Pool() as pool:
            for P,X in pool.imap(probability_dist, Time_steps):
                unfilted_data.append(P)
                ix+=1
                print(ix)

        X_unfilt = X
        PDF_unfilt_mean = np.mean(unfilted_data,axis=0)


        #PDF of filtered data
        data = pd.read_csv('LPF_data_ww.csv')

        data_max = data.to_numpy().max()
        data_min = data.to_numpy().min()

        filtered_data = True
        ix = 0
        with Pool() as pool:
            for P,X in pool.imap(probability_dist, Time_steps):
                PDF_data_ww["{}".format(Time[ix])] = P
                ix+=1
                print(ix)


        PDF_ww_mean = PDF_data_ww.mean(axis=1)
        PDF_data_ww["mean"] = PDF_ww_mean

        PDF_data_ww['X'] = X

        PDF_data_ww.to_csv('PDF_data_ww.csv',index=False)

        plt.rcParams['font.size'] = 12

        PDF_ww_mean = np.array(PDF_ww_mean)

        CDF_i = 0
        CDF = []
        dx = X[1]-X[0]
        for f in PDF_ww_mean:
            CDF_i+=f*dx
            CDF.append(CDF_i)

        LSS_idx = np.searchsorted(CDF,0.3); HSS_idx = np.searchsorted(CDF,0.7)
        print("DD = ", X[LSS_idx], "UD = ", X[HSS_idx], "mean = ", np.mean(u))

        fig = plt.figure(figsize=(14,8))
        plt.plot(X,CDF)
        plt.xlabel("Fluctuating Vertical velocity [m/s]",fontsize=16)
        plt.ylabel("Probability filtered [-]",fontsize=16)
        plt.xticks(np.arange(floor(np.min(X)),ceil(np.max(X)),1))
        plt.title("CDF averaged over final 1000s",fontsize=18)
        plt.tight_layout()
        plt.savefig("CDF_Vertical_velocity.png")
        plt.close()

        fig = plt.figure(figsize=(14,8))
        plt.plot(X,PDF_ww_mean,"b-")
        plt.plot(X_unfilt,PDF_unfilt_mean,"r--")
        plt.axvline(X[LSS_idx],color="k",linestyle="--"); plt.axvline(X[HSS_idx],color="k",linestyle="--")
        plt.axvline(np.mean(u),color="k",linestyle="--")
        plt.xlabel("Fluctuating Vertical velocity [m/s]",fontsize=16)
        plt.ylabel("Probability [-]",fontsize=16)
        plt.xticks(np.arange(floor(np.min(X)),ceil(np.max(X)),1))
        plt.title("PDF averaged over final 1000s",fontsize=18)
        plt.legend(["filtered", "unfiltered"],fontsize=12)
        plt.tight_layout()
        plt.savefig("PDF_Vertical_velocity.png")
        plt.close()