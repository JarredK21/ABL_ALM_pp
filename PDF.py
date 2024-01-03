from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import netCDF4 as nc
from scipy.fft import fft2, fftfreq, fftshift
import numpy as np
from multiprocessing import Pool
import math
import pandas as pd


def butterwort_low_pass_filer(f):

    M,N = f.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = 1/30 #cut off frequency

    delx = 10
    freqs = fftshift(fftfreq(x, d=delx)) #mirror frequencies are equal and opposite sign in the Re, Im are zero.
    freq2d = np.array(np.meshgrid( freqs, freqs)) #mirror frequencies are equal in the Re, Im are zero
    freqs2d = np.sqrt( freq2d[0]**2 + freq2d[1]**2)

    for u in range(M):
        for v in range(N):
            D = freqs2d[u,v]
            if D >= D0:
                H[u,v] = 0
            else:
                H[u,v] = 1

    return H


def Horizontal_velocity(it):
    mag_horz_vel = u[it]*np.cos(np.radians(29)) + v[it]*np.sin(np.radians(29))
    return mag_horz_vel


def two_dim_LPF(it):

    U = u[it].reshape(x,y)

    #FFT
    ufft = np.fft.fftshift(np.fft.fft2(U))

    ##multiply filter
    H = butterwort_low_pass_filer(U)
    ufft_filt = ufft * H

    #IFFT
    ufft_filt_shift = np.fft.ifftshift(ufft_filt)
    iufft_filt = np.abs(np.fft.ifft2(ufft_filt_shift))

    return iufft_filt


def unfiltered_contour(it):

    #plot example contour w/o filtering
                    
    u_plane = u[it].reshape(x,y)
    X,Y = np.meshgrid(xs,ys)

    Z = u_plane

    T = Time[0]

    fig = plt.figure(figsize=(50,37.5))
    plt.rcParams['font.size'] = 40

    cs = plt.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    plt.xlabel("X axis [m]",fontsize=50)
    plt.ylabel("Y axis [m]",fontsize=50)

    cb = plt.colorbar(cs)

    if velocity_field_u == True:
        Title = "Horizontal Plane at 90m. \n Horizontal velocity [m/s], Time = {0}[s]".format(round(T,4),fontsize=50)
        filename = "Horizontal_plane_90m_Horizontal_velocity_{0}.png".format(round(T,4))
    elif velocity_field_w == True:
        Title = "Horizontal Plane at 90m. \n Vertical velocity [m/s], Time = {0}[s]".format(round(T,4),fontsize=50)
        filename = "Horizontal_plane_90m_Vertical_velocity_{0}.png".format(round(T,4))

    plt.title(Title)
    plt.savefig(out_dir+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)


def filtered_contour():

    #plot example contour w filtering
    U = iufft #velocity time step it

    levels = np.linspace(cmin,cmax,nlevs,dtype=int)
                    
    u_plane = U.reshape(x,y)
    X,Y = np.meshgrid(xs,ys)

    Z = u_plane

    T = Time[0]

    fig = plt.figure(figsize=(50,37.5))
    plt.rcParams['font.size'] = 40

    cs = plt.contourf(X,Y,Z,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)

    plt.xlabel("X axis [m]",fontsize=50)
    plt.ylabel("Y axis [m]",fontsize=50)

    cb = plt.colorbar(cs)

    if velocity_field_u == True:
        Title = "Horizontal Plane at 90m. \n Horizontal velocity [m/s], Filtered at k = 7, Time = {0}[s]".format(round(T,4),fontsize=50)
        filename = "Horizontal_plane_90m_Filtered_Horizontal_velocity_{0}.png".format(round(T,4))
    elif velocity_field_w == True:
        Title = "Horizontal Plane at 90m. \n Vertical velocity [m/s], Filtered at k = 7, Time = {0}[s]".format(round(T,4),fontsize=50)
        filename = "Horizontal_plane_90m_Filtered_Vertical_velocity_{0}.png".format(round(T,4))

    plt.title(Title)
    plt.savefig(out_dir+filename)
    plt.cla()
    cb.remove()
    plt.close(fig)


def probability_dist(it):
    
    if filtered_data == True:
        y = data[str(Time[it])]
    else:
        y = u[it]

    mu = np.mean(y)
    var = np.var(y)
    no_bin = 1000
    X = np.linspace(data_min,data_max,no_bin)
    dx = X[1]-X[0]
    P = []
    p = 0
    i = 0
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
        p+=(num/denom)*dx
        i+=1
    print(p)
    return P,X


in_dir = "../../ABL_precursor_2/"
out_dir = in_dir+"plots/"


a = Dataset(in_dir+"sampling_l_85.nc")

p = a.groups["p_l"]

#time options
Time = np.array(a.variables["time"])
tstart = 32700
tstart_idx = np.searchsorted(Time,tstart)
tend = 33700
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

#velocity field
velocity_field_w = True
velocity_field_u = True


if velocity_field_u == True:
    u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
    v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

    with Pool() as pool:
        u_hvel = []
        for u_hvel_it in pool.imap(Horizontal_velocity,Time_steps):
            
            u_hvel.append(u_hvel_it)
            print(len(u_hvel))
    u = np.array(u_hvel); del u_hvel; del v; del a

    cmin = math.floor(np.min(u))
    cmax = math.ceil(np.max(u))
                        
    nlevs = (cmax-cmin)
    levels = np.linspace(cmin,cmax,nlevs,dtype=int)

    unfiltered_contour(it=0)

    iufft = two_dim_LPF(it=0)

    cmin = math.floor(np.min(iufft))
    cmax = math.ceil(np.max(iufft))
                        
    nlevs = (cmax-cmin)
    levels = np.linspace(cmin,cmax,nlevs,dtype=int)

    filtered_contour()


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
    data = pd.read_csv(in_dir+'LPF_data_uu.csv')

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

    plt.rcParams['font.size'] = 12

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
    plt.xlabel("Horizontal velocity [m/s]",fontsize=16)
    plt.ylabel("Probability filtered [-]",fontsize=16)
    plt.title("CDF averaged over final 1000s",fontsize=18)
    plt.tight_layout()
    plt.savefig(out_dir+"CDF_Horizontal_velocity.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(X,PDF_uu_mean,"b-")
    plt.plot(X_unfilt,PDF_unfilt_mean,"r--")
    plt.axvline(X[LSS_idx],color="k",linestyle="--"); plt.axvline(X[HSS_idx],color="k",linestyle="--")
    plt.axvline(np.mean(u),color="k",linestyle="--")
    plt.xlabel("Horizontal velocity [m/s]",fontsize=16)
    plt.ylabel("Probability [-]",fontsize=16)
    plt.title("PDF averaged over final 1000s",fontsize=18)
    plt.legend(["filtered", "unfiltered"],fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_Horizontal_velocity.png")
    plt.close()


if velocity_field_w == True:
    u = np.array(p.variables["velocityz"][tstart_idx:tend_idx])

    cmin = math.floor(np.min(u))
    cmax = math.ceil(np.max(u))
                        
    nlevs = (cmax-cmin)
    levels = np.linspace(cmin,cmax,nlevs,dtype=int)

    unfiltered_contour(it=0)

    iufft = two_dim_LPF(it=0)

    cmin = math.floor(np.min(iufft))
    cmax = math.ceil(np.max(iufft))
                        
    nlevs = (cmax-cmin)
    levels = np.linspace(cmin,cmax,nlevs,dtype=int)

    filtered_contour()


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
    data = pd.read_csv(in_dir+'LPF_data_ww.csv')

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

    plt.rcParams['font.size'] = 12

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
    plt.xlabel("Vertical velocity [m/s]",fontsize=16)
    plt.ylabel("Probability filtered [-]",fontsize=16)
    plt.title("CDF averaged over final 1000s",fontsize=18)
    plt.tight_layout()
    plt.savefig(out_dir+"CDF_Vertical_velocity.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.plot(X,PDF_ww_mean,"b-")
    plt.plot(X_unfilt,PDF_unfilt_mean,"r--")
    plt.axvline(X[LSS_idx],color="k",linestyle="--"); plt.axvline(X[HSS_idx],color="k",linestyle="--")
    plt.axvline(np.mean(u),color="k",linestyle="--")
    plt.xlabel("Vertical velocity [m/s]",fontsize=16)
    plt.ylabel("Probability [-]",fontsize=16)
    plt.title("PDF averaged over final 1000s",fontsize=18)
    plt.legend(["filtered", "unfiltered"],fontsize=12)
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_Vertical_velocity.png")
    plt.close()