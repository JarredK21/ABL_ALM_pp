from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq, fftshift
import numpy as np
from multiprocessing import Pool
import math
from scipy import interpolate
import time


def probability_dist(y):

    mu = np.mean(y)
    var = np.var(y)
    sd = np.std(y)
    no_bin = 1000
    X = np.linspace(np.min(y),np.max(y),no_bin)
    dX = X[1] - X[0]
    P = []
    for x in X:
        denom = np.sqrt(var*2*np.pi)
        num = np.exp(-((x-mu)**2)/(2*var))
        P.append(num/denom)
    print(np.sum(P)*dX)
    return P,X


def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist


def Horizontal_velocity(it):
    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    twist_h = f(height)
    ux_mean = f_ux(height)
    mag_horz_vel = np.array(u[it]*np.cos(twist_h) + v[it]*np.sin(twist_h))
    mag_fluc_horz_vel = np.array(np.subtract(mag_horz_vel,ux_mean))
    
    return mag_horz_vel,mag_fluc_horz_vel


def butterwort_low_pass_filer(f):

    M,N = f.shape
    H = np.zeros((M,N), dtype=np.float32)
    D0 = cutoff #cut off frequency

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


def two_dim_LPF(it):

    U = u[it].reshape(x,y)

    U_pri = u_pri[it].reshape(x,y)

    #FFT
    ufft = np.fft.fftshift(np.fft.fft2(U))
    ufft_pri = np.fft.fftshift(np.fft.fft2(U_pri))


    #multiply filter
    H = butterwort_low_pass_filer(U)
    ufft_filt = ufft * H
    ufft_pri_filt = ufft_pri * H

    #IFFT
    ufft_filt_shift = np.fft.ifftshift(ufft_filt)
    iufft_filt = np.real(np.fft.ifft2(ufft_filt_shift))

    ufft_pri_filt_shift = np.fft.ifftshift(ufft_pri_filt)
    iufft_pri_filt = np.real(np.fft.ifft2(ufft_pri_filt_shift))


    Z = iufft_filt.reshape(x,y)
    Z_pri = iufft_pri_filt.reshape(x,y)

    return Z, Z_pri


def high_Speed_eddy(it):

    filt_U = filt_u[it]
    filt_U_PRI =  filt_u_pri[it]
    CS = plt.contour(X, Y, filt_U_PRI, levels=levels_pos)

    f = interpolate.interp2d(xs,ys,filt_U,kind="linear")
    f_pri = interpolate.interp2d(xs,ys,filt_U_PRI,kind="linear")
    
    D_high = []
    Tau_high = []
    Ux_avg_high = []
    lines = CS.allsegs[0] #plot only threshold velocity
    for line in lines:
        if xleft in np.around(line[:,0],1) or xright in np.around(line[:,0],1) or ytop in np.around(line[:,1],1) or ybottom in np.around(line[:,1],1):
            continue
        else:
            x_pri = np.subtract( line[:,0] * np.cos(np.radians(-29)), line[:,1] * np.sin(np.radians(-29)) )

            Centroid = [np.sum(line[:,0])/len(line[:,0]), np.sum(line[:,1])/len(line[:,1])]

            Dist = np.max(x_pri) - np.min(x_pri)

            if f_pri(Centroid[0],Centroid[1]) < 0.7 or Dist < 1/cutoff:
                continue
            else:

                xmin = np.min(line[:,0]); xmax = np.max(line[:,0])
                x_array = np.arange(xmin+5,xmax-5,10)


                coordinates = []
                for xr in x_array:
            
                    xidx = (line[:,0]>(xr-5))*(line[:,0]<xr+5)
                    xidxlist = np.where(xidx)
                    if len(xidxlist[0]) == 0:
                        continue

                    ymin = np.min(line[xidxlist[0],1]); ymax = np.max(line[xidxlist[0],1])

                    if ymin+10 < ymax-10:
                        ylist = np.arange(ymin,ymax,10)
                        
                        for yr in ylist:
                            coordinates.append([xr,yr])

                Ux_avg = []
                for coordinate in coordinates:

                    ux = f(coordinate[0],coordinate[1])
                    ux_pri = f_pri(coordinate[0],coordinate[1])
                    if ux_pri >= 0.7 and cmin <= ux <= cmax:
                        Ux_avg.append(ux)


                if len(Ux_avg) == 0:
                    continue
                else:
                    D_high.append(Dist)
                    Ux_avg_high.append(np.average(Ux_avg))
                    Tau_high.append(Dist/np.average(Ux_avg))
    
    return D_high, Ux_avg_high, Tau_high


def low_Speed_eddy(it):

    filt_U = filt_u[it]
    filt_U_PRI = filt_u_pri[it]

    CZ = plt.contour(X,Y,filt_U_PRI, levels=levels_neg)

    f = interpolate.interp2d(xs,ys,filt_U,kind="linear")
    f_pri = interpolate.interp2d(xs,ys,filt_U_PRI,kind="linear")

    D_low = []
    Tau_low = []
    Ux_avg_low = []
    lines = CZ.allsegs[-1] #plot only threshold velocity
    for line in lines:
        if xleft in np.around(line[:,0],1) or xright in np.around(line[:,0],1) or ytop in np.around(line[:,1],1) or ybottom in np.around(line[:,1],1):
            continue
        else:
            x_pri = np.subtract( line[:,0] * np.cos(np.radians(-29)), line[:,1] * np.sin(np.radians(-29)) )

            Centroid = [np.sum(line[:,0])/len(line[:,0]), np.sum(line[:,1])/len(line[:,1])]

            Dist = np.max(x_pri) - np.min(x_pri)


            if f_pri(Centroid[0],Centroid[1]) > -0.7 or Dist < 1/cutoff:
                continue
            else:

                xmin = np.min(line[:,0]); xmax = np.max(line[:,0])
                x_array = np.arange(xmin+5,xmax-5,10)


                coordinates = []
                for xr in x_array:
            
                    xidx = (line[:,0]>(xr-5))*(line[:,0]<xr+5)
                    xidxlist = np.where(xidx)
                    if len(xidxlist[0]) == 0:
                        continue

                    ymin = np.min(line[xidxlist[0],1]); ymax = np.max(line[xidxlist[0],1])

                    if ymin+10 < ymax-10:
                        ylist = np.arange(ymin,ymax,10)
                        
                        for yr in ylist:
                            coordinates.append([xr,yr])

                Ux_avg = []
                for coordinate in coordinates:
                    ux = f(coordinate[0],coordinate[1])
                    ux_pri = f_pri(coordinate[0],coordinate[1])
                    if ux_pri <= -0.7 and cmin <= ux <= cmax:
                        Ux_avg.append(ux)


                if len(Ux_avg) == 0:
                    continue
                else:
                    D_low.append(Dist)
                    Ux_avg_low.append(np.average(Ux_avg))
                    Tau_low.append(Dist/np.average(Ux_avg))
    
    return D_low, Ux_avg_low, Tau_low



start_time = time.time()

t_start = 38000; t_end = 39201
#defining twist angles with height from precursor
precursor = Dataset("abl_statistics70000.nc")
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start_idx = np.searchsorted(precursor.variables["time"],t_start)
t_end_idx = np.searchsorted(precursor.variables["time"],t_end)
u = np.average(mean_profiles.variables["u"][t_start_idx:t_end_idx],axis=0)
v = np.average(mean_profiles.variables["v"][t_start_idx:t_end_idx],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del mean_profiles; del u; del v; del t_start_idx; del t_end_idx

print("line 252",time.time()-start_time)

offsets = [22.5,85,142.5]
filter_cutoffs = [6.5e-03,3e-03,1.1e-03]
for i in np.arange(0,len(offsets)):
    a = Dataset("sampling_l_{}.nc".format(offsets[i]))
    height = offsets[i]+7.5
    p = a.groups["p_l"]

    #time options
    Time = np.array(a.variables["time"])
    tstart_idx = np.searchsorted(Time,t_start)
    tend_idx = np.searchsorted(Time,t_end)
    Time_steps = np.arange(0, tend_idx-tstart_idx)
    Time = Time[tstart_idx:tend_idx]


    x = p.ijk_dims[0] #no. data points
    y = p.ijk_dims[1] #no. data points


    #define plotting axes
    coordinates = np.array(p.variables["coordinates"])

    xs = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x)
    ys = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y)

    print("line 277",time.time()-start_time)

    #velocity field
    u = np.array(p.variables["velocityx"][tstart_idx:tend_idx])
    v = np.array(p.variables["velocityy"][tstart_idx:tend_idx])

    u[u<0]=0; v[v<0]=0 #remove negative velocities

    print(len(u),len(v))

    with Pool() as pool:
        u_hvel = []; u_pri = []
        for u_hvel_it, u_hvel_pri_it in pool.imap(Horizontal_velocity,Time_steps):
            
            u_hvel.append(u_hvel_it)
            u_pri.append(u_hvel_pri_it)
            print(len(u_hvel))
    u = np.array(u_hvel); u_pri = np.array(u_pri); del u_hvel; del v


    D_low_array = []; Ux_avg_low_array = []; Tau_low_array = []
    D_high_array = []; Ux_avg_high_array = []; Tau_high_array = []
    filter_cutoff = [(1*filter_cutoffs[i]),(1.5*filter_cutoffs[i]),(2*filter_cutoffs[i]),(2.5*filter_cutoffs[i]),(3*filter_cutoffs[i])]
    for cutoff in filter_cutoff:

        print("cuttoff = ",round(1/cutoff,0),time.time()-start_time)

        filt_u = []; filt_u_pri = []
        with Pool() as pool:
            for filt_u_it, filt_u_pri_it in pool.imap(two_dim_LPF,Time_steps):
                filt_u.append(filt_u_it)
                filt_u_pri.append(filt_u_pri_it)
                print(len(filt_u))
        filt_u = np.array(filt_u); filt_u_pri = np.array(filt_u_pri)


        #find vmin and vmax for isocontour plots            
        #min and max over data
        cmin_pri = math.floor(np.min(filt_u_pri))
        cmax_pri = math.ceil(np.max(filt_u_pri))

        cmin = math.floor(np.min(filt_u))
        cmax = math.ceil(np.max(filt_u))


        nlevs = int((cmax_pri-cmin_pri)/2)
        if nlevs>abs(cmin_pri) or nlevs>cmax_pri:
            nlevs = min([abs(cmin_pri),cmax_pri])+1

        levs_min = np.linspace(cmin_pri,0,nlevs,dtype=int); levs_max = np.linspace(0,cmax_pri,nlevs,dtype=int)
        levels = np.concatenate((levs_min,levs_max[1:]))
        print("line 326", levels)


        nlevs = int((cmax_pri-cmin_pri)/2)
        if nlevs>abs(cmin_pri) or nlevs>cmax_pri:
            nlevs = min([abs(cmin_pri),cmax_pri])+1

        #define thresholds with number of increments
        levels_pos = np.linspace(0.7,cmax_pri,4)
        print("line 335", levels_pos)
        levels_neg = np.linspace(cmin_pri,-0.7,4)
        print("line 337", levels_neg)



        xleft = np.min(xs); xright = np.max(xs)
        ytop = np.max(ys); ybottom = np.min(ys)
        X,Y = np.meshgrid(xs,ys)

        print("line 345",time.time()-start_time)


        D_high_time_arr = []; Ux_avg_high_time_arr = []; Tau_high_time_arr = []
        with Pool() as pool:
            ix = 0
            for D_high_it, Ux_avg_high_it, Tau_high_it in pool.imap(high_Speed_eddy,Time_steps):
                D_high_time_arr.extend(D_high_it)
                Ux_avg_high_time_arr.extend(Ux_avg_high_it)
                Tau_high_time_arr.extend(Tau_high_it)
                print(ix,time.time()-start_time)
                ix+=1

        D_high_array.append(D_high_time_arr); Ux_avg_high_array.append(Ux_avg_high_time_arr); Tau_high_array.append(Tau_high_time_arr)
        del D_high_time_arr; del Ux_avg_high_time_arr; del Tau_high_time_arr


        D_low_time_arr = []; Ux_avg_low_time_arr = []; Tau_low_time_arr = []
        with Pool() as pool:
            ix = 0
            for D_low_it, Ux_avg_low_it, Tau_low_it in pool.imap(low_Speed_eddy,Time_steps):
                D_low_time_arr.extend(D_low_it)
                Ux_avg_low_time_arr.extend(Ux_avg_low_it)
                Tau_low_time_arr.extend(Tau_low_it)
                print(ix,time.time()-start_time)

        D_low_array.append(D_low_time_arr); Ux_avg_low_array.append(Ux_avg_low_time_arr); Tau_low_array.append(Tau_low_time_arr)
        del D_low_time_arr; del Ux_avg_low_time_arr; del Tau_low_time_arr



    colors = ["g","c","b","r","k"]
    plt.rcParams['font.size'] = 12
    fig = plt.figure(figsize=(14,8))
    for j in np.arange(0,len(filter_cutoff)):
        cutoff = filter_cutoff[j]
        Tau_high = Tau_high_array[j]
        Tau_low = Tau_low_array[j]
        min_Tau_high = round(np.min(Tau_high),1); min_Tau_low = round(np.min(Tau_low),1)
        max_Tau_high = round(np.max(Tau_high),1); max_Tau_low = round(np.max(Tau_low),1)
        mean_Tau_high = round(np.mean(Tau_high),1); mean_Tau_low = round(np.mean(Tau_low),1)
        std_Tau_high = round(np.std(Tau_high),1); std_Tau_low = round(np.std(Tau_low),1) 
        P,X = probability_dist(Tau_high)
        plt.plot(X,P,"-",color=colors[j],label="High speed: cutoff = {}\nMin = {}, Max = {}\nMean = {}, Std = {}".format(round(1/cutoff,0),min_Tau_high,max_Tau_high,mean_Tau_high,std_Tau_high))
        P,X = probability_dist(Tau_low)
        plt.plot(X,P,"--",color=colors[j],label="Low speed: cuttoff = {}\nMin = {}, Max = {}\nMean = {}, Std = {}".format(round(1/cutoff,0),min_Tau_low,max_Tau_low,mean_Tau_low,std_Tau_low))
    plt.xlabel("Eddy passage time [s]")
    plt.ylabel("Probability [-]")
    plt.title("Height from surface {}m".format(height))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("Eddy_passage_time_{}m.png".format(height))
    plt.close()

    fig = plt.figure(figsize=(14,8))
    for j in np.arange(0,len(filter_cutoff)):
        cutoff = filter_cutoff[j]
        Ux_avg_high = Ux_avg_high_array[j]
        Ux_avg_low = Ux_avg_low_array[j]
        min_Ux_high = round(np.min(Ux_avg_high),2); min_Ux_low = round(np.min(Ux_avg_low),2)
        max_Ux_high = round(np.max(Ux_avg_high),2); max_Ux_low = round(np.max(Ux_avg_low),2)
        mean_Ux_high = round(np.mean(Ux_avg_high),2); mean_Ux_low = round(np.mean(Ux_avg_low),2)
        std_Ux_high = round(np.std(Ux_avg_high),2); std_Ux_low = round(np.std(Ux_avg_low),2) 
        P,X = probability_dist(Ux_avg_high)
        plt.plot(X,P,"-",color=colors[j],label="High speed: cutoff = {}\nMin = {}, Max = {}\nMean = {}, Std = {}".format(round(1/cutoff,0),min_Ux_high,max_Ux_high,mean_Ux_high,std_Ux_high))
        P,X = probability_dist(Ux_avg_low)
        plt.plot(X,P,"--",color=colors[j],label="Low speed: cutoff = {}\nMin = {}, Max = {}\nMean = {}, Std = {}".format(round(1/cutoff,0),min_Ux_low,max_Ux_low,mean_Ux_low,std_Ux_low))
    plt.xlabel("Area average of eddy velocity [m/s]")
    plt.ylabel("Probability [-]")
    plt.title("Height from surface {}m".format(height))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("Eddy_velocity_{}m.png".format(height))
    plt.close()

    fig = plt.figure(figsize=(14,8))
    for j in np.arange(0,len(filter_cutoff)):
        cutoff = filter_cutoff[j]
        D_high = D_high_array[j]
        D_low = D_low_array[j]
        min_D_high = round(np.min(D_high),0); min_D_low = round(np.min(D_low),0)
        max_D_high = round(np.max(D_high),0); max_D_low = round(np.max(D_low),0)
        mean_D_high = round(np.mean(D_high),0); mean_D_low = round(np.mean(D_low),0)
        std_D_high = round(np.std(D_high),0); std_D_low = round(np.std(D_low),0) 
        P,X = probability_dist(D_high)
        plt.plot(X,P,"-",color=colors[j],label="High speed: cutoff = {}\nMin = {}, Max = {}\nMean = {}, Std = {}".format(round(1/cutoff,0),min_D_high,max_D_high,mean_D_high,std_D_high))
        P,X = probability_dist(D_low)
        plt.plot(X,P,"--",color=colors[j],label="Low speed: cutoff = {}\nMin = {}, Max = {}\nMean = {}, Std = {}".format(round(1/cutoff,0),min_D_low,max_D_low,mean_D_low,std_D_low))
    plt.xlabel("Eddy length x' direction [m]")
    plt.ylabel("Probability [-]")
    plt.title("Height from surface {}m".format(height))
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig("Eddy_length_{}m.png".format(height))
    plt.close()


