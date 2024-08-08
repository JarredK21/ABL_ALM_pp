from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq, fftshift
import numpy as np
from multiprocessing import Pool
import math
from scipy import interpolate
import time
from matplotlib import cm
import pandas as pd


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


def two_dim_LPF():

    U = u.reshape(x,y)

    U_pri = u_pri.reshape(x,y)

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

    fig = plt.figure(figsize=(50,30))
    plt.rcParams['font.size'] = 40

    cs = plt.contourf(X,Y,filt_U_PRI,levels=levels, cmap=cm.coolwarm,vmin=cmin,vmax=cmax)
    cb = plt.colorbar(cs)
    
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
                    plt.plot(line[:,0],line[:,1],"-k")


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
                    plt.plot(line[:,0],line[:,1])


start_time = time.time()
t_start = 32500
#defining twist angles with height from precursor
in_dir="../../ABL_precursor_2/"
precursor = Dataset(in_dir+"abl_statistics60000.nc")
Time = np.array(precursor.variables["time"])
mean_profiles = precursor.groups["mean_profiles"] #create variable to hold mean profiles
t_start_idx = np.searchsorted(precursor.variables["time"],t_start)
u = np.average(mean_profiles.variables["u"][t_start_idx:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start_idx:],axis=0)
h = mean_profiles["h"][:]
twist = coriolis_twist(u,v) #return twist angle in radians for precursor simulation
ux_mean_profile = []
for i in np.arange(0,len(twist)):
    ux_mean_profile.append(u[i] * np.cos(twist[i]) + v[i] * np.sin(twist[i]))
del precursor; del mean_profiles; del u; del v; del t_start_idx

print("line 252",time.time()-start_time)

offsets = [85]


for offset in offsets:
    print(offset)
    a = Dataset(in_dir+"sampling_l_{}.nc".format(offset))
    height = offset+7.5
    p = a.groups["p_l"]


    filter_cutoff = [(2*3e-03)]

    print(filter_cutoff)

    #time options
    Time = np.array(a.variables["time"])
    tstart_idx = np.searchsorted(Time,t_start)


    x = p.ijk_dims[0] #no. data points
    y = p.ijk_dims[1] #no. data points


    #define plotting axes
    coordinates = np.array(p.variables["coordinates"])

    xs = np.linspace(p.origin[0],p.origin[0]+p.axis1[0],x)
    ys = np.linspace(p.origin[1],p.origin[1]+p.axis2[1],y)

    print("line 277",time.time()-start_time)

    #velocity field
    u = np.array(p.variables["velocityx"][tstart_idx])
    v = np.array(p.variables["velocityy"][tstart_idx])

    u[u<0]=0; v[v<0]=0 #remove negative velocities

    print(len(u),len(v))

    f = interpolate.interp1d(h,twist)
    f_ux = interpolate.interp1d(h,ux_mean_profile)
    twist_h = f(height)
    ux_mean = f_ux(height)
    u = np.array(u*np.cos(twist_h) + v*np.sin(twist_h))
    u_pri = np.array(np.subtract(u,ux_mean))
    del v


    for cutoff in filter_cutoff:

        print("cuttoff = ",round(1/cutoff,0),time.time()-start_time)

        filt_u, filt_u_pri = two_dim_LPF()

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

        #high speed eddies
        it = 0
        filt_U = filt_u
        filt_U_PRI =  filt_u_pri
        CS = plt.contour(X, Y, filt_U_PRI, levels=levels_pos)
        CZ = plt.contour(X,Y,filt_U_PRI, levels=levels_neg)

        f = interpolate.interp2d(xs,ys,filt_U,kind="linear")
        f_pri = interpolate.interp2d(xs,ys,filt_U_PRI,kind="linear")

        fig = plt.figure(figsize=(50,30))
        plt.rcParams['font.size'] = 40

        cs = plt.contourf(X,Y,filt_U_PRI,levels=levels, cmap=cm.coolwarm,vmin=cmin_pri,vmax=cmax_pri)
        cb = plt.colorbar(cs)
        
        lines = CS.allsegs[0] #plot only threshold velocity
        #for line in lines:
        line = lines[0]

        x_pri = np.subtract( line[:,0] * np.cos(np.radians(-29)), line[:,1] * np.sin(np.radians(-29)) )

        Centroid = [np.sum(line[:,0])/len(line[:,0]), np.sum(line[:,1])/len(line[:,1])]

        Dist = np.max(x_pri) - np.min(x_pri)

        # if f_pri(Centroid[0],Centroid[1]) < 0.7:
        #     continue
        # else:

        xmin = np.min(line[:,0]); xmax = np.max(line[:,0])
        x_array = np.arange(xmin+5,xmax-5,10)

        plt.plot(line[:,0],line[:,1],"-k")
        # coordinates = []
        # for xr in x_array:
        if 5.0 in line[:,0] or 5.0 in line[:,1]:

            c = line[np.concatenate((np.where(line[:,0]==5.0)[0],np.where(line[:,1]==5.0)[0])),0]
            print(c)

        xlist = []
        for xr in line[:,0]:
            if xr in xlist:
                continue
            else:
                xlist.append(xr)
        xlist = np.sort(xlist)
        for xr in xlist:
            xc = np.where(line[:,0]==xr)[0]
            print(xc)
            xline = [xr,xr]
            print(xline)
            yc = line[xc,1]
            print(yc)

            if len(yc) > 2 and len(yc)%2 != 0:
                yc = np.sort(yc)
                print(yc)
                if np.min(c) <= xr <= np.max(c):
                    yline = np.insert(yc,0,5.0)
                    n = int(len(yline)/2)
                    ylines = np.reshape(yline,(n,2))
                for i in np.arange(0,len(ylines)):
                    print(ylines[i])
                    plt.plot(xline,ylines[i],"-k")
            else:
                if np.min(c) <= xr <= np.max(c):
                    yline = [5.0,np.max(yc)]
                else:
                    yline = [np.min(yc),np.max(yc)]
                print(yline)
                plt.plot(xline,yline,"-k")
    
        #     xidx = (line[:,0]>(xr-5))*(line[:,0]<xr+5)
        #     xidxlist = np.where(xidx)
        #     if len(xidxlist[0]) == 0:
        #         continue

        #     ymin = np.min(line[xidxlist[0],1]); ymax = np.max(line[xidxlist[0],1])

        #     if ymin+10 < ymax-10:
        #         ylist = np.arange(ymin,ymax,10)
                
        #         for yr in ylist:
        #             coordinates.append([xr,yr])

        # Ux_avg = []
        # Ux_pri_avg = []
        # for coordinate in coordinates:
        #     ux = f(coordinate[0],coordinate[1])
        #     ux_pri = f_pri(coordinate[0],coordinate[1])
        #     #if ux_pri >= 0.7 and cmin <= ux <= cmax:
        #     if cmin <= ux <= cmax:
        #         Ux_avg.append(ux)
        #         Ux_pri_avg.append(ux_pri)
        #         plt.plot(coordinate[0],coordinate[1],"ok",markersize=1)


        # if len(Ux_avg) == 0 or np.mean(Ux_pri_avg) < 0.7:
        #     continue
        # else:
        #     plt.plot(line[:,0],line[:,1],"-k")





        #low speed eddies

        # lines = CZ.allsegs[-1] #plot only threshold velocity
        # #for line in lines:
        # line = lines[0]
        # x_pri = np.subtract( line[:,0] * np.cos(np.radians(-29)), line[:,1] * np.sin(np.radians(-29)) )

        # Centroid = [np.sum(line[:,0])/len(line[:,0]), np.sum(line[:,1])/len(line[:,1])]

        # Dist = np.max(x_pri) - np.min(x_pri)


        # # if f_pri(Centroid[0],Centroid[1]) > -0.7:
        # #     continue
        # # else:

        # xmin = np.min(line[:,0]); xmax = np.max(line[:,0])
        # x_array = np.arange(xmin+5,xmax-5,10)


        # coordinates = []
        # for xr in x_array:
    
        #     xidx = (line[:,0]>(xr-5))*(line[:,0]<xr+5)
        #     xidxlist = np.where(xidx)
        #     if len(xidxlist[0]) == 0:
        #         continue

        #     ymin = np.min(line[xidxlist[0],1]); ymax = np.max(line[xidxlist[0],1])

        #     if ymin+10 < ymax-10:
        #         ylist = np.arange(ymin,ymax,10)
                
        #         for yr in ylist:
        #             coordinates.append([xr,yr])

        # Ux_avg = []
        # Ux_pri_avg = []
        # for coordinate in coordinates:
        #     ux = f(coordinate[0],coordinate[1])
        #     ux_pri = f_pri(coordinate[0],coordinate[1])
        #     #if ux_pri <= -0.7 and cmin <= ux <= cmax:
        #     if cmin <= ux <= cmax:
        #         Ux_avg.append(ux)
        #         Ux_pri_avg.append(ux_pri)
        #         plt.plot(coordinate[0],coordinate[1],"ok",markersize=1)


        # if len(Ux_avg) == 0 or np.mean(Ux_pri_avg) > -0.7:
        #     continue
        # else:
        #     plt.plot(line[:,0],line[:,1],"--k")


        plt.xlabel("x axis [m]")
        plt.ylabel("y axis [m]")
        plt.title("{}m plane from surface, {}m filtered Streamwise fluctuating velocity, T = {}".format(offset,round(1/cutoff,0),0.0))
        plt.show()