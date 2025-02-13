from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt
from scipy import interpolate
import pyFAST.input_output as io
import time
from multiprocessing import Pool


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt


def tranform_fixed_frame(Y_pri,Z_pri,Theta):

    Y = Y_pri*np.cos(Theta) - Z_pri*np.sin(Theta)
    Z = Y_pri*np.sin(Theta) + Z_pri*np.cos(Theta)

    return Y,Z



def low_pass_filter(signal, cutoff,dt):  

    fs = 1/dt     # sample rate, Hz      
    nyq = 0.5 * fs  # Nyquist Frequency      
    order = 3  # sin wave can be approx represented as 3rd order polynomial

    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='lowpass', analog=False)

    low_pass_signal = filtfilt(b, a, signal)

    return low_pass_signal


def correlation_coef(x,y):
    
    r = (np.sum(((x-np.mean(x))*(y-np.mean(y)))))/(np.sqrt(np.sum(np.square(x-np.mean(x)))*np.sum(np.square(y-np.mean(y)))))

    return r


def energy_contents_check(Var,e_fft,signal,dt):

    E = (1/dt)*np.sum(e_fft)

    q = np.sum(np.square(signal))

    E2 = q

    print(Var, E, E2, abs(E2/E))    


def temporal_spectra(signal,dt,Var):

    fs =1/dt
    n = len(signal) 
    if n%2==0:
        nhalf = int(n/2+1)
    else:
        nhalf = int((n+1)/2)
    frq = np.arange(nhalf)*fs/n
    Y   = np.fft.fft(signal)
    PSD = abs(Y[range(nhalf)])**2 /(n*fs) # PSD
    PSD[1:-1] = PSD[1:-1]*2


    energy_contents_check(Var,PSD,signal,dt)

    return frq, PSD


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


def moments(y):
    mu = np.mean(y)
    std = np.std(y)
    N = len(y)

    skewness = (np.sum(np.power(np.subtract(y,mu),3)))/(N*std**3)
    kurotsis = (np.sum(np.power(np.subtract(y,mu),4)))/(N*std**4)

    return mu, std, skewness,kurotsis


def dt_calc(y,dt):
    d_dt = []
    for i in np.arange(0,len(Time_OF)-1):
        d_dt.append((y[i+1]-y[i])/dt)

    return d_dt


def actuator_asymmetry_calc(it):
    hvelB1 = np.add(np.cos(np.radians(29))*uvelB1[it], np.sin(np.radians(29))*vvelB1[it])
    IyB1 = hvelB1*R*np.cos(np.radians(Azimuth[it]))
    IyB1_75 = IyB1[225]
    IyB1 = np.sum(IyB1)
    IzB1 = hvelB1*R*np.sin(np.radians(Azimuth[it]))
    IzB1_75 = IzB1[225]
    IzB1 = np.sum(IzB1)

    hvelB2 = np.add(np.cos(np.radians(29))*uvelB2[it], np.sin(np.radians(29))*vvelB2[it])
    AzB2 = Azimuth[it] + 120
    if AzB2 >= 360:
        AzB2-=360

    IyB2 = hvelB2*R*np.cos(np.radians(AzB2))
    IzB2 = hvelB2*R*np.sin(np.radians(AzB2))
    IyB2_75 = IyB2[225]
    IyB2 = np.sum(IyB2)
    IzB2_75 = IzB2[225]
    IzB2 = np.sum(IzB2)

    hvelB3 = np.add(np.cos(np.radians(29))*uvelB3[it], np.sin(np.radians(29))*vvelB3[it])
    AzB3 = Azimuth[it] + 240
    if AzB3 >= 360:
        AzB3-=360

    IyB3 = hvelB3*R*np.cos(np.radians(AzB3))
    IzB3 = hvelB3*R*np.sin(np.radians(AzB3))
    IyB3_75 = IyB3[225]
    IyB3 = np.sum(IyB3)
    IzB3_75 = IzB3[225]
    IzB3 = np.sum(IzB3)

    IB1 = np.sqrt(np.add(np.square(IyB1),np.square(IzB1)))
    IB2 = np.sqrt(np.add(np.square(IyB2),np.square(IzB2)))
    IB3 = np.sqrt(np.add(np.square(IyB3),np.square(IzB3)))

    return IyB1+IyB2+IyB3, IzB1+IzB2+IzB3, IyB1_75+IyB2_75+IyB3_75, IzB1_75+IzB2_75+IzB3_75,IB1,IB2,IB3,IyB1_75, IyB2_75, IyB3_75, IzB1_75, IzB2_75, IzB3_75

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"


df = Dataset(in_dir+"WTG01a.nc")

# num_act_points = 300

xco = np.array(df.variables["xco"])
yco = np.array(df.variables["yco"])
zco = np.array(df.variables["zco"])

R = np.linspace(0,63.0,300)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(xco, yco, zco, marker="o")
# ax.plot(xco[267], yco[267], zco[267],"*r",markersize=10)
# ax.plot(xco[567], yco[567], zco[567],"*r",markersize=10)
# ax.plot(xco[867], yco[867], zco[867],"*r",markersize=10)
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')





Time = np.array(df.variables["time"])
dt = Time[1] - Time[0]
Tstart_idx = np.searchsorted(Time,200)
T_end_idx = np.searchsorted(Time,1199.6361)+1
Time = Time[Tstart_idx:T_end_idx]


a = Dataset(in_dir+"Dataset.nc")
Time_sampling = np.array(a.variables["Time_sampling"])
T_start_sampling_idx = np.searchsorted(Time_sampling,200)
Time_sampling = Time_sampling[T_start_sampling_idx:]
dt_sampling = Time_sampling[1] - Time_sampling[0]

Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]

Rotor_avg_vars = Rotor_avg_vars.groups["5.5"]
Ux_R = np.array(Rotor_avg_vars.variables["Ux"][T_start_sampling_idx:])
Ux_mean = np.mean(Ux_R)

Iy = np.array(Rotor_avg_vars.variables["Iy"][T_start_sampling_idx:])
Iz = np.array(Rotor_avg_vars.variables["Iz"][T_start_sampling_idx:])
I_2 = np.sqrt(np.add(np.square(Iy),np.square(Iz)))
I_2_LPF = low_pass_filter(I_2,0.3,dt_sampling)

Rotor_avg_vars = a.groups["Rotor_Avg_Variables"]
Rotor_avg_vars = Rotor_avg_vars.groups["63.0"]
Ux_D = np.array(Rotor_avg_vars.variables["Ux"][T_start_sampling_idx:])
Ux_D_mean = np.mean(Ux_D)

Iy_3 = np.array(Rotor_avg_vars.variables["Iy"][T_start_sampling_idx:])
Iz_3 = np.array(Rotor_avg_vars.variables["Iz"][T_start_sampling_idx:])
I_3 = np.sqrt(np.add(np.square(Iy_3),np.square(Iz_3)))
I_3_LPF = low_pass_filter(I_3,0.3,dt_sampling)


a = Dataset(in_dir+"Dataset.nc")

Time_OF = np.array(a.variables["Time_OF"][Tstart_idx:T_end_idx])

OF_vars = a.groups["OpenFAST_Variables"]

Azimuth = np.radians(np.array(OF_vars.variables["Azimuth"][Tstart_idx:T_end_idx]))

RtAeroFyh = np.array(OF_vars.variables["RtAeroFyh"][Tstart_idx:T_end_idx])/1000
RtAeroFzh = np.array(OF_vars.variables["RtAeroFzh"][Tstart_idx:T_end_idx])/1000

RtAeroFys, RtAeroFzs = tranform_fixed_frame(RtAeroFyh,RtAeroFzh,Azimuth)


RtAeroMyh = np.array(OF_vars.variables["RtAeroMyh"][Tstart_idx:T_end_idx])/1000
RtAeroMzh = np.array(OF_vars.variables["RtAeroMzh"][Tstart_idx:T_end_idx])/1000

RtAeroMys, RtAeroMzs = tranform_fixed_frame(RtAeroMyh,RtAeroMzh,Azimuth)

RtAeroMR = np.sqrt( np.add(np.square(RtAeroMys), np.square(RtAeroMzs)) ) 


RtAeroFxa_E = np.array(OF_vars.variables["LSShftFxa"][Tstart_idx:T_end_idx])

RtAeroFys_EE = np.array(OF_vars.variables["LSShftFys"][Tstart_idx:T_end_idx])
RtAeroFzs_EE = np.array(OF_vars.variables["LSShftFzs"][Tstart_idx:T_end_idx])

RtAeroMxa_EE = np.array(OF_vars.variables["LSShftMxa"][Tstart_idx:T_end_idx])

RtAeroMys_EE = np.array(OF_vars.variables["LSSTipMys"][Tstart_idx:T_end_idx])
RtAeroMzs_EE = np.array(OF_vars.variables["LSSTipMzs"][Tstart_idx:T_end_idx])

#Filtering MR
LPF_1_MR = low_pass_filter(RtAeroMR,0.3,dt)
LPF_2_MR = low_pass_filter(RtAeroMR,0.9,dt)
LPF_3_MR = low_pass_filter(RtAeroMR,1.5,dt)
BPF_MR = np.subtract(LPF_2_MR,LPF_1_MR)
HPF_MR = np.subtract(RtAeroMR,LPF_3_MR)
HPF_MR = np.array(low_pass_filter(HPF_MR,40,dt))

dHPF_MR = dt_calc(HPF_MR,dt)

zero_crossings_index_HPF_MR = np.where(np.diff(np.sign(dHPF_MR)))[0]
Time_zero_crossings_HPF_MR = Time_OF[zero_crossings_index_HPF_MR]



LSSTipMys = np.array(OF_vars.variables["LSSTipMys"][Tstart_idx:T_end_idx])
LSSTipMzs = np.array(OF_vars.variables["LSSTipMzs"][Tstart_idx:T_end_idx])
LSSTipMR = np.sqrt( np.add(np.square(LSSTipMys), np.square(LSSTipMzs)) )

#Total radial aerodynamic bearing force aeroFBR
L1 = 1.912; L2 = 2.09

FBMy = RtAeroMzs/L2; FBFy = -RtAeroFys*((L1+L2)/L2)
FBMz = -RtAeroMys/L2; FBFz = -RtAeroFzs*((L1+L2)/L2)

FBy = -(FBMy + FBFy); FBz = -(FBMz + FBFz)


FBR = np.sqrt(np.add(np.square(FBy),np.square(FBz)))
#Filtering FBR
LPF_3_FBR = low_pass_filter(FBR,1.5,dt)
HPF_FBR = np.subtract(FBR,LPF_3_FBR)
HPF_FBR = np.array(low_pass_filter(HPF_FBR,40,dt))

dHPF_FBR = dt_calc(HPF_FBR,dt)

zero_crossings_index_HPF_FBR = np.where(np.diff(np.sign(dHPF_FBR)))[0]


Azimuth = np.array(OF_vars.variables["Azimuth"][Tstart_idx:T_end_idx])




uvelB1 = np.array(df.variables["uvel"][Tstart_idx:T_end_idx,1:301])
vvelB1 = np.array(df.variables["vvel"][Tstart_idx:T_end_idx,1:301])
uvelB1[uvelB1<0]=0; vvelB1[vvelB1<0]=0 #remove negative velocities
hvelB1 = np.add(np.cos(np.radians(29))*uvelB1, np.sin(np.radians(29))*vvelB1)
uvelB2 = np.array(df.variables["uvel"][Tstart_idx:T_end_idx,301:601])
vvelB2 = np.array(df.variables["vvel"][Tstart_idx:T_end_idx,301:601])
uvelB2[uvelB2<0]=0; vvelB2[vvelB2<0]=0 #remove negative velocities
hvelB2 = np.add(np.cos(np.radians(29))*uvelB2, np.sin(np.radians(29))*vvelB2)
uvelB3 = np.array(df.variables["uvel"][Tstart_idx:T_end_idx,601:901])
vvelB3 = np.array(df.variables["vvel"][Tstart_idx:T_end_idx,601:901])
uvelB3[uvelB3<0]=0; vvelB3[vvelB3<0]=0 #remove negative velocities
hvelB3 = np.add(np.cos(np.radians(29))*uvelB3, np.sin(np.radians(29))*vvelB3)

#normalize HPF FBR
HPF_FBR_norm = HPF_FBR/(1079*((L1+L2)/L2))

#HPF FBR calc
FBR_HPF = []
dF_FB_HPF = []
Time_ux_HPF = []
Time_FB_HPF = []
maxUx = []
for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

    it_1 = zero_crossings_index_HPF_FBR[i]
    it_2 = zero_crossings_index_HPF_FBR[i+1]

    Time_ux_HPF.append(Time_OF[it_1])
    Time_ux_HPF.append(Time_OF[it_2])

    Time_FB_HPF.append(Time_OF[it_1])
    FBR_HPF.append(HPF_FBR[it_1])

    dF_FB_HPF.append(abs(HPF_FBR[it_2]-HPF_FBR[it_1]))

    dUxB1 = abs(hvelB1[it_2,225]-hvelB1[it_1,225])
    dUxB2 = abs(hvelB2[it_2,225]-hvelB2[it_1,225])
    dUxB3 = abs(hvelB3[it_2,225]-hvelB3[it_1,225])

    max_dUx = np.max([dUxB1,dUxB2,dUxB3])
    if dUxB1 == max_dUx:
        maxUx.append(hvelB1[it_1,225]); maxUx.append(hvelB1[it_2,225])
    elif dUxB2 == max_dUx:
        maxUx.append(hvelB2[it_1,225]); maxUx.append(hvelB2[it_2,225])
    elif dUxB3 == max_dUx:
        maxUx.append(hvelB3[it_1,225]); maxUx.append(hvelB3[it_2,255])


fig, ax = plt.subplots(figsize=(14,8))
ax.plot(Time_OF,hvelB1[:,225],"-b",label="B1")
ax.plot(Time_OF,hvelB2[:,225],"-r",label="B2")
ax.plot(Time_OF,hvelB3[:,225],"-g",label="B3")
ax.legend()
ax.axhline(y=Ux_mean,linestyle="--",color="k")
ax2=ax.twinx()
ax2.plot(Time_OF,HPF_FBR,"-k")
sigma_events = 0
for i in np.arange(0,len(dF_FB_HPF)):
    if dF_FB_HPF[i] >= 2*np.std(dF_FB_HPF):
        ax2.plot(Time_FB_HPF[i],FBR_HPF[i],"ok")
        sigma_events +=1
ax.grid()
plt.show()

def eddy_type(maxUx_1,maxUx_2):

    if maxUx_1 <= Ux_mean and maxUx_2 <= Ux_mean:
        return "LSS_only"
    elif maxUx_1 <= Ux_mean and maxUx_2 >= Ux_mean:
        return "LSS_HSR"
    elif maxUx_2 <= Ux_mean and maxUx_1 >= Ux_mean:
        return "LSS_HSR"
    elif maxUx_1 > Ux_mean and maxUx_2 > Ux_mean:
        return "HSR_only"

LSS_array = []
HSR_array = []
LSS_HSR_array = []
for perc in np.linspace(0.5,1.0,6):
    LSS_only = 0
    HSR_only = 0
    LSS_HSR = 0
    for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

        it_1 = zero_crossings_index_HPF_FBR[i]
        it_2 = zero_crossings_index_HPF_FBR[i+1]


        dF_FB_HPF_i = abs(HPF_FBR[it_2]-HPF_FBR[it_1])
        if dF_FB_HPF_i >= 2*np.std(dF_FB_HPF):

            dUxB1 = abs(hvelB1[it_2,225]-hvelB1[it_1,225])
            dUxB2 = abs(hvelB2[it_2,225]-hvelB2[it_1,225])
            dUxB3 = abs(hvelB3[it_2,225]-hvelB3[it_1,225])

            max_dUx = np.max([dUxB1,dUxB2,dUxB3])

            if dUxB1 >= perc*max_dUx:
                maxUx_1 = hvelB1[it_1,225]; maxUx_2 = hvelB1[it_2,225]
                eddyB1 = eddy_type(maxUx_1,maxUx_2)
            else:
                eddyB1 = "None"
            
            if dUxB2 >= perc*max_dUx:
                maxUx_1 = hvelB2[it_1,225]; maxUx_2 = hvelB2[it_2,225]
                eddyB2 = eddy_type(maxUx_1,maxUx_2)
            else:
                eddyB2 = "None"

            if dUxB3 >= perc*max_dUx:
                maxUx_1 = hvelB3[it_1,225]; maxUx_2 = hvelB3[it_2,225]
                eddyB3 = eddy_type(maxUx_1,maxUx_2)
            else:
                eddyB3 = "None"
            

            if eddyB1 == "LSS_only" and eddyB2 == "LSS_only" and eddyB3 == "LSS_only":
                LSS_only+=1
            elif eddyB1 == "LSS_only" and eddyB2 == "None" and eddyB3 == "None":
                LSS_only+=1
            elif eddyB2 == "LSS_only" and eddyB1 == "None" and eddyB3 == "None":
                LSS_only+=1
            elif eddyB3 == "LSS_only" and eddyB2 == "None" and eddyB1 == "None":
                LSS_only+=1
            elif eddyB1 == "HSR_only" and eddyB2 == "HSR_only" and eddyB3 == "HSR_only":
                HSR_only+=1
            elif eddyB1 == "HSR_only" and eddyB2 == "None" and eddyB3 == "None":
                HSR_only+=1
            elif eddyB2 == "HSR_only" and eddyB1 == "None" and eddyB3 == "None":
                HSR_only+=1
            elif eddyB3 == "HSR_only" and eddyB2 == "None" and eddyB1 == "None":
                HSR_only+=1
            else:
                LSS_HSR+=1
    
    print(sigma_events)
    print(LSS_only+HSR_only+LSS_HSR)

    LSS_array.append(LSS_only/sigma_events); HSR_array.append(HSR_only/sigma_events); LSS_HSR_array.append(LSS_HSR/sigma_events)

xaxis = np.linspace(0.5,1.0,6)
plt.rcParams['font.size'] = 16
out_dir=in_dir+"High_frequency_analysis/"
fig = plt.figure(figsize=(14,8))
plt.plot(xaxis,LSS_array,"-b",label="LSS only")
plt.plot(xaxis,HSR_array,"-r",label="HSR only")
plt.plot(xaxis,LSS_HSR_array,"-g",label="LSS and HSR")
plt.xlabel("Fraction threshold to include blade change in velocity [-]")
plt.ylabel("Fraction of $2\sigma$ events in the HPF $F_B$ sigma [-]")
plt.legend()
plt.grid()
plt.title("Total od 837 $2\sigma$ events in 1000s")
plt.tight_layout()
plt.savefig(out_dir+"perc_HPF_eddy_response.png")
plt.close()

# xlabels = ["LSS Only", "HSR Only", "Both HSR and LSS"]
# heights = [LSS_only/sigma_events,HSR_only/sigma_events,LSS_HSR/sigma_events]
# bar_colors = ["tab:blue","tab:red","tab:green"]
# print(sigma_events)
# print(LSS_only+HSR_only+LSS_HSR)
# fig = plt.figure(figsize=(14,8))
# plt.bar(xlabels,heights,color=bar_colors)
# plt.ylabel("Fraction of $2\sigma$ events in the HPF $F_B$ sigma by type of eddy [-]")
# plt.title("Total of 837 $2\sigma$ events in 1000s")
# plt.tight_layout()
# plt.savefig(out_dir+"HPF_eddy_response_v2.png")
# plt.close()


# LPF_3_hvelB1 = low_pass_filter(hvelB1[:,225],1.5,dt)
# HPF_hvelB1 = np.subtract(hvelB1[:,225],LPF_3_hvelB1)
# HPF_hvelB1 = np.array(low_pass_filter(HPF_hvelB1,40,dt))

# LPF_3_hvelB2 = low_pass_filter(hvelB2[:,225],1.5,dt)
# HPF_hvelB2 = np.subtract(hvelB2[:,225],LPF_3_hvelB2)
# HPF_hvelB2 = np.array(low_pass_filter(HPF_hvelB2,40,dt))

# LPF_3_hvelB3 = low_pass_filter(hvelB3[:,225],1.5,dt)
# HPF_hvelB3 = np.subtract(hvelB3[:,225],LPF_3_hvelB3)
# HPF_hvelB3 = np.array(low_pass_filter(HPF_hvelB3,40,dt))

# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,HPF_hvelB1,"-b",label="B1")
# ax.plot(Time_OF,HPF_hvelB2,"-r",label="B2")
# ax.plot(Time_OF,HPF_hvelB3,"-g",label="B3")
# ax.grid()
# ax.legend()
# ax2=ax.twinx()
# ax2.plot(Time_OF,HPF_FBR,"-k")


# #normalize HPF FBR
# HPF_FBR_norm = HPF_FBR/(1079*((L1+L2)/L2))

# #HPF FBR calc
# dF_mag_HPF = []
# dUxB1 = []
# dUxB2 = []
# dUxB3 = []
# Time_mag_HPF = []
# FBR_HPF = []
# for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

#     it_1 = zero_crossings_index_HPF_FBR[i]
#     it_2 = zero_crossings_index_HPF_FBR[i+1]

#     Time_mag_HPF.append(Time_OF[it_1])

#     FBR_HPF.append(HPF_FBR_norm[it_1])

#     dF_mag_HPF.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1]))
#     dUxB1.append(abs(HPF_hvelB1[it_2]-HPF_hvelB1[it_1]))
#     dUxB2.append(abs(HPF_hvelB2[it_2]-HPF_hvelB2[it_1]))
#     dUxB3.append(abs(HPF_hvelB3[it_2]-HPF_hvelB3[it_1]))

# plt.rcParams['font.size'] = 16
# out_dir=in_dir+"High_frequency_analysis/velocity_cc_FBR/"

# max_dUx = []
# for i in np.arange(0,len(dUxB1)):
#     max_dUx.append(np.max([dUxB1[i],dUxB2[i],dUxB3[i]]))


# fig = plt.figure(figsize=(14,8))
# plt.scatter(dF_mag_HPF,max_dUx)
# dF_threshold = []
# dUx_threshold = []
# for i in np.arange(0,len(dF_mag_HPF)):
#     if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
#         dF_threshold.append(dF_mag_HPF[i]); dUx_threshold.append(max_dUx[i])
# plt.scatter(dF_threshold,dUx_threshold,color="blue")
# cc = round(correlation_coef(dF_threshold,dUx_threshold),2)
# plt.xlabel("$|dF_B|$ [kN]")
# plt.ylabel("$max[|du_{x',i}|]$ B1,B2,B3 [m/s]")
# plt.title("correlation coefficient $|dF_B|$"+",$max[|du_{x',i}|]$"+" = {}".format(cc))
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"scatter_dF_max_HPF_dUx.png")
# plt.close()

# Time_mag_HPF_interp = np.linspace(Time_mag_HPF[0],Time_mag_HPF[-1],len(Time_OF))
# f = interpolate.interp1d(Time_mag_HPF,dF_mag_HPF)
# dF_mag_HPF_interp = f(Time_mag_HPF_interp)
# f = interpolate.interp1d(Time_mag_HPF,max_dUx)
# max_dUx_interp = f(Time_mag_HPF_interp)

# cc1 = round(correlation_coef(max_dUx,dF_mag_HPF),2)
# #cc2 = round(correlation_coef(HPF_FBR,HPF_I_75),2)
# fig,ax1 = plt.subplots(figsize=(14,8),sharex=True)
# ax1.plot(Time_mag_HPF,dF_mag_HPF,"-or",markersize=3)
# ax1.set_ylabel("$|dF_B|$ [kN]")
# ax2=ax1.twinx()
# ax2.plot(Time_mag_HPF,max_dUx,"-ob",markersize=3)
# ax2.set_ylabel("$max[|du_{x',i}|]$ B1,B2,B3 [m/s]")
# ax1.grid()
# ax1.set_title("correlation coefficient $|dF_B|$"+",$max[|du_{x',i}|]$"+" = {}\ncorrelation coefficient HPF $F_B$,HPF $I_B$ = {}".format(cc1,0.64))
# fig.supxlabel("Time [s]")

# # idx = np.searchsorted(Time_mag_HPF_interp,Time_mag_HPF_interp[0]+10)
# # cc = []
# # for it in np.arange(0,len(Time_mag_HPF_interp)-idx):
# #     cc.append(correlation_coef(dF_mag_HPF_interp[it:it+idx],max_dUx_interp[it:it+idx]))

# # ax3.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)],cc,"-k")
# # ax3.set_ylabel("Local correlation T=10s")
# # ax3.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"cc_max_HPF_dUx_FB.png")
# plt.close()



R = np.linspace(0,63,300)

def actuator_asymmetry_calc_75(it):
    hvelB1 = np.add(np.cos(np.radians(29))*uvelB1[it], np.sin(np.radians(29))*vvelB1[it])
    IyB1 = hvelB1*R*np.cos(np.radians(Azimuth[it]))
    IyB1_75 = IyB1[j]
    IzB1 = hvelB1*R*np.sin(np.radians(Azimuth[it]))
    IzB1_75 = IzB1[j]

    hvelB2 = np.add(np.cos(np.radians(29))*uvelB2[it], np.sin(np.radians(29))*vvelB2[it])
    AzB2 = Azimuth[it] + 120
    if AzB2 >= 360:
        AzB2-=360

    IyB2 = hvelB2*R*np.cos(np.radians(AzB2))
    IzB2 = hvelB2*R*np.sin(np.radians(AzB2))
    IyB2_75 = IyB2[j]
    IzB2_75 = IzB2[j]

    hvelB3 = np.add(np.cos(np.radians(29))*uvelB3[it], np.sin(np.radians(29))*vvelB3[it])
    AzB3 = Azimuth[it] + 240
    if AzB3 >= 360:
        AzB3-=360

    IyB3 = hvelB3*R*np.cos(np.radians(AzB3))
    IzB3 = hvelB3*R*np.sin(np.radians(AzB3))
    IyB3_75 = IyB3[j]
    IzB3_75 = IzB3[j]

    return IyB1_75+IyB2_75+IyB3_75, IzB1_75+IzB2_75+IzB3_75




#analysis options
High_frequency_analysis = False
planar_asymmetry_analysis = False
velocity_analysis = False
cc_radius = False


if cc_radius == True:
    cc = []
    for j in np.linspace(0,299,63,dtype=int):
        Iy_75 = []
        Iz_75 = []
        ix=0
        with Pool() as pool:
            for Iy_75_it, Iz_it_75 in pool.imap(actuator_asymmetry_calc_75,np.arange(0,len(Time))):
                Iy_75.append(Iy_75_it); Iz_75.append(Iz_it_75)
                print(ix)
                ix+=1

        I_75 = np.sqrt(np.add(np.square(Iy_75),np.square(Iz_75)))

        cc.append(correlation_coef(I_75,RtAeroMR))

    x = np.linspace(0,63,63,dtype=int)
    out_dir=in_dir+"High_frequency_analysis/"
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(14,8))
    plt.plot(x,cc)
    plt.xlabel("Blade span [m]")
    plt.ylabel("Correlation coefficient Actuator asymmetry\n,OOPBM")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"ccIB_MR_R.png")
    plt.close()


if velocity_analysis == True:

    dA = 0.3125*0.3125
    Iy = []
    Iz = []
    Iy_75 = []
    Iz_75 = []
    IB1_arr = []
    IB2_arr = []
    IB3_arr = []
    IyB1_arr = []
    IyB2_arr = []
    IyB3_arr = []
    IzB1_arr = []
    IzB2_arr = []
    IzB3_arr = []
    ix=0
    with Pool() as pool:
        for Iy_it, Iz_it, Iy_75_it, Iz_it_75,IB1_it,IB2_it,IB3_it,IyB1_75_it,IyB2_75_it,IyB3_75_it,IzB1_75_it,IzB2_75_it,IzB3_75_it in pool.imap(actuator_asymmetry_calc,np.arange(0,len(Time))):
            Iy.append(Iy_it); Iz.append(Iz_it); Iy_75.append(Iy_75_it); Iz_75.append(Iz_it_75); IB1_arr.append(IB1_it); IB2_arr.append(IB2_it); IB3_arr.append(IB3_it)
            IyB1_arr.append(IyB1_75_it); IyB2_arr.append(IyB2_75_it); IyB3_arr.append(IyB3_75_it); IzB1_arr.append(IzB1_75_it); IzB2_arr.append(IzB2_75_it); IzB3_arr.append(IzB3_75_it)
            print(ix)
            ix+=1

    I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

    #Filtering IB
    LPF_3_I = low_pass_filter(I,1.5,dt)
    HPF_I = np.subtract(I,LPF_3_I)
    HPF_I = np.array(low_pass_filter(HPF_I,40,dt))

    I_75 = np.sqrt(np.add(np.square(Iy_75),np.square(Iz_75)))
    LPF_3_I_75 = low_pass_filter(I_75,1.5,dt)
    HPF_I_75 = np.subtract(I_75,LPF_3_I_75)
    HPF_I_75 = np.array(low_pass_filter(HPF_I_75,40,dt))

    #HPF calc
    dF_mag_HPF = []
    Time_mag_HPF = []
    MR_HPF = []
    IB_HPF = []
    for i in np.arange(0,len(zero_crossings_index_HPF_MR)-1):

        it_1 = zero_crossings_index_HPF_MR[i]
        it_2 = zero_crossings_index_HPF_MR[i+1]

        Time_mag_HPF.append(Time_OF[it_1])

        MR_HPF.append(HPF_MR[it_1])

        IB_HPF.append(HPF_I[it_1])

        dF_mag_HPF.append(HPF_MR[it_2] - HPF_MR[it_1])

    #normalize HPF FBR
    HPF_FBR_norm = HPF_FBR/(1079*((L1+L2)/L2))

    #HPF FBR calc
    dF_mag_HPF = []
    Time_mag_HPF = []
    FBR_HPF = []
    for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

        it_1 = zero_crossings_index_HPF_FBR[i]
        it_2 = zero_crossings_index_HPF_FBR[i+1]

        Time_mag_HPF.append(Time_OF[it_1])

        FBR_HPF.append(HPF_FBR_norm[it_1])

        dF_mag_HPF.append(HPF_FBR[it_2] - HPF_FBR[it_1])


    Time_OF = Time_OF-4.6
    Time_mag_HPF = np.subtract(Time_mag_HPF,4.6)

    out_dir=in_dir+"High_frequency_analysis/velocity_FBR/"
    plt.rcParams['font.size'] = 16

    cc = round(correlation_coef(HPF_I_75,HPF_FBR),2)

    times = np.arange(200-4.6,1210-4.6,10)

    for j in np.arange(0,len(times)-1):

        it_1 = np.searchsorted(Time_OF,times[j])
        it_2 = np.searchsorted(Time_OF,times[j+1])

        fig,(ax,ax3)=plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax.plot(Time_OF[it_1:it_2],IyB1_arr[it_1:it_2],"-b",label="B1")
        ax.plot(Time_OF[it_1:it_2],IyB2_arr[it_1:it_2],"-r",label="B2")
        ax.plot(Time_OF[it_1:it_2],IyB3_arr[it_1:it_2],"-g",label="B3")
        ax.plot(Time_OF[it_1:it_2],HPF_I_75[it_1:it_2],"--k",label="HPF $I_B$")
        ax.set_ylabel("Asymmetry y 75% span\nlocation [m/s]")
        ax.legend()
        ax2=ax.twinx()
        ax2.set_ylabel("HPF Bearing force magntiude\nNormalized on rotor weight [-]")
        ax2.grid()
        ax2.plot(Time_OF[it_1:it_2],HPF_FBR_norm[it_1:it_2],"-k")
        ax.set_title("correlation coefficient HPF $I_B$, HPF FB = {}".format(cc))
        for i in np.arange(0,len(dF_mag_HPF)):
            if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
                if times[j] <= Time_mag_HPF[i] <= times[j+1]:
                    ax2.plot(Time_mag_HPF[i],FBR_HPF[i],"ob")

        ax3.plot(Time_OF[it_1:it_2],hvelB1[it_1:it_2,225],"-b",label="B1")
        ax3.plot(Time_OF[it_1:it_2],hvelB2[it_1:it_2,225],"-r",label="B2")
        ax3.plot(Time_OF[it_1:it_2],hvelB3[it_1:it_2,225],"-g",label="B3")
        ax3.set_ylabel("Streamwise velocity [m/s]")
        ax3.legend()
        ax4=ax3.twinx()
        ax4.set_ylabel("HPF Bearing force magnitude\nNormalized on rotor weight [-]")
        ax4.grid()
        ax4.plot(Time_OF[it_1:it_2],HPF_FBR_norm[it_1:it_2],"-k")
        for i in np.arange(0,len(dF_mag_HPF)):
            if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
                if times[j] <= Time_mag_HPF[i] <= times[j+1]:
                    ax4.plot(Time_mag_HPF[i],FBR_HPF[i],"ob")

        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(out_dir+"Iy_{}_{}.png".format(times[j]+4.6,times[j+1]+4.6))
        plt.close()


        fig,(ax,ax3)=plt.subplots(2,1,figsize=(14,8),sharex=True)
        ax.plot(Time_OF[it_1:it_2],IzB1_arr[it_1:it_2],"-b",label="B1")
        ax.plot(Time_OF[it_1:it_2],IzB2_arr[it_1:it_2],"-r",label="B2")
        ax.plot(Time_OF[it_1:it_2],IzB3_arr[it_1:it_2],"-g",label="B3")
        ax.plot(Time_OF[it_1:it_2],HPF_I_75[it_1:it_2],"--k",label="HPF $I_B$")
        ax.set_ylabel("Asymmetry z 75% span\nlocation [m/s]")
        ax.set_title("correlation coefficient HPF $I_B$, HPF OOPBM = {}".format(cc))
        ax.legend()
        ax2=ax.twinx()
        ax2.set_ylabel("HPF Bearing force magnitude\nNormalized on rotor weight [-]")
        ax2.grid()
        ax2.plot(Time_OF[it_1:it_2],HPF_FBR_norm[it_1:it_2],"-k")
        for i in np.arange(0,len(dF_mag_HPF)):
            if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
                if times[j] <= Time_mag_HPF[i] <= times[j+1]:
                    ax2.plot(Time_mag_HPF[i],FBR_HPF[i],"ob")

        ax3.plot(Time_OF[it_1:it_2],hvelB1[it_1:it_2,225],"-b",label="B1")
        ax3.plot(Time_OF[it_1:it_2],hvelB2[it_1:it_2,225],"-r",label="B2")
        ax3.plot(Time_OF[it_1:it_2],hvelB3[it_1:it_2,225],"-g",label="B3")
        ax3.set_ylabel("Streamwise velocity [m/s]")
        ax3.legend()
        ax4=ax3.twinx()
        ax4.set_ylabel("HPF Bearing force magnitude\nNormalized on rotor weight [-]")
        ax4.grid()
        ax4.plot(Time_OF[it_1:it_2],HPF_FBR_norm[it_1:it_2],"-k")
        for i in np.arange(0,len(dF_mag_HPF)):
            if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
                if times[j] <= Time_mag_HPF[i] <= times[j+1]:
                    ax4.plot(Time_mag_HPF[i],FBR_HPF[i],"ob")

        fig.supxlabel("Time [s]")
        plt.tight_layout()
        plt.savefig(out_dir+"Iz_{}_{}.png".format(times[j]+4.6,times[j+1]+4.6))
        plt.close()

    #plt.show()



if planar_asymmetry_analysis == True:

    a = Dataset(in_dir+"Dataset_Planar_asymmetry.nc")

    Planar_asymmetry = a.groups["Planar_Asymmetry_Variables"]

    offsets = [5.5,63.0]

    for offset in offsets:
        Planar_asymmetry_P = Planar_asymmetry.groups["{}".format(offset)]

        IyP = np.array(Planar_asymmetry_P.variables["Iy"][Tstart_idx:T_end_idx])
        IzP = -np.array(Planar_asymmetry_P.variables["Iz"][Tstart_idx:T_end_idx])
        IP = np.sqrt(np.add(np.square(IyP),np.square(IzP)))

        if offset == 63.0:
            Time_shift_idx = np.searchsorted(Time,Time[0]+4.6)
            Time = Time[:-Time_shift_idx]

            Iy = Iy[Time_shift_idx:]
            Iz = Iz[Time_shift_idx:]
            I = I[Time_shift_idx:]

            IyP = IyP[:-Time_shift_idx]
            IzP = IzP[:-Time_shift_idx]
            IP = IP[:-Time_shift_idx]


        #Filtering I
        LPF_1_I = low_pass_filter(I,0.3,dt)
        LPF_2_I = low_pass_filter(I,0.9,dt)
        LPF_3_I = low_pass_filter(I,1.5,dt)
        BPF_I = np.subtract(LPF_2_I,LPF_1_I)
        HPF_I = np.subtract(I,LPF_3_I)
        HPF_I = np.array(low_pass_filter(HPF_I,40,dt))


        #Filtering I
        LPF_1_IP = low_pass_filter(IP,0.3,dt)
        LPF_2_IP = low_pass_filter(IP,0.9,dt)
        LPF_3_IP = low_pass_filter(IP,1.5,dt)
        BPF_IP = np.subtract(LPF_2_IP,LPF_1_IP)
        HPF_IP = np.subtract(IP,LPF_3_IP)
        HPF_IP = np.array(low_pass_filter(HPF_IP,40,dt))


        out_dir=in_dir+"High_frequency_analysis/planar_blade_asymmetry/"

        plt.rcParams.update({'font.size': 18})

        print("{} IyP {}".format(offset,moments(IyP)))
        print("{} Iy {}".format(offset,moments(Iy)))

        cc = round(correlation_coef(IyP,Iy),2)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,IyP,"-r",label="Blade asymmetry\nfrom planar data")
        plt.plot(Time,Iy,"-b",label="Blade asymmetry\nfrom actuator data")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around y axis [$m^2/s$]")
        plt.legend()
        plt.title("correlation coefficient = {}".format(cc))
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"Iy_{}.png".format(offset))
        plt.close()

        fig = plt.figure(figsize=(14,8))
        frq,PSD = temporal_spectra(IyP,dt,Var="IyP")
        plt.loglog(frq,PSD,"-r",label="Blade asymmetry\nfrom planar data")
        frq,PSD = temporal_spectra(Iy,dt,Var="Iy")
        plt.loglog(frq,PSD,"-b",label="Blade asymmetry\nfrom actuator data")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Asymmetry around y axis [$m^2/s$]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"spectra_Iy_{}.png".format(offset))
        plt.close()

        print("{} IzP {}".format(offset,moments(IzP)))
        print("{} Iz {}".format(offset,moments(Iz)))

        cc = round(correlation_coef(IzP,Iz),2)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,IzP,"-r",label="Blade asymmetry\nfrom planar data")
        plt.plot(Time,Iz,"-b",label="Blade asymmetry\nfrom actuator data")
        plt.xlabel("Time [s]")
        plt.ylabel("Asymmetry around z axis [$m^2/s$]")
        plt.title("correlation coefficient = {}".format(cc))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"Iz_{}.png".format(offset))
        plt.close()

        fig = plt.figure(figsize=(14,8))
        frq,PSD = temporal_spectra(IzP,dt,Var="IzP")
        plt.loglog(frq,PSD,"-r",label="Blade asymmetry\nfrom planar data")
        frq,PSD = temporal_spectra(Iz,dt,Var="Iz")
        plt.loglog(frq,PSD,"-b",label="Blade asymmetry\nfrom actuator data")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Asymmetry around z axis [$m^2/s$]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"spectra_Iz_{}.png".format(offset))
        plt.close()


        print("{} IP {}".format(offset,moments(IP)))
        print("{} I {}".format(offset,moments(I)))

        cc = round(correlation_coef(IP,I),2)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,IP,"-r",label="Blade asymmetry\nfrom planar data")
        plt.plot(Time,I,"-b",label="Blade asymmetry\nfrom actuator data")
        plt.xlabel("Time [s]")
        plt.ylabel("Magnitude Asymmetry [$m^2/s$]")
        plt.title("correlation coefficient = {}".format(cc))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"I_{}.png".format(offset))
        plt.close()

        print("{} LPF IP {}".format(offset,moments(LPF_1_IP)))
        print("{} LPF I {}".format(offset,moments(LPF_1_I)))

        cc = round(correlation_coef(LPF_1_IP,LPF_1_I),2)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,LPF_1_IP,"-r",label="LPF 0.3Hz Blade asymmetry\nfrom planar data")
        plt.plot(Time,LPF_1_I,"-b",label="LPF 0.3Hz Blade asymmetry\nfrom actuator data")
        plt.xlabel("Time [s]")
        plt.ylabel("LPF 0.3Hz Magnitude Asymmetry [$m^2/s$]")
        plt.title("correlation coefficient = {}".format(cc))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"LPF_I_{}.png".format(offset))
        plt.close()
                
        print("{} BPF IP {}".format(offset,moments(BPF_IP)))
        print("{} BPF I {}".format(offset,moments(BPF_I)))

        cc = round(correlation_coef(BPF_IP,BPF_I),2)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,BPF_IP,"-r",label="BPF 0.3-0.9Hz Blade asymmetry\nfrom planar data")
        plt.plot(Time,BPF_I,"-b",label="BPF 0.3-0.9Hz Blade asymmetry\nfrom actuator data")
        plt.xlabel("Time [s]")
        plt.ylabel("BPF 0.3-0.9Hz Magnitude Asymmetry [$m^2/s$]")
        plt.title("correlation coefficient = {}".format(cc))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"BPF_I_{}.png".format(offset))
        plt.close()


        print("{} HPF IP {}".format(offset,moments(HPF_IP)))
        print("{} HPF I {}".format(offset,moments(HPF_I)))
        
        cc = round(correlation_coef(HPF_IP,HPF_I),2)
        fig = plt.figure(figsize=(14,8))
        plt.plot(Time,HPF_IP,"-r",label="HPF 1.5Hz Blade asymmetry\nfrom planar data")
        plt.plot(Time,HPF_I,"-b",label="HPF 1.5Hz Blade asymmetry\nfrom actuator data")
        plt.xlabel("Time [s]")
        plt.ylabel("HPF 1.5Hz Magnitude Asymmetry [$m^2/s$]")
        plt.title("correlation coefficient = {}".format(cc))
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"HPF_I_{}.png".format(offset))
        plt.close()


        fig = plt.figure(figsize=(14,8))
        frq,PSD = temporal_spectra(IP,dt,Var="IP")
        plt.loglog(frq,PSD,"-r",label="Blade asymmetry\nfrom planar data")
        frq,PSD = temporal_spectra(I,dt,Var="I")
        plt.loglog(frq,PSD,"-b",label="Blade asymmetry\nfrom actuator data")
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Magnitude Asymmetry [$m^2/s$]")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"spectra_I_{}.png".format(offset))
        plt.close()



if High_frequency_analysis == True:
    out_dir=in_dir+"High_frequency_analysis/"
    plt.rcParams.update({'font.size': 18})

    cc = round(correlation_coef(I,I_75),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time,I,"-b")
    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("Magnitude Asymmetry vector [$m^2/s$] $I_B$")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Time,I_75,"-r")
    ax2.set_ylabel("Magnitude Asymmetry vector at 75% span [$m^2/s$] $I_B$")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"cc_IB_I_75.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(I_75,dt,Var="I_75")
    plt.loglog(frq,PSD,"-r",label="$I_B$ 75% span")
    frq,PSD = temporal_spectra(I,dt,Var="I_B")
    plt.loglog(frq,PSD,"-b",label="$I_B$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title("Spectra $I_B at 47m span [m^2/s]\,I_B [m^2/s]$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"spectra_IB_IB_75.png")
    plt.close()

    cc = round(correlation_coef(Iy,RtAeroMys),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time,Iy,"-b")
    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("Asymmetry around y axis [$m^4/s$] $I_{B_y}$")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Time,LSSTipMys,"-r")
    ax2.set_ylabel("Rotor Aerodynamic moment y component [kN-m]")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"cc_Iy_My.png")
    plt.close()

    cc = round(correlation_coef(Iz,RtAeroMzs),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time,Iz,"-b")
    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("Asymmetry around z axis [$m^4/s$] $I_{B_z}$")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Time,LSSTipMzs,"-r")
    ax2.set_ylabel("Rotor aerodynamic moment z component [kN-m]")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"cc_Iz_Mz.png")
    plt.close()

    cc = round(correlation_coef(I,RtAeroMR),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time,I,"-b")
    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("Magnitude Asymmetry vector [$m^4/s$] $I_B$")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Time,RtAeroMR,"-r")
    ax2.set_ylabel("Magnitude Aerodynamic Rotor moment [kN-m]")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"cc_I_MR.png")
    plt.close()

    cc = round(correlation_coef(I_75,RtAeroMR),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time,I_75,"-b")
    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("Magnitude Asymmetry vector at 75% span [$m^2/s$] $I_B$")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Time,RtAeroMR,"-r")
    ax2.set_ylabel("Magnitude Aerodynamic Rotor moment [kN-m]")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"cc_IB_75_MR.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(RtAeroMR,dt,Var="MR")
    plt.loglog(frq,PSD,"-r",label="OOPBM")
    frq,PSD = temporal_spectra(FBR,dt,Var="MR")
    plt.loglog(frq,np.multiply(PSD,1e+05),"-g",label="$F_{B_R}$")
    frq,PSD = temporal_spectra(I,dt,Var="I_B")
    plt.loglog(frq,np.true_divide(PSD,1e+05),"-b",label="$I_B$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title("Spectra OOPBM $[kN-m]\,F_{B_R} [kN] \,I_B [m^4/s]$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"spectra_IB_MR_FBR.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(I_3,dt_sampling,Var="I")
    plt.loglog(frq,PSD,"-r",label="I")
    frq,PSD = temporal_spectra(I,dt,Var="I_B")
    plt.loglog(frq,PSD,"-b",label="$I_B$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title("Spectra $I [m^4/s]\, I_B [m^4/s]$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"spectra_IB_I.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    frq,PSD = temporal_spectra(I_3,dt_sampling,Var="I")
    plt.loglog(frq,np.true_divide(PSD,2000),"-r",label="I")
    frq,PSD = temporal_spectra(I,dt,Var="I_B")
    plt.loglog(frq,np.multiply(PSD,100),"-b",label="$I_B$")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("PSD")
    plt.title("Spectra $I [m^4/s]\, I_B [m^4/s]$")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+"spectra_IB_I_2.png")
    plt.close()

    #filtering 
    LPF_1_MR = low_pass_filter(RtAeroMR,0.3,dt)
    LPF_2_MR = low_pass_filter(RtAeroMR,0.9,dt)
    LPF_3_MR = low_pass_filter(RtAeroMR,1.5,dt)

    HPF_MR = np.subtract(RtAeroMR,LPF_3_MR)
    HPF_MR = np.array(low_pass_filter(HPF_MR,40,dt))
    BPF_MR = np.subtract(LPF_2_MR,LPF_1_MR)

    LPF_1_I = low_pass_filter(I,0.3,dt)
    LPF_2_I = low_pass_filter(I,0.9,dt)
    LPF_3_I = low_pass_filter(I,1.5,dt)

    HPF_I = np.subtract(I,LPF_3_I)
    HPF_I = np.array(low_pass_filter(HPF_I,40,dt))
    BPF_I = np.subtract(LPF_2_I,LPF_1_I)


    cc = round(correlation_coef(LPF_1_MR,LPF_1_I),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time,LPF_1_I,"-b")
    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("LPF 0.3Hz Magnitude Asymmetry vector [$m^4/s$] $I_B$")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Time,LPF_1_MR,"-r")
    ax2.set_ylabel("LPF 0.3Hz Magnitude Rotor moment [kN-m]")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"LPF_cc_I_MR.png")
    plt.close()

    cc = round(correlation_coef(BPF_MR,BPF_I),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time,BPF_I,"-b")
    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("BPF 0.3-0.9Hz Magnitude Asymmetry vector [$m^4/s$] $I_B$")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Time,BPF_MR,"-r")
    ax2.set_ylabel("BPF 0.3-0.9Hz Magnitude Rotor moment [kN-m]")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"BPF_cc_I_MR.png")
    plt.close()

    cc = round(correlation_coef(HPF_MR,HPF_I),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time,HPF_I,"-b")
    ax.yaxis.label.set_color('blue') 
    ax.set_ylabel("HPF 1.5-40Hz Magnitude Asymmetry vector [$m^4/s$] $I_B$")
    ax.grid()
    ax2=ax.twinx()
    ax2.plot(Time,HPF_MR,"-r")
    ax2.set_ylabel("HPF 1.5-40Hz Magnitude Rotor moment [kN-m]")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    fig.supxlabel("Time [s]")
    plt.tight_layout()
    plt.savefig(out_dir+"HPF_cc_I_MR.png")
    plt.close()


    times = np.arange(200,1300,100)
    for i in np.arange(0,len(times)-1):
        idx1 = np.searchsorted(Time,times[i])
        idx2 = np.searchsorted(Time,times[i+1])
        fig,ax = plt.subplots(figsize=(14,8))
        #ax2=ax.twinx()

        ax.plot(Time[idx1:idx2],I[idx1:idx2],"-k",label="$I_B,tot$")

        #ax2.plot(Time[idx1:idx2],RtAeroMR[idx1:idx2],"--k",label="OOPBM")

        ax.plot(Time[idx1:idx2],LPF_1_I[idx1:idx2],"-b",label="LPF $I_B$")
        
        #ax2.plot(Time[idx1:idx2],LPF_1_MR[idx1:idx2],"--b",label="LPF OOPBM")

        ax.plot(Time[idx1:idx2],BPF_I[idx1:idx2],"-r",label="BPF $I_B$")

        #ax2.plot(Time[idx1:idx2],BPF_MR[idx1:idx2],"--r",label="BPF OOPBM")

        ax.plot(Time[idx1:idx2],np.subtract(HPF_I[idx1:idx2],10000),"-g",label="HPF $I_B -10,000m^2/s$")

        #ax2.plot(Time[idx1:idx2],HPF_MR[idx1:idx2],"--g",label="HPF OOPBM")

        ax.yaxis.label.set_color('blue') 
        ax.set_ylabel("Magnitude Asymmetry vector [$m^2/s$] $I_B$")
        ax.grid()
        #ax2.set_ylabel("Magnitude aerodynamic Rotor moment [kN-m]")
        #ax2.yaxis.label.set_color('red') 
        fig.supxlabel("Time [s]")
        ax.legend(loc="upper right")
        #ax2.legend(loc="upper right")
        plt.tight_layout()
        plt.savefig(out_dir+"I_all_times/I_{}_{}.png".format(times[i],times[i+1]))
        plt.close()


    f = interpolate.interp1d(Time,LPF_1_I)
    LPF_1_I_interp = f(Time_sampling)
    cc = round(correlation_coef(LPF_1_I_interp,I_2_LPF),2)
    fig,ax = plt.subplots(figsize=(14,8))
    ax.plot(Time_sampling,LPF_1_I_interp,"-b")
    ax.set_ylabel("LPF 0.3Hz Actuator Magnitude Asymmetry vector [$m^4/s$] $I_B$")
    ax.yaxis.label.set_color('blue') 
    ax2=ax.twinx()
    ax2.plot(Time_sampling,I_2_LPF,"-r")
    ax2.set_ylabel("LPF 0.3Hz Magnitude Asymmetry vector [$m^4/s$] $I$")
    ax2.yaxis.label.set_color('red') 
    fig.suptitle("correlation coefficient = {}".format(cc))
    ax.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"cc_I_I2.png")
    plt.close()

    times = np.arange(200,1300,100)
    for i in np.arange(0,len(times)-1):
        idx1 = np.searchsorted(Time,times[i])
        idx2 = np.searchsorted(Time,times[i+1])
        fig,ax = plt.subplots(figsize=(14,8))
        ax.plot(Time[idx1:idx2],I[idx1:idx2],"-b",label="$I_{B,tot}$")
        idx1 = np.searchsorted(Time_sampling,times[i])
        idx2 = np.searchsorted(Time_sampling,times[i+1])
        ax.plot(Time_sampling[idx1:idx2],LPF_1_I_interp[idx1:idx2],"-k",label="LPF $I_B$")
        ax.set_ylabel("Actuator Magnitude Asymmetry vector [$m^4/s$] $I_B$")
        ax.legend()
        ax.yaxis.label.set_color('blue') 
        ax2=ax.twinx()
        ax2.plot(Time_sampling[idx1:idx2],I_2[idx1:idx2],"-r")
        ax2.set_ylabel("Magnitude Asymmetry vector [$m^4/s$] $I$")
        ax2.yaxis.label.set_color('red') 
        ax.grid()
        plt.tight_layout()
        plt.savefig(out_dir+"I_IB_all_times/cc_I_I2_{}_{}.png".format(times[i],times[i+1]))
        plt.close()



    dHPF_MR = np.array(dt_calc(HPF_MR,dt))

    zero_crossings_index_HPF_MR = np.where(np.diff(np.sign(dHPF_MR)))[0]
    #HPF calc
    dF_mag_HPF = []
    dt_mag_HPF = []
    dF_I_HPF = []
    for i in np.arange(0,len(zero_crossings_index_HPF_MR)-1):

        it_1 = zero_crossings_index_HPF_MR[i]
        it_2 = zero_crossings_index_HPF_MR[i+1]

        dF_mag_HPF.append(HPF_MR[it_2] - HPF_MR[it_1])

        dt_mag_HPF.append(Time[it_2] - Time[it_1])

        dF_I_HPF.append(HPF_I[it_2] - HPF_I[it_1])


    fig = plt.figure(figsize=(14,8))
    plt.scatter(dF_I_HPF,dF_mag_HPF)
    plt.xlabel("$dI_B [m^2/s]$")
    plt.ylabel("dF [kN-m]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dIB_dF_HPF.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_HPF,dF_mag_HPF,c=dF_I_HPF,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN-m]$")
    plt.title("HPF $M_R$")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"MR_dF_dt_HPF_I.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    P,X = probability_dist(dF_mag_HPF)
    plt.plot(X,P)
    plt.xlabel("Jump in Magnitude Rotor moment [kN-m]")
    plt.ylabel("Probabilty [-]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_dF_MR.png")
    plt.close()

    fig = plt.figure(figsize=(14,8))
    P,X = probability_dist(dt_mag_HPF)
    plt.plot(X,P)
    plt.xlabel("Time step [s]")
    plt.ylabel("Probabilty [-]")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"PDF_dt_MR.png")
    plt.close()


    #HPF calc
    dF_mag_HPF_threshold = []
    dt_mag_HPF_threshold = []
    dF_I_HPF_threshold = []
    for i in np.arange(0,len(zero_crossings_index_HPF_MR)-1):

        it_1 = zero_crossings_index_HPF_MR[i]
        it_2 = zero_crossings_index_HPF_MR[i+1]

        if HPF_MR[it_2] - HPF_MR[it_1] > 3*np.std(dF_mag_HPF) or HPF_MR[it_2] - HPF_MR[it_1] < -3*np.std(dF_mag_HPF):

            dF_mag_HPF_threshold.append(HPF_MR[it_2] - HPF_MR[it_1])

            dt_mag_HPF_threshold.append(Time[it_2] - Time[it_1])

            dF_I_HPF_threshold.append(HPF_I[it_2] - HPF_I[it_1])


    fig = plt.figure(figsize=(14,8))
    plt.scatter(dF_I_HPF_threshold,dF_mag_HPF_threshold)
    plt.xlabel("$dI_B [m^2/s]$")
    plt.ylabel("dF [kN-m]")
    plt.title("HPF $M_R$ Threshold on 3x std")
    plt.grid()
    plt.tight_layout()
    plt.savefig(out_dir+"dIB_dF_HPF_threshold.png")
    plt.close()

    print(len(dF_mag_HPF_threshold))
    fig = plt.figure(figsize=(14,8))
    plt.scatter(dt_mag_HPF_threshold,dF_mag_HPF_threshold,c=dF_I_HPF_threshold,cmap="viridis")
    plt.xlabel("$dt$ [s]")
    plt.ylabel("$dF [kN-m]$")
    plt.title("HPF $M_R$ Threshold on 3x std")
    plt.grid()
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(out_dir+"MR_dF_dt_HPF_threshold_I.png")
    plt.close()




dA = 0.3125*0.3125
Iy = []
Iz = []
Iy_75 = []
Iz_75 = []
IB1_arr = []
IB2_arr = []
IB3_arr = []
IyB1_arr = []
IyB2_arr = []
IyB3_arr = []
IzB1_arr = []
IzB2_arr = []
IzB3_arr = []
ix=0
with Pool() as pool:
    for Iy_it, Iz_it, Iy_75_it, Iz_it_75,IB1_it,IB2_it,IB3_it,IyB1_75_it,IyB2_75_it,IyB3_75_it,IzB1_75_it,IzB2_75_it,IzB3_75_it in pool.imap(actuator_asymmetry_calc,np.arange(0,len(Time))):
        Iy.append(Iy_it); Iz.append(Iz_it); Iy_75.append(Iy_75_it); Iz_75.append(Iz_it_75); IB1_arr.append(IB1_it); IB2_arr.append(IB2_it); IB3_arr.append(IB3_it)
        IyB1_arr.append(IyB1_75_it); IyB2_arr.append(IyB2_75_it); IyB3_arr.append(IyB3_75_it); IzB1_arr.append(IzB1_75_it); IzB2_arr.append(IzB2_75_it); IzB3_arr.append(IzB3_75_it)
        print(ix)
        ix+=1

I = np.sqrt(np.add(np.square(Iy),np.square(Iz)))

#Filtering IB
LPF_1_IB = low_pass_filter(I,0.3,dt)
LPF_2_IB = low_pass_filter(I,0.9,dt)
LPF_3_IB = low_pass_filter(I,1.5,dt)
BPF_IB = np.subtract(LPF_2_IB,LPF_1_IB)
HPF_IB = np.subtract(I,LPF_3_IB)
HPF_IB = np.array(low_pass_filter(HPF_IB,40,dt))

# out_dir=in_dir+"Blade_asymmetry_analysis/"
# plt.rcParams['font.size'] = 16
# cc = round(correlation_coef(I,RtAeroMR),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,I,"-b")
# ax.grid()
# ax.set_ylabel("Blade Asymmetry [$m^2/s$]")
# ax2=ax.twinx()
# ax2.plot(Time_OF,RtAeroMR,"-r")
# ax2.set_ylabel("Out-of-plane bending moment magnitude [kN-m]")
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficent = {}".format(cc))
# plt.tight_layout()
# plt.savefig(out_dir+"cc_IB_MR.png")
# plt.close()

# cc = round(correlation_coef(LPF_1_IB,LPF_1_MR),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,LPF_1_IB,"-b")
# ax.grid()
# ax.set_ylabel("LPF Blade Asymmetry [$m^2/s$]")
# ax2=ax.twinx()
# ax2.plot(Time_OF,LPF_1_MR,"-r")
# ax2.set_ylabel("LPF Out-of-plane bending moment magnitude [kN-m]")
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficent = {}".format(cc))
# plt.tight_layout()
# plt.savefig(out_dir+"cc_LPF_IB_MR.png")
# plt.close()

# cc = round(correlation_coef(BPF_IB,BPF_MR),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,BPF_IB,"-b")
# ax.grid()
# ax.set_ylabel("BPF Blade Asymmetry [$m^2/s$]")
# ax2=ax.twinx()
# ax2.plot(Time_OF,BPF_MR,"-r")
# ax2.set_ylabel("BPF Out-of-plane bending moment magnitude [kN-m]")
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficent = {}".format(cc))
# plt.tight_layout()
# plt.savefig(out_dir+"cc_BPF_IB_MR.png")
# plt.close()

# cc = round(correlation_coef(HPF_IB,HPF_MR),2)
# fig,ax = plt.subplots(figsize=(14,8))
# ax.plot(Time_OF,HPF_IB,"-b")
# ax.grid()
# ax.set_ylabel("HPF Blade Asymmetry [$m^2/s$]")
# ax2=ax.twinx()
# ax2.plot(Time_OF,HPF_MR,"-r")
# ax2.set_ylabel("HPF Out-of-plane bending moment magnitude [kN-m]")
# fig.supxlabel("Time [s]")
# fig.suptitle("Correlation coefficent = {}".format(cc))
# plt.tight_layout()
# plt.savefig(out_dir+"cc_HPF_IB_MR.png")
# plt.close()

# fig = plt.figure(figsize=(14,8))
# frq,PSD = temporal_spectra(I,dt,Var="IB")
# plt.loglog(frq,PSD,label="Blade Asymmetry")
# frq,PSD = temporal_spectra(RtAeroMR,dt,Var="MR")
# plt.loglog(frq,PSD,label="OOPBM")
# plt.xlabel("Frequency [Hz]")
# plt.ylabel("PSD")
# plt.legend()
# plt.grid()
# plt.tight_layout()
# plt.savefig(out_dir+"spectra_IB_MR.png")
# plt.close()


#Filtering IB 75
I_75 = np.sqrt(np.add(np.square(Iy_75),np.square(Iz_75)))
LPF_3_I_75 = low_pass_filter(I_75,1.5,dt)
HPF_I_75 = np.subtract(I_75,LPF_3_I_75)
HPF_I_75 = np.array(low_pass_filter(HPF_I_75,40,dt))

cc2 = round(correlation_coef(HPF_I_75,HPF_FBR))


#normalize HPF FBR
HPF_FBR_norm = HPF_FBR/(1079*((L1+L2)/L2))

#HPF FBR calc
dF_mag_HPF = []
dUxB1 = []
dUxB2 = []
dUxB3 = []
Time_mag_HPF = []
FBR_HPF = []
for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

    it_1 = zero_crossings_index_HPF_FBR[i]
    it_2 = zero_crossings_index_HPF_FBR[i+1]

    Time_mag_HPF.append(Time_OF[it_1])

    FBR_HPF.append(HPF_FBR_norm[it_1])

    dF_mag_HPF.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1]))
    dUxB1.append(hvelB1[it_2,225]-hvelB1[it_1,225])
    dUxB2.append(hvelB2[it_2,225]-hvelB2[it_1,225])
    dUxB3.append(hvelB3[it_2,225]-hvelB3[it_1,225])


max_dUx = []
for i in np.arange(0,len(dUxB1)):
    abs_max_dUx = np.max([abs(dUxB1[i]),abs(dUxB2[i]),abs(dUxB3[i])])
    if abs(dUxB1[i]) == abs_max_dUx:
        max_dUx.append(dUxB1[i])
    elif abs(dUxB2[i]) == abs_max_dUx:
        max_dUx.append(dUxB2[i])
    elif abs(dUxB3[i]) == abs_max_dUx:
        max_dUx.append(dUxB3[i])

# plt.plot(Time_mag_HPF,max_dUx)
# plt.show()


plt.rcParams['font.size'] = 16
out_dir=in_dir+"High_frequency_analysis/velocity_cc_FBR/"

max_dUx = []
for i in np.arange(0,len(dUxB1)):
    max_dUx.append(np.max([abs(dUxB1[i]),abs(dUxB2[i]),abs(dUxB3[i])]))


fig = plt.figure(figsize=(14,8))
plt.scatter(dF_mag_HPF,max_dUx)
dF_threshold = []
dUx_threshold = []
for i in np.arange(0,len(dF_mag_HPF)):
    if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
        dF_threshold.append(dF_mag_HPF[i]); dUx_threshold.append(max_dUx[i])
plt.scatter(dF_threshold,dUx_threshold,color="blue")
cc = round(correlation_coef(dF_threshold,dUx_threshold),2)
plt.xlabel("$|dF_B|$ [kN]")
plt.ylabel("$max[|du_{x',i}|]$ B1,B2,B3 [m/s]")
plt.title("correlation coefficient $|dF_B|$"+",$max[|du_{x',i}|]$"+" = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"scatter_dF_max_dUx.png")
plt.close()

Time_mag_HPF_interp = np.linspace(Time_mag_HPF[0],Time_mag_HPF[-1],len(Time_OF))
f = interpolate.interp1d(Time_mag_HPF,dF_mag_HPF)
dF_mag_HPF_interp = f(Time_mag_HPF_interp)
f = interpolate.interp1d(Time_mag_HPF,max_dUx)
max_dUx_interp = f(Time_mag_HPF_interp)

cc1 = round(correlation_coef(max_dUx,dF_mag_HPF),2)
cc2 = round(correlation_coef(HPF_FBR,HPF_I_75),2)
fig,(ax1,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax1.plot(Time_mag_HPF,dF_mag_HPF,"-or",markersize=3)
ax1.set_ylabel("$|dF_B|$ [kN]")
ax2=ax1.twinx()
ax2.plot(Time_mag_HPF,max_dUx,"-ob",markersize=3)
ax2.set_ylabel("$max[|du_{x',i}|]$ B1,B2,B3 [m/s]")
ax1.grid()
ax1.set_title("correlation coefficient $|dF_B|$"+",$max[|du_{x',i}|]$"+" = {}\ncorrelation coefficient HPF $F_B$,HPF $I_B$ = {}".format(cc1,cc2))
fig.supxlabel("Time [s]")

idx = np.searchsorted(Time_mag_HPF_interp,Time_mag_HPF_interp[0]+10)
cc = []
for it in np.arange(0,len(Time_mag_HPF_interp)-idx):
    cc.append(correlation_coef(dF_mag_HPF_interp[it:it+idx],max_dUx_interp[it:it+idx]))

ax3.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)],cc,"-k")
ax3.set_ylabel("Local correlation T=10s")
ax3.grid()
plt.tight_layout()
plt.savefig(out_dir+"cc_max_dUx_FB.png")
plt.close()


# cc1 = round(correlation_coef(max_dUx,dF_mag_HPF),2)
# #cc2 = round(correlation_coef(HPF_FBR,HPF_I_75),2)
# fig,(ax1,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
# ax1.plot(Time_OF,HPF_FBR_norm,"-k")
# ax1.set_ylabel("HPF FB nomalized on rotor weight[-]")
# ax1.plot(Time_mag_HPF,FBR_HPF,"ok",markersize=3)
# ax2=ax1.twinx()
# ax2.plot(Time_OF,hvelB1[:,225],"-b",label="B1")
# ax2.plot(Time_OF,hvelB2[:,225],"-r",label="B2")
# ax2.plot(Time_OF,hvelB3[:,225],"-g",label="B3")
# ax2.set_ylabel("$u_{x'}$ [m/s]")
# ax2.legend()
# ax1.grid()
# ax1.set_title("correlation coefficient $dF_B,max[dux]$ = {}".format(cc1))
# fig.supxlabel("Time [s]")

# idx = np.searchsorted(Time_mag_HPF_interp,Time_mag_HPF_interp[0]+10)
# cc = []
# for it in np.arange(0,len(Time_mag_HPF_interp)-idx):
#     cc.append(correlation_coef(dF_mag_HPF_interp[it:it+idx],max_dUx_interp[it:it+idx]))

# ax3.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)],cc,"-k")
# ax3.set_ylabel("Local correlation T=10s")
# ax3.grid()



# cc1 = round(correlation_coef(max_dUx,dF_mag_HPF),2)
# #cc2 = round(correlation_coef(HPF_FBR,HPF_I_75),2)
# fig,(ax1,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
# ax1.set_ylabel("HPF FB [kN]")
# ax1.plot(Time_OF,HPF_FBR,"-k")
# ax1.grid()
# ax1.set_title("correlation coefficient $dF_B,max[dux]$ = {}".format(cc1))
# fig.supxlabel("Time [s]")

# idx = np.searchsorted(Time_mag_HPF_interp,Time_mag_HPF_interp[0]+10)
# cc = []
# for it in np.arange(0,len(Time_mag_HPF_interp)-idx):
#     cc.append(correlation_coef(dF_mag_HPF_interp[it:it+idx],max_dUx_interp[it:it+idx]))

# ax3.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)],cc,"-k")
# ax3.set_ylabel("Local correlation T=10s")
# ax3.grid()


fig,ax=plt.subplots(figsize=(14,8))
ax.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)],cc,"-r")
ax.set_ylabel("Local correlation cc($|dF_B|,max[|du_{x',i}|]$) T=10s")
ax.grid()
idx = np.searchsorted(Time_OF,Time_OF[0]+10)
std = []
for it in np.arange(0,len(Time_OF)-idx):
    std.append(np.std(HPF_FBR[it:it+idx]))
ax2=ax.twinx()
ax2.plot(Time_OF[int(idx/2):-int(idx/2)-1],std,"-b")
ax2.set_ylabel("Local standard devation $F_B$ T=10s")
cc_std = round(correlation_coef(std[:-1],cc))
fig.suptitle("correlation coefficient = {}".format(cc_std))
fig.supxlabel("Time [s]")
plt.tight_layout()
plt.savefig(out_dir+"cc_local_cc_local_std.png")
plt.close()

# fig = plt.figure(figsize=(14,8))
# plt.scatter(std[:-1],cc)
# plt.xlabel("local standard deviation T=10s")
# plt.ylabel("local correlation coefficient T=10s")


#HPF FBR calc
dF_mag_HPF = []
dUxB1 = []
dUxB2 = []
dUxB3 = []
Time_mag_HPF = []
FBR_HPF = []
for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

    it_1 = zero_crossings_index_HPF_FBR[i]
    it_2 = zero_crossings_index_HPF_FBR[i+1]

    Time_mag_HPF.append(Time_OF[it_1])

    FBR_HPF.append(HPF_FBR_norm[it_1])

    dF_mag_HPF.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1]))
    dUxB1.append(hvelB1[it_2,225]-hvelB1[it_1,225])
    dUxB2.append(hvelB2[it_2,225]-hvelB2[it_1,225])
    dUxB3.append(hvelB3[it_2,225]-hvelB3[it_1,225])

avg_dUx = abs(np.average([dUxB1,dUxB2,dUxB3],axis=0))


f = interpolate.interp1d(Time_mag_HPF,avg_dUx)
avg_dUx_interp = f(Time_mag_HPF_interp)

cc1 = round(correlation_coef(avg_dUx,dF_mag_HPF),2)
cc2 = round(correlation_coef(HPF_FBR,HPF_I_75),2)
fig,(ax1,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax1.plot(Time_mag_HPF,dF_mag_HPF,"-or",markersize=3)
ax1.set_ylabel("$|dF_B|$ [kN]")
ax2=ax1.twinx()
ax2.plot(Time_mag_HPF,avg_dUx,"-ob",markersize=3)
ax2.set_ylabel("$|avg[du_{x',i}]|$ B1,B2,B3 [m/s]")
ax1.grid()
ax1.set_title("correlation coefficient $dF_B$"+",$|avg[du_{x',i}]|$"+" = {}\ncorrelation coefficient HPF $F_B$,HPF $I_B$ = {}".format(cc1,cc2))
fig.supxlabel("Time [s]")

idx = np.searchsorted(Time_mag_HPF_interp,Time_mag_HPF_interp[0]+10)
cc = []
for it in np.arange(0,len(Time_mag_HPF_interp)-idx):
    cc.append(correlation_coef(dF_mag_HPF_interp[it:it+idx],avg_dUx_interp[it:it+idx]))

ax3.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)],cc,"-k")
ax3.set_ylabel("Local correlation T=10s")
ax3.grid()
plt.tight_layout()
plt.savefig(out_dir+"cc_avg_dux_FB.png")


# cc1 = round(correlation_coef(avg_dUx,dF_mag_HPF),2)
# #cc2 = round(correlation_coef(HPF_FBR,HPF_I_75),2)
# fig,(ax1,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
# ax1.plot(Time_OF,HPF_FBR_norm,"-k")
# ax1.set_ylabel("HPF FB nomalized on rotor weight[-]")
# ax1.plot(Time_mag_HPF,FBR_HPF,"ok",markersize=3)
# ax2=ax1.twinx()
# ax2.plot(Time_OF,hvelB1[:,225],"-b",label="B1")
# ax2.plot(Time_OF,hvelB2[:,225],"-r",label="B2")
# ax2.plot(Time_OF,hvelB3[:,225],"-g",label="B3")
# ax2.set_ylabel("$u_{x'}$ [m/s]")
# ax2.legend()
# ax1.grid()
# ax1.set_title("correlation coefficient $dF_B,avg[dux]$ = {}".format(cc1))
# fig.supxlabel("Time [s]")

# idx = np.searchsorted(Time_mag_HPF_interp,Time_mag_HPF_interp[0]+10)
# cc = []
# for it in np.arange(0,len(Time_mag_HPF_interp)-idx):
#     cc.append(correlation_coef(dF_mag_HPF_interp[it:it+idx],avg_dUx_interp[it:it+idx]))

# ax3.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)],cc,"-k")
# ax3.set_ylabel("Local correlation T=10s")
# ax3.grid()


fig = plt.figure(figsize=(14,8))
plt.scatter(dF_mag_HPF,avg_dUx)
dF_threshold = []
dUx_threshold = []
for i in np.arange(0,len(dF_mag_HPF)):
    if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
        dF_threshold.append(dF_mag_HPF[i]); dUx_threshold.append(avg_dUx[i])
plt.scatter(dF_threshold,dUx_threshold,color="blue")
cc = round(correlation_coef(dF_threshold,dUx_threshold),2)
plt.xlabel("$|dF_B|$ [kN]")
plt.ylabel("$|avg[du_{x',i}]|$ B1,B2,B3 [m/s]")
plt.title("correlation coefficient $dF_B$"+",$|avg[du_{x',i}]|$"+" = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"scatter_FB_avg_dux.png")
plt.close()


#HPF FBR calc
dF_mag_HPF = []
ddUx = []
Time_mag_HPF = []
FBR_HPF = []
for i in np.arange(0,len(zero_crossings_index_HPF_FBR)-1):

    it_1 = zero_crossings_index_HPF_FBR[i]
    it_2 = zero_crossings_index_HPF_FBR[i+1]

    Time_mag_HPF.append(Time_OF[it_1])

    FBR_HPF.append(HPF_FBR_norm[it_1])

    dF_mag_HPF.append(abs(HPF_FBR[it_2] - HPF_FBR[it_1]))
    dUxB = [hvelB1[it_2,225]-hvelB1[it_1,225],hvelB2[it_2,225]-hvelB2[it_1,225],hvelB3[it_2,225]-hvelB3[it_1,225]]
    ddUx.append(abs(np.max(dUxB)-np.min(dUxB)))


cc1 = round(correlation_coef(ddUx,dF_mag_HPF),2)
cc2 = round(correlation_coef(HPF_FBR,HPF_I_75),2)
fig,(ax1,ax3) = plt.subplots(2,1,figsize=(14,8),sharex=True)
ax1.plot(Time_mag_HPF,dF_mag_HPF,"-or",markersize=3)
ax1.set_ylabel("$|dF|$ [kN]")
ax2=ax1.twinx()
ax2.plot(Time_mag_HPF,ddUx,"-ob",markersize=3)
ax2.set_ylabel("$|max[du_{x',i}]-min[du_{x',i}]|$ B1,B2,B3 [m/s]")
ax1.grid()
ax1.set_title("correlation coefficient $dF_B,ddU_x$ = {}\ncorrelation coefficient HPF $F_B$,HPF $I_B$ = {}".format(cc1,cc2))
fig.supxlabel("Time [s]")

f = interpolate.interp1d(Time_mag_HPF,ddUx)
ddUx_interp = f(Time_mag_HPF_interp)
idx = np.searchsorted(Time_mag_HPF_interp,Time_mag_HPF_interp[0]+10)
cc = []
for it in np.arange(0,len(Time_mag_HPF_interp)-idx):
    cc.append(correlation_coef(dF_mag_HPF_interp[it:it+idx],ddUx_interp[it:it+idx]))

ax3.plot(Time_mag_HPF_interp[int(idx/2):-int(idx/2)],cc,"-k")
ax3.set_ylabel("Local correlation T=10s")
ax3.grid()
plt.tight_layout()
plt.savefig(out_dir+"cc_FB_ddUx.png")
plt.close()


fig = plt.figure(figsize=(14,8))
plt.scatter(dF_mag_HPF,ddUx)
dF_threshold = []
dUx_threshold = []
for i in np.arange(0,len(dF_mag_HPF)):
    if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
        dF_threshold.append(dF_mag_HPF[i]); dUx_threshold.append(ddUx[i])
plt.scatter(dF_threshold,dUx_threshold,color="blue")
cc = round(correlation_coef(dF_threshold,dUx_threshold),2)
plt.xlabel("$|dF_B|$ [kN]")
plt.ylabel("$|max[du_{x',i}]-min[du_{x',i}]|$ B1,B2,B3 [m/s]")
plt.title("correlation coefficient $dF_B,ddU_x$ = {}".format(cc))
plt.grid()
plt.tight_layout()
plt.savefig(out_dir+"scatter_dF_ddUx.png")
plt.close()

# Time_OF = Time_OF-4.6
# Time_mag_HPF = np.subtract(Time_mag_HPF,4.6)



# fig,(ax,ax3)=plt.subplots(2,1,figsize=(14,8),sharex=True)
# ax.plot(Time_OF,IyB1_arr,"-b",label="B1")
# ax.plot(Time_OF,IyB2_arr,"-r",label="B2")
# ax.plot(Time_OF,IyB3_arr,"-g",label="B3")
# ax.plot(Time_OF,HPF_I_75,"--k",label="HPF $I_B$")
# ax.set_ylabel("Asymmetry y 75% span\nlocation [m/s]")
# ax.legend()
# ax2=ax.twinx()
# ax2.set_ylabel("HPF Bearing force magntiude\nNormalized on rotor weight [-]")
# ax2.grid()
# ax2.plot(Time_OF,HPF_FBR_norm,"-k")
# for i in np.arange(0,len(dF_mag_HPF)):
#     if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
#         ax2.plot(Time_mag_HPF[i],FBR_HPF[i],"ob")

# ax3.plot(Time_OF,hvelB1[:,225],"-b",label="B1")
# ax3.plot(Time_OF,hvelB2[:,225],"-r",label="B2")
# ax3.plot(Time_OF,hvelB3[:,225],"-g",label="B3")
# ax3.set_ylabel("Streamwise velocity [m/s]")
# ax3.legend()
# ax4=ax3.twinx()
# ax4.set_ylabel("HPF Bearing force magnitude\nNormalized on rotor weight [-]")
# ax4.grid()
# ax4.plot(Time_OF,HPF_FBR_norm,"-k")
# for i in np.arange(0,len(dF_mag_HPF)):
#     if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
#         ax4.plot(Time_mag_HPF[i],FBR_HPF[i],"ob")


# fig,(ax,ax3)=plt.subplots(2,1,figsize=(14,8),sharex=True)
# ax.plot(Time_OF,IzB1_arr,"-b",label="B1")
# ax.plot(Time_OF,IzB2_arr,"-r",label="B2")
# ax.plot(Time_OF,IzB3_arr,"-g",label="B3")
# ax.plot(Time_OF,HPF_I_75,"--k",label="HPF $I_B$")
# ax.set_ylabel("Asymmetry z 75% span\nlocation [m/s]")
# ax.legend()
# ax2=ax.twinx()
# ax2.set_ylabel("HPF Bearing force magnitude\nNormalized on rotor weight [-]")
# ax2.grid()
# ax2.plot(Time_OF,HPF_FBR_norm,"-k")
# for i in np.arange(0,len(dF_mag_HPF)):
#     if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
#         ax2.plot(Time_mag_HPF[i],FBR_HPF[i],"ob")

# ax3.plot(Time_OF,hvelB1[:,225],"-b",label="B1")
# ax3.plot(Time_OF,hvelB2[:,225],"-r",label="B2")
# ax3.plot(Time_OF,hvelB3[:,225],"-g",label="B3")
# ax3.set_ylabel("Streamwise velocity [m/s]")
# ax3.legend()
# ax4=ax3.twinx()
# ax4.set_ylabel("HPF Bearing force magnitude\nNormalized on rotor weight [-]")
# ax4.grid()
# ax4.plot(Time_OF,HPF_FBR_norm,"-k")
# for i in np.arange(0,len(dF_mag_HPF)):
#     if abs(dF_mag_HPF[i]) >= np.mean(dF_mag_HPF)+2*np.std(dF_mag_HPF):
#         ax4.plot(Time_mag_HPF[i],FBR_HPF[i],"ob")

# plt.show()