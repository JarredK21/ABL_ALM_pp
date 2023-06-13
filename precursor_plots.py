from math import ceil, floor
from netCDF4 import Dataset
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import os, glob

dir = "../../ABL_precursor/post_processing/plots/"

case = "../../ABL_precursor"

a = Dataset("./{0}/post_processing/abl_statistics60000.nc".format(case))

mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

z = mean_profiles["h"][:]

t_start = np.searchsorted(a.variables["time"],32300)
t_end = np.searchsorted(a.variables["time"],33500)

def coriolis_twist(u,v):
    twist = np.arctan(np.true_divide(v,u))

    return twist

def TI(case):

    a = Dataset("{0}/post_processing/abl_statistics60000.nc".format(case))

    mean_profiles = a.groups["mean_profiles"] #create variable to hold mean profiles

    tstart = np.searchsorted(a.variables["time"],32500)
    t_end = np.searchsorted(a.variables["time"],33700)

    u_var = np.average(mean_profiles["u'u'_r"][tstart:t_end][:],axis=0)
    v_var = np.average(mean_profiles["v'v'_r"][tstart:t_end][:],axis=0)
    w_var = np.average(mean_profiles["w'w'_r"][tstart:t_end][:],axis=0)

    U_var = np.average(mean_profiles["u"][tstart:t_end][:],axis=0)
    V_var = np.average(mean_profiles["v"][tstart:t_end][:],axis=0)
    W_var = np.average(mean_profiles["w"][tstart:t_end][:],axis=0)

    u_pri = np.sqrt((1/3)*(u_var + v_var + w_var))
    U_bar = np.sqrt((U_var**2 + V_var**2 + W_var**2))

    I = np.round((u_pri/U_bar),decimals=2)

    return I

fig = plt.figure()

hub_height_ind = np.searchsorted(z,90)
u = np.array(mean_profiles.variables["u"])
v = np.array(mean_profiles.variables["v"])

u_2 = np.average(u[t_start:t_end],axis=0)
v_2 = np.average(v[t_start:t_end],axis=0)

u = u[t_start:t_end,hub_height_ind]
v = v[t_start:t_end,hub_height_ind]


twist = coriolis_twist(u,v)


#coriolis twist plots
fig = plt.figure()
plt.rcParams.update({'font.size': 12})
twist_2 = coriolis_twist(u=u_2,v=v_2)
plt.plot(twist_2,z,linewidth=2)
plt.xlabel("$\Theta$ - twist induced by coriolis [deg]")
plt.ylabel("Distance from the surface [m]")
plt.grid()
plt.tight_layout()       
path = dir + "coriolis_twist_avg.png"
plt.savefig(path)
plt.close(fig)

z_zi = z/1007
z_zi_loc = [0.1,0.8,1.1,1.2]
text = ["$0.1z_i$","$0.8z_i$","$1.1z_i$","$1.2z_i$"]
fig = plt.figure()
plt.rcParams.update({'font.size': 12})
for loc in z_zi_loc:
    z_zi_idx = np.searchsorted(z_zi,loc)
    u_loc = u_2[z_zi_idx]
    v_loc = v_2[z_zi_idx]
    plt.plot(u_loc,v_loc,"o",markersize=10)
plt.xlabel("U [m/s]")
plt.ylabel("V [m/s]")
plt.legend(text)

for loc in z_zi_loc:
    z_zi_idx = np.searchsorted(z_zi,loc)
    u_loc = u_2[z_zi_idx]
    v_loc = v_2[z_zi_idx]
    plt.arrow(0,0,u_loc,v_loc,length_includes_head=True,head_width=0.25,color="k")

plt.grid()
plt.tight_layout()       
path = dir + "hodograph_avg.png"
plt.savefig(path)
plt.close(fig)



hvelmag = np.add( np.multiply(u,np.cos(twist[hub_height_ind])) , np.multiply(v,np.sin(twist[hub_height_ind])) )


print(hvelmag)
plt.plot(a.variables["time"][t_start:t_end],hvelmag)
#plt.show()


u = np.average(mean_profiles.variables["u"][t_start:t_end][:],axis=0)
v = np.average(mean_profiles.variables["v"][t_start:t_end][:],axis=0)

twist = coriolis_twist()

hvelmag = []
for i in np.arange(0,len(u),1):

    hvel = u[i]*np.cos(twist[i]) + v[i]*np.sin(twist[i])
    hvelmag.append(hvel)


w = np.average(mean_profiles.variables["w"][t_start:t_end][:],axis=0)

theta = np.average(mean_profiles.variables["theta"][t_start:t_end][:],axis=0)

u_w_r = np.average(mean_profiles.variables["u'w'_r"][t_start:t_end][:],axis=0)


fig = plt.figure()
plt.rcParams.update({'font.size': 12})

plt.plot(w,z,"b-")

plt.xlabel("Ensemble averaged vertical velocity [m/s]")
plt.ylabel("Distance from surface [m]") 
plt.grid()
plt.tight_layout()
        
path = dir + "w_avg.png"
plt.savefig(path)
plt.close(fig)

fig = plt.figure()
plt.rcParams.update({'font.size': 12})

plt.plot(theta,z,"b-")
plt.xlabel("Ensemble averaged Potential temperature [K]")
plt.ylabel("Distance from surface [m]") 
plt.grid()
plt.tight_layout()
        
path = dir + "theta_avg.png"
plt.savefig(path)
plt.close(fig)


fig = plt.figure()
plt.rcParams.update({'font.size': 12})

plt.plot(hvelmag,z,"b-")
plt.grid()
plt.xlabel("Ensemble averaged Horizontal velocity [m/s]")
plt.ylabel("Distance from surface [m]")

plt.axhline(y=90, color="k", linestyle='--') 
plt.tight_layout()
path = dir + "Horz_vel_avg.png"
plt.savefig(path)
plt.close(fig)


fig = plt.figure()
plt.rcParams.update({'font.size': 12})

plt.plot(hvelmag,z,"b-")
plt.grid()
plt.xlabel("Ensemble averaged Horizontal velocity [m/s]")
plt.ylabel("Distance from surface [m]")

plt.axhline(y=90, color="k", linestyle='--')


hub_height_ind = np.searchsorted(z,90)
hub_height_var = np.round(hvelmag[hub_height_ind],decimals=2)
upper_height_ind = np.searchsorted(z,90+63)
upper_height_var = hvelmag[upper_height_ind]
upper_height = z[upper_height_ind]
lower_height_ind = np.searchsorted(z,90-63)
lower_height_var = hvelmag[lower_height_ind]
lower_height = z[lower_height_ind]
alpha = np.round(np.log((upper_height_var/lower_height_var))/
                    np.log((upper_height/lower_height)),decimals=2)

I = TI(case)
I = I[hub_height_ind]*100

plt.text(8, 90+10,"velocity x = {0} [m/s]  \nShear exp = {1}  \nTurbulence intensity = {2}%".format(hub_height_var,alpha,I),fontsize=12)
            
            
plt.ylim([0,90+100])  

plt.tight_layout()
path = dir + "Horz_vel_avg_hub_height.png"
plt.savefig(path)
plt.close(fig)