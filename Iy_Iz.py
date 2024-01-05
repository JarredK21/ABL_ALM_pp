import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

in_dir = "../../NREL_5MW_MCBL_R_CRPM_3/post_processing/"

a = Dataset(in_dir+"Dataset.nc")

Time_sampling = np.array(a.variables["time_sampling"])
Time_sampling = Time_sampling - Time_sampling[0]

offset = "63.0"
group = a.groups["{}".format(offset)]
Ux = np.array(group.variables["Ux"])
Uz = np.array(group.variables["Uz"])
IA = np.array(group.variables["IA"])
Iy = np.array(group.variables["Iy"])
Iz = np.array(group.variables["Iz"])

fig,(ax1,ax2) = plt.subplots(2,figsize=(14,8),sharey=True)

ax1.plot(Time_sampling,Iy,'-b')
ax1.set_ylabel("Asymmetry around y axis",fontsize=14)

ax2.plot(Time_sampling,-Iz,"-r")
ax2.set_ylabel("Asymmretry around z axis",fontsize=14)

ax1.grid()
ax2.grid()
plt.xlabel("Time [s]",fontsize=16)
plt.tight_layout()
plt.show()