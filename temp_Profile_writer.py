from netCDF4 import Dataset

in_dir = "../../ABL_precursor_2_restart/"
a = Dataset(in_dir+"abl_statistics70000.nc")
mean_profiles = a.groups["mean_profiles"]
z = mean_profiles.variables["h"]
theta = mean_profiles.variables["theta"][6000]
print(a.variables["time"][6000])
nz = len(z)

f= open(in_dir+"tempProfile.txt","w+")

f.write("{}\n".format(nz))

for i in range(nz):
    f.write("{0}    {1}\n".format(z[i],theta[i]))

f.close()