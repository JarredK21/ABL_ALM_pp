from netCDF4 import Dataset

ncfile = Dataset("test_dataset.nc",mode="w",format='NETCDF4')
group = ncfile.createGroup("OpenFAST_Variables")

group_2 = group.createGroup("Inner OpenFAST_Variables")

print(ncfile)
print(ncfile.groups)
print(group.groups)