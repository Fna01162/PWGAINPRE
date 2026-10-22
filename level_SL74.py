import xarray as xr

path = r"dataset/level1_SL74/level1/SL74_l1/20171007/SL74_ppi3_l1_20171007_004200.nc"
ds = xr.open_dataset(path)

print(ds)
print("\n变量信息：")
for name, var in ds.variables.items():
    print(name, var.dims, var.shape, var.dtype)