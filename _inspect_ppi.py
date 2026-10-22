import netCDF4 as nc
import numpy as np
from pathlib import Path

root = Path(r"dataset/level1_SL74/level1/SL74_l1")

sample_files = [
    root / "20171007" / "SL74_ppi3_l1_20171007_014200.nc",
    root / "20171008" / "SL74_ppi3_l1_20171008_004200.nc",
    root / "20171015" / "SL74_ppi3_l1_20171015_004200.nc",
]

for f in sample_files:
    if not f.exists():
        print(f"NOT FOUND: {f}")
        continue
    ds = nc.Dataset(str(f))
    n_rays = ds.dimensions["NUMBER_OF_RAYS"].size
    n_gates = ds.dimensions["NUMBER_OF_GATES"].size
    az = ds.variables["azimuth"][:]
    el = ds.variables["elevation"][:]
    gc = ds.variables["gate_centers"][:]
    rv = ds.variables["radial_velocity"][:]
    intensity = ds.variables["intensity"][:]
    print(f"{f.name}:")
    print(f"  rays={n_rays}, gates={n_gates}")
    print(f"  az=[{az.min():.2f}, {az.max():.2f}], unique={len(np.unique(az))}")
    print(f"  elev=[{el.min():.3f}, {el.max():.3f}]")
    print(f"  gate_centers: {gc[0]:.0f} to {gc[-1]:.0f}m, step={gc[1]-gc[0]:.0f}m")
    print(f"  RV: [{rv.min():.3f}, {rv.max():.3f}]")
    print(f"  Intensity: [{intensity.min():.4f}, {intensity.max():.4f}]")
    print(f"  scan_type: {ds.scan_type}")
    ds.close()

print("\n--- Dimension statistics across all PPI files ---")
rays_list = []
gates_list = []
az_min_list = []
az_max_list = []
count = 0
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    for f in d.glob("SL74_ppi3_*.nc"):
        ds = nc.Dataset(str(f))
        rays_list.append(ds.dimensions["NUMBER_OF_RAYS"].size)
        gates_list.append(ds.dimensions["NUMBER_OF_GATES"].size)
        az = ds.variables["azimuth"][:]
        az_min_list.append(float(az.min()))
        az_max_list.append(float(az.max()))
        ds.close()
        count += 1
        if count % 500 == 0:
            print(f"  processed {count} files...")

print(f"Total PPI files: {count}")
print(f"Rays: min={min(rays_list)}, max={max(rays_list)}, unique={sorted(set(rays_list))}")
print(f"Gates: min={min(gates_list)}, max={max(gates_list)}, unique={sorted(set(gates_list))}")
print(f"Az min: min={min(az_min_list):.2f}, max={max(az_min_list):.2f}")
print(f"Az max: min={min(az_max_list):.2f}, max={max(az_max_list):.2f}")