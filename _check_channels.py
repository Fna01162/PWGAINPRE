import numpy as np

arr = np.load(r"dataset/split_32/test.npy")
print("Shape:", arr.shape)
names = ["V(radial_vel)", "CNR", "SW(spectral_width)", "Skew", "Range_norm", "sin_az", "cos_az", "obs_mask", "rel_mask"]
for c in range(9):
    v = arr[..., c]
    finite = v[np.isfinite(v)]
    print(f"Ch{c} {names[c]:20s}: min={finite.min():10.4f}, max={finite.max():10.4f}, mean={finite.mean():10.4f}, std={finite.std():10.4f}")

print("\n--- Checking geometry channels ---")
print("Ch4 (Range) at gate 0:", arr[0, 0, 0, 4], "(should be 0)")
print("Ch4 (Range) at gate 84:", arr[0, 84, 0, 4], "(should be ~0.94)")
print("Ch5 (sin_az) at ray 0:", arr[0, 0, 0, 5], "(should be 0)")
print("Ch5 (sin_az) at ray 31:", arr[0, 0, 31, 5], "(should be ~sin(90))")
print("Ch6 (cos_az) at ray 0:", arr[0, 0, 0, 6], "(should be 1)")
print("Ch6 (cos_az) at ray 31:", arr[0, 0, 31, 6], "(should be ~cos(90))")