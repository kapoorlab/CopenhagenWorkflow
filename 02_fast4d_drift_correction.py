import os
import numpy as np
from pathlib import Path
from tifffile import imread, imwrite
import dask.array as da
from napari_fast4dreg._fast4Dreg_functions import (
    get_xy_drift,
    get_z_drift,
    apply_xy_drift,
    apply_z_drift,
    crop_data
)

# --- Configuration ---
raw_timelapse_path = "/path/to/raw.tif"
seg_timelapse_path = "/path/to/seg.tif"
output_dir = "/path/to/drift_corrected"

crop_output = True
# Output structure
output_seg_dir = os.path.join(output_dir, "StarDist")
Path(output_dir).mkdir(parents=True, exist_ok=True)
Path(output_seg_dir).mkdir(parents=True, exist_ok=True)

# --- Load timelapse and wrap as Dask arrays (assume 1 channel) ---
raw_np = imread(raw_timelapse_path)  # shape: (T, Z, Y, X)
seg_np = imread(seg_timelapse_path)  # shape: (T, Z, Y, X)

# Add channel axis: shape -> (C=1, T, Z, Y, X)
raw_dask = da.from_array(raw_np[np.newaxis, ...], chunks=("auto", 1, "auto", "auto", "auto"))
seg_dask = da.from_array(seg_np[np.newaxis, ...], chunks=("auto", 1, "auto", "auto", "auto"))

# --- Drift correction: XY ---
xy_drift = get_xy_drift(raw_dask, ref_channel=0)
z_drift = get_z_drift(raw_dask, ref_channel=0)
raw_xy_corrected = apply_xy_drift(raw_dask, xy_drift)
seg_xy_corrected = apply_xy_drift(seg_dask, xy_drift)
raw_xyz_corrected = apply_z_drift(raw_xy_corrected, z_drift)
seg_xyz_corrected = apply_z_drift(seg_xy_corrected, z_drift)

# --- Crop edges to remove border artifacts ---
if crop_output:
        raw_xyz_corrected = crop_data(raw_xyz_corrected, xy_drift=xy_drift, z_drift = z_drift)
        seg_xyz_corrected = crop_data(seg_xyz_corrected, xy_drift=xy_drift, z_drift = z_drift)
    

# --- Save results ---
raw_out = os.path.join(output_dir, "timelapse_name_drift_corrected.tif")
seg_out = os.path.join(output_seg_dir, "timelapse_name_drift_corrected.tif")

imwrite(raw_out, raw_xyz_corrected.compute()[0].astype(np.uint8))  # Drop channel axis
imwrite(seg_out, seg_xyz_corrected.compute()[0].astype(np.uint16))

print("✅ Full XY and Z drift correction complete.")
