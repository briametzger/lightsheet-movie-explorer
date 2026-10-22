Put one folder per movie here, e.g. `data/<dataset>/`, containing the segmentation/tracking pipeline outputs:

- `segments.zarr` — labels (T, Z, Y, X), label == track_id
- `tracks_fluo.csv` — track_id, mean_fluo, t, channel
- `bonus_files/tracks.csv` — track_id, t, z, y, x, parent_track_id (pixel units)
- `bonus_files/cell_class_df.csv` — t, track_id, frame_class (optional)
- `bonus_files/mask_features.csv` — label, frame, volume, solidity, elongation_ratio, bbox_z (optional)
- `MIPs/` — maximum-intensity-projection movies per channel (.mov, .tif, .czi), display only

Everything in this folder except this README is ignored by git.
