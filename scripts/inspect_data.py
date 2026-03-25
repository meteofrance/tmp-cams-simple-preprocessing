"""Script that opens sample data then opens pdb to allow you to inspect it.
The sample data opened is one random raw input, target, processed input and
target files.
"""

from pathlib import Path
from random import choice

import xarray as xr

# Path constants
CAMS_DATASET_DIR: Path = Path("./dataset")
RAW_DATA_DIR: Path = CAMS_DATASET_DIR / "raw"
PROCESSED_DATA_DIR: Path = CAMS_DATASET_DIR / "processed"

# Build file paths
raw_input_path: Path = choice(list(RAW_DATA_DIR.glob("**/*.grib")))
raw_target_path: Path = choice(list(RAW_DATA_DIR.glob("**/*.netcdf")))
processed_input_paths: list[Path] = list(PROCESSED_DATA_DIR.glob("input/*.netcdf"))
processed_target_paths: list[Path] = list(PROCESSED_DATA_DIR.glob("target/*.netcdf"))


# Open random raw files
raw_input = xr.open_dataarray(raw_input_path)
raw_target: xr.DataArray = xr.open_dataarray(raw_target_path)

# Open random processed files
if len(processed_input_paths):
    processed_input = xr.open_dataarray(choice(processed_input_paths))
if len(processed_target_paths):
    processed_target = xr.open_dataarray(choice(processed_target_paths))

# Open pdb
breakpoint()
