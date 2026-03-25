"""Preprocessing script for a CAMS dataset.

Usage:
```bash
python scripts/preprocessing.py
```
"""

import datetime as dt
import pickle as pkl
from pathlib import Path
from warnings import warn

import gribapi
import numpy as np
import xarray as xr
from tqdm import tqdm

CAMS_DATASET_DIR: Path = Path("./dataset")
RAW_DATA_DIR: Path = CAMS_DATASET_DIR / "raw"
PROCESSED_DATA_DIR: Path = CAMS_DATASET_DIR / "processed"

MODEL_NAMES = [
    "MATCH",
    "MINNI",
    "CHIMERE",
    "MOCAGE",
    "MONARCH",
    "EURADIM",
    "EMEP",
    "GEMAQ",
    "SILAM",
    "DEHM",
    "LOTOS",
]
PMACC_MODEL_NAMES = ["PMACC" + model_name for model_name in MODEL_NAMES]
KILOGRAM_TO_MICROGRAM = 10**9


def _process_input_date(
    run_date_string: str,
    lat_coordinates: xr.DataArray,
    lon_coordinates: xr.DataArray,
) -> None:
    """Process input data for a run date.
    Some of the input data are not on the same grid as the others.
    The difference is by a very small distance, so we normalize them
    all on the same latitude and longitude.

    Args:
        run_date_string: The run date string, written as YYYY_MM_DD.
        lat_coordinates: Latitude coordinates to normalize data to.
        lon_coordinates: Longitude coordinates to normalize data to.
    """

    # Check if the processed file exists
    save_path = PROCESSED_DATA_DIR / "input" / f"{run_date_string}.netcdf"
    if save_path.exists():
        return

    def preprocess_input(dataset: xr.Dataset) -> xr.Dataset:
        """Function called by xr.open_mfdataset to preprocess the
        `.netcdf` files before merging them.

        Args:
            dataset: The dataset object passed by xr.open_mfdataset.
        """

        # Gather informations about which file is being processed
        path = Path(dataset.encoding["source"])
        model_name: str = path.parent.stem[5:]  # Remove PMACC from model name

        # Drop unused coordinates
        for variable_name in [
            "valid_time",
            "step",
            "valid_time",
            "heightAboveGround",
            "time",
            "surface",
        ]:
            if variable_name in list(dataset.coords.keys()):
                dataset = dataset.drop_vars(variable_name)

        # Add dimention and coordinates needed to be merged
        dataset = dataset.expand_dims(dim=["model"], axis=[0])
        dataset = dataset.assign_coords({"model": [model_name]})

        # 2 of the CTM models are on slightly different grids.
        # We simply redefine their longitude and latitude to be the
        # same as the other models, creating an accepted imprecision.
        if model_name in ("LOTOS", "SILAM"):
            dataset.coords["latitude"] = lat_coordinates.latitude.values
            dataset.coords["longitude"] = lon_coordinates.longitude.values

        # Round latitude coordinates
        rounded_lat = np.round(dataset.latitude.values, decimals=2)
        rounded_lon = np.round(dataset.longitude.values, decimals=2)
        if not np.allclose(dataset.latitude, rounded_lat) or not np.allclose(
            dataset.longitude, rounded_lon
        ):
            warn(
                "Rounded longitude or latitude is not close to the "
                f"original coordinate for {path}."
            )
        dataset = dataset.assign_coords(
            latitude=np.round(dataset.coords["latitude"].values, decimals=2),
            longitude=np.round(dataset.coords["longitude"].values, decimals=2),
        )

        # Convert from kg/m3 to micro gram per cubic meter
        dataset = dataset.assign_attrs(units="µg/m3")
        dataset *= KILOGRAM_TO_MICROGRAM

        return dataset

    try:
        # Open grib files as xr.Dataset and classify them based on the weather
        # parameter they represent.
        output_dataset = xr.open_mfdataset(
            paths=list(RAW_DATA_DIR.glob(f"**/{run_date_string}*.grib")),
            preprocess=preprocess_input,
            coords="minimal",  # type: ignore[reportArgumentType]
            compat="equals",
            join="outer",
        )

        # Add run_date coordinate
        output_dataset = output_dataset.assign_coords(
            run_date=np.datetime64(run_date_string.replace("_", "-"))
        )

        # Save
        output_dataset.to_netcdf(save_path)

    except gribapi.errors.WrongGridError as error:
        # Catch in consistent input data, and log it
        print(f"Error on {run_date_string}: {error}")


def _process_target_month(
    required_dates: list[dt.datetime],
) -> None:
    """Processes some raw netcdf files of monthly target (reanalysis) data.
    Split the reanalysis monthly files into one file for each hour of the month
    they contain.

    Args:
        required_dates: List of dates expected to be extracted
            from one month of reanalysis.
    """

    # Define the date month
    date_month = dt.date(required_dates[0].year, required_dates[0].month, 1)

    # Find the netcdf files for the given month
    file_paths: list[Path] = list(
        RAW_DATA_DIR.glob(
            f"ensemble/**/{date_month.year}_{date_month.month:02}_*m.netcdf"
        )
    )
    if file_paths == []:
        return

    file_path = file_paths[0]

    # Open the month's netcdf file
    month_dataarray: xr.DataArray = xr.load_dataarray(file_path)

    # Rename variables
    month_dataarray = month_dataarray.rename(
        {
            "time": "valid_date",
            "lat": "latitude",
            "lon": "longitude",
        }
    )

    # Add units attributes
    month_dataarray = month_dataarray.assign_attrs(units="µg/m3")

    # Vertical flip, reindex the latitudes in reverse order
    month_dataarray = month_dataarray.reindex(
        latitude=list(reversed(month_dataarray.latitude))
    )

    # Round latitude coordinates
    month_dataarray = month_dataarray.assign_coords(
        {
            "latitude": np.round(month_dataarray.coords["latitude"].values, decimals=2),
            "longitude": np.round(
                month_dataarray.coords["longitude"].values, decimals=2
            ),
        }
    )

    # Save a target file for each dates required
    for date in required_dates:
        # Check if output path exists
        save_path = (
            PROCESSED_DATA_DIR / "target" / f"{date.strftime(r'%Y_%m_%d_%H')}.netcdf"
        )
        if save_path.exists():
            continue

        # Select the right date from the month data array
        hour_dataarray = month_dataarray.sel(valid_date=date)

        # Save
        hour_dataarray.name = date.strftime(r"%Y_%m_%d_%H reanalisis")
        hour_dataarray.to_netcdf(save_path)


def process() -> None:
    """Prepares a CAMS dataset for use in training."""

    # Create dirs
    (PROCESSED_DATA_DIR / "input").mkdir(exist_ok=True, parents=True)
    (PROCESSED_DATA_DIR / "target").mkdir(exist_ok=True, parents=True)

    # Gather dates
    run_date_strings: set[str] = set(
        file_path.stem[:10] for file_path in RAW_DATA_DIR.glob(r"**/*.grib")
    )

    # ---------------------------------------------------------------------
    # -------                      input                           --------
    # ---------------------------------------------------------------------

    # Open reference MACCGE01 grid.
    with open("dataset/MACCGE01.pkl", "br") as file:
        lat, lon = pkl.load(file)

    # Process the inputs
    for run_date_string in tqdm(run_date_strings, desc="Input processing"):
        _process_input_date(run_date_string, lat, lon)

    # ---------------------------------------------------------------------
    # -------                    target                            --------
    # ---------------------------------------------------------------------

    # Gather months existing in the input
    required_dates: set[dt.datetime] = set(
        dt.datetime.strptime(path.stem[:10], r"%Y_%m_%d")
        for path in PROCESSED_DATA_DIR.glob(r"input/*.netcdf")
    )
    required_months: set[dt.datetime] = set(
        dt.datetime.strptime(path.stem[:7], r"%Y_%m")
        for path in PROCESSED_DATA_DIR.glob(r"input/*.netcdf")
    )

    # Process the target
    for date_month in tqdm(required_months, desc="Target processing"):
        _process_target_month(
            required_dates=[
                date
                for date in required_dates
                if (date.year == date_month.year and date.month == date_month.month)
            ]
        )

    # ---------------------------------------------------------------------
    # -------                   cleanup                            --------
    # ---------------------------------------------------------------------

    # Delete processed input files that do not have an associated target file
    for input_path in tqdm(
        list((PROCESSED_DATA_DIR / "input").glob("*.netcdf")), desc="Cleanup"
    ):
        date = dt.datetime.strptime(input_path.stem, r"%Y_%m_%d")
        target_file_path: Path = (
            PROCESSED_DATA_DIR / "target" / date.strftime(r"%Y_%m_%d_%H.netcdf")
        )
        if not target_file_path.exists():
            input_path.unlink()


if __name__ == "__main__":
    process()
