import datetime as dt
from pathlib import Path

import cartopy
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import torch
import xarray as xr
from cartopy.crs import PlateCarree
from matplotlib.axes import Axes
from mfai.pytorch.namedtensor import NamedTensor

# Setup cache dir for cartopy to avoid downloading data each time
cartopy_cache_dir = Path("/scratch/shared/cartopy")
if cartopy_cache_dir.exists():
    cartopy.config["data_dir"] = str(cartopy_cache_dir)

# Constants
MOSAIC: list[list[str]] = [
    ["MATCH", "MINNI", "CHIMERE", "MEDIAN", "MEDIAN", "TARGET", "TARGET", "TARGET"],
    ["MOCAGE", "MONARCH", "EURADIM", "MEDIAN", "MEDIAN", "TARGET", "TARGET", "TARGET"],
    ["EMEP", "GEMAQ", "SILAM", "DEHM", "LOTOS", "TARGET", "TARGET", "TARGET"],
]
UNITS = {"O3": "Ozone (µg/m3)"}
CMAP = "turbo"
EXTENT = (-24.95, 44.95, 30.05, 71.95)


def format_axis(ax: Axes, title: str) -> None:
    """Formats a given plot axis with title, labels, ticks and coastlines.

    Args:
        ax: A matplotlib Axes.
        title: The title of this axes.
    """
    ax.set_title(title)
    ax.set(xticklabels=[], yticklabels=[])
    ax.tick_params(bottom=False, left=False)
    ax.set_aspect(1.8)
    ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="grey", linewidth=1)  # type: ignore[reportAttributeAccessIssue]
    ax.coastlines(resolution="50m", color="black", linewidth=1)  # type: ignore[reportAttributeAccessIssue]


def plot(date: dt.datetime) -> None:
    """Plots a sample's input and target data for one parameter only.

    Args:
        date: date of the input/target processed data couple to plot.
    """

    species_name = "O3"
    date_str = date.strftime(r"%Y_%m_%d")
    processed_path = Path("dataset") / "processed"

    # Open input
    input_path = processed_path / "input" / f"{date_str}.netcdf"
    data = xr.open_dataarray(input_path)
    tensor = torch.Tensor(data.to_numpy())
    # For now, we work with all models, the first species, level, and leadtime:
    names = [name.replace("PMACC", "") for name in data.model.values]
    x = NamedTensor(tensor, ["features", "lat", "lon"], names)

    # Open target
    target_path = processed_path / "target" / f"{date_str}_00.netcdf"
    data = xr.open_dataarray(target_path)
    tensor = torch.Tensor(data.to_numpy())
    tensor = tensor.unsqueeze(dim=0)  # Add feature dimension
    y = NamedTensor(tensor, ["features", "lat", "lon"], ["O3"])

    median = torch.median(x.tensor, dim=0).values
    vmin, vmax = None, None

    # Create the different subfigures
    scale = 2.5
    subplot_kw = {"projection": PlateCarree()}
    fig, axs = plt.subplot_mosaic(
        mosaic=MOSAIC,  # type: ignore[reportArgumentType]
        layout="constrained",
        figsize=(8 * scale, 3.2 * scale),
        subplot_kw=subplot_kw,
    )

    # Render the 11 models to their corresponding plot cell
    cell_name: str
    ax: Axes
    for cell_name, ax in axs.items():
        if cell_name in ["MEDIAN", "TARGET"]:
            continue
        ax.imshow(x[cell_name][0], cmap=CMAP, vmin=vmin, vmax=vmax, extent=EXTENT)
        format_axis(ax, cell_name)

    # Render the median to its corresponding plot cell
    axs["MEDIAN"].imshow(median, cmap=CMAP, vmin=vmin, vmax=vmax, extent=EXTENT)
    format_axis(axs["MEDIAN"], "Median Ensemble = Baseline")

    # Render the target to its corresponding plot cell
    img = axs["TARGET"].imshow(
        y.tensor.squeeze(0), cmap=CMAP, vmin=vmin, vmax=vmax, extent=EXTENT
    )
    format_axis(axs["TARGET"], "Analysis = Target")

    # Add Colorbar
    cbar = fig.colorbar(img, ax=axs["TARGET"])
    cbar.set_label(UNITS[species_name], size=13)

    # Add the plot's title
    run_str = date.strftime(r"%Y-%m-%d %Hh")
    title = f"{species_name} - Run {run_str} - Leadtime +15h"
    fig.suptitle(title, size=16)

    plt.savefig("plot.png")
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--date", type=str, default="2023_04_01")
    args = parser.parse_args()

    date = dt.datetime.strptime(args.date, r"%Y_%m_%d")

    plot(date)
