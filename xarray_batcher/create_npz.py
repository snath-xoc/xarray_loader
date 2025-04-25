import os

import numpy as np
import xarray as xr
from torch.utils.data import DataLoader

from .batch_helper_functions import get_spherical
from .loading import get_IMERG_year
from .torch_batcher import BatchDataset, BatchTruth
from .utils import get_paths

_, _, CONSTANTS_PATH = get_paths()
elev = xr.open_dataset(CONSTANTS_PATH + "elev.nc")


def collate_fn(batch, elev=elev, reg_dict={}):

    lat_batch = np.round(batch.lat.values, decimals=2)
    lon_batch = np.round(batch.lon.values, decimals=2)
    lat_values, lon_values = np.meshgrid(lat_batch, lon_batch)
    elev_values = elev.sel({"lat": lat_batch, "lon": lon_batch}, method="nearest")
    elev_values = np.squeeze(elev_values.elevation.values) / 10000.0
    spherical_coords = get_spherical(
        lat_values, lon_values, elev_values, return_hstacked=False
    )

    i = 0
    reg_sel = None
    for reg in reg_dict.keys():
        if np.array_equal(reg_dict[reg]["spherical_coords"], spherical_coords):
            reg_sel = reg
            break
        i += 1

    if reg_sel is None:
        reg_sel = i
        reg_dict[reg_sel] = {
            "elevation": elev_values,
            "spherical_coords": spherical_coords,
            "precipitation": [],
        }
    reg_dict[reg_sel]["precipitation"].append(batch.precipitation.values)

    return reg_dict


def TruthDataloader_to_Npz(
    out_path,
    years=[2018, 2019, 2020, 2021, 2023, 2024],
    centre=[-1.25, 36.80],
    months=None,
    window_size=3,
    collate_fn=collate_fn,
):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if months is None:
        months = np.arange(1, 13).tolist()
    elev = xr.open_dataset(CONSTANTS_PATH + "elev.nc")
    for year in years:

        ds = get_IMERG_year(year, months=months, centre=centre, window_size=window_size)
        if not isinstance(ds, xr.Dataset):
            ds = ds.to_dataset()

        # load in truth to batcher without any weighting in sampler
        ds = BatchTruth(
            ds, batch_size=[1, 128, 128], weighted_sampler=False, return_dataset=True
        )
        reg_dict = {}

        for batch in ds:
            reg_dict = collate_fn(batch, elev=elev, reg_dict=reg_dict)

        spherical_coords = np.stack(
            [reg_dict[key]["spherical_coords"] for key in reg_dict.keys()]
        )
        elevation = np.stack([reg_dict[key]["elevation"] for key in reg_dict.keys()])
        precipitation = np.stack(
            [np.vstack(reg_dict[key]["precipitation"]) for key in reg_dict.keys()]
        )

        np.savez(
            out_path + f"{year}_30min_IMERG_Nairobi_windowsize={window_size}.npz",
            spherical_coords=spherical_coords,
            elevation=elevation,
            precipitation=precipitation,
        )
