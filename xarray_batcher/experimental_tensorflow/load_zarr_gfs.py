# DEPRECATED functions from GFS load in using kerchunk and MultiZarrtoZarr
import glob
import os
import numpy as np
import xarray as xr
import xesmf
from kerchunk.combine import MultiZarrToZarr
from kerchunk.zarr import ZarrToZarr

from ..utils import get_lonlat


def regridding(type="conservative"):
    """
    Perform regridding using xesmf

    Input
    -----

    lonlatbox: list of int with length 4
               bottom, left, top, right corners of lon-lat box
    fcst_spat_res: float
               spatial resolution of forecasts
    TRUTH_PATH: str
               path to truth data so we can infer grid type
    type: str
              type of regridding to be done, default is conservative

    Output
    ------

    xesmf regridder object to go from forecast to truth grids
    """

    (
        lon_reg,
        lat_reg,
        lon_reg_b,
        lat_reg_b,
        lon_reg_TRUTH,
        lat_reg_TRUTH,
        lon_reg_TRUTH_b,
        lat_reg_TRUTH_b,
    ) = get_lonlat()

    grid_in = {"lon": lon_reg, "lat": lat_reg, "lon_b": lon_reg_b, "lat_b": lat_reg_b}

    # output grid has a larger coverage and finer resolution
    grid_out = {
        "lon": lon_reg_TRUTH,
        "lat": lat_reg_TRUTH,
        "lon_b": lon_reg_TRUTH_b,
        "lat_b": lat_reg_TRUTH_b,
    }

    regridder = xesmf.Regridder(grid_in, grid_out, type)

    return regridder


def load_zarr_store(ds_path):
    z2z = [ZarrToZarr(ds).translate() for ds in ds_path]

    # somehow the grib.idx files are not all identical so need to first
    # extract similar ones into xarray then concat
    mode_length = np.array([len(z.keys()) for z in z2z]).flatten()
    modals, counts = np.unique(mode_length, return_counts=True)
    index = np.argmax(counts)

    return [z for z in z2z if len(z.keys()) == modals[index]]


def load_da_from_zarr_store(z2zs, field, all_fcst_levels, from_idx=False):
    (
        lon_reg,
        lat_reg,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = get_lonlat()

    if len(z2zs) == 0:
        print(field)

        return

    if from_idx:
        z2zs = load_zarr_store(z2zs)
    try:
        mzz = MultiZarrToZarr(
            z2zs,
            concat_dims=["time"],
            coo_map={"time": "INDEX"},
            identical_dims=["latitude", "longitude", all_fcst_levels[field]],
        )

        ref = mzz.translate()

        backend_kwargs = {
            "consolidated": False,
            "storage_options": {
                "fo": ref,
            },
        }

        ds = xr.open_dataset(
            "reference://", engine="zarr", backend_kwargs=backend_kwargs
        ).sel({"latitude": lat_reg, "longitude": lon_reg})

        if ds["time"].values.dtype == "object":
            ds["time"] = ds["time"].values.astype("datetime64[ns]")

    except:
        print("Not sure what happened with", field, z2zs)

        return

    if all_fcst_levels[field] == "isobaricInhPa":
        ds = ds.sel({all_fcst_levels[field]: 200})

    short_name = [var for var in ds.data_vars]

    # TO DO:drop forecast level so there are no
    # conflicts in case we merge with other variables
    return ds[short_name].expand_dims({"field": [short_name]}, axis=0)


def load_all_single_field_forecasts(
    field, yearsordates, FCST_PATH, regrid=True, return_store=False
):
    """

    Load all the forecast data for a given field at a given year
    that are available within the forecast directory

    Inputs
    ------

    field: str
           Name of field to get forecast output for

    year: int
          Integer of year for which to extract forecast for

    FCST_PATH: str
          Directory containing forecasts (assumes all forecasts are in
          there and no subdirectories,to be potentially modified
          to year subdirectories

    Outputs
    -------

    xr.Dataset (time, lat, lon,)


    """

    ds_path = []

    for yearordate in yearsordates:

        ds_path += glob.glob(
            FCST_PATH
            + f"gfs{str(yearordate).replace('-', '')}\
            *{field.replace(' ', '-')}*.zarr"
        )

    ds_path = sorted(
        ds_path, key=lambda x: int(os.path.basename(x).split("_")[0].split("s")[1])
    )

    if return_store:
        return ds_path

    else:
        z2zs = load_zarr_store(ds_path[-186:])
        return load_da_from_zarr_store(z2zs, field)
