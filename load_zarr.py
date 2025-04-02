## Load in all zarr, thank god for kerchunk
import glob
import os

import numpy as np
import xarray as xr
import datetime

from kerchunk.zarr import ZarrToZarr
from kerchunk.combine import MultiZarrToZarr
from tqdm import tqdm
import xesmf

import utils
import normalise as nm

import importlib

importlib.reload(utils)
importlib.reload(nm)

(
    all_fcst_fields,
    all_fcst_levels,
    accumulated_fields,
    nonnegative_fields,
) = utils.get_config()

FCST_PATH, FCST_PATH_IFS, TRUTH_PATH, CONSTANTS_PATH, TFRECORDS_PATH = utils.get_paths()

fcst_time_res, time_res, lonlatbox, fcst_spat_res = utils.get_metadata()


def daterange(start_date, end_date):
    """
    Generator to get date range for a given time period from start_date to end_date
    """

    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def get_lonlat():
    """
    Function to get longitudes and latitudes of forecast and truth data

    Input
    ------

    lonlatbox: list of int with length 4
               bottom, left, top, right corners of lon-lat box
    fcst_spat_res: float
               spatial resolution of forecasts

    Output
    ------

    centres of forecast (lon_reg, lat_reg), and truth (lon__reg_TRUTH, lat_reg_TRUTH), and their box
    edges: (lon_reg_b, lat_reg_b) and (lon__reg_TRUTH, lat_reg_TRUTH) for forecasts and truth resp.

    """
    assert len(lonlatbox) == 4

    lat_reg_b = np.arange(lonlatbox[0], lonlatbox[2], fcst_spat_res) - fcst_spat_res / 2
    lat_reg = 0.5 * (lat_reg_b[1:] + lat_reg_b[:-1])

    lon_reg_b = np.arange(lonlatbox[1], lonlatbox[3], fcst_spat_res) - fcst_spat_res / 2
    lon_reg = 0.5 * (lon_reg_b[1:] + lon_reg_b[:-1])

    data_path = glob.glob(TRUTH_PATH + "*.nc")

    ds = xr.open_mfdataset(data_path[0])
    # print(ds)
    ##infer spatial resolution of truth, we assume a standard lon lat grid!

    lat_reg_TRUTH = ds.latitude.values
    lon_reg_TRUTH = ds.longitude.values

    TRUTH_RES = np.abs(lat_reg_TRUTH[1] - lat_reg_TRUTH[0])

    lat_reg_TRUTH_b = np.append(
        (lat_reg_TRUTH - TRUTH_RES / 2), lat_reg_TRUTH[-1] + TRUTH_RES / 2
    )
    lon_reg_TRUTH_b = np.append(
        (lon_reg_TRUTH - TRUTH_RES / 2), lon_reg_TRUTH[-1] + TRUTH_RES / 2
    )

    return (
        lon_reg,
        lat_reg,
        lon_reg_b,
        lat_reg_b,
        lon_reg_TRUTH,
        lat_reg_TRUTH,
        lon_reg_TRUTH_b,
        lat_reg_TRUTH_b,
    )


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

    ## somehow the grib.idx files are not all identical so need to first extract similar ones into xarray then concat
    mode_length = np.array([len(z.keys()) for z in z2z]).flatten()
    modals, counts = np.unique(mode_length, return_counts=True)
    index = np.argmax(counts)

    return [z for z in z2z if len(z.keys()) == modals[index]]


def load_da_from_zarr_store(z2zs, field, from_idx=False):
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
        
        if ds['time'].values.dtype=='object':
            ds['time'] = ds['time'].values.astype('datetime64[ns]')

        print(ds.time.values)
        
    except:
        print("Not sure what happened with", field, z2zs)

        return

    if all_fcst_levels[field] == "isobaricInhPa":
        ds = ds.sel({all_fcst_levels[field]: 200})

    short_names = list(ds.variables.keys())

    short_name = [
        short_name
        for short_name in short_names
        if short_name
        not in [
            "latitude",
            "longitude",
            "step",
            all_fcst_levels[field],
            "surface",
            "time",
            "valid_time",
        ]
    ][0]

    return ds[short_name].drop(all_fcst_levels[field]).expand_dims({'field':[short_name]},axis=0)


def load_all_single_field_forecasts(
    field, yearsordates, regrid=True, return_store=False
):
    """

    Load all the forecast data for a given field at a given year that are available within the forecast directory

    Inputs
    ------

    field: str
           Name of field to get forecast output for

    year: int
          Integer of year for which to extract forecast for

    FCST_PATH: str
          Directory containing forecasts (assumes all forecasts are in there and no subdirectories,
          to be potentially modified to year subdirectories

    Outputs
    -------

    xr.Dataset (time, lat, lon,)


    """

    ds_path = []

    for yearordate in yearsordates:

            
        ds_path += glob.glob(
            FCST_PATH
            + f"gfs{str(yearordate).replace('-','')}*{field.replace(' ','-')}_{all_fcst_levels[field]}.zarr"
        )
        
    ds_path = sorted(ds_path,
            key=lambda x: int(os.path.basename(x).split('_')[0].split('s')[1]))

    if return_store:
        return ds_path

    else:
        z2zs = load_zarr_store(ds_path[-186:])
        return load_da_from_zarr_store(z2zs, field)


import warnings

def var_and_norm_over_day(da):

    """
    Gets mean and variance over first and second half of day assuming da contains all values 
    over a day
    """

    n_steps = len(da.step.values)
    steps_1 = da.step.values[: n_steps // 2]
    steps_2 = da.step.values[n_steps // 2 :]

    da_to_concat = [
        da.sel({"step": steps_1})
        .mean("step", skipna=True)
        .expand_dims(dim={"i_x": [0]}, axis=0),
        da.sel({"step": steps_2})
        .mean("step", skipna=True)
        .expand_dims(dim={"i_x": [2]}, axis=0),
    ]

    da_to_concat.append(
        da.sel({"step": steps_1})
        .std("step", skipna=True)
        .expand_dims(dim={"i_x": [1]}, axis=0)
    )
    da_to_concat.append(
        da.sel({"step": steps_2})
        .std("step", skipna=True)
        .expand_dims(dim={"i_x": [3]}, axis=0)
    )

    return xr.concat(da_to_concat, dim="i_x")

def hourly_split(da, split_steps=[0,3], mean=True, fcst_time_res=fcst_time_res):

    """
    Returns data split into time windows defined by split steps, if mean then the mean within each time window is
    also provided

    Note: split steps should be changed according to time resolution, e.g. we want 6-hourly predictions
    and have 3-hourly forecasts so we take all three forecasts leading up to and including hour prediction valid time
    """

    if not isinstance(split_steps, list):

        split_steps = [split_steps]

    #if len(da.step.values)>9:
    #    da = da.sel({'step':da.step.values[::3]})

    if len(split_steps)==1:

        split_steps.append(da.step.values[-1])

    
    da_complete = []
    for start_step, stop_step in zip(split_steps[:-1],split_steps[1:]):
        da_to_concat = []
        da_sel = da.isel({'step':slice(start_step,stop_step+1)}).rename({"step":"i_x"}).drop("valid_time")

        ## for consistency of dim names
        da_sel['i_x'] = np.arange((stop_step+1)-start_step)
                                  
        da_to_concat.append(da_sel)

        if mean:
    
            da_to_concat.append(da.isel({'step':slice(start_step,stop_step+1)}).mean("step", skipna=True)
                                .expand_dims(dim={"i_x": [(stop_step-start_step)+1]}, axis=0))
        
        da_to_concat = xr.concat(da_to_concat, dim="i_x")
        
        da_to_concat['time']=da_to_concat['time'].values+np.timedelta64(1, "D") + np.timedelta64(start_step*fcst_time_res+6, "h")

        da_complete.append(da_to_concat)
    
    return xr.concat(da_complete, 'time')



def streamline_and_normalise_zarr(field, da, regrid=True, norm=True, log_prec=True, streamline_type="hourly_split", split_steps=None):
    """
    Streamlines zarr file by calculating daily (or ensemble) mean and std

    Input
    -----

    df: xr.Dataset (time, step, lon, lat,)

    regrid: Boolean

    norm: Boolean

    log_prec: Boolean

    Outputs
    -------

    xr.Dataset with mean calculated over time steps, and depending on norm and regrid specified,
    normalised and regridded
    """

    location_of_vals = [0,2]
    
    if streamline_type == "hourly_split":

        if split_steps is None:

            print("Need split steps for hourly split!")

            return

        location_of_vals = np.arange((split_steps[1]-split_steps[0])+2)
        
        da = hourly_split(da, split_steps=split_steps)
    
    if streamline_type == "var_and_norm_over_day":

        da = var_and_norm_over_day(da)
    
    if field in nonnegative_fields:
        da = nm.nonnegative(da)

    da = nm.convert_units(da, field, log_prec, m_to_mm=False)

    if norm:
        
        da = nm.get_norm(da, field, location_of_vals=location_of_vals)
            
    warnings.filterwarnings("ignore", category=UserWarning)
    if regrid:
        regridder = regridding()
        da = regridder(da)
    return da.where(np.isfinite(da), 0).sortby("time")

def streamline_and_normalise_ifs(field, da, log_prec=True, norm=True, split_steps = [5,6,7,8,9]):

    all_data_mean = da[f"{field}_mean"].values
    all_data_sd = da[f"{field}_sd"].values

    times = np.hstack(([time[split_steps[0]:split_steps[-1]] for time in da.fcst_valid_time.values]))

    data = []
    
    for start, end in zip(split_steps[:4],split_steps[1:5]):
        if field in ["tp","cp","ssr"]:
            # return mean, sd, 0, 0.  zero fields are so that each field returns a 4 x ny x nx array.
            # accumulated fields have been pre-processed s.t. data[:, j, :, :] has accumulation between times j and j+1
            data1 = all_data_mean[:,start:end,:,:].reshape(-1,all_data_mean.shape[2],all_data_mean.shape[3])
            data2 = all_data_sd[:,start:end,:,:].reshape(-1,all_data_sd.shape[2],all_data_sd.shape[3])
            data3 = np.zeros(data1.shape)
            data.append(np.stack([data1, data2, data3, data3], axis=-1)[:,None,:,:,:])
        else:
            # return mean_start, sd_start, mean_end, sd_end
            temp_data_mean_start = all_data_mean[:,start:end,:,:].reshape(-1,all_data_mean.shape[2],all_data_mean.shape[3])
            temp_data_mean_end = all_data_mean[:,end:end+1,:,:].reshape(-1,all_data_mean.shape[2],all_data_mean.shape[3])
            temp_data_sd_start = all_data_sd[:,start:end,:,:].reshape(-1,all_data_sd.shape[2],all_data_sd.shape[3])
            temp_data_sd_end = all_data_sd[:,end:end+1,:,:].reshape(-1,all_data_sd.shape[2],all_data_sd.shape[3])
    
            data.append(np.stack([temp_data_mean_start, temp_data_sd_start, temp_data_mean_end, temp_data_sd_end], axis=-1)[:,None,:,:,:])

    data = np.hstack((data)).reshape(-1,da.latitude.shape[0],da.longitude.shape[0],4)

    da = xr.DataArray(data=data,
                        dims=["time","lat","lon","i_x"],
                        coords=dict(
                            lon=da.longitude.values,
                            lat=da.latitude.values,
                            time=times,
                            i_x=np.arange(4),
                        ),
    )


    if field in ['cape', 'cp', 'mcc', 'sp', 'ssr', 't2m', 'tciw', 'tclw', 'tcrw', 'tcw', 'tcwv', 'tp']:
        da = nm.nonnegative(da)

    da = nm.convert_units(da, field, log_prec, m_to_mm=True)

    if norm:
        da = nm.get_norm(da, field)

    return da.where(np.isfinite(da), 0).sortby("time")
        


def load_truth_and_mask(dates, time_idx=[5,6,7,8], log_precip=True, normalise=True):
    """
    Returns a single (truth, mask) item of data.
    Parameters:
        date: forecast start date
        time_idx: forecast 'valid time' array index
        log_precip: whether to apply log10(1+x) transformation
    """
    ds_to_concat = []

    for date in tqdm(dates):
        date = str(date).split("T")[0].replace("-", "")

        # convert date and time_idx to get the correct truth file
        fcst_date = datetime.datetime.strptime(date, "%Y%m%d")

        for idx_t in time_idx:
            valid_dt = fcst_date + datetime.timedelta(
                hours=int(idx_t) * time_res
            )  # needs to change for 12Z forecasts
    
            fname = valid_dt.strftime("%Y%m%d_%H")
            #print(fname)
    
            data_path = glob.glob(TRUTH_PATH + f"{date[:4]}/{fname}.nc")
            if len(data_path)<1:
                break
            # ds = xr.concat([xr.open_dataset(dataset).expand_dims(dim={'time':i}, axis=0)
            # for i,dataset in enumerate(data_path)],dim='time').mean('time')
            ds = xr.open_dataset(data_path[0])
    
            if log_precip:
                ds["precipitation"] = nm.logprec(ds["precipitation"])
            
            # mask: False for valid truth data, True for invalid truth data
            # (compatible with the NumPy masked array functionality)
            # if all data is valid:
            mask = ~np.isfinite(ds["precipitation"])
            ds["mask"] = mask
    
            ds_to_concat.append(ds)

    return xr.concat(ds_to_concat, dim="time")


def load_hires_constants(batch_size=1):
    """

    Get elevation and land sea mask on IMERG resolution

    """

    oro_path = CONSTANTS_PATH + "elev.nc"

    lsm_path = CONSTANTS_PATH + "lsm.nc"

    ds = xr.open_mfdataset([lsm_path, oro_path])

    # LSM is already 0:1
    ds["elevation"] = ds["elevation"] / 10000.0

    return ds.expand_dims(dim={"time": batch_size}).compute()
