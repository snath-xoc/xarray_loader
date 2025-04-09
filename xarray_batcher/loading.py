import datetime
import glob

import numpy as np
import xarray as xr
from tqdm import tqdm

from .normalise import convert_units, get_norm, logprec, nonnegative
from .utils import get_config, get_metadata, get_paths

(
    all_fcst_fields,
    all_fcst_levels,
    accumulated_fields,
    nonnegative_fields,
) = get_config()

(FCST_PATH, FCST_PATH_IFS, TRUTH_PATH, CONSTANTS_PATH, TFRECORDS_PATH) = get_paths()

(fcst_time_res, time_res, lonlatbox, fcst_spat_res) = get_metadata()


def daterange(start_date, end_date):
    """
    Generator to get date range for a given time period
    from start_date to end_date
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

    centres of forecast (lon_reg, lat_reg),
    and truth (lon__reg_TRUTH, lat_reg_TRUTH), and their box
    edges: (lon_reg_b, lat_reg_b) and (lon__reg_TRUTH,
    lat_reg_TRUTH) for forecasts and truth resp.

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


def streamline_and_normalise_ifs(
    field, da, log_prec=True, norm=True, split_steps=[5, 6, 7, 8, 9]
):

    all_data_mean = da[f"{field}_mean"].values
    all_data_sd = da[f"{field}_sd"].values

    times = np.hstack(
        ([time[split_steps[0] : split_steps[-1]] for time in da.fcst_valid_time.values])
    )

    data = []

    for start, end in zip(split_steps[:4], split_steps[1:5]):
        if field in ["tp", "cp", "ssr"]:
            # return mean, sd, 0, 0.  zero fields are so
            # that each field returns a 4 x ny x nx array.
            # accumulated fields have been pre-processed
            # s.t. data[:, j, :, :] has accumulation between times j and j+1
            data1 = all_data_mean[:, start:end, :, :].reshape(
                -1, all_data_mean.shape[2], all_data_mean.shape[3]
            )
            data2 = all_data_sd[:, start:end, :, :].reshape(
                -1, all_data_sd.shape[2], all_data_sd.shape[3]
            )
            data3 = np.zeros(data1.shape)
            data.append(
                np.stack([data1, data2, data3, data3], axis=-1)[:, None, :, :, :]
            )
        else:
            # return mean_start, sd_start, mean_end, sd_end
            temp_data_mean_start = all_data_mean[:, start:end, :, :].reshape(
                -1, all_data_mean.shape[2], all_data_mean.shape[3]
            )
            temp_data_mean_end = all_data_mean[:, end : end + 1, :, :].reshape(
                -1, all_data_mean.shape[2], all_data_mean.shape[3]
            )
            temp_data_sd_start = all_data_sd[:, start:end, :, :].reshape(
                -1, all_data_sd.shape[2], all_data_sd.shape[3]
            )
            temp_data_sd_end = all_data_sd[:, end : end + 1, :, :].reshape(
                -1, all_data_sd.shape[2], all_data_sd.shape[3]
            )

            data.append(
                np.stack(
                    [
                        temp_data_mean_start,
                        temp_data_sd_start,
                        temp_data_mean_end,
                        temp_data_sd_end,
                    ],
                    axis=-1,
                )[:, None, :, :, :]
            )

    data = np.hstack((data)).reshape(-1, da.latitude.shape[0], da.longitude.shape[0], 4)

    da = xr.DataArray(
        data=data,
        dims=["time", "lat", "lon", "i_x"],
        coords=dict(
            lon=da.longitude.values,
            lat=da.latitude.values,
            time=times,
            i_x=np.arange(4),
        ),
    )

    if field in [
        "cape",
        "cp",
        "mcc",
        "sp",
        "ssr",
        "t2m",
        "tciw",
        "tclw",
        "tcrw",
        "tcw",
        "tcwv",
        "tp",
    ]:
        da = nonnegative(da)

    da = convert_units(da, field, log_prec, m_to_mm=True)

    if norm:
        da = get_norm(da, field)

    return da.where(np.isfinite(da), 0).sortby("time")


def load_truth_and_mask(dates, time_idx=[5, 6, 7, 8], log_precip=True, normalise=True):
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
            # print(fname)

            data_path = glob.glob(TRUTH_PATH + f"{date[:4]}/{fname}.nc")
            if len(data_path) < 1:
                break
            # ds = xr.concat([xr.open_dataset(dataset).expand_dims(dim={'time':i},
            # axis=0)
            # for i,dataset in enumerate(data_path)],dim='time').mean('time')
            ds = xr.open_dataset(data_path[0])

            if log_precip:
                ds["precipitation"] = logprec(ds["precipitation"])

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
