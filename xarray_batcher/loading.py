import datetime
import glob

import numpy as np
import xarray as xr
from tqdm import tqdm

from .normalise import convert_units, get_norm, logprec, nonnegative
from .utils import get_metadata, get_paths


(FCST_PATH_IFS, TRUTH_PATH, CONSTANTS_PATH, TFRECORDS_PATH) = get_paths()

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

def get_IMERG_lonlat():

     # A single IMERG data file to get latitude and longitude
    IMERG_file_name = "../example_datasets/3B-HHR.MS.MRG.3IMERG.20180116-S120000-E122959.0720.V07B.HDF5"

    # HDF5 in the ICPAC region
    h5_file = h5py.File(IMERG_file_name)
    latitude = h5_file['Grid']["lat"][763:1147]
    longitude = h5_file['Grid']["lon"][1991:2343]
    h5_file.close()

    return latitude, longitude

def prepare_year_and_month_input(years, months):

    if not isinstance(years, list):
        assert isinstance(years, int)
        years = [years, years]
    if not isinstance(months, list):
        assert isinstance(months, int)
        months = [months, months]

    assert len(years)>1
    assert len(months)>1

    years = np.sort(years)
    months = np.sort(months)

    assert years[-1]>=years[0]
    assert months[-1]>=months[0]

    year_beg = years[0]
    year_end = years[-1]

    month_beg = months[0]
    month_end = months[-1]

    if month_end==12:
        year_end+=1
    else:
        month_end+=1

    return year_beg, year_end, month_beg, month_end


def retrieve_vars_ifs(field, all_data_mean, all_data_sd, start=1, end=2):

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
            data = np.stack([data1, data2, data3, data3], axis=-1)[:, None, :, :, :]
            
    else:
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

        data = np.stack(
                [
                    temp_data_mean_start,
                    temp_data_sd_start,
                    temp_data_mean_end,
                    temp_data_sd_end,
                ],
                axis=-1,
            )[:, None, :, :, :]
        
    return data


def streamline_and_normalise_ifs(
    field, da, log_prec=True, norm=True, time_idx=None, split_steps=[5, 6, 7, 8, 9],
):
    '''
    Streamline IFS date to:
    * Have appropriate valid time from time of forecast initialization
    * If time_idx are provided then we directly select based on that
    * Otherwise we select based on split_steps (default 30 - 54 hour lead time)

    Inputs
    ------

    field: str
           field to select, needed to check accumulated or non-
           negative field
    da: xr.DataArray or xr.Dataset
        data over which to streamline and normalise
    log_prec: boolean
              whether to calculate the log of precipitation,
              default=True.
    norm: boolean
          whether to normalise or not, default = True
    split_steps: list or 1-D array
                 valid_time steps to iterate over
                 default=[5,6,7,8,9]
    time_idx: 1-D array or None
              instead of split-steps if we have a more randomised
              selection of valid time to operate on for each 
              initialisation time available

    Outputs
    -------

    xr.DataArray or xr.Dataset of streamline and normalised values
    
    NOTE: We replace the time with the valid time NOT initia
    
    '''

    all_data_mean = da[f"{field}_mean"].values
    all_data_sd = da[f"{field}_sd"].values

    if time_idx is None:
        times = np.hstack(
            ([time[split_steps[0] : split_steps[-1]] for time in da.fcst_valid_time.values])
        )
    else:
        assert da.fcst_valid_time.values.shape[0]==time_idx.shape[0]
        times = np.hstack(
            ([time[[idx_t]] for time, 
              idx_t in zip(da.fcst_valid_time.values,time_idx)])
        )

    data = []

    if time_idx is None:
        for start, end in zip(split_steps[:4], split_steps[1:5]):

            data.append(retrieve_vars_ifs(field, all_data_mean, 
                                          all_data_sd, start=start, end=end))
    else:
        
        for i_row, start in enumerate(time_idx):
            
            data.append(retrieve_vars_ifs(field, all_data_mean[[i_row]], 
                                          all_data_sd[[i_row]], 
                                           start=start, end=start+1))

        
        
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

def get_IMERG_year(years, months=[3,4,5,6]):

    year_beg, year_end, month_beg, month_end = prepare_year_and_month_input(years, months)

    latitude, longitude = get_IMERG_lonlat()

    # Load the IMERG data averaged over 6h periods
    d = datetime(year_beg,month_beg,1,6)
    d_end = datetime(year_end,month_end,1,6)
    # Number of 30 minutes rainfall periods
    num_time_pts = (d_end - d).days*48

    # The 6h average rainfall
    rain_IMERG = np.full([num_time_pts, len(longitude), len(latitude)],np.nan)

    start_time = time.time()

    time_idx = 0
    progbar = Progbar(int((d_end-d).days)*2*24)
    while (d<d_end):

        if d.month not in np.arange(month_beg,month_end):
            progbar.add(1)
            # Move to the next timesetp
            d += timedelta(minutes=30)
            time_idx += 1
            continue

        # Load an IMERG file with the current date
        d2 = d + timedelta(seconds=30*60-1)
        # Number of minutes since 00:00
        count = int((d - datetime(d.year, d.month, d.day)).seconds / 60)
        IMERG_file_name = TRUTH_PATH+"/%s/%s/"%(str(d.year),str(d.strftime('%b')))+\
        f"3B-HHR.MS.MRG.3IMERG.{d.year}{d.month:02d}{d.day:02d}-S{d.hour:02d}{d.minute:02d}00-"+\
        f"E{d2.hour:02d}{d2.minute:02d}{d2.second:02d}.{count:04d}.V07B.HDF5"

        h5_file = h5py.File(IMERG_file_name)
        times = h5_file['Grid']["time"][:]

        # Check the time is correct
        #if (d != datetime(1970,1,1) + timedelta(seconds=int(times[0]))):
        #    print(f"Incorrect time for {d}", datetime(1970,1,1) + timedelta(seconds=int(times[0])))

        # Accumulate the rainfall
        rain_IMERG[time_idx,:,:] = h5_file['Grid']["precipitation"][0,1991:2343,763:1147]
        h5_file.close()

        # Move to the next timesetp
        d += timedelta(minutes=30)

        # Move to the next time index
        time_idx += 1
        progbar.add(1)

    # Put into the same order as the IFS and cGAN data
    rain_IMERG = np.moveaxis(rain_IMERG, [0, 1, 2], [0, 2, 1])

    obs = xr.DataArray(data=rain_IMERG.reshape(-1,len(latitude), len(longitude)),
                         dims=['time','latitude','longitude'],
                         coords = {
                             'time':np.arange(
                                    "%s-%s-01"%(str(year_beg), str(month_beg).zfill(2)),
                                    "%s-%s-01"%(str(year_end), str(month_end).zfill(2)),
                                    np.timedelta64(30, "m"),
                                    dtype="datetime64[ns]"),
                             'latitude':latitude,
                             'longitude':longitude,
                         },
                          attrs=dict(
                          description="IMERG 30 min precipitation",
                          units="mm"))

    print('Finished loading in IMERG data in ----%.2f s-----'%(time.time()-start_time))


    return obs.dropna('time', how='all')
    
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
