import glob
import time

import dask
import numpy as np
import xarray as xr

from .loading import (
    load_hires_constants,
    load_truth_and_mask,
    streamline_and_normalise_ifs,
)
from .utils import get_paths, get_valid_dates, match_fcst_to_valid_time

FCST_PATH_IFS, TRUTH_PATH, CONSTANTS_PATH = get_paths()


def open_mfzarr(
    file_names,
    use_modify=False,
    centre=[-1.25, 36.80],
    window_size=2,
    lats=None,
    lons=None,
    months=[3, 4, 5, 6],
    dates=None,
    time_idx=None,
    split_steps=[5, 6, 7, 8, 9],
    clip_to_window=True,
):
    """
    Open multiple files using dask delayed

    Inputs
    ------
    file_names: list
                list of file names to open typically one
                file for each variable
    kwargs: to be passed on to modify

    Outputs
    -------
    List of xr.DataArray or xr.Dataset we avoid concatenation
    as this takes to long, but through modify, we are confident
    that the times align.
    """

    # this is basically what open_mfdataset does
    open_kwargs = dict(decode_cf=True, decode_times=True)
    open_tasks = [dask.delayed(xr.open_dataset)(f, **open_kwargs) for f in file_names]

    tasks = [
        dask.delayed(modify)(
            task,
            use_modify=use_modify,
            centre=centre,
            window_size=window_size,
            lats=lats,
            lons=lons,
            months=months,
            dates=dates,
            time_idx=time_idx,
            split_steps=split_steps,
            clip_to_window=clip_to_window,
        )
        for task in open_tasks
    ]

    datasets = dask.compute(tasks)  # get a list of xarray.Datasets
    combined = datasets[0]  # or some combination of concat, merge
    dates = []
    for dataset in combined:
        dates += list(np.unique(dataset.time.values.astype("datetime64[D]")))

    return combined, dates


def modify(
    ds,
    use_modify=False,
    centre=[-1.25, 36.80],
    window_size=2,
    lats=None,
    lons=None,
    months=[3, 4, 5, 6],
    dates=None,
    time_idx=None,
    split_steps=[5, 6, 7, 8, 9],
    clip_to_window=True,
):

    """
    Modification function to wrap around dask delayed compute

    Inputs
    ------

    use_modify: boolean
                whether to apply modification function at all.
    centre: list or tuple
            centre around which to select a region when looking at sub-domains
    window_size: integer
                 window size around centre to use in sub-domain selection
    lats: ndarray or list
          alternatively, latitudes can be given to sub-select
    lons: ndarray or list
          Ditto as lats
    months: list
            months to select if we are doing seasonal/monthly training
    dates: list or 1-D array
           dates to sub-select, typically is all months are used but too
           expensive to downlaod. If time_idx is None, we assume these are
           the dates of forecast issue
    time_idx: list or 1-D array
              valid time indices to select in index form, not absolute
              value
    Outputs
    -------

    xr.Dataset or xr.DataArray with modifications if use_midify=True or simply
    without modifications

    **Note: when time_idx is provided, split_steps is ignored**
    """

    if use_modify:
        name = [var for var in ds.data_vars]
        if lats is not None and lons is not None:
            lat_batch = np.round(lats, decimals=2)
            lon_batch = np.round(lons, decimals=2)

            ds = ds.sel(time=ds.time.dt.month.isin(months)).sel(
                {
                    "latitude": lat_batch,
                    "longitude": lon_batch,
                }
            )
        elif clip_to_window:

            ds = ds.sel(time=ds.time.dt.month.isin(months)).sel(
                {
                    "latitude": slice(centre[0] - window_size, centre[0] + window_size),
                    "longitude": slice(
                        centre[1] - window_size, centre[1] + window_size
                    ),
                }
            )
        if dates is not None:
            _, dates_intersect, _ = np.intersect1d(
                ds.time.values.astype("datetime64[D]"), dates, return_indices=True
            )
            ds = ds.isel(time=dates_intersect)

        ds = streamline_and_normalise_ifs(
            name[1].split("_")[0], ds, time_idx=time_idx, split_steps=split_steps
        ).to_dataset(name=name[1].split("_")[0])
        return ds

    else:
        return ds


def stream_ifs(truth_batch, offset=24, variables=None):
    """
    Input
    -----
    truth_batch: xr.DataArray or xr.Dataset
                 truth values of a single batch item
                 to match and load
                 fcst data for
    offset: int
            day offset to factor in, should be in hours

    variables: list or None
               variables to load, if None then all are
               loaded.
    Output
    ------

    forecast batch as ndarray
    """

    batch_time = truth_batch.time.values
    batch_lats = truth_batch.lat.values
    batch_lons = truth_batch.lon.values

    if not isinstance(batch_time, np.ndarray):
        batch_time = np.asarray(batch_time)

    # Get hours in the truth_batch times object
    # First need to convert to format with base units of hours to extract hour offset
    hour = batch_time.astype("datetime64[h]").astype(object)[0].hour

    # Note that if hour is 0 then we add 24 as this
    # is the offset+24
    hour = hour + 24 * (hour == 0) + offset

    fcst_date, time_idx = match_fcst_to_valid_time(batch_time, hour)

    year = fcst_date.astype("datetime64[D]").astype(object)[0].year
    month = fcst_date.astype("datetime64[D]").astype(object)[0].month

    if variables is None:
        files = sorted(glob.glob(FCST_PATH_IFS + str(year) + "/" + "*.nc"))
    else:
        files = [FCST_PATH_IFS + str(year) + "/" + "%s.nc" % var for var in variables]

    ds, dates_modified = open_mfzarr(
        files,
        use_modify=True,
        lats=batch_lats,
        lons=batch_lons,
        months=[month],
        dates=fcst_date,
        time_idx=time_idx,
        clip_to_window=False,
    )
    assert batch_time.astype("datetime64[D]") == np.unique(dates_modified)
    return ds


def get_whole_year_ifs(
    years,
    centre=[-1.25, 36.80],
    window_size=30,
    months=[3, 4, 5, 6],
    n_days=None,
    split_steps=[5, 6, 7, 8, 9],
    ignore_truth=False,
    variables=None,
    clip_to_window=True,
):
    dates_all = []

    if ignore_truth:
        dates_year = [
            list(
                np.arange(
                    "%i-01-01" % year,
                    "%i-01-01" % (year + 1),
                    np.timedelta64(1, "D"),
                    dtype="datetime64[D]",
                ).astype("str")
            )
            for year in years
        ]
    else:
        dates_year = [get_valid_dates(year, raw_list=True) for year in years]

    for dates in dates_year:
        start_time = time.time()

        if len(months) == 12 and n_days is not None:
            dates_sel = np.random.choice(
                np.array(dates, dtype="datetime64[D]"), n_days, replace=False
            )
        else:
            dates_sel = None

        dates_final = np.array(dates.copy(), dtype="datetime64[D]")
        year = dates_final[0].astype(object).year

        if variables is None:
            files = sorted(glob.glob(FCST_PATH_IFS + str(year) + "/" + "*.nc"))
        else:
            files = [
                FCST_PATH_IFS + str(year) + "/" + "%s.nc" % var for var in variables
            ]

        ds, dates_modified = open_mfzarr(
            files,
            use_modify=True,
            centre=centre,
            window_size=window_size,
            months=months,
            dates=dates_sel,
            split_steps=split_steps,
            clip_to_window=clip_to_window,
        )

        if year == years[0]:
            ds_vars = ds
        else:
            ds_vars = [
                xr.concat([ds_1, ds_2], "time") for ds_1, ds_2 in zip(ds_vars, ds)
            ]

        dates_final = np.append(dates_final, dates_modified, axis=0)

        del ds

        print(
            "Extracted all %i variables in ----" % len(files),
            time.time() - start_time,
            "s---- for year",
            year,
        )

        dates_final, dates_count = np.unique(dates_final, return_counts=True)
        dates_idx = np.squeeze(np.argwhere(dates_count == (len(files) + 1)))
        dates_final = dates_final[dates_idx]
        dates_all += [str(date) for date in dates_final]
        # print(len(dates)-len(dates_final)," missing dates in year", year)

    if ignore_truth:
        return ds_vars

    print("Now doing truth values")
    start_time = time.time()
    # time_idx is hard-coded in here as forecast is made to have time as valid_time
    ds_truth_and_mask = load_truth_and_mask(
        np.array(dates_all, dtype="datetime64[ns]").flatten(),
        time_idx=[1, 2, 3, 4],
    )
    if dates_sel is not None:
        # Because of 6 hour offset when streamlining select dates 6AM to midnight is used
        # Meaning that the next day midnight is in truth but no other time step in that date.
        # Therefore, need to guarantee alignment in times.
        ds_truth_and_mask = ds_truth_and_mask.drop_duplicates("time")
        times_sel = np.intersect1d(
            ds_vars[0].time.values, ds_truth_and_mask.time.values
        )
        ds_truth_and_mask = ds_truth_and_mask.sel({"time": times_sel})
    ds_constants = load_hires_constants(batch_size=1)

    print(
        "Finished retrieving truth values in ----",
        time.time() - start_time,
        "s---- for year",
        year,
    )

    return (
        ds_vars,
        ds_truth_and_mask.rename({"latitude": "lat", "longitude": "lon"})
        .sel(time=ds_truth_and_mask.time.dt.month.isin(months))
        .sel(
            {
                "lat": slice(centre[0] - window_size, centre[0] + window_size),
                "lon": slice(centre[1] - window_size, centre[1] + window_size),
            }
        ),
        ds_constants.sel(
            {
                "lat": slice(centre[0] - window_size, centre[0] + window_size),
                "lon": slice(centre[1] - window_size, centre[1] + window_size),
            }
        ),
    )


def get_all(
    years,
    model="ifs",
    offset=None,
    stream=False,
    truth_batch=None,
    time_idx=[5, 6, 7, 8],
    split_steps=[5, 6, 7, 8, 9],
    ignore_truth=False,
    variables=None,
    months=None,
    n_days=10,
    centre=[-1.25, 36.80],
    window_size=30,
    clip_to_window=False,
    log_precip=True,
):

    """
    Wrapper function to return either:

    * IFS data alongside truth data fully loaded into memory.
    This is recommended if an npz file is being created for
    example.
    * stream IFS data in which case truth_batch should be given
    and offset is the no. days (in hours) lead time used to
    obtain the initialisation time from truth batch valid time
    (See stream_ifs for further details)
    * Obtain only truth data (when model="truth")

    Inputs
    ------
    years: list
           list of years to calculate over
    model: str
           either truth or ifs (later additions possible)
           default='ifs'
    offset: int or None
            Hour offset that accounts for no. days lead time
            needed to match back to intiialisation time
            during streaming.
    stream: boolean
            whether or not to stream data in which case
            truth batch should be provided
    truth_batch: xr.DataArray or xr.Dataset
                 truth batch to obtain forecast variables from
    time_idx: list
              time_idx to obtain truth
    split_steps: list
                 fcst lead times to take when loading in IFS
    ignore_truth: boolean
                  passed on to get_whole_year_ifs, whether to
                  only load in the fcst and ignore turht
    variables: list or None
               variables to load, if None then all are
               loaded.
    months: ndarray or list or None
            months to load in in case of seasonal/sub-seasonal
            training, if None all months are used
     centre: list or tuple
             centre around which to select a region when looking
             at sub-domains, only used when clip_to_window is true
    window_size: integer
                 window size around centre to use in sub-domain
                 selection, only used when clip_to_window is true
    clip_to_window: boolean
                    passed on to get_whole_year_ifs or simply
                    getting truth, to signal when to clip to a
                    window (size window_size) around centre
    log_precip: boolean
                passed on to load_truth_and_mask when model is "truth
                default=True

    Outputs:
    -------
    list of xr.DataArray or streamed or whole IFS+truth or simply truth data
    """

    if months is None:
        months = np.arange(1, 13).tolist()

    if model == "ifs":
        if not stream:
            return get_whole_year_ifs(
                years,
                split_steps=split_steps,
                ignore_truth=ignore_truth,
                variables=variables,
                months=months,
                n_days=n_days,
                centre=centre,
                window_size=window_size,
                clip_to_window=clip_to_window,
            )

        else:
            assert truth_batch is not None
            assert offset is not None
            return stream_ifs(truth_batch, offset=offset, variables=variables)

    elif model == "truth":
        dates_all = []
        for year in years:
            dates = get_valid_dates(year)
            dates_all += [date.strftime("%Y-%m-%d") for date in dates]

        print("Only getting truth values over,", len(dates_all), "dates")
        start_time = time.time()
        ds_truth_and_mask = load_truth_and_mask(
            np.array(dates_all, dtype="datetime64[ns]").flatten(),
            time_idx=time_idx,
            log_precip=log_precip,
        )

        print(
            "Finished retrieving truth values in ----",
            time.time() - start_time,
            "s---- for years",
            years,
        )
        if clip_to_window:
            ds_truth_and_mask = ds_truth_and_mask.sel(
                {
                    "latitude": slice(centre[0] - window_size, centre[0] + window_size),
                    "longitude": slice(
                        centre[1] - window_size, centre[1] + window_size
                    ),
                }
            )

        return ds_truth_and_mask.sel(
            time=ds_truth_and_mask.time.dt.month.isin(months)
        ).rename({"latitude": "lat", "longitude": "lon"})
