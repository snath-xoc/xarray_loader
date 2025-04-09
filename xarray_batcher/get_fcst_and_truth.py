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
from .utils import get_config, get_paths, get_valid_dates

FCST_PATH, FCST_PATH_IFS, TRUTH_PATH, CONSTANTS_PATH, TFRECORDS_PATH = get_paths()


def open_mfzarr(
    file_names,
    use_modify=False,
    centre=[-1.25, 36.80],
    window_size=2,
    months=[3, 4, 5, 6],
    dates=None,
):
    # client = Client()
    # client.restart()
    def modify(
        ds,
        use_modify=False,
        centre=[-1.25, 36.80],
        window_size=2,
        months=[3, 4, 5, 6],
        dates=None,
    ):

        if use_modify:
            name = [var for var in ds.data_vars]
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
            ds = streamline_and_normalise_ifs(name[1].split("_")[0], ds).to_dataset(
                name=name[1].split("_")[0]
            )
            return ds

        else:
            return ds

    # this is basically what open_mfdataset does
    open_kwargs = dict(decode_cf=True, decode_times=True)
    # cb = TqdmCallback(desc="global")
    # cb.register()
    open_tasks = [dask.delayed(xr.open_dataset)(f, **open_kwargs) for f in file_names]

    tasks = [
        dask.delayed(modify)(
            task,
            use_modify=use_modify,
            centre=centre,
            window_size=window_size,
            months=months,
            dates=dates,
        )
        for task in open_tasks
    ]

    datasets = dask.compute(tasks)  # get a list of xarray.Datasets
    combined = datasets[0]  # .result()  # or some combination of concat, merge
    dates = []
    for dataset in combined:
        dates += list(np.unique(dataset.time.values.astype("datetime64[D]")))

    return combined, dates


def get_all_ifs(
    years,
    centre=[-1.25, 36.80],
    window_size=30,
    months=[3, 4, 5, 6],
    ignore_truth=False,
    variables=None,
):
    dates_all = []

    for year in years:

        start_time = time.time()

        if ignore_truth:
            dates = list(
                np.arange(
                    "%i-01-01" % year,
                    "%i-01-01" % (year + 1),
                    np.timedelta64(1, "D"),
                    dtype="datetime64[D]",
                ).astype("str")
            )
        else:
            dates = get_valid_dates(year)
            dates = [date.strftime("%Y-%m-%d") for date in dates]
            if len(months) == 12:
                dates_sel = np.random.choice(
                    np.array(dates, dtype="datetime64[D]"), 60, replace=False
                )
            else:
                dates_sel = None

        dates_final = np.array(dates.copy(), dtype="datetime64[D]")

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
    ds_truth_and_mask = load_truth_and_mask(
        np.array(dates_all, dtype="datetime64[ns]").flatten(), time_idx=[1, 2, 3, 4]
    )
    if dates_sel is not None:
        # Because of 6 hour offset when streamlining select dates 6AM to midnight is used
        # Meaning that the next day midnight is in truth but no other time step in that date
        # need to guarantee therefore alignment in times
        ds_truth_and_mask = ds_truth_and_mask.sel({"time": ds_vars[0].time.values})
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


def get_all(years, model="ifs", **kwargs):

    if model == "ifs":
        return get_all_ifs(years, **kwargs)

    elif model == "truth":
        dates_all = []

        months = kwargs["months"]

        for year in years:
            dates = get_valid_dates(year)
            dates_all += [date.strftime("%Y-%m-%d") for date in dates]

        print("Only getting truth values over,", len(dates_all), "dates")
        start_time = time.time()
        ds_truth_and_mask = load_truth_and_mask(
            np.array(dates_all, dtype="datetime64[ns]").flatten(), time_idx=[1, 2, 3, 4]
        )

        print(
            "Finished retrieving truth values in ----",
            time.time() - start_time,
            "s---- for years",
            years,
        )
        return ds_truth_and_mask.sel(
            time=ds_truth_and_mask.time.dt.month.isin(months)
        ).rename({"latitude": "lat", "longitude": "lon"})
