## Utils needed in loading zarr and batching
import datetime
import os
import pickle

import numpy as np

## Put all forecast fields, their levels (can be None also), and specify categories of accumulated fields
accumulated_fields = ["ssr", "cp", "tp"]


## Put other user-specification i.e., lon-lat box, spatial and temporal resolution (in hours)
TIME_RES = 6

LONLATBOX = [-14, 19, 25.25, 54.75]
FCST_SPAT_RES = 0.25
FCST_TIME_RES = 3

## Put all directories here

TRUTH_PATH = (
    "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IMERG_V07/ICPAC_region/6h/"
)
FCST_PATH_IFS = (
    "/network/group/aopp/predict/TIP021_MCRAECOOPER_IFS/IFS-regICPAC-meansd/"
)

CONSTANTS_PATH = (
    "/network/group/aopp/predict/TIP022_NATH_GFSAIMOD/cGAN/constants-regICPAC/"
)


def get_metadata():

    """
    Returns time resolution (in hours), lonlat box (bottom, left, top, right) and the forecast's spatial resolution
    """

    return FCST_TIME_RES, TIME_RES, LONLATBOX, FCST_SPAT_RES


def get_paths():

    return FCST_PATH_IFS, TRUTH_PATH, CONSTANTS_PATH


import pickle


def load_fcst_norm(year=2018):

    fcstnorm_path = os.path.join(
        CONSTANTS_PATH.replace("-regICPAC", "_IFS"), f"FCSTNorm{year}.pkl"
    )

    with open(fcstnorm_path, "rb") as f:
        return pickle.load(f)


def daterange(start_date, end_date):

    """
    Generator to get date range for a given time period from start_date to end_date
    """

    for n in range(int((end_date - start_date).days)):
        yield start_date + datetime.timedelta(days=n)


def get_valid_dates(
    year,
    TIME_RES=TIME_RES,
    start_hour=30,
    end_hour=60,
    raw_list=False,
):

    """
    Returns list of valid forecast start dates for which 'truth' data
    exists, given the other input parameters. If truth data is not available
    for certain days/hours, this will not be the full year. Dates are returned
    as a list of YYYYMMDD strings.

    Parameters:
        year (list): forecasts starting in this year
        start_hour (int): Lead time of first forecast desired
        end_hour (int): Lead time of last forecast desired
    """

    # sanity checks for our dataset
    assert year in (2018, 2019, 2020, 2021, 2022, 2023, 2024)
    assert start_hour >= 0
    assert start_hour % TIME_RES == 0
    assert end_hour % TIME_RES == 0
    assert end_hour > start_hour

    # Build "cache" of truth data dates/times that exist as well as forecasts
    valid_dates = []

    start_date = datetime.date(year, 1, 1)
    end_date = datetime.date(
        year + 1, 1, end_hour // TIME_RES + 2
    )  # go a bit into following year

    for curdate in daterange(start_date, end_date):
        datestr = curdate.strftime("%Y%m%d")
        valid = True

        ## then check for truth data at the desired lead time
        for hr in np.arange(start_hour, end_hour, TIME_RES):
            datestr_true = curdate + datetime.timedelta(hours=6)
            datestr_true = datestr_true.strftime("%Y%m%d_%H")
            fname = f"{datestr_true}"  # {hr:02}

            if not os.path.exists(
                os.path.join(TRUTH_PATH, f"{datestr_true[:4]}/{fname}.nc")
            ):
                valid = False
                break

        if valid:
            valid_dates.append(curdate)

    if raw_list:
        # Need to get it from datetime to numpy readable format
        valid_dates = [date.strftime("%Y-%m-%d") for date in valid_dates]

    return valid_dates


def match_fcst_to_valid_time(valid_times, time_idx, step_type="h"):

    """
    Inputs
    ------
    valid_times: ndarray or datetime64 object
                 array of dates as data type datetime64[ns]
    TIME_RES: integer
              hourly timesteps which the fcst. is in
              default = 6
    time_idx: int
              array of prediction timedelta of same shape
              as valid_dates, should be in hours,
              data type = int.
    step_type: str
                 type of fcst step e.g., D for day
                 default: 'h' for hour

    Outputs
    -------
    fcst_dates: ndarray
                i.e., valid_dates-time_idx
    valid_date_idx: ndarray
                    to select
    """

    time_offset = np.timedelta64(time_idx, step_type)
    fcst_times = valid_times - time_offset

    valid_date_idx = np.asarray([int(time_offset.astype(int) / TIME_RES)])

    return fcst_times, valid_date_idx
