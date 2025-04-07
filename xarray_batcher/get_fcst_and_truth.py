import glob
import os
import time
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import h5py
import dask
from tqdm import tqdm
from tqdm.dask import TqdmCallback
from tensorflow.keras.utils import Progbar
from distributed import Client

import sys
from .utils import get_paths, get_valid_dates, get_config
from .load_zarr import (load_all_single_field_forecasts, 
streamline_and_normalise_zarr, streamline_and_normalise_ifs,
load_truth_and_mask, load_hires_constants)


FCST_PATH, FCST_PATH_IFS, TRUTH_PATH, CONSTANTS_PATH, TFRECORDS_PATH = get_paths()

def get_IMERG_lonlat():

     # A single IMERG data file to get latitude and longitude
    IMERG_file_name = "/home/n/nath/cGAN/shruti/xarray_batcher/example_datasets/3B-HHR.MS.MRG.3IMERG.20180116-S120000-E122959.0720.V07B.HDF5"

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
def get_var(variable, dates, years, return_store=False):
    """
    Input
    -----

    var: str
         Variable name to extract

    dates: ndarray, dtype=datetime64[ns]
           valid dates to keep

    year: list
          years for which to extract variables for

    generator: Boolean, default=False


    Output
    ------

    xr.DataArray (time,lon,lat)
    """

    if return_store:
        return load_all_single_field_forecasts(
            variable, years, return_store=return_store
        )

    else:
        da = load_all_single_field_forecasts(
            variable, years, return_store=return_store
        )
        
        dates = [date for date in dates if np.array([date], dtype='datetime64[D]') in da.time.values.astype('datetime64[D]')]
        da = da.sel({"time": dates})

        da = streamline_and_normalise_zarr(variable, da, split_steps=[int(x) for x in np.arange(0,49,2)])

        return da, list(np.unique(dates))

def open_mfzarr(file_names, use_modify=False, centre = [-1.25, 36.80], window_size = 2, months=[3,4,5,6], dates=None):
    #client = Client()
    #client.restart()
    def modify(ds, use_modify=False,centre = [-1.25, 36.80], window_size = 2, months=[3,4,5,6], dates=None):

        if use_modify:
            name = [var for var in ds.data_vars]
            ds = ds.sel(time=ds.time.dt.month.isin(months)).sel({'latitude':slice(centre[0]-window_size,centre[0]+window_size ),
                     'longitude':slice(centre[1]-window_size,centre[1]+window_size)})
            if dates is not None:
                ds = ds.sel(time=ds.time.dt.date.isin(dates),method='nearest')
            ds = streamline_and_normalise_ifs(name[1].split("_")[0],ds).to_dataset(name=name[1].split("_")[0])
            return ds

        else:
            return ds
        
    
    # this is basically what open_mfdataset does
    open_kwargs = dict(decode_cf=True, decode_times=True)
    #cb = TqdmCallback(desc="global")
    #cb.register()
    open_tasks = [dask.delayed(xr.open_dataset)(f, **open_kwargs) for f in file_names]
    
    tasks = [dask.delayed(modify)(task,use_modify=use_modify,
                                  centre = centre, window_size = window_size,
                                 months=months,dates=dates) for task in open_tasks]
    
    datasets = dask.compute(tasks)  # get a list of xarray.Datasets
    combined = datasets[0]#.result()  # or some combination of concat, merge
    dates = []
    for dataset in combined:
        dates+=list(np.unique(dataset.time.values.astype('datetime64[D]')))
    
    return combined, dates

def get_all_ifs(years,centre = [-1.25, 36.80], window_size = 30, months=[3,4,5,6], ignore_truth=False, variables = None):
    dates_all = []
    
    for year in years:

        start_time = time.time()

        if ignore_truth:
            dates = list(np.arange("%i-01-01"%year,"%i-01-01"%(year+1),np.timedelta64(1,"D"),dtype='datetime64[D]').astype("str"))
        else:
            dates = get_valid_dates(year)
            dates = [date.strftime("%Y-%m-%d") for date in dates]
            if len(months)==12:
                dates_sel = np.random.choice(np.array(dates,dtype='datetime64[D]'),60,replace=False)
            else:
                dates_sel=None
        dates_final = np.array(dates.copy(), dtype='datetime64[D]')

        if variables is None:
            files = sorted(glob.glob(FCST_PATH_IFS+str(year)+'/'+'*.nc'))
        else:
            files = [FCST_PATH_IFS+str(year)+'/'+'%s.nc'%var for var in variables]

        ds, dates_modified = open_mfzarr(files, use_modify=True, centre = centre, 
                                         window_size = window_size, months=months,dates=dates_sel)

        if year == years[0]:
            ds_vars = ds
        else:
            ds_vars = [xr.concat([ds_1,ds_2],"time") for ds_1,ds_2 in zip(ds_vars,ds)]
        
        dates_final = np.append(dates_final,dates_modified, axis=0)

        del ds

        print(
                "Extracted all %i variables in ----" % len(files),
                time.time() - start_time,
                "s---- for year", year,
            )
        
        dates_final, dates_count = np.unique(dates_final, return_counts=True)
        dates_idx = np.squeeze(np.argwhere(dates_count==len(files)+1))
        dates_final = dates_final[dates_idx]
        dates_all+=[str(date) for date in dates_final]
        #print(len(dates)-len(dates_final)," missing dates in year", year)

    if ignore_truth:
        return ds_vars
    
    print("Now doing truth values")
    start_time = time.time()

    ds_truth_and_mask =  load_truth_and_mask(np.array(dates_all, dtype="datetime64[ns]").flatten(),
                                                       time_idx=[1,2,3,4])
    ds_constants = load_hires_constants(batch_size=1)

    print(
            "Finished retrieving truth values in ----",
            time.time() - start_time,
            "s---- for year", year
        )

    return (
            ds_vars,
            ds_truth_and_mask.rename({"latitude": "lat", "longitude": "lon"}).sel(\
                time=ds_truth_and_mask.time.dt.month.isin(months)).sel({'lat':slice(centre[0]-window_size,centre[0]+window_size ),
                     'lon':slice(centre[1]-window_size,centre[1]+window_size)}),
            ds_constants.sel({'lat':slice(centre[0]-window_size,centre[0]+window_size ),
                     'lon':slice(centre[1]-window_size,centre[1]+window_size)}),
        )

def get_all_gfs(years, generator=False, ignore_truth=False):
    """
    Input
    -----
    year: list
          years for which to extract variables for

    Output
    ------

    xr.Dataset of variables (variable time,lon,lat)
    xr.Dataset of truth values alongside a valid mask
    """

    if not isinstance(years, list):
        years = [years]

    (
        all_fcst_fields,
        all_fcst_levels,
        accumulated_fields,
        nonnegative_fields,
    ) = get_config()

    dates_all = []

    print("Getting all variables for years: ", years)
    start_time = time.time()

    #print(dates_all)
    if generator:
        
        for year in years:
            dates = get_valid_dates(year)
            dates = [date.strftime("%Y-%m-%d") for date in dates]
            dates_all += dates
        
        dates_all = np.array(dates_all, dtype="datetime64[D]")
        zarr_store = []

        for var in all_fcst_fields:
            zarr_store.append(get_var(var, dates_all, years, return_store=True))

        print(
            "Extracted all %i variables in ----" % len(all_fcst_fields),
            time.time() - start_time,
            "s---- as zarr store list now working on truth",
        )
        start_time = time.time()

        ds_truth_and_mask = load_truth_and_mask(np.unique(dates_all))
        ds_constants = load_hires_constants(batch_size=len(dates_all))
        ds_constants["time"] = dates_all

        print(
            "Finished retrieving truth values in ----",
            time.time() - start_time,
            "s----",
        )

        return (
            zarr_store,
            ds_truth_and_mask.rename({"latitude": "lat", "longitude": "lon"}),
            ds_constants,
        )
    
    else:

        for year in years:
            
            start_time = time.time()

            if ignore_truth:
                dates = list(np.arange("%i-06-29"%year,"%i-01-01"%(year+1),np.timedelta64(1,"D"),dtype='datetime64[D]').astype("str"))
            else:
                dates = utils.get_valid_dates(year)
                dates = [date.strftime("%Y-%m-%d") for date in dates]

            if not os.path.exists(FCST_PATH+str(year)+'/'):
                os.makedirs(FCST_PATH+str(year)+'/')
                
            dates_final = np.array(dates.copy(), dtype='datetime64[D]')
            for var in all_fcst_fields.keys():
    
                if os.path.exists(FCST_PATH+str(year)+'/'+all_fcst_fields[var]+'_second_half.zarr'):
                    continue
                else:
                    print('Creating consolidated file for variable', var)
                    da, _ = get_var(var, dates, years)
                    da.to_zarr(FCST_PATH+str(year)+'/'+da['field'].values[0]+'_second_half.zarr')
                    da.close()
    
            files = glob.glob(FCST_PATH+str(year)+'/'+'*.zarr')
            ds, dates_modified = open_mfzarr(files)
            dates_final = np.append(dates_final,dates_modified, axis=0)
            
            
            if year == years[0]:
                ds_vars = ds
    
            else:
                ds_vars = [xr.concat([ds_1,ds_2],"time") for ds_1,ds_2 in zip(ds_vars,ds)]
            
            print(
                "Extracted all %i variables in ----" % len(all_fcst_fields),
                time.time() - start_time,
                "s---- for year", year,
            )
            #[d.close() for d in ds]
            
            #start_time = time.time()
    
            #ds_vars = xr.concat(ds_vars, dim="time")
    
            #print(
            #    "Finished consolidation in ----",
            #    time.time() - start_time,
            #    "s---- now working on truth",
            #)
            #start_time = time.time()
    
            dates_final, dates_count = np.unique(dates_final, return_counts=True)
            dates_idx = np.squeeze(np.argwhere(dates_count==len(all_fcst_fields.keys())+1))
            dates_final = dates_final[dates_idx]
            dates_all+=[str(date) for date in dates_final]
            print(len(dates)-len(dates_final)," missing dates in year", year)

        if ignore_truth:
            return ds_vars
            
        print("Now doing truth")
        start_time = time.time()
        
        ds_truth_and_mask = load_truth_and_mask(np.array(dates_all, dtype="datetime64[ns]").flatten(),
                                                  time_idx=[1,2,3,4])

        ds_constants = load_hires_constants(batch_size=len(ds_truth_and_mask.time.values))
        ds_constants["time"] = ds_truth_and_mask.time.values

        print(
            "Finished retrieving truth values in ----",
            time.time() - start_time,
            "s---- for year", year
        )

        return (
            ds_vars,
            ds_truth_and_mask.rename({"latitude": "lat", "longitude": "lon"}),
            ds_constants,
        )

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
    
def get_all(years, model='gfs', **kwargs):

    if model=='gfs':
        return get_all_gfs(years, **kwargs)
    elif model=='ifs':
        return get_all_ifs(years, **kwargs)
    
    elif model=='truth':
        dates_all = []

        months = kwargs['months']

        for year in years:
            dates = get_valid_dates(year)
            dates_all += [date.strftime("%Y-%m-%d") for date in dates]

        print("Only getting truth values over,",len(dates_all),"dates")
        start_time = time.time()
        ds_truth_and_mask =  load_truth_and_mask(np.array(dates_all, dtype="datetime64[ns]").flatten(),
                                                           time_idx=[1,2,3,4])

        print(
            "Finished retrieving truth values in ----",
            time.time() - start_time,
            "s---- for years", years
            )
        return ds_truth_and_mask.sel(time=ds_truth_and_mask.time.dt.month.isin(months)).rename({"latitude": "lat", "longitude": "lon"})
            
                
                
