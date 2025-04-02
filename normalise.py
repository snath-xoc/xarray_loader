## Normalisation functions, note to self you can apply universal functions from numpy as scipy that operate element wise on xarray
## not too many function comments as I feel like they are self-explanatory

import numpy as np
from .utils import get_config, load_fcst_norm, get_metadata



## Unfortunately need to have this look up table, not sure what a work around is
precip_fields = ["Convective precipitation (water)", "Total Precipitation", "cp", "tp"]

(
    _,
    _,
    accumulated_fields,
    nonnegative_fields,
) = get_config()

## Normalisation to apply !!! make sure a field doesn't appear twice!!!
standard_scaling = ["Surface pressure", "2 metre temperature","sp","t2m"]
maximum_scaling = [
    "Convective available potential energy",
    "Upward short-wave radiation flux",
    "Downward short-wave radiation flux",
    "Cloud water",
    "Precipitable water",
    "Ice water mixing ratio",
    "Cloud mixing ratio",
    "Rain mixing ratio",
    "cape",
    "ssr", 
    "tciw", 
    "tclw", 
    "tcrw", 
    "tcw", 
    "tcwv", 
]
absminimum_maximum_scaling = ["U component of wind", "V component of wind","u700", "v700"]

#fcst_norm = load_fcst_norm(model="ifs", year=2018)
## get some standard stuff from utils
fcst_time_res, time_res, lonlatbox, fcst_spat_res = get_metadata()


def logprec(data, threshold=0.1, fill_value=0.02,mean=0.051, std=0.266):
    log_scale = np.log10(1e-1+data).astype(np.float32)
    if threshold is not None:
        log_scale.where(log_scale > np.log10(threshold),np.log10(fill_value))
    
    log_scale.fillna(np.log10(fill_value))
    log_scale -= mean
    log_scale /= std
    return log_scale


def nonnegative(data):
    return np.maximum(data, 0.0)  # eliminate any data weirdness/regridding issues


def m_to_mm_per_hour(data, time_res):
    data *= 1000
    return data / time_res  # convert to mm/hr


def to_per_second(data, time_res):
    # for all other accumulated fields [just ssr for us]
    return data / (
        time_res * 3600
    )  # convert from a 6-hr difference to a per-second rate


def centre_at_mean(data, field):
    # these are bounded well away from zero, so subtract mean from ens mean (but NOT from ens sd!)
    return data - fcst_norm[field]["mean"]


def change_to_unit_std(data, field):
    return data / fcst_norm[field]["std"]


def max_scaling(data, field):
    return (data) / (
        fcst_norm[field]["max"]
    )


def absmin_max_scaling(data, field):
    return data / max(-fcst_norm[field]["min"], fcst_norm[field]["max"])


def convert_units(data, field, log_prec, m_to_mm=True):
    if field in precip_fields:

        if m_to_mm:
            data = m_to_mm_per_hour(data, time_res)

        if log_prec:
            return logprec(data)

        else:
            return data

    elif field in accumulated_fields:
        data = to_per_second(data, time_res)

        return data

    else:
        return data


def get_norm(data, field, location_of_vals=[0,2]):
    
    if field in precip_fields:
        return data

    if field in standard_scaling:
        data.loc[{"i_x": location_of_vals}] = centre_at_mean(data.sel({"i_x": location_of_vals}), field)

        return change_to_unit_std(data, field)

    if field in maximum_scaling:
        return max_scaling(data, field)

    if field in absminimum_maximum_scaling:
        return absmin_max_scaling(data, field)

    else:
        return data
