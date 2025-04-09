

Xarray Loader for forecast data
===
![xarray](https://img.shields.io/badge/Xarray-royalblue)
![xbatcher](https://img.shields.io/badge/Xarray-batcher-gold)
![PRs](https://img.shields.io/badge/PRs-welcome!-green)
![pytorch](https://img.shields.io/badge/pytorch-torchvision-purple)
![zarr](https://img.shields.io/badge/zarr-hotpink)
![lightning](https://img.shields.io/badge/lighting-violet)


## Gist of it.

Forecast data is multi-dimensional, now that we have lovely zarr data it typically looks like this (in a simplified manner):

```bash!
.
├── level
   ├── isobaricinhPa
   :
   └── surface
             |
             variable
             ├── 2m Temperature
             :
             └── Total Precipitation
                                    |
                                    array
                                    └── (time,timestep,member,lat.,lon.)
```

Where for inference we need to match the forecast to the valid time i.e.,

```bash!
time+timedelta(timestep)==>valid_time
```

Or for example if it is to simulate irregularly sampled of discontinuous data, it may be desirable to be able stocahstically generate discontinuous paths.

Moreover to calculate graph networks we may want a **NearestNeighbour** search based on the radial coordinates:

```python!
x = elev * np.cos(lat) * np.cos(lon)
y = elev * np.cos(lat) * np.sin(lon)
z = elev * np.sin(lat)
```

This package grew out of this necessity to switch between different model configurations and inputs (i.e., discontinuous, graph or simply forecast data with varying number of input features).

Data loading flow
---
When loading in turth data, there is no need to stream/lazy load as the magnitude of data is low and optimisation within pytorch's DataLoader is enough.

We plan to utilise pytorch's IterableDataset class to stream in batches, but there's a catch! What if we want dynamic sampling of data? One could of course create a sampler, however for more flexibility we plan the following:


```graphviz
digraph graphname {
W [label="Initial load-in of truth"]
S [label="Create custom sampler file" ]
R [label="Set attributes within IterableDataset in e.g., lightning datamodule"]
G [label="__iter__: dask load-in multiple variables and match valid time and latxlon patch"]
W->S
S->R
R->G
}
```

Commands e.g. for_NJ can be used to generate irregularly sampled fields or do greedy nearest neighbour searches

The following link: https://drive.google.com/file/d/116xsLEtRntjWOljMG4yE71_uelgIXoox/view?usp=sharing
contains rainfall data from IMERG, sampled at a 30 minute frequency for the Greater horn of Africa region.

One can download and set it's path under the utils.py file:

```python!
TRUTH_PATH = (
    "../example_datasets/"
)
```
Then as shown in test_xbatcher.ipynb, you can generate on-the-fly

```python=
import xarray_batcher as xb

dl = xb.DataModule()

for d in dl:
    print(d)
    break

```
This will call the generator function to randomly sample discontinuous paths of rainfall images. Patching is done within to return 128x128 images or the whole domain for validaiton set.

## Timeline

Check the issue tracker [here](https://github.com/snath-xoc/xarray_loader/issues). The most pressing changes:

1) Deprecate initial GFS load in and Tensorflow batcher modalities using kerchunk (moved to experimental_tensorflow)
2) Add IterableDataset switch: not sure if a one-size-fits all here is possible but when we switch to having both forecast and turth data we need to switch to data streaming (potentially with reload after n epoch but this us overkill from prelim. experiments). Main steps here is to refactor the dask based open_mfzarr and modify functions to allow slicing on single files.
3) Allow custom-collate function that returns numpy function and implement np.savez for JAX
4) Allow load in from pre-created npz file with memore mapping


## Appendix and FAQ

:::info
**Find this document incomplete?** Leave a comment!
:::
