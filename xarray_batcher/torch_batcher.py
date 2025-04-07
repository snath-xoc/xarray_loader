import torch
import xbatcher
import dask
import numpy as np
from tqdm.dask import TqdmCallback
from tqdm import tqdm
from scipy.spatial import KDTree

import concurrent.futures
import multiprocessing
from scipy.ndimage import convolve

class Antialiasing:
    def __init__(self):
        (x,y) = np.mgrid[-2:3,-2:3]
        self.kernel = np.exp(-0.5*(x**2+y**2)/(0.5**2))
        self.kernel /= self.kernel.sum()
        self.edge_factors = {}
        self.img_smooth = {}
        num_threads = multiprocessing.cpu_count()
        self.executor = concurrent.futures.ThreadPoolExecutor(num_threads)

    def __call__(self, img):
        if img.ndim<3:
            img = img[None,None,:,:]
        elif img.ndim<4:
            img = img[None,:,:,:]
        img_shape = img.shape[-2:]
        if img_shape not in self.edge_factors:
            s = convolve(np.ones(img_shape, dtype=np.float32),
                self.kernel, mode="constant")
            s = 1.0/s
            self.edge_factors[img_shape] = s
        else:
            s = self.edge_factors[img_shape]
        
        if img.shape not in self.img_smooth:
            img_smooth = np.empty_like(img)
            self.img_smooth[img_shape] = img_smooth
        else:
            img_smooth = self.img_smooth[img_shape]

        def _convolve_frame(i,j):
            convolve(img[i,j,:,:], self.kernel, 
                mode="constant", output=img_smooth[i,j,:,:])
            img_smooth[i,j,:,:] *= s

        futures = []
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                args = (_convolve_frame, i, j)
                futures.append(self.executor.submit(*args))
        concurrent.futures.wait(futures)

        return img_smooth


def get_spherical(lat,lon, elev):

    """
    Get spherical coordinates of lat and lon, not assuming unit ball for radius
    So we also take elev into account

    Inputs
    ------

    lat: np.array or xr.DataArray (n_lats,n_lons)
         meshgrid of latitude points

    lon: np.array or xr.DataArray (n_lats,n_lons)
         meshgrid of longitude points

    elev: np.array or xr.DataArray (n_lats,n_lons)
          altitude values in m

    Output
    ------

    r, sigma and phi
    See: https://en.wikipedia.org/wiki/Spherical_coordinate_system
    for more details
    """
    
    lat, lon = np.deg2rad(lat), np.deg2rad(lon)

    x = elev * np.cos(lat) * np.cos(lon)
    y = elev * np.cos(lat) * np.sin(lon)
    z = elev * np.sin(lat)
    
    return np.hstack((x.reshape(-1,1),y.reshape(-1,1),z.reshape(-1,1)))

class BatchDataset(torch.utils.data.Dataset):

    """
    class for iterating over a dataset
    """
    def __init__(self, X, y, constants, batch_size=[4, 128, 128], weighted_sampler=True, for_NJ=False, for_val=False, antialiasing=False):

        self.batch_size=batch_size
        self.X_generator = X
        self.y_generator = xbatcher.BatchGenerator(y,
                {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[2]},
                input_overlap={"lat": int(batch_size[1]/32), "lon": int(batch_size[2]/32)})
        constants["lat"]=np.round(y.lat.values,decimals=2)
        constants["lon"]=np.round(y.lon.values,decimals=2)
                                                  
        self.constants_generator = constants
        
        self.variables = [list(x.data_vars)[0] for x in X]
        self.constants = list(constants.data_vars)
        self.for_NJ = for_NJ
        self.for_val = for_val
        self.antialiasing = antialiasing

        if weighted_sampler:
            y_train = [self.y_generator[i].precipitation.mean(["time","lat","lon"], skipna=False)\
                       for i in range(len(self.y_generator))]
            class_sample_count = np.array(
                                [len(np.where(np.round(y_train,decimals=1) == t)[0])\
                                 for t in np.unique(np.round(y_train,decimals=1))
                                ]
            )
            weight = 1. / class_sample_count
            samples_weight = np.zeros_like(y_train)
            for i_t,t in \
            enumerate(\
                np.unique(np.round(y_train,decimals=1))):
                idx = np.squeeze(np.argwhere(np.round(y_train,decimals=1) == t))
                samples_weight[idx]=weight[i_t]
                
            self.samples_weight = torch.from_numpy(np.asarray(samples_weight))
            self.sampler = torch.utils.data.WeightedRandomSampler(\
                self.samples_weight.type('torch.DoubleTensor'), 
                len(samples_weight)
            )
                
    def __len__(self) -> int:
        return len(self.y_generator)

    def __getitem__(self, idx):

        y_batch = self.y_generator[idx]
        time_batch = y_batch.time.values
        lat_batch = np.round(y_batch.lat.values, decimals=2)
        lon_batch = np.round(y_batch.lon.values, decimals=2)

        X_batch = []
        for x,variable in zip(self.X_generator,self.variables):
            try:
                X_batch.append(x[variable].sel({"time":time_batch,
                                          "lat":lat_batch,
                                         "lon":lon_batch}).values)
            except:
                print(variable,time_batch)
                X_batch.append(np.zeros([len(time_batch),len(lat_batch),len(lon_batch),4]))    
            

        X_batch = torch.tensor(
            np.concatenate(X_batch, axis=-1,
                ), dtype=torch.float32)

        constant_batch = torch.tensor(
            np.stack([self.constants_generator[constant].sel({"lat":lat_batch,
                                     "lon":lon_batch}).values for constant in self.constants], axis=-1,
                    ), dtype=torch.float32
        )

        

        if self.for_NJ:
            
            elev_values = np.squeeze(constant_batch[:,:,0]).reshape(-1,1)
            lat_values, lon_values = np.meshgrid(lat_batch, lon_batch)
            spherical_coords = get_spherical(lat_values.reshape(-1,1),lon_values.reshape(-1,1),elev_values)
            
            kdtree = KDTree(spherical_coords)

            pairs = []

            for i_coord, coord in enumerate(spherical_coords):
                pairs.append(np.vstack((np.full(3,fill_value=i_coord).reshape(1,-1),kdtree.query(coord, k=3)[1])))

            pairs = np.hstack((pairs))
                
            rainfall_path = torch.cat((torch.tensor(\
                y_batch.precipitation.fillna(0).values.reshape(self.batch_size[0],-1,1), dtype=torch.float32
                ),X_batch.reshape(self.batch_size[0],-1,len(self.variables)*4))
                                      ,dim=-1)
            obs_dates = np.ones(self.batch_size[0]).reshape(1,-1)
            n_obs = np.array([self.batch_size[0]])
            if self.for_val:
                obs_dates = np.zeros(self.batch_size[0]).reshape(1,-1)
                n_obs = np.random.randint(1,self.batch_size[0]-8,1)
                obs_dates[:n_obs[0]]=1
            
            return{"idx": idx, "rainfall_path": rainfall_path[None,:,:,:],
                "observed_dates": obs_dates, 
                "nb_obs": n_obs, "dt": 1, "edge_indices": pairs,
                "obs_noise": None}

        else:
            
            if self.antialiasing:
                antialiaser = Antialiasing()
                y_batch = y_batch.precipitation.fillna(np.log10(0.02)).values
                y_batch = antialiaser(y_batch)
                y_batch = torch.tensor(np.moveaxis(y_batch,0,-1), dtype=torch.float32)

            else:
                y_batch = torch.tensor(
                y_batch.precipitation.fillna(np.log10(0.02)).values[:,:,:,None], dtype=torch.float32
                )
            return (torch.cat((X_batch,constant_batch),dim=-1), y_batch)

class BatchTruth(torch.utils.data.Dataset):

    """
    class for iterating over a dataset
    """
    def __init__(self, y, batch_size=[4, 128, 128], weighted_sampler=True, for_NJ=False, for_val=False, length=None, antialiasing=False,
                transform=None):

        self.batch_size=batch_size
        self.for_NJ = for_NJ
        self.for_val = for_val
        self.length = length
        self.antialiasing = antialiasing
        self.transform = transform

        if for_NJ:
            self.y_generator = xbatcher.BatchGenerator(y,
                {"latitude": batch_size[1], "longitude": batch_size[2]},
                input_overlap={"latitude": int(batch_size[1]-8), "longitude": int(batch_size[2]-8)})
        else:
            self.y_generator = xbatcher.BatchGenerator(y,
                {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[2]},
                input_overlap={"lat": int(batch_size[1]//8), "lon": int(batch_size[2]//8)})

        if weighted_sampler:
            if self.for_NJ:
                y_train = [self.y_generator[i].mean(["time","latitude","longitude"], skipna=False)\
                           for i in range(len(self.y_generator))]
            else:
                y_train = [self.y_generator[i].precipitation.mean(["time","lat","lon"], skipna=False)\
                           for i in range(len(self.y_generator))]
            class_sample_count = np.array(
                                [len(np.where(np.round(y_train,decimals=1) == t)[0])\
                                 for t in np.unique(np.round(y_train,decimals=1))
                                ]
            )
            #class_sample_count = np.array(
            #                    [len(np.where(np.searchsorted([25,40,50,60],y_train) == t)[0])\
            #                     for t in np.unique(np.searchsorted([25,40,50,60],y_train))
            #                    ]
            #)
            weight = 1. / class_sample_count
            samples_weight = np.zeros_like(y_train)
            for i_t,t in \
            enumerate(\
                np.unique(np.round(y_train,decimals=1))):
                idx = np.squeeze(np.argwhere(np.round(y_train,decimals=1) == t))
                samples_weight[idx]=weight[i_t]
                
            self.samples_weight = torch.from_numpy(np.asarray(samples_weight))
            self.sampler = torch.utils.data.WeightedRandomSampler(\
                self.samples_weight.type('torch.DoubleTensor'), 
                len(samples_weight)
            )
                
    def __len__(self) -> int:
        return len(self.y_generator)

    def __getitem__(self, idx):

        y_batch = self.y_generator[idx]

        if self.for_NJ:

            def generate(y_batch, length, stop=None):

                rng = np.random.default_rng()
                random_year = rng.choice(np.unique(y_batch['time.year'].values),1)[0]
                ds_sel = y_batch.sel({'time':slice('%i-01-01'%random_year,'%i-01-01'%(random_year+1))})
                
                time_of_event = rng.choice(ds_sel.time.values[length:-length], 1)[0]
                
                time_to_event = rng.choice(np.arange(length), 1)[0]
                time_after_event = length-time_to_event-1
            
                rainfall_path = ds_sel.sel({
                    'time':slice(time_of_event-np.timedelta64(time_to_event*30, "m"), 
                                 time_of_event+np.timedelta64(time_after_event*30, "m")),
                })
                times_rainfall = rainfall_path.time.values
                rainfall_path = rainfall_path.fillna(0).values[None,:,:,:]
                
                if stop is not None:
                    # limit observations to once a day
                    nb_obs_single = stop
                    obs_ptr = np.arange(1,nb_obs_single)
                    
                else:
                    nb_obs_single = length
                    obs_ptr = np.arange(1,length)
                
                observed_date = np.zeros(rainfall_path.shape[1])
                observed_date[0] = 1    
                
                for i_obs in obs_ptr:
            
                    observed_date[i_obs] = 1

                return rainfall_path, observed_date, nb_obs_single

            rainfall_paths = []
            observed_dates = []
            n_obs = []

            stop = None
            batch_size=50
            if self.for_val:
                rng = np.random.default_rng()
                stop = rng.choice(np.arange(2,self.length-100), 1)[0]
                batch_size=2
            
            
            for i in range(batch_size):
                rainfall_path, observed_date, nb_obs = generate(y_batch, self.length, stop=stop)
                rainfall_paths.append(rainfall_path)
                observed_dates.append(observed_date)
                n_obs.append(nb_obs)
    
            rainfall_paths = np.vstack(rainfall_paths)
            observed_dates = np.stack(observed_dates)
            n_obs = np.asarray(n_obs)
            
            return{"idx": idx, "rainfall_path": torch.tensor(rainfall_paths[:,:,:,:,None],dtype=torch.float32),
                    "observed_dates": observed_dates, 
                    "nb_obs": n_obs, "dt": 1,
                    "obs_noise": None}
               
        else:
            if self.antialiasing:
                antialiaser = Antialiasing()
                y_batch = y_batch.precipitation.fillna(np.log10(0.02)).values
                y_batch = antialiaser(y_batch)
                y_batch = torch.tensor(np.moveaxis(y_batch,0,-1), dtype=torch.float32)

            else:
                y_batch = torch.tensor(
                y_batch.precipitation.fillna(np.log10(0.02)).values[:,:,:,None], dtype=torch.float32
                )
            if self.transform:
                y_batch = self.transform(y_batch) 
            
            return y_batch

        
        

        
        

        

