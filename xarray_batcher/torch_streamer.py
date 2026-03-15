import glob
import os

import dask
import joblib
import numpy as np
import torch
import xarray as xr
import xbatcher
from scipy.spatial import KDTree
from tqdm import tqdm

from xarray_batcher.get_fcst_and_truth import get_all

from .batch_helper_functions import Antialiasing, get_spherical
from .normalise import fcst_norm, logprec

seeps_dataset = xr.open_dataset("../SEEPS_tests.nc")
OUT_PATH = (
    "/network/group/aopp/predict/AWH024_COOPERNATH_IFS/cGAN_gefs/zarr/samples-v2/"
)


class StreamDataset(torch.utils.data.IterableDataset):

    """
    Similar as BatchDataset, see torch_batcher.py apart
    from the new workflow to assist in streaming:

    1) Start using only truth data
    2) Calculate sampler
    3) When iterating through truth, load in the fcst.
    data on-the-fly

    """

    def __init__(
        self,
        y,
        variables,
        constants,
        batch_size: list[int] = [4, 128, 128],
        offset: int = 24,
        batches_per_epoch=1200,
        weighted_sampler: bool = True,
        create_dataset: bool = False,
        create_sampler: bool = True,
        for_NJ: bool = False,
        for_val: bool = False,
        antialiasing: bool = False,
        batch_type: str = "24-hourly",
        return_seeps=False,
        fill_value=np.log10(0.02),
        log_precip=False,
    ):
        self.offset = offset
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch
        self.variables = variables
        self.log_to_sqrt = log_to_sqrt
        self.return_seeps = return_seeps
        self.seeps_ds = seeps_dataset
        self.log_precip = log_precip

        if y is not None:
            self.y_generator = xbatcher.BatchGenerator(
                y,
                {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[2]},
                input_overlap={
                    "lat": 128 // 8,
                    "lon": 128 // 8,
                },
            )
            constants["lat"] = np.round(y.lat.values, decimals=2)
            constants["lon"] = np.round(y.lon.values, decimals=2)
            self.seeps_ds["latitude"] = np.round(y.lat.values, decimals=2)
            self.seeps_ds["longitude"] = np.round(y.lon.values, decimals=2)

            self.constants_generator = constants

            self.constants = list(constants.data_vars)
        self.for_NJ = for_NJ
        self.for_val = for_val
        self.antialiasing = antialiasing
        self.batch_type = batch_type
        self.fill_value = fill_value

        if create_dataset:
            print(
                "DataGenerator initialised and can be called using __sample__(n_samp) to yield n_samp batches for storage"
            )
            if create_sampler:
                if not os.path.exists(f"{OUT_PATH}/sampler.npz") and y is not None:
                    print(
                        f"Creating and storing sampler file with mean and max y values under {OUT_PATH}/sampler.npz"
                    )
                    y_train_mean = [
                        self.y_generator[i].precipitation.mean(
                            ["time", "lat", "lon"], skipna=False
                        )
                        for i in range(len(self.y_generator))
                    ]
                    y_train_max = [
                        self.y_generator[i].precipitation.max(
                            ["time", "lat", "lon"], skipna=False
                        )
                        for i in range(len(self.y_generator))
                    ]
                    np.savez(
                        f"{OUT_PATH}/sampler.npz",
                        mean=y_train_mean,
                        maximum=y_train_max,
                    )
        else:

            samples = glob.glob(f"{OUT_PATH}/*.npz")
            self.samples_present = np.sort(
                [
                    int(f.split("/")[-1].split("_")[-1].split(".npz")[0])
                    for f in samples
                    if f.split("/")[-1].split("_")[-1].split(".npz")[0] != "sampler"
                ]
            )
            if weighted_sampler:
                if not os.path.exists(f"{OUT_PATH}/sampler.npz"):
                    print("Sampler file was not created, cannot use weighted sampler")
                    return
                else:
                    sampler = np.load(f"{OUT_PATH}/sampler.npz")
                    rounded_y_train = np.round(sampler["mean"], decimals=1)
                    unique_classes = np.unique(rounded_y_train)
                    class_sample_count = np.bincount(
                        np.digitize(rounded_y_train, unique_classes) - 1
                    )
                    weight = 1.0 / class_sample_count
                    samples_weight = weight[
                        np.digitize(rounded_y_train, unique_classes) - 1
                    ]
                    self.sample_weights = torch.from_numpy(np.asarray(samples_weight))
                    self.sample_weights = self.sample_weights[self.samples_present]
                    self.sample_weights /= self.sample_weights.sum()
            self.batch_dir = OUT_PATH

    def __len__(self):

        return self.batches_per_epoch

    def __iter__(self):
        while True:
            idx_samp = np.random.choice(
                self.samples_present, replace=False, p=self.sample_weights
            )
            data = np.load(self.batch_dir + f"sample_{idx_samp}.npz")
            X = torch.from_numpy(data["X"]).float()
            y = torch.from_numpy(data["y"]).float()
            seep = torch.from_numpy(data["seep"]).float()
            data.close()
            yield (X, y, seep)

    def __sample__(self, n_samp=1):
        if isinstance(n_samp, int):
            n_samp = [n_samp]

        idx_samp = n_samp

        if len(n_samp) == 1:
            idx_samp = n_samp[0]
            y_batch = self.y_generator[int(idx_samp)]
            time_batch = y_batch.time.values
            lat_batch = np.round(y_batch.lat.values, decimals=2)
            lon_batch = np.round(y_batch.lon.values, decimals=2)
            month_batch = y_batch.time.dt.month.values
        else:
            y_batch = [self.y_generator[int(idx_s)] for idx_s in idx_samp]
            times = np.hstack([t.time.values for t in y_batch])
            idx_ordered = np.argsort(times)
            y_batch = [y_batch[idx_order] for idx_order in idx_ordered]
            lat_batch = [np.round(t.lat.values, decimals=2) for t in y_batch]
            lon_batch = [np.round(t.lon.values, decimals=2) for t in y_batch]
            month_batch = [t.time.dt.month.values for t in y_batch]

        (X_generator, dates_modified), client = get_all(
            None,
            model="ifs",
            truth_batch=y_batch,
            stream=True,
            offset=self.offset,
            variables=self.variables,
            batch_type=self.batch_type,
            log_precip=self.log_precip,
        )

        X_batch = []
        for x, variable in zip(X_generator, self.variables):
            X_batch.append(x[variable].values)

        X_batch = torch.from_numpy(
            np.concatenate(
                X_batch,
                axis=-1,
            )
        ).float()

        if X_batch.ndim == 4:
            X_batch = X_batch[:, None, :, :, :]

        if isinstance(lat_batch, np.ndarray) and isinstance(lon_batch, np.ndarray):
            constant_batch = torch.from_numpy(
                np.stack(
                    [
                        self.constants_generator[constant]
                        .sel({"lat": lat_batch, "lon": lon_batch})
                        .values
                        for constant in self.constants
                    ],
                    axis=-1,
                )
            ).float()
            if self.return_seeps:
                seeps_batch = torch.from_numpy(
                    np.stack(
                        [
                            self.seeps_ds[seep_var]
                            .sel(
                                {
                                    "month": month_batch,
                                    "latitude": lat_batch,
                                    "longitude": lon_batch,
                                }
                            )
                            .values
                            for seep_var in ["p1", "p3", "t2", "t3"]
                        ],
                        axis=0,
                    )
                ).float()
        elif isinstance(lat_batch, list) and isinstance(lon_batch, list):
            constant_batch = [
                torch.from_numpy(
                    np.stack(
                        [
                            self.constants_generator[constant]
                            .sel({"lat": la_batch, "lon": lo_batch})
                            .values
                            for constant in self.constants
                        ],
                        axis=-1,
                    )
                ).float()
                for la_batch, lo_batch in zip(lat_batch, lon_batch)
            ]
            constant_batch = torch.stack(constant_batch, axis=0)
            if self.return_seeps:
                seeps_batch = [
                    torch.from_numpy(
                        np.stack(
                            [
                                self.seeps_ds[seep_var]
                                .sel(
                                    {
                                        "month": month_batch[j],
                                        "latitude": la_batch,
                                        "longitude": lo_batch,
                                    }
                                )
                                .values
                                for seep_var in ["p1", "p3", "t2", "t3"]
                            ],
                            axis=0,
                        )
                    ).float()
                    for j, (la_batch, lo_batch) in enumerate(zip(lat_batch, lon_batch))
                ]
                seeps_batch = torch.stack(seeps_batch, axis=0)
        workers = list(client.scheduler_info()["workers"])
        client.retire_workers(workers=workers, close_workers=True)
        client.shutdown()

        if self.for_NJ:

            elev_values = np.squeeze(constant_batch[:, :, 0]).reshape(-1, 1)
            lat_values, lon_values = np.meshgrid(lat_batch, lon_batch)
            spherical_coords = get_spherical(
                lat_values.reshape(-1, 1), lon_values.reshape(-1, 1), elev_values
            )

            kdtree = KDTree(spherical_coords)

            pairs = []

            for i_coord, coord in enumerate(spherical_coords):
                pairs.append(
                    np.vstack(
                        (
                            np.full(3, fill_value=i_coord).reshape(1, -1),
                            kdtree.query(coord, k=3)[1],
                        )
                    )
                )

            pairs = np.hstack((pairs))

            rainfall_path = torch.cat(
                (
                    torch.from_numpy(
                        y_batch.precipitation.fillna(0).values.reshape(
                            self.batch_size[0], -1, 1
                        )
                    ).float(),
                    X_batch.reshape(self.batch_size[0], -1, len(self.variables) * 4),
                ),
                dim=-1,
            )
            obs_dates = np.ones(self.batch_size[0]).reshape(1, -1)
            n_obs = np.array([self.batch_size[0]])
            if self.for_val:
                obs_dates = np.zeros(self.batch_size[0]).reshape(1, -1)
                n_obs = np.random.randint(1, self.batch_size[0] - 8, 1)
                obs_dates[: n_obs[0]] = 1

            return {
                "idx": idx,
                "rainfall_path": rainfall_path[None, :, :, :],
                "observed_dates": obs_dates,
                "nb_obs": n_obs,
                "dt": 1,
                "edge_indices": pairs,
                "obs_noise": None,
            }

        else:

            if self.antialiasing:
                if isinstance(y_batch, list):

                    y_batch = torch.stack(
                        [
                            torch.from_numpy(
                                y.precipitation.fillna(self.fill_value).values
                            ).float()
                            for y in y_batch
                            if y.time.values in dates_modified
                        ],
                        axis=0,
                    )
                else:
                    y_batch = y_batch.precipitation.fillna(self.fill_value).values

                antialiaser = Antialiasing()
                y_batch = antialiaser(y_batch)
                y_batch = torch.from_numpy(np.moveaxis(y_batch, 1, -1)).float()

            else:
                if isinstance(y_batch, list):

                    y_batch = torch.stack(
                        [
                            torch.from_numpy(
                                y.precipitation.fillna(self.fill_value).values[
                                    :, :, :, None
                                ]
                            ).float()
                            for y in y_batch
                            if y.time.values in dates_modified
                        ],
                        axis=0,
                    )

                else:
                    y_batch = torch.from_numpy(
                        y_batch.precipitation.fillna(self.fill_value).values[
                            :, :, :, None
                        ]
                    ).float()
            if not self.return_seeps:
                if len(n_samp) == 1:
                    return (X_batch, y_batch)
                else:
                    return [
                        (X[0], y[0])
                        for X, y in zip(
                            torch.cat((X_batch, constant_batch), dim=-1).chunk(
                                len(n_samp), dim=0
                            ),
                            y_batch.chunk(len(n_samp), dim=0),
                        )
                    ]
            else:
                if len(n_samp) == 1:
                    return (X_batch, y_batch, seeps_batch)
                else:
                    return [
                        (X[0], y[0], seep[0])
                        for X, y, seep in zip(
                            torch.cat((X_batch, constant_batch), dim=-1).chunk(
                                len(n_samp), dim=0
                            ),
                            y_batch.chunk(len(n_samp), dim=0),
                            seeps_batch.chunk(len(n_samp), dim=0),
                        )
                    ]


class StreamTruth(torch.utils.data.Dataset):

    """
    class for iterating over a dataset
    """

    def __init__(
        self,
        y,
        batch_size=[4, 128, 128],
        weighted_sampler=True,
        for_NJ=False,
        for_val=False,
        length=None,
        antialiasing=False,
        transform=None,
        return_dataset=False,
        fill_value=np.log10(0.02),
    ):

        self.batch_size = batch_size
        self.for_NJ = for_NJ
        self.for_val = for_val
        self.length = length
        self.antialiasing = antialiasing
        self.transform = transform
        self.return_dataset = return_dataset
        self.fill_value = fill_value
        overlap = (
            {"latitude": int(batch_size[1] - 8), "longitude": int(batch_size[2] - 8)}
            if for_NJ
            else {"lat": int(batch_size[1] // 8), "lon": int(batch_size[2] // 8)}
        )
        self.y_generator = xbatcher.BatchGenerator(
            y,
            {
                "time": batch_size[0],
                "latitude" if for_NJ else "lat": batch_size[1],
                "longitude" if for_NJ else "lon": batch_size[2],
            },
            input_overlap=overlap,
        )

        if weighted_sampler:
            if self.for_NJ:
                y_train = [
                    self.y_generator[i].mean(
                        ["time", "latitude", "longitude"], skipna=False
                    )
                    for i in range(len(self.y_generator))
                ]
            else:
                y_train = [
                    self.y_generator[i].precipitation.max(
                        ["time", "lat", "lon"], skipna=False
                    )
                    for i in range(len(self.y_generator))
                ]
            rounded_y_train = np.round(y_train, decimals=1)
            unique_classes = np.unique(rounded_y_train)
            class_sample_count = np.bincount(
                np.digitize(rounded_y_train, unique_classes) - 1
            )
            weight = 1.0 / class_sample_count
            samples_weight = weight[np.digitize(rounded_y_train, unique_classes) - 1]
            self.samples_weight = torch.from_numpy(np.asarray(samples_weight))
            self.sampler = torch.utils.data.WeightedRandomSampler(
                self.samples_weight.type("torch.DoubleTensor"), len(samples_weight)
            )

    def __len__(self) -> int:
        return len(self.y_generator)

    def __getitem__(self, idx):

        y_batch = self.y_generator[idx]

        if self.return_dataset:
            return y_batch

        if self.for_NJ:

            def generate(y_batch, length, stop=None):

                rng = np.random.default_rng()
                random_year = rng.choice(np.unique(y_batch["time.year"].values), 1)[0]
                ds_sel = y_batch.sel(
                    {
                        "time": slice(
                            "%i-01-01" % random_year, "%i-01-01" % (random_year + 1)
                        )
                    }
                )

                time_of_event = rng.choice(ds_sel.time.values[length:-length], 1)[0]

                time_to_event = rng.choice(np.arange(length), 1)[0]
                time_after_event = length - time_to_event - 1

                rainfall_path = ds_sel.sel(
                    {
                        "time": slice(
                            time_of_event - np.timedelta64(time_to_event * 30, "m"),
                            time_of_event + np.timedelta64(time_after_event * 30, "m"),
                        ),
                    }
                )
                times_rainfall = rainfall_path.time.values
                rainfall_path = rainfall_path.fillna(0).values[None, :, :, :]

                if stop is not None:
                    # limit observations to once a day
                    nb_obs_single = stop
                    obs_ptr = np.arange(1, nb_obs_single)

                else:
                    nb_obs_single = length
                    obs_ptr = np.arange(1, length)

                observed_date = np.zeros(rainfall_path.shape[1])
                observed_date[0] = 1

                for i_obs in obs_ptr:

                    observed_date[i_obs] = 1

                return rainfall_path, observed_date, nb_obs_single

            rainfall_paths = []
            observed_dates = []
            n_obs = []

            stop = None
            batch_size = 50
            if self.for_val:
                rng = np.random.default_rng()
                stop = rng.choice(np.arange(2, self.length - 100), 1)[0]
                batch_size = 2

            for i in range(batch_size):
                rainfall_path, observed_date, nb_obs = generate(
                    y_batch, self.length, stop=stop
                )
                rainfall_paths.append(rainfall_path)
                observed_dates.append(observed_date)
                n_obs.append(nb_obs)

            rainfall_paths = np.vstack(rainfall_paths)
            observed_dates = np.stack(observed_dates)
            n_obs = np.asarray(n_obs)

            return {
                "idx": idx,
                "rainfall_path": torch.tensor(
                    rainfall_paths[:, :, :, :, None], dtype=torch.float32
                ),
                "observed_dates": observed_dates,
                "nb_obs": n_obs,
                "dt": 1,
                "obs_noise": None,
            }

        else:
            if self.antialiasing:
                antialiaser = Antialiasing()
                y_batch = y_batch.precipitation.fillna(self.fill_value).values
                y_batch = antialiaser(y_batch)
                y_batch = torch.tensor(np.moveaxis(y_batch, 0, -1), dtype=torch.float32)

            else:
                y_batch = torch.tensor(
                    y_batch.precipitation.fillna(self.fill_value).values[:, :, :, None],
                    dtype=torch.float32,
                )
            if self.transform:
                y_batch = self.transform(y_batch)

            return y_batch
