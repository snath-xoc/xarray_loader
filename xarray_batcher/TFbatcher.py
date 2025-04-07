from typing import Any, Callable, Optional, Tuple
import numpy as np
import xarray as xr
import xbatcher
import tensorflow as tf

import load_zarr as lz
import utils

import importlib

importlib.reload(lz)
importlib.reload(utils)

def load_fcst_truth_mask(idx_zarr, zarr_store, df_truth_and_mask, df_constants, all_fcst_fields,
                         streamline_type="hourly_split", batch_size=[2, 128, 128], **kwargs):
    
    X = xr.concat(
        [
            lz.streamline_and_normalise_zarr(
                field,
                lz.load_da_from_zarr_store(
                    zarr_store_var[idx_zarr : idx_zarr + batch_size[0] + 1],
                    field,
                    from_idx=True,
                ), streamline_type = streamline_type, **kwargs,
            )
            .expand_dims(dim={"field": [field]}, axis=0)
            for field, zarr_store_var in zip(all_fcst_fields, zarr_store)
        ],
        dim="field",
    ).to_dataset(dim="field")

    X = X.sortby("time").drop_duplicates("time").isel({"time": np.arange(batch_size[0])})
    
    if streamline_type=="hourly_split":
        df_truth_and_mask_batch = df_truth_and_mask.sel(
            {"time": X.time.values}
        )
    else:
        df_truth_and_mask_batch = df_truth_and_mask.sel(
            {"time": X.time.values + np.timedelta64(1, "D") + np.timedelta64(6, "h")}
        )
        
    df_constants_batch = df_constants.isel({"time": np.arange(batch_size[0])})
    
    variables = [var for var in X.data_vars]
    constants = [constant for constant in df_constants_batch.data_vars]

    return X, df_truth_and_mask_batch, df_constants_batch, variables, constants
    

def data_format(X, df_truth_and_mask_batch, df_constants_batch, variables, constants, batch_size=[2, 128, 128], full=False):

    if full:
    
        return (
                {
                    "lo_res_inputs": np.moveaxis(
                        np.hstack(
                            (
                                [
                                    X[variable]
                                    .fillna(0)
                                    .values
                                    .reshape(
                                        batch_size[0],-1, len(X.lat.values), len(X.lon.values)
                                    )
                                    for variable in variables
                                ]
                            )
                        ),
                        1,
                        -1,
                    ),
                    "hi_res_inputs": np.moveaxis(
                        np.stack(
                            (
                                [
                                    df_constants_batch[constant]
                                    .fillna(0)
                                    .values
                                    .reshape(
                                        -1, len(X.lat.values), len(X.lon.values)
                                    )
                                    for constant in constants
                                ]
                            )
                        ),
                        0,
                        -1,
                    ),
                },
                {
                    "output": df_truth_and_mask_batch.precipitation.values,
                    "mask": df_truth_and_mask_batch.mask.values,
                },
            )
    
    else:
        return (
            xbatcher.BatchGenerator(
                X.fillna(0),
                {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[1]},
                input_overlap={"lat": batch_size[1] - 1, "lon": batch_size[1] - 1},
            ),
            xbatcher.BatchGenerator(
                df_constants_batch.fillna(0),
                {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[1]},
                input_overlap={"lat": batch_size[1] - 1, "lon": batch_size[1] - 1},
            ),
            xbatcher.BatchGenerator(
                df_truth_and_mask_batch,
                {"time": batch_size[0], "lat": batch_size[1], "lon": batch_size[1]},
                input_overlap={"lat": batch_size[1] - 1, "lon": batch_size[1] - 1},
            ),
        )

def zarr_store_loader(
    zarr_store, df_truth_and_mask, df_constants, batch_size=[2, 128, 128], full=False, hourly=True,
):
    if not isinstance(batch_size, list):
        batch_size = [batch_size]

    zarr_len_all = np.array([len(zarr_store_var) for zarr_store_var in zarr_store])
    
    all_fcst_fields, _, _, _ = utils.get_config()

    for idx_zarr in range(8, zarr_len_all.max() - batch_size[0]):

        if hourly:

            for split_steps in [[0,2],[2,4],[4,6],[6,8]]:

                X, df_constants_batch, df_truth_and_mask_batch, variables, constants = load_fcst_truth_mask(
                    idx_zarr, zarr_store, df_truth_and_mask, df_constants, all_fcst_fields,
                    streamline_type="hourly_split", batch_size=batch_size, split_steps=split_steps)
                yield data_format(X, df_constants_batch, df_truth_and_mask_batch, variables, constants, full=full, batch_size=batch_size)
        else:

            X, df_constants_batch, df_truth_and_mask_batch, variables, constants = load_fcst_truth_mask(
                    idx_zarr, zarr_store, df_truth_and_mask, df_constants, all_fcst_fields,
                streamline_type="var_and_norm_over_day", batch_size=batch_size)
            yield data_format(X, df_constants_batch, df_truth_and_mask_batch, variables, constants, full=full)
            
            


class CustomTFDataset(tf.keras.utils.Sequence):
    def __init__(
        self,
        X_generator,
        constant_generator,
        y_generator,
    ) -> None:
        """
        Keras Dataset adapter for Xbatcher

        Parameters
        ----------
        X_generator : xbatcher.BatchGenerator
        y_generator : xbatcher.BatchGenerator
        transform : callable, optional
            A function/transform that takes in an array and returns a transformed version.
        target_transform : callable, optional
            A function/transform that takes in the target and transforms it.
        """
        self.X_generator = X_generator
        self.constant_generator = constant_generator
        self.y_generator = y_generator

    def __len__(self) -> int:
        return len(self.X_generator)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        variables = [var for var in self.X_generator[0].data_vars]
        constants = [constant for constant in self.constant_generator[0].data_vars]

        X_batch = tf.convert_to_tensor(
            np.moveaxis(
                np.hstack(
                    ([self.X_generator[idx][variable].values for variable in variables])
                ),
                1,
                -1,
            )
        )
        constant_batch = tf.convert_to_tensor(
            np.moveaxis(
                np.stack(
                    (
                        [
                            self.constant_generator[idx][constant].values
                            for constant in constants
                        ]
                    )
                ),
                0,
                -1,
            )
        )
        y_batch = tf.convert_to_tensor(
            self.y_generator[idx].precipitation.fillna(0).values[:,:,:,None]
        )
        mask_batch = tf.convert_to_tensor(self.y_generator[idx].mask.values[:,:,:,None])

        return (
            {"lo_res_inputs": X_batch, "hi_res_inputs": constant_batch},
            {"output": y_batch, "mask": mask_batch},
        )


def batch_from_zarr_store(df_vars, df_truth, df_constants, **kwargs):
    loader_time = zarr_store_loader(df_vars, df_truth, df_constants, **kwargs)

    for X_gen, constants_gen, y_gen in loader_time:
        loader = CustomTFDataset(X_gen, constants_gen, y_gen)
        print(len(loader))

        for batch in loader:
            yield batch


