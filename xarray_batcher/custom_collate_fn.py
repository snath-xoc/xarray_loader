import numpy as np
import torch

def _get_func(name):
    """
    transform a function given as str to a python function
    :param name: str, correspond to a function,
            supported: 'exp', 'power-x' (x the wanted power)
    :return: numpy fuction
    """
    if name in ['exp', 'exponential']:
        return np.exp
    if 'power-' in name:
        x = float(name.split('-')[1])
        def pow(input):
            return np.power(input, x)
        return pow
    else:
        try:
            return eval(name)
        except Exception:
            return None


def _get_X_with_func_appl(X, functions, axis):
    """
    apply a list of functions to the paths in X and append X by the outputs
    along the given axis
    :param X: np.array, with the data,
    :param functions: list of functions to be applied
    :param axis: int, the data_dimension (not batch and not time dim) along
            which the new paths are appended
    :return: np.array
    """
    Y = X
    for f in functions:
        Y = np.concatenate([Y, f(X)], axis=axis)
    return Y


def CustomCollateFnGen(func_names=None):
    """
    a function to get the costume collate function that can be used in
    torch.DataLoader with the wanted functions applied to the data as new
    dimensions
    -> the functions are applied on the fly to the dataset, and this additional
    data doesn't have to be saved

    :param func_names: list of str, with all function names, see _get_func
    :return: collate function, int (multiplication factor of dimension before
                and after applying the functions)
    """
    # get functions that should be applied to X, additionally to identity 
    functions = []
    if func_names is not None:
        for func_name in func_names:
            f = _get_func(func_name)
            if f is not None:
                functions.append(f)
    mult = len(functions) + 1

    def custom_collate_fn(batch):
        dt = batch[0]['dt']
        stock_paths = np.concatenate([b['rainfall_path'] for b in batch], axis=0)
        observed_dates = np.concatenate([b['observed_dates'] for b in batch],
                                        axis=0)
        #edge_indices = np.concatenate([b['edge_indices'] for b in batch], axis=0)
        obs_noise = None
        if batch[0]["obs_noise"] is not None:
            obs_noise = np.concatenate([b['obs_noise'] for b in batch], axis=0)
        masked = False
        mask = None
        if len(observed_dates.shape) == 3:
            masked = True
            mask = observed_dates
            observed_dates = observed_dates.max(axis=1)
        nb_obs = torch.tensor(
            np.concatenate([b['nb_obs'] for b in batch], axis=0))

        # here axis=1, since we have elements of dim
        #    [batch_size, data_dimension] => add as new data_dimensions
        sp = stock_paths[:,0]
        if obs_noise is not None:
            sp = stock_paths[:, :, 0] + obs_noise[:, :, 0]
        start_X = torch.tensor(
            _get_X_with_func_appl(sp, functions, axis=1),
            dtype=torch.float32)
        X = []
        if masked:
            M = []
            start_M = torch.tensor(mask[:,:,0], dtype=torch.float32).repeat(
                (1,mult))
        else:
            M = None
            start_M = None
        times = []
        time_ptr = [0]
        obs_idx = []
        current_time = 0.
        counter = 0
        for t in range(1, observed_dates.shape[-1]):
            current_time += dt
            if observed_dates[:, t].sum() > 0:
                times.append(current_time)
                for i in range(observed_dates.shape[0]):
                    if observed_dates[i, t] == 1:
                        counter += 1
                        # here axis=0, since only 1 dim (the data_dimension),
                        #    i.e. the batch-dim is cummulated outside together
                        #    with the time dimension
                        sp = stock_paths[i, t]
                        if obs_noise is not None:
                            sp = stock_paths[i, :, t] + obs_noise[i, :, t]
                        X.append(_get_X_with_func_appl(sp, functions, axis=0))
                        if masked:
                            M.append(np.tile(mask[i, :, t], reps=mult))
                        obs_idx.append(i)
                time_ptr.append(counter)
        # if obs_noise is not None:
        #     print("noisy observations used")

        assert len(obs_idx) == observed_dates[:, 1:].sum()
        if masked:
            M = torch.tensor(np.array(M), dtype=torch.float32)
        res = {'times': np.array(times), 'time_ptr': np.array(time_ptr),
               'obs_idx': torch.tensor(obs_idx, dtype=torch.long),
               'start_X': start_X, 'n_obs_ot': nb_obs,
               'X': torch.tensor(np.array(X), dtype=torch.float32).permute(3,1,2,0),
               'true_paths': stock_paths, 'observed_dates': observed_dates,
               'true_mask': mask, 'obs_noise': obs_noise, #'edge_indices': torch.from_numpy(edge_indices).long().contiguous(),
               'M': M, 'start_M': start_M}
        return res

    return custom_collate_fn, mult
