import numpy as np
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
