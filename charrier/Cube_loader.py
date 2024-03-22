import numpy as np
import os
from mjd2date import mjd2date  # /ST_RELEASE/UTILITIES/PYTHON/mjd2date.py
import xarray as xr
import pandas as pd
import dask
from pyproj import Proj, Transformer, CRS
import rasterio.enums
from datetime import date

class cube_data_class:

    def __init__(self):
        self.filedir = ''
        self.filename = ''
        self.nx = 250
        self.ny = 250
        self.nz = 0
        self.author = ''
        self.ds = xr.Dataset({})

    def subset(self, proj, subset):
        """
        Crop according to 4 coordinates
        :param proj: EPSG system of the coordinates given in subset
        :param subset: list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :return: nothing, crop self.ds without the need of returning it
        """
        if CRS(self.ds.proj4) != CRS(proj):
            transformer = Transformer.from_crs(CRS(proj), CRS(self.ds.proj4)) #convert the coordinates from proj to self.ds.proj4
            lon1, lat1 = transformer.transform(subset[2], subset[1])
            lon2, lat2 = transformer.transform(subset[3], subset[1])
            lon3, lat3 = transformer.transform(subset[2], subset[1])
            lon4, lat4 = transformer.transform(subset[3], subset[0])
            self.ds = self.ds.sel(x=slice(np.min([lon1, lon2, lon3, lon4]), np.max([lon1, lon2, lon3, lon4])),
                                  y=slice(np.max([lat1, lat2, lat3, lat4]), np.min([lat1, lat2, lat3, lat4])))
            del lon1, lon2, lon3, lon4, lat1, lat2, lat3, lat4
        else:
            self.ds = self.ds.sel(x=slice(np.min([subset[0], subset[1]]), np.max([subset[0], subset[1]])),
                                  y=slice(np.max([subset[2], subset[3]]), np.min([subset[2], subset[3]])))

    def buffer(self, proj, buffer):
        """
        Crop the dataset around a given pixel, according to a given buffer
        :param proj: EPSG system of the coordinates given in subset
        :param buffer:  a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :return: nothing, crop self.ds without the need of returning it
        """
        if CRS(self.ds.proj4) != CRS(proj):
            transformer = Transformer.from_crs(CRS(proj), CRS(self.ds.proj4)) #convert the coordinates from proj to self.ds.proj4
            i1, j1 = transformer.transform(buffer[1] + buffer[2],
                                           buffer[0] - buffer[2])
            i2, j2 = transformer.transform(buffer[1] - buffer[2],
                                           buffer[0] + buffer[2])
            i3, j3 = transformer.transform(buffer[1] + buffer[2],
                                           buffer[0] + buffer[2])
            i4, j4 = transformer.transform(buffer[1] - buffer[2],
                                           buffer[0] - buffer[2])
            self.ds = self.ds.sel(x=slice(np.min([i1, i2, i3, i4]), np.max([i1, i2, i3, i4])),
                                  y=slice(np.max([j1, j2, j3, j4]), np.min([j1, j2, j3, j4])))
            del i3, i4, j3, j4
        else:
            i1, j1 = buffer[0] - buffer[2], buffer[1] + buffer[2]
            i2, j2 = buffer[0] + buffer[2], buffer[1] - buffer[2]
            self.ds = self.ds.sel(x=slice(np.min([i1, i2]), np.max([i1, i2])),
                                  y=slice(np.max([j1, j2]), np.min([j1, j2])))
        del i1, i2, j1, j2, buffer

    # ====== = ====== LOAD DATASET ====== = ======
    def load_itslive(self, filepath, conf=False, pick_date=None, subset=None,
                     pick_sensor=None, pick_temp_bas=None, buffer=None,
                     verbose=False, proj='EPSG:4326'):  # {{{
        """
        Load a cube dataset written by ITS_LIVE
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """
        if verbose:
            print(filepath)

        self.filedir = os.path.dirname(filepath) # path were is stored the netcdf file
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        self.ds = self.ds.assign_attrs({'proj4': self.ds['mapping'].proj4text})
        self.author = self.ds.author.split(', a NASA')[0]
        self.source = self.ds.url

        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)

        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        if pick_date is not None:
            self.ds = self.ds.where(((self.ds['acquisition_date_img1'] >= np.datetime64(pick_date[0])) & (
                    self.ds['acquisition_date_img2'] <= np.datetime64(pick_date[1]))).compute(), drop=True)

        self.nx = self.ds.dims['x']
        self.ny = self.ds.dims['y']
        self.nz = self.ds.dims['mid_date']

        if conf:
            minconfx = np.nanmin(self.ds['vx_error'].values[:])
            maxconfx = np.nanmax(self.ds['vx_error'].values[:])
            minconfy = np.nanmin(self.ds['vy_error'].values[:])
            maxconfy = np.nanmax(self.ds['vy_error'].values[:])

        date1 = np.array([np.datetime64(date_str, 'ns') for date_str in self.ds['acquisition_date_img1'].values])
        date2 = np.array([np.datetime64(date_str, 'ns') for date_str in self.ds['acquisition_date_img2'].values])
        sensor = np.core.defchararray.add(np.char.strip(self.ds['mission_img1'].values.astype(str), '�'),
                                          np.char.strip(self.ds['satellite_img1'].values.astype(str), '�')
                                          ).astype(
            'U10')  # np.char.strip is used to remove the null character ('�') from each elemen and np.core.defchararray.add to concatenate array of different types
        sensor[sensor == 'L7'] = 'Landsat-7'
        sensor[sensor == 'L8'] = 'Landsat-8'
        sensor[sensor == 'L9'] = 'Landsat-9'
        sensor[np.isin(sensor, ['S1A', 'S1B'])] = 'Sentinel-1'
        sensor[np.isin(sensor, ['S2A', 'S2B'])] = 'Sentinel-2'

        if conf:  # normalize the error between 0 and 1, and convert error in confidence
            errorx = 1 - (self.ds['vx_error'].values - minconfx) / (maxconfx - minconfx)
            errory = 1 - (self.ds['vy_error'].values - minconfy) / (maxconfy - minconfy)
        else:
            errorx = self.ds['vx_error'].values
            errory = self.ds['vy_error'].values

        # Drop variables not in the specified list
        variables_to_keep = ['vx', 'vy', 'mid_date', 'x', 'y']
        self.ds = self.ds.drop_vars([var for var in self.ds.variables if var not in variables_to_keep])
        # Drop attributes not in the specified list
        attributes_to_keep = ['date_created', 'mapping', 'author', 'proj4']
        self.ds.attrs = {attr: self.ds.attrs[attr] for attr in attributes_to_keep if attr in self.ds.attrs}

        # self.ds = self.ds.unify_chunks()  # to avoid error ValueError: Object has inconsistent chunks along dimension mid_date. This can be fixed by calling unify_chunks().
        # Create new variable and chunk them
        self.ds['sensor'] = xr.DataArray(sensor, dims='mid_date').chunk(chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['date1'] = xr.DataArray(date1, dims='mid_date').chunk(chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['date2'] = xr.DataArray(date2, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['source'] = xr.DataArray(['ITS_LIVE'] * self.nz, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['errorx'] = xr.DataArray(
            errorx,
            dims=['mid_date'],
            coords={'mid_date': self.ds.mid_date}).chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds = self.ds.unify_chunks()
        self.ds['errory'] = xr.DataArray(
            errory,
            dims=['mid_date'],
            coords={'mid_date': self.ds.mid_date}).chunk(
            chunks=self.ds.chunks['mid_date'])

        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds['sensor'].isin(pick_sensor))
        if pick_temp_bas is not None:
            temp = ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))
            self.ds = self.ds.where(((pick_temp_bas[0] < temp) & (temp < pick_temp_bas[1])).compute(), drop=True)
            del temp
        self.ds = self.ds.unify_chunks()

    def load_millan(self, filepath, conf=False, pick_date=None, subset=None,
                    pick_sensor=None, pick_temp_bas=None, buffer=None,
                    verbose=False, proj='EPSG:4326'):
        """
        Load a cube dataset written by R. Millan et al.
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """

        if verbose:
            print(filepath)
        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        self.author = 'IGE'  # name of the author
        self.source = self.ds.source
        del filepath

        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)

        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        # Uniformization of the name and format of the time coordinate
        self.ds = self.ds.rename({'z': 'mid_date'})
        date1 = [mjd2date(date_str) for date_str in self.ds['date1'].values]  # convertion in date
        date2 = [mjd2date(date_str) for date_str in self.ds['date2'].values]
        self.ds['date1'] = xr.DataArray(np.array(date1).astype('datetime64[ns]'), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['date2'] = xr.DataArray(np.array(date2).astype('datetime64[ns]'), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        del date1, date2

        if pick_date is not None:  # Temporal subset between two dates
            self.ds = self.ds.where(
                ((self.ds['date1'] >= np.datetime64(pick_date[0])) & (
                        self.ds['date2'] <= np.datetime64(pick_date[1]))).compute(),
                drop=True)
        del pick_date

        self.ds = self.ds.assign_coords(
            mid_date=np.array(self.ds['date1'] + (self.ds['date2'] - self.ds['date1']) // 2))

        self.nx = self.ds.sizes['x']
        self.ny = self.ds.sizes['y']
        self.nz = self.ds.sizes['mid_date']

        if conf and 'confx' not in self.ds.data_vars:  # convert the errors into confidence indicators between 0 and 1
            minconfx = np.nanmin(self.ds['error_vx'].values[:])
            maxconfx = np.nanmax(self.ds['error_vx'].values[:])
            minconfy = np.nanmin(self.ds['error_vy'].values[:])
            maxconfy = np.nanmax(self.ds['error_vy'].values[:])
            errorx = 1 - (self.ds['error_vx'].values - minconfx) / (maxconfx - minconfx)
            errory = 1 - (self.ds['error_vy'].values - minconfy) / (maxconfy - minconfy)
        else:
            errorx = self.ds['error_vx'].values[:]
            errory = self.ds['error_vy'].values[:]

        # Homogenize sensors names
        sensor = np.char.strip(self.ds['sensor'].values.astype(str),
                               '�')  # np.char.strip is used to remove the null character ('�') from each element
        sensor[np.isin(sensor, ['S1'])] = 'Sentinel-1'
        sensor[np.isin(sensor, ['S2'])] = 'Sentinel-2'
        sensor[np.isin(sensor, ['landsat-8', 'L8', 'L8. '])] = 'Landsat-8'

        # Drop variables not in the specified list
        self.ds = self.ds.drop_vars(
            [var for var in self.ds.variables if var not in ['vx', 'vy', 'mid_date', 'x', 'y', 'date1', 'date2']])
        self.ds = self.ds.transpose('mid_date', 'y', 'x')

        # Store the variable in xarray dataset
        self.ds['sensor'] = xr.DataArray(sensor, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        del sensor
        self.ds['source'] = xr.DataArray(['IGE'] * self.nz, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['errorx'] = xr.DataArray(np.tile(errorx[:, np.newaxis, np.newaxis], (1, self.ny, self.nx)),
                                         dims=['mid_date', 'y', 'x'],
                                         coords={'mid_date': self.ds.mid_date, 'y': self.ds.y, 'x': self.ds.x}).chunk(
            chunks=self.ds.chunks)
        self.ds['errory'] = xr.DataArray(np.tile(errory[:, np.newaxis, np.newaxis], (1, self.ny, self.nx)),
                                         dims=['mid_date', 'y', 'x'],
                                         coords={'mid_date': self.ds.mid_date, 'y': self.ds.y, 'x': self.ds.x}).chunk(
            chunks=self.ds.chunks)
        del errorx, errory

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds['sensor'].isin(pick_sensor))
        if pick_temp_bas is not None:
            self.ds = self.ds.sel(
                mid_date=(pick_temp_bas[0] < ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))) & (
                        ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D')) < pick_temp_bas[1]))
        self.ds = self.ds.unify_chunks()

    def load_ducasse(self, filepath, conf=False, pick_date=None, subset=None,
                     pick_sensor=None, pick_temp_bas=None, buffer=None,
                     verbose=False, proj='EPSG:4326'):
        """
        Load a cube dataset written by E. Ducasse et al.
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """

        if verbose:
            print(filepath)
        self.ds = self.ds.chunk({'x': 125, 'y': 125, 'time': 2000})  # set chunk
        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        # self.author = self.ds.author  # name of the author
        self.author = 'IGE'  # name of the author
        del filepath

        # Spatial subset
        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)

        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        # Uniformization of the name and format of the time coordinate
        self.ds = self.ds.rename({'time': 'mid_date'})
        date1 = [date_str.split(' ')[0] for date_str in self.ds['mid_date'].values]
        date2 = [date_str.split(' ')[1] for date_str in self.ds['mid_date'].values]
        self.ds['date1'] = xr.DataArray(np.array(date1).astype('datetime64[ns]'), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['date2'] = xr.DataArray(np.array(date2).astype('datetime64[ns]'), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        del date1, date2

        if pick_date is not None:  # Temporal subset between two dates
            self.ds = self.ds.where(
                ((self.ds['date1'] >= np.datetime64(pick_date[0])) & (
                            self.ds['date2'] <= np.datetime64(pick_date[1]))).compute(),
                drop=True)
        del pick_date

        self.ds = self.ds.assign_coords(
            mid_date=np.array(self.ds['date1'] + (self.ds['date2'] - self.ds['date1']) // 2))

        self.nx = self.ds.dims['x']
        self.ny = self.ds.dims['y']
        self.nz = self.ds.dims['mid_date']

        # Drop variables not in the specified list
        variables_to_keep = ['vx', 'vy', 'mid_date', 'x', 'y', 'date1', 'date2']
        self.ds = self.ds.drop_vars([var for var in self.ds.variables if var not in variables_to_keep])
        self.ds = self.ds.transpose('mid_date', 'y', 'x')

        # Store the variable in xarray dataset
        self.ds['sensor'] = xr.DataArray(['Pleiades'] * len(self.ds['mid_date']), dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['source'] = xr.DataArray(['IGE'] * self.nz, dims='mid_date').chunk(
            chunks=self.ds.chunks['mid_date'])
        self.ds['vy'] = -self.ds['vy']

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds['sensor'].isin(pick_sensor))
        if pick_temp_bas is not None:
            self.ds = self.ds.sel(
                mid_date=(pick_temp_bas[0] < ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))) & (
                        ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D')) < pick_temp_bas[1]))

    def load_charrier(self, filepath, conf=False, pick_date=None, subset=None,
                            pick_sensor=None, pick_temp_bas=None, buffer=None,
                      verbose=False, proj='EPSG:4326'):
        """
        Load a cube dataset written by L.Charrier et al.
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """

        if verbose:
            print(filepath)
        self.filedir = os.path.dirname(filepath)
        self.filename = os.path.basename(filepath)  # name of the netcdf file
        if self.ds.author == 'J. Mouginot, R.Millan, A.Derkacheva_aligned':
            self.author = 'IGE'  # name of the author
        else:
            self.author = self.ds.author

        self.source = self.ds.source
        del filepath

        if subset is not None:  # crop according to 4 coordinates
            self.subset(proj, subset)

        elif buffer is not None:  # crop the dataset around a given pixel, according to a given buffer
            self.buffer(proj, buffer)

        if pick_date is not None:  # Temporal subset between two dates
            self.ds = self.ds.where(
                ((self.ds['date1'] >= np.datetime64(pick_date[0])) & (
                            self.ds['date2'] <= np.datetime64(pick_date[1]))).compute(),
                drop=True)
        del pick_date

        self.nx = self.ds.dims['x']
        self.ny = self.ds.dims['y']
        self.nz = self.ds.dims['mid_date']

        # Pick sensors or temporal baselines
        if pick_sensor is not None:
            self.ds = self.ds.sel(mid_date=self.ds['sensor'].isin(pick_sensor))

        if pick_temp_bas is not None:
            self.ds = self.ds.sel(
                mid_date=(pick_temp_bas[0] < ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D'))) & (
                        ((self.ds['date2'] - self.ds['date1']) / np.timedelta64(1, 'D')) < pick_temp_bas[1]))

        if conf and 'confx' not in self.ds.data_vars:  # convert the errors into confidence indicators between 0 and 1
            minconfx = np.nanmin(self.ds['errorx'].values[:])
            maxconfx = np.nanmax(self.ds['errorx'].values[:])
            minconfy = np.nanmin(self.ds['errory'].values[:])
            maxconfy = np.nanmax(self.ds['errory'].values[:])
            errorx = 1 - (self.ds['errorx'].values - minconfx) / (maxconfx - minconfx)
            errory = 1 - (self.ds['errory'].values - minconfy) / (maxconfy - minconfy)
            self.ds['errorx'] = xr.DataArray(errorx,
                                             dims=['mid_date', 'y', 'x'],
                                             coords={'mid_date': self.ds.mid_date, 'y': self.ds.y,
                                                     'x': self.ds.x}).chunk(
                chunks=self.ds.chunks)
            self.ds['errory'] = xr.DataArray(errory,
                                             dims=['mid_date', 'y', 'x'],
                                             coords={'mid_date': self.ds.mid_date, 'y': self.ds.y,
                                                     'x': self.ds.x}).chunk(
                chunks=self.ds.chunks)

        # For cube writen with write_result_TICOI
        if 'source' not in self.ds.variables:
            self.ds['source'] = xr.DataArray([self.ds.author] * self.nz, dims='mid_date').chunk(
                chunks=self.ds.chunks['mid_date'])
        if 'sensor' not in self.ds.variables:
            self.ds['sensor'] = xr.DataArray([self.ds.sensor] * self.nz, dims='mid_date').chunk(
                chunks=self.ds.chunks['mid_date'])

    def load(self, filepath=None, conf=False, pick_date=None, subset=None,
             pick_sensor=None, pick_temp_bas=None, buffer=None, proj=None, chunks={},
             verbose=True):
        """
        Load a cube dataset which could be in format netcdf or zarr
        :param filepath: str or None, filepath of the dataset, if None the code will search which
        :param conf: True or False, if True convert the error in confidence between 0 and 1
        :param pick_date: a list of 2 string yyyy-mm-dd, pick the data between these two date
        :param subset: a list of 4 float, these values are used to give a subset of the dataset : [xmin,xmax,ymax,ymin]
        :param pick_sensor: a list of strings, pick only the corresponding sensors
        :param pick_temp_bas: a list of 2 integer, pick only the data which have a temporal baseline between these two integers
        :param buffer: a list of 3 float, the first is the longitude, the second the latitude of the central point, the last is the buffer around which the subset will be performed (in pixels)
        :param proj: str, projection of the buffer or subset which is given, e.g. EPSG:4326
        :param chunks: dictionary with the size of chunks for each dimension, if chunks=-1 loads the dataset with dask using a single chunk for all arrays. chunks={} loads the dataset with dask using engine preferred chunks if exposed by the backend, otherwise with a single chunk for all arrays, chunks='auto' will use dask auto chunking taking into account the engine preferred chunks.
        :param verbose: bool, display some text
        :return: cube_data_class object where cube_data_class.ds is an xarray.DataArray
        """
        self.__init__()
        with dask.config.set(**{'array.slicing.split_large_chunks': True}):#To avoid creating the large chunks
            if filepath.split('.')[-1] == 'nc':
                self.ds = xr.open_dataset(filepath, engine="netcdf4", chunks=chunks)
            elif filepath.split('.')[-1] == 'zarr':
                if chunks=={}:chunks='auto'#change the default value to auto

                self.ds = xr.open_dataset(filepath, decode_timedelta=False, engine='zarr', consolidated=True,
                                              chunks=chunks)

        if verbose: print('file open')
        if 'Author' in self.ds.attrs:  # uniformization of the attribute Author to author
            self.ds.attrs['author'] = self.ds.attrs.pop('Author')

        dico_load = {'ITS_LIVE, a NASA MEaSUREs project (its-live.jpl.nasa.gov)': self.load_itslive,
                     'J. Mouginot, R.Millan, A.Derkacheva': self.load_millan,
                     'J. Mouginot, R.Millan, A.Derkacheva_aligned': self.load_charrier,
                     'L. Charrier': self.load_charrier, 'E. Ducasse': self.load_ducasse}
        dico_load[self.ds.author](filepath, pick_date=pick_date, subset=subset, conf=conf, pick_sensor=pick_sensor,
                                  pick_temp_bas=pick_temp_bas, buffer=buffer, proj=proj)
        if verbose: print(self.ds.author)
