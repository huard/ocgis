import os
import glob
import shapefile
import netCDF4
import numpy as np
import datetime

from ocgis.test.base import TestBase, attr
from ocgis import OcgOperations, RequestDataset, RequestDatasetCollection, env
env.DIR_GEOMCABINET=os.path.abspath(os.path.dirname(__file__))
env.OVERWRITE=True

def create_shapefile(parts,save_structure):
    w = shapefile.Writer(shapefile.POLYGON)
    w.autoBalance = 1
    w.poly(parts)
    # Character field with max length of 40, Field names seem to be limited to
    # 10 characters
    w.field('ID','N','11',0)
    w.field('FEAT_NAME','C','40')
    w.field('SUB_NAME','C','40')
    w.record(1,'First feature name','First sub feature name')
    w.save(save_structure)

def delete_shapefile(save_structure):
    shp_files = glob.glob(save_structure+'.*')
    for shp_file in shp_files:
        os.remove(shp_file)

def create_dummy_netcdf(nc_file,nc_format='NETCDF4_CLASSIC',use_time=True,
                        use_level=False,use_lat=True,use_lon=True,
                        use_station=False,use_ycxc=False,
                        time_size=None,time_num_values=1,level_size=1,
                        lat_size=1,lon_size=1,station_size=1,yc_size=1,
                        xc_size=1,time_dtype='i2',
                        time_units='days since 2001-01-01 00:00:00',
                        time_calendar='gregorian',level_dtype='f4',
                        level_units='Pa',level_positive='up',
                        var_name='dummy',var_dtype='f4',data_scale_factor=1.0,
                        data_add_offset=0.0,fill_mode='random',
                        time_values=None,lon_values=None,lat_values=None,
                        verbose=False):
    """
    Create a dummy NetCDF file on disk.

    Parameters
    ----------
    nc_file : str
        Name (and path) of the file to write.
    nc_format : str
    use_time : bool
    use_level : bool
    use_lat : bool
    use_lon : bool
    use_station : bool
    use_ycxc : bool
    time_size : int or None
    time_num_values : int
    level_size : int
    lat_size : int
    lon_size : int
    station_size : int
    yc_size : int
    xc_size : int
    time_dtype : str
    time_units : str
    time_calendar : str
    level_dtype : str
    level_units : str
    level_positive : str
    var_name : str
    var_dtype : str
    data_scale_factor : float
    data_add_offset : float
    fill_mode : str
    time_values : numpy.ndarray
    lon_values : numpy.ndarray
    lat_values : numpy.ndarray
    verbose : bool

    Notes
    -----
    Features that would make this more flexible in the future:
    1. insert_annual_cycle=True: fake an annual cycle in the data.
    2. chunksizes
    3. stations,yc,xc
    4. missing values
    5. consider rlat and rlon as dimensions
    6. default level values, allow user defined level_values
    7. allow user defined grids
    8. create files larger than machine memory

    """

    # Template for CF-1.6 convention in python, using netCDF4.
    # http://cfconventions.org/

    # Aliases for default fill values
    #defi2 = netCDF4.default_fillvals['i2']
    #defi4 = netCDF4.default_fillvals['i4']
    #deff4 = netCDF4.default_fillvals['f4']

    # Create netCDF file
    # Valid formats are 'NETCDF4', 'NETCDF4_CLASSIC', 'NETCDF3_CLASSIC' and
    # 'NETCDF3_64BIT'
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    nc1 = netCDF4.Dataset(nc_file,'w',format=nc_format)

    # 2.6.1 Identification of Conventions
    nc1.Conventions = 'CF-1.6'

    # 2.6.2. Description of file contents
    nc1.title = 'Dummy NetCDF file'
    nc1.history = "%s: File creation." % (now,)

    # Create netCDF dimensions
    if use_time:
        nc1.createDimension('time',time_size)
    if use_level:
        nc1.createDimension('level',level_size)
    if use_lat:
        nc1.createDimension('lat',lat_size)
    if use_lon:
        nc1.createDimension('lon',lon_size)
    if use_station:
        nc1.createDimension('station',station_size)
    if use_ycxc:
        nc1.createDimension('yc',yc_size)
        nc1.createDimension('xc',xc_size)

    # Create netCDF variables
    # Compression parameters include:
    # zlib=True,complevel=9,least_significant_digit=1
    # Set the fill value (shown with the 'f4' default value here) using:
    # fill_value=netCDF4.default_fillvals['f4']
    # In order to also follow COARDS convention, it is suggested to enforce the
    # following rule (this is used, for example, in nctoolbox for MATLAB):
    #     Coordinate Variables:
    #     1-dimensional netCDF variables whose dimension names are identical to
    #     their variable names are regarded as "coordinate variables"
    #     (axes of the underlying grid structure of other variables defined on
    #     this dimension).

    if use_time:
        # 4.4. Time Coordinate
        time = nc1.createVariable('time',time_dtype,('time',),zlib=True)
        time.axis = 'T'
        time.units = time_units
        time.long_name = 'time'
        time.standard_name = 'time'
        # 4.4.1. Calendar
        time.calendar = time_calendar
        if time_values is None:
            time[:] = list(range(time_num_values))
        else:
            time[:] = time_values[:]

    if use_level:
        # 4.3. Vertical (Height or Depth) Coordinate
        level = nc1.createVariable('level',level_dtype,('level',),zlib=True)
        level.axis = 'Z'
        level.units = level_units
        level.positive = level_positive
        #level.long_name = 'air_pressure'
        #level.standard_name = 'air_pressure'
        raise NotImplementedError()  # need to fill level[:]

    if use_lat:
        # 4.1. Latitude Coordinate
        lat = nc1.createVariable('lat','f4',('lat',),zlib=True)
        lat.axis = 'Y'
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'
        lat.standard_name = 'latitude'
        if lat_values is None:
            dlat = 180.0/(lat_size+1)
            lat[:] = np.arange(-90.0+dlat,90.0-dlat/2.0,dlat)
        else:
            lat[:] = lat_values[:]

    if use_lon:
        # 4.2. Longitude Coordinate
        lon = nc1.createVariable('lon','f4',('lon',),zlib=True)
        lon.axis = 'X'
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        lon.standard_name = 'longitude'
        if lon_values is None:
            dlon = 360.0/lon_size
            lon[:] = np.arange(0.0,360.0-dlon/2.0,dlon)
        else:
            lon[:] = lon_values[:]

    if use_station:
        lat = nc1.createVariable('lat','f4',('station',),zlib=True)
        lat.axis = 'Y'
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'
        lat.standard_name = 'latitude'
        if lat_values is None:
            dlat = 180.0/(lat_size+1)
            lat[:] = np.arange(-90.0+dlat,90.0-dlat/2.0,dlat)
        else:
            lat[:] = lat_values[:]
        lon = nc1.createVariable('lon','f4',('station',),zlib=True)
        lon.axis = 'X'
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        lon.standard_name = 'longitude'
        if lon_values is None:
            dlon = 360.0/lon_size
            lon[:] = np.arange(0.0,360.0-dlon/2.0,dlon)
        else:
            lon[:] = lon_values[:]

    if use_ycxc:
        yc = nc1.createVariable('yc','f4',('yc',))
        yc[:] = range(yc_size)
        lat = nc1.createVariable('lat','f4',('yc','xc'),zlib=True)
        lat.axis = 'Y'
        lat.units = 'degrees_north'
        lat.long_name = 'latitude'
        lat.standard_name = 'latitude'
        if lat_values is None:
            raise NotImplementedError()
            #dlat = 180.0/(lat_size+1)
            #lat[:] = np.arange(-90.0+dlat,90.0-dlat/2.0,dlat)
        else:
            lat[:,:] = lat_values[:,:]
        xc = nc1.createVariable('xc','f4',('xc',))
        xc[:] = range(xc_size)
        lon = nc1.createVariable('lon','f4',('yc','xc'),zlib=True)
        lon.axis = 'X'
        lon.units = 'degrees_east'
        lon.long_name = 'longitude'
        lon.standard_name = 'longitude'
        if lon_values is None:
            raise NotImplementedError()
            #dlon = 360.0/lon_size
            #lon[:] = np.arange(0.0,360.0-dlon/2.0,dlon)
        else:
            lon[:,:] = lon_values[:,:]

    # 2.3. Naming Conventions
    # 2.4 Dimensions
    #     If any or all of the dimensions of a variable have the
    #     interpretations of "date or time" (T), "height or depth" (Z),
    #     "latitude" (Y), or "longitude" (X) then we recommend, but do not
    #     require, those dimensions to appear in the relative order T, then Z,
    #     then Y, then X
    # Chunksizes should be set to the expected input/output pattern and be of
    # the order of 1000000 (that is the product of the chunksize in each
    # dimention). e.g. for daily data, this might be 30 (a month) or
    # 365 (a year) in time, and then determine the spatial chunks to obtain
    # ~1000000.
    var_dims = []
    if use_time:
        var_dims.append('time')
    if use_level:
        var_dims.append('level')
    if use_lat:
        var_dims.append('lat')
    if use_lon:
        var_dims.append('lon')
    if use_station:
        var_dims.append('station')
    if use_ycxc:
        var_dims.append('yc')
        var_dims.append('xc')
    var1 = nc1.createVariable(var_name,var_dtype,tuple(var_dims),zlib=True,
                              chunksizes=None,
                              fill_value=netCDF4.default_fillvals[var_dtype])
    # 3.1. Units
    var1.units = '1'
    # 3.2. Long Name
    var1.long_name = 'dummy_variable'
    # 3.3. Standard Name
    var1.standard_name = 'dummy_variable'

    if fill_mode == 'random':
        data1 = np.random.rand(*var1.shape)*data_scale_factor+data_add_offset
        num_values = data1.size
        if hasattr(data1,'count'):
            masked_values = num_values-data1.count()
        else:
            masked_values = 0
        data_size_mb = data1.nbytes/1000000.0
        var1[...] = data1[...]
    elif fill_mode == 'gradient':
        data1 = np.arange(0,1.0+0.5/(var1.size-1),1.0/(var1.size-1))
        data1 = data1*data_scale_factor+data_add_offset
        if hasattr(data1,'count'):
            masked_values = num_values-data1.count()
        else:
            masked_values = 0
        data_size_mb = data1.nbytes/1000000.0
        var1[...] = ma.reshape(data1,var1.shape)

    nc1.close()

    if verbose:
        str1 = "%s values, %s masked, for %s Mb of uncompressed data."
        print(str1 % (str(num_values),str(masked_values),str(data_size_mb)))

@attr('spatialop')
class TestSpatialOp(TestBase):

    def setUp(self):
        env.DIR_GEOMCABINET = os.path.abspath(os.path.dirname(__file__))
        env.OVERWRITE = True

    def tearDown(self):
        pass

    def test_ocgis_spatialop_01(self):
        # Trivial case: regular grid, polygon is a square that exactly overlaps
        # the grid tiles.

        # Create a temporary shapefile
        save_structure = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),'polygon_tmp')
        parts=[[[289.5,49.5],
                [289.5,51.5],
                [291.5,51.5],
                [291.5,49.5]]]
        create_shapefile(parts,save_structure)
    
        # create a temporary NetCDF file
        nc_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'sample_tmp.nc')
        create_dummy_netcdf(nc_file,
                            nc_format='NETCDF4_CLASSIC',use_time=True,
                            use_level=False,use_lat=True,use_lon=True,
                            use_station=False,use_ycxc=False,
                            time_size=None,time_num_values=5,level_size=1,
                            lat_size=179,lon_size=360,station_size=1,yc_size=1,
                            xc_size=1,time_dtype='i2',
                            time_units='days since 2001-01-01 00:00:00',
                            time_calendar='gregorian',level_dtype='f4',
                            level_units='Pa',level_positive='up',
                            var_name='dummy',var_dtype='f4',
                            data_scale_factor=1.0,
                            data_add_offset=0.0,fill_mode='random',
                            time_values=None,lon_values=None,lat_values=None,
                            verbose=False)
        nc = netCDF4.Dataset(nc_file,'a')
        ncvar = nc.variables['dummy']
        lon = nc.variables['lon']
        lat = nc.variables['lat']
        new_data = np.zeros([5,4,4])
        for t in range(ncvar.shape[0]):
            tmul = 10**t
            new_data[t,1,1] = tmul*10
            new_data[t,1,2] = tmul*1
            new_data[t,2,2] = tmul*2
            new_data[t,2,1] = tmul*3
        ncvar[:,138:142,289:293] = new_data[:,:,:]
        nc.close()
    
        # Do the test
        NCS = {nc_file:'dummy'}
        rdc = RequestDatasetCollection(
            [RequestDataset(uri,var) for uri,var in NCS.iteritems()])
        ops = OcgOperations(dataset=rdc,spatial_operation='clip',
                            aggregate=True,geom='polygon_tmp')
        ret = ops.execute()
        result = ret[1]['dummy'].variables['dummy'].value
        
        # cleanup (really, this should be done in a unittest setUp + tearDown)
        delete_shapefile(save_structure)
        os.remove(nc_file)
        
        assert result[0,0,0,0,0] == 4.0
        assert result[0,2,0,0,0] == 400.0
    
    
    def test_ocgis_spatialop_02(self):
        # Regular grid, triangle polygon
    
        # Create a temporary shapefile
        save_structure = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),'polygon_tmp')
        parts=[[[289.5,49.5],
                [289.5,51.5],
                [291.5,49.5]]]
        create_shapefile(parts,save_structure)
    
        # create a temporary NetCDF file
        nc_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'sample_tmp.nc')
        create_dummy_netcdf(nc_file,
                            nc_format='NETCDF4_CLASSIC',use_time=True,
                            use_level=False,use_lat=True,use_lon=True,
                            use_station=False,use_ycxc=False,
                            time_size=None,time_num_values=5,level_size=1,
                            lat_size=179,lon_size=360,station_size=1,yc_size=1,
                            xc_size=1,time_dtype='i2',
                            time_units='days since 2001-01-01 00:00:00',
                            time_calendar='gregorian',level_dtype='f4',
                            level_units='Pa',level_positive='up',
                            var_name='dummy',var_dtype='f4',
                            data_scale_factor=1.0,
                            data_add_offset=0.0,fill_mode='random',
                            time_values=None,lon_values=None,lat_values=None,
                            verbose=False)
        nc = netCDF4.Dataset(nc_file,'a')
        ncvar = nc.variables['dummy']
        lon = nc.variables['lon']
        lat = nc.variables['lat']
        new_data = np.zeros([5,4,4])
        for t in range(ncvar.shape[0]):
            tmul = 10**t
            new_data[t,1,1] = tmul*10
            new_data[t,1,2] = tmul*1
            new_data[t,2,2] = tmul*2
            new_data[t,2,1] = tmul*3
        ncvar[:,138:142,289:293] = new_data[:,:,:]
        nc.close()
    
        # Do the test
        NCS = {nc_file:'dummy'}
        rdc = RequestDatasetCollection(
            [RequestDataset(uri,var) for uri,var in NCS.iteritems()])
        ops = OcgOperations(dataset=rdc,spatial_operation='clip',
                            aggregate=True,geom='polygon_tmp')
        ret = ops.execute()
        result = ret[1]['dummy'].variables['dummy'].value
        
        # cleanup (really, this should be done in a unittest setUp + tearDown)
        delete_shapefile(save_structure)
        os.remove(nc_file)
        
        print(result[0,0,0,0,0])
        assert result[0,0,0,0,0] == 3.0
        assert result[0,2,0,0,0] == 300.0
    
    
    def test_ocgis_spatialop_03(self):
        # Regular grid, square polygon overlaps with masked values
    
        # Create a temporary shapefile
        save_structure = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),'polygon_tmp')
        parts=[[[289.5,49.5],
                [289.5,51.5],
                [293.5,51.5],
                [293.5,49.5]]]
        create_shapefile(parts,save_structure)
    
        # create a temporary NetCDF file
        nc_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'sample_tmp.nc')
        create_dummy_netcdf(nc_file,
                            nc_format='NETCDF4_CLASSIC',use_time=True,
                            use_level=False,use_lat=True,use_lon=True,
                            use_station=False,use_ycxc=False,
                            time_size=None,time_num_values=5,level_size=1,
                            lat_size=179,lon_size=360,station_size=1,yc_size=1,
                            xc_size=1,time_dtype='i2',
                            time_units='days since 2001-01-01 00:00:00',
                            time_calendar='gregorian',level_dtype='f4',
                            level_units='Pa',level_positive='up',
                            var_name='dummy',var_dtype='f4',
                            data_scale_factor=1.0,
                            data_add_offset=0.0,fill_mode='random',
                            time_values=None,lon_values=None,lat_values=None,
                            verbose=False)
        nc = netCDF4.Dataset(nc_file,'a')
        ncvar = nc.variables['dummy']
        lon = nc.variables['lon']
        lat = nc.variables['lat']
        new_data = np.zeros([5,4,4])
        for t in range(ncvar.shape[0]):
            tmul = 10**t
            new_data[t,1,1] = tmul*10
            new_data[t,1,2] = tmul*1
            new_data[t,2,2] = tmul*2
            new_data[t,2,1] = tmul*3
        ncvar[:,138:142,289:293] = new_data[:,:,:]
        nc.close()
    
        # Do the test
        NCS = {nc_file:'dummy'}
        rdc = RequestDatasetCollection(
            [RequestDataset(uri,var) for uri,var in NCS.iteritems()])
        ops = OcgOperations(dataset=rdc,spatial_operation='clip',
                            aggregate=True,geom='polygon_tmp')
        ret = ops.execute()
        result = ret[1]['dummy'].variables['dummy'].value
        
        # cleanup (really, this should be done in a unittest setUp + tearDown)
        delete_shapefile(save_structure)
        os.remove(nc_file)
        
        assert result[0,0,0,0,0] == 16/6.
        assert result[0,2,0,0,0] == 1600/6.
    
    
    def test_ocgis_spatialop_04(self):
        # Regular grid, grid & polygon have longitudes with/without -180,180 range
    
        # Create a temporary shapefile
        save_structure = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),'polygon_tmp')
        parts=[[[-70.5,49.5],
                [-70.5,51.5],
                [-68.5,51.5],
                [-68.5,49.5]]]
        create_shapefile(parts,save_structure)
    
        # create a temporary NetCDF file
        nc_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'sample_tmp.nc')
        create_dummy_netcdf(nc_file,
                            nc_format='NETCDF4_CLASSIC',use_time=True,
                            use_level=False,use_lat=True,use_lon=True,
                            use_station=False,use_ycxc=False,
                            time_size=None,time_num_values=5,level_size=1,
                            lat_size=179,lon_size=360,station_size=1,yc_size=1,
                            xc_size=1,time_dtype='i2',
                            time_units='days since 2001-01-01 00:00:00',
                            time_calendar='gregorian',level_dtype='f4',
                            level_units='Pa',level_positive='up',
                            var_name='dummy',var_dtype='f4',
                            data_scale_factor=1.0,
                            data_add_offset=0.0,fill_mode='random',
                            time_values=None,lon_values=None,lat_values=None,
                            verbose=False)
        nc = netCDF4.Dataset(nc_file,'a')
        ncvar = nc.variables['dummy']
        lon = nc.variables['lon']
        lat = nc.variables['lat']
        new_data = np.zeros([5,4,4])
        for t in range(ncvar.shape[0]):
            tmul = 10**t
            new_data[t,1,1] = tmul*10
            new_data[t,1,2] = tmul*1
            new_data[t,2,2] = tmul*2
            new_data[t,2,1] = tmul*3
        ncvar[:,138:142,289:293] = new_data[:,:,:]
        nc.close()
    
        # Do the test
        NCS = {nc_file:'dummy'}
        rdc = RequestDatasetCollection(
            [RequestDataset(uri,var) for uri,var in NCS.iteritems()])
        ops = OcgOperations(dataset=rdc,spatial_operation='clip',
                            aggregate=True,geom='polygon_tmp')
        ret = ops.execute()
        result = ret[1]['dummy'].variables['dummy'].value
        
        # cleanup (really, this should be done in a unittest setUp + tearDown)
        delete_shapefile(save_structure)
        os.remove(nc_file)
        
        assert result[0,0,0,0,0] == 4.0
        assert result[0,2,0,0,0] == 400.0
    
    
    def test_ocgis_spatialop_05(self):
        # Regular grid over a subregion, polygon outside region
    
        # Create a temporary shapefile
        save_structure = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),'polygon_tmp')
        parts=[[[289.5,9.5],
                [289.5,11.5],
                [291.5,11.5],
                [291.5,9.5]]]
        create_shapefile(parts,save_structure)
    
        # create a temporary NetCDF file
        nc_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'sample_tmp.nc')
        create_dummy_netcdf(nc_file,
                            nc_format='NETCDF4_CLASSIC',use_time=True,
                            use_level=False,use_lat=True,use_lon=True,
                            use_station=False,use_ycxc=False,
                            time_size=None,time_num_values=5,level_size=1,
                            lat_size=61,lon_size=101,station_size=1,yc_size=1,
                            xc_size=1,time_dtype='i2',
                            time_units='days since 2001-01-01 00:00:00',
                            time_calendar='gregorian',level_dtype='f4',
                            level_units='Pa',level_positive='up',
                            var_name='dummy',var_dtype='f4',
                            data_scale_factor=1.0,
                            data_add_offset=0.0,fill_mode='random',
                            time_values=None,lon_values=range(220,321),
                            lat_values=range(20,81),verbose=False)
        nc = netCDF4.Dataset(nc_file,'a')
        ncvar = nc.variables['dummy']
        lon = nc.variables['lon']
        lat = nc.variables['lat']
        ncvar[:,:,:] = np.ones(ncvar.shape)
        nc.close()
    
        # Do the test
        NCS = {nc_file:'dummy'}
        rdc = RequestDatasetCollection(
            [RequestDataset(uri,var) for uri,var in NCS.iteritems()])
        ops = OcgOperations(dataset=rdc,spatial_operation='clip',
                            aggregate=True,geom='polygon_tmp')
        # This should be done with unittest assertRaise
        try:
            ret = ops.execute()
            assert False
        except:
            # cleanup (really, this should be done in a unittest setUp + tearDown)
            delete_shapefile(save_structure)
            os.remove(nc_file)
            assert True
    
    
    def test_ocgis_spatialop_06(self):
        # Regular grid over a subregion, polygon partly in region
    
        # Create a temporary shapefile
        save_structure = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),'polygon_tmp')
        parts=[[[320.5,49.5],
                [320.5,51.5],
                [322.5,51.5],
                [322.5,49.5]]]
        create_shapefile(parts,save_structure)
    
        # create a temporary NetCDF file
        nc_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'sample_tmp.nc')
        create_dummy_netcdf(nc_file,
                            nc_format='NETCDF4_CLASSIC',use_time=True,
                            use_level=False,use_lat=True,use_lon=True,
                            use_station=False,use_ycxc=False,
                            time_size=None,time_num_values=5,level_size=1,
                            lat_size=61,lon_size=101,station_size=1,yc_size=1,
                            xc_size=1,time_dtype='i2',
                            time_units='days since 2001-01-01 00:00:00',
                            time_calendar='gregorian',level_dtype='f4',
                            level_units='Pa',level_positive='up',
                            var_name='dummy',var_dtype='f4',
                            data_scale_factor=1.0,
                            data_add_offset=0.0,fill_mode='random',
                            time_values=None,lon_values=range(220,321),
                            lat_values=range(20,81),verbose=False)
        nc = netCDF4.Dataset(nc_file,'a')
        ncvar = nc.variables['dummy']
        lon = nc.variables['lon']
        lat = nc.variables['lat']
        ncvar[:,:,:] = np.ones(ncvar.shape)
        nc.close()
    
        # Do the test
        NCS = {nc_file:'dummy'}
        rdc = RequestDatasetCollection(
            [RequestDataset(uri,var) for uri,var in NCS.iteritems()])
        ops = OcgOperations(dataset=rdc,spatial_operation='clip',
                            aggregate=True,geom='polygon_tmp')
        # This should be done with unittest assertRaise
        try:
            ret = ops.execute()
            # cleanup (really, this should be done in a unittest setUp + tearDown)
            delete_shapefile(save_structure)
            os.remove(nc_file)
            assert False
        except:
            # cleanup (really, this should be done in a unittest setUp + tearDown)
            delete_shapefile(save_structure)
            os.remove(nc_file)
            assert True
    
    
    def test_ocgis_spatialop_07(self):
        # Irregular grid over a subregion
    
        # Create a temporary shapefile
        save_structure = os.path.join(
            os.path.abspath(os.path.dirname(__file__)),'polygon_tmp')
        parts=[[[289.5,49.5],
                [289.5,51.5],
                [291.5,51.5],
                [291.5,49.5]]]
        create_shapefile(parts,save_structure)
    
        # create a temporary NetCDF file
        nc_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                               'sample_tmp.nc')
        mylons = np.zeros([60,100])
        mylats = np.zeros([60,100])
        for j in range(60):
            for i in range(100):
                mylons[j,i] = 220+i+j/2.0
                mylats[j,i] = 20+j
        create_dummy_netcdf(nc_file,
                            nc_format='NETCDF4_CLASSIC',use_time=True,
                            use_level=False,use_lat=False,use_lon=False,
                            use_station=False,use_ycxc=True,
                            time_size=None,time_num_values=5,level_size=1,
                            lat_size=1,lon_size=1,station_size=1,yc_size=60,
                            xc_size=100,time_dtype='i2',
                            time_units='days since 2001-01-01 00:00:00',
                            time_calendar='gregorian',level_dtype='f4',
                            level_units='Pa',level_positive='up',
                            var_name='dummy',var_dtype='f4',
                            data_scale_factor=1.0,
                            data_add_offset=0.0,fill_mode='random',
                            time_values=None,lon_values=mylons,lat_values=mylats,
                            verbose=False)
        nc = netCDF4.Dataset(nc_file,'a')
        ncvar = nc.variables['dummy']
        lon = nc.variables['lon']
        lat = nc.variables['lat']
        ncvar[:,:,:] = np.ones(ncvar.shape)
        nc.close()
    
        # Do the test
        NCS = {nc_file:'dummy'}
        rdc = RequestDatasetCollection(
            [RequestDataset(uri,var) for uri,var in NCS.iteritems()])
        ops = OcgOperations(dataset=rdc,spatial_operation='clip',
                            aggregate=True,geom='polygon_tmp')
        ret = ops.execute()
        result = ret[1]['dummy'].variables['dummy'].value
        
        # cleanup (really, this should be done in a unittest setUp + tearDown)
        delete_shapefile(save_structure)
        os.remove(nc_file)
        
        assert result[0,0,0,0,0] == 1.0
        assert result[0,2,0,0,0] == 1.0
