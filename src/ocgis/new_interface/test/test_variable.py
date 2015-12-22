from copy import deepcopy

import numpy as np

from ocgis import RequestDataset
from ocgis import constants
from ocgis.exc import VariableInCollectionError, EmptySubsetError, NoUnitsError
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable, SourcedVariable, VariableCollection, BoundedVariable
from ocgis.test.base import attr
from ocgis.util.helpers import get_bounds_from_1d
from ocgis.util.units import get_units_object, get_are_units_equal


class TestBoundedVariable(AbstractTestNewInterface):
    def get_boundedvariable(self, with_bounds=True):
        value = np.array([4, 5, 6], dtype=float)
        if with_bounds:
            value_bounds = get_bounds_from_1d(value)
            bounds = Variable('x_bounds', value=value_bounds)
        else:
            bounds = None
        var = BoundedVariable('x', value=value, bounds=bounds)
        return var

    def test(self):
        # Test property update cascade.
        bv = self.get_boundedvariable()
        self.assertIsNone(bv.units)
        self.assertIsNone(bv.bounds.units)
        bv.units = 'celsius'
        self.assertIsNone(bv.conform_units_to)
        self.assertEqual(bv.units, bv.bounds.units)

    @attr('data')
    def test_with_data(self):
        # Test units are left on bounds if we are conforming.
        rd = self.get_request_dataset()
        bounds = SourcedVariable(name='lat_bnds', request_dataset=rd)
        bv = BoundedVariable(name='lat', request_dataset=rd, conform_units_to='celsius', units='K', bounds=bounds)
        self.assertIsNone(bv._value)
        self.assertIsNone(bv.bounds._value)
        self.assertEqual(bv.bounds.units, 'K')
        assert bv.value is not None
        self.assertEqual(bv.units, 'celsius')
        self.assertEqual(bv.bounds.units, 'K')

    def test_init(self):
        bv = self.get_boundedvariable()
        self.assertEqual(bv.shape, (3,))

        # Test loading from source.
        request_dataset = self.get_request_dataset()
        bounds = SourcedVariable(request_dataset=request_dataset, name='time_bnds')
        bv = BoundedVariable(bounds=bounds, name='time', request_dataset=request_dataset)[30:50]
        self.assertEqual(bv.ndim, 1)
        self.assertEqual(bv.dtype, np.float64)
        self.assertEqual(bv.bounds.dtype, np.float64)
        self.assertEqual(bv.shape, (20,))
        self.assertEqual(bv.bounds.shape, (20, 2))
        self.assertEqual(len(bv.dimensions), 1)
        self.assertEqual(len(bv.bounds.dimensions), 2)
        self.assertIsNone(bv.bounds._value)
        self.assertIsNone(bv._value)

        # Test with two dimensions.
        y_value = [[40.0, 40.0, 40.0], [41.0, 41.0, 41.0], [42.0, 42.0, 42.0], [43.0, 43.0, 43.0]]
        y_corners = [[[39.5, 39.5, 40.5, 40.5], [39.5, 39.5, 40.5, 40.5], [39.5, 39.5, 40.5, 40.5]],
                     [[40.5, 40.5, 41.5, 41.5], [40.5, 40.5, 41.5, 41.5], [40.5, 40.5, 41.5, 41.5]],
                     [[41.5, 41.5, 42.5, 42.5], [41.5, 41.5, 42.5, 42.5], [41.5, 41.5, 42.5, 42.5]],
                     [[42.5, 42.5, 43.5, 43.5], [42.5, 42.5, 43.5, 43.5], [42.5, 42.5, 43.5, 43.5]]]
        y_bounds = Variable(value=y_corners, name='y_corners')
        y_bounds.create_dimensions(names=['y', 'x', 'cbnds'])
        self.assertEqual(y_bounds.ndim, 3)
        y = BoundedVariable(value=y_value, bounds=y_bounds, name='y')
        y.create_dimensions(names=['y', 'x'])
        suby = y[1:3, 1]
        self.assertEqual(suby.bounds.shape, (2, 1, 4))
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            suby.write_netcdf(ds)

    def test_dimensions(self):
        bv = self.get_boundedvariable()
        self.assertEqual(bv.dimensions[0], bv.bounds.dimensions[0])
        self.assertEqual(bv.bounds.dimensions[1], Dimension(constants.OCGIS_BOUNDS, 2))

    @attr('cfunits')
    def test_cfunits_conform(self):
        bv = BoundedVariable(value=[5., 10., 15.], units='celsius', name='tas')
        bv.set_extrapolated_bounds()
        self.assertEqual(bv.bounds.units, 'celsius')
        bv.cfunits_conform(get_units_object('kelvin'))
        self.assertEqual(bv.bounds.units, 'kelvin')
        self.assertNumpyAll(bv.bounds.value, np.ma.array([[275.65, 280.65], [280.65, 285.65], [285.65, 290.65]]))

        # Test conforming without bounds.
        bv = BoundedVariable(value=[5., 10., 15.], units='celsius', name='tas')
        bv.cfunits_conform('kelvin')
        self.assertNumpyAll(bv.value, np.ma.array([278.15, 283.15, 288.15]))

    def test_getitem(self):
        bv = self.get_boundedvariable()
        sub = bv[1]
        self.assertEqual(sub.bounds.shape, (1, 2))
        self.assertNumpyAll(sub.bounds.value, bv.bounds[1, :].value)

    def test_get_between(self):
        bv = BoundedVariable('foo', value=[0])
        with self.assertRaises(EmptySubsetError):
            bv.get_between(100, 200)

        bv = BoundedVariable('foo', value=[100, 200, 300, 400])
        vdim_between = bv.get_between(100, 200)
        self.assertEqual(vdim_between.shape[0], 2)

    def test_get_between_bounds(self):
        value = [0., 5., 10.]
        bounds = [[-2.5, 2.5], [2.5, 7.5], [7.5, 12.5]]

        # A reversed copy of these bounds are created here.
        value_reverse = deepcopy(value)
        value_reverse.reverse()
        bounds_reverse = deepcopy(bounds)
        bounds_reverse.reverse()
        for ii in range(len(bounds)):
            bounds_reverse[ii].reverse()

        data = {'original': {'value': value, 'bounds': bounds},
                'reversed': {'value': value_reverse, 'bounds': bounds_reverse}}
        for key in ['original', 'reversed']:
            bounds = Variable('hello_bounds', value=data[key]['bounds'])
            vdim = BoundedVariable('hello', value=data[key]['value'], bounds=bounds)

            vdim_between = vdim.get_between(1, 3)
            self.assertEqual(len(vdim_between), 2)
            if key == 'original':
                self.assertEqual(vdim_between.bounds.value.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x04\xc0\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x1e@')
            else:
                self.assertEqual(vdim_between.bounds.value.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04\xc0')
            self.assertEqual(vdim.resolution, 5.0)

            # Preference is given to the lower bound in the case of "ties" where the value could be assumed part of the
            # lower or upper cell.
            vdim_between = vdim.get_between(2.5, 2.5)
            self.assertEqual(len(vdim_between), 1)
            if key == 'original':
                self.assertNumpyAll(vdim_between.bounds.value, np.ma.array([[2.5, 7.5]]))
            else:
                self.assertNumpyAll(vdim_between.bounds.value, np.ma.array([[7.5, 2.5]]))

            # If the interval is closed and the subset range falls only on bounds value then the subset will be empty.
            with self.assertRaises(EmptySubsetError):
                vdim.get_between(2.5, 2.5, closed=True)

            vdim_between = vdim.get_between(2.5, 7.5)
            if key == 'original':
                self.assertEqual(vdim_between.bounds.value.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00)@')
            else:
                self.assertEqual(vdim_between.bounds.value.tostring(),
                                 '\x00\x00\x00\x00\x00\x00)@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x04@')

    def test_get_between_use_bounds(self):
        value = [3., 5.]
        bounds = [[2., 4.], [4., 6.]]
        bounds = Variable('bounds', bounds)
        vdim = BoundedVariable('foo', value=value, bounds=bounds)
        ret = vdim.get_between(3, 4.5, use_bounds=False)
        self.assertNumpyAll(ret.value, np.ma.array([3.]))
        self.assertNumpyAll(ret.bounds.value, np.ma.array([[2., 4.]]))

    def test_set_extrapolated_bounds(self):
        bv = self.get_boundedvariable(with_bounds=False)
        self.assertIsNone(bv.bounds)
        self.assertFalse(bv._has_extrapolated_bounds)
        bv.set_extrapolated_bounds()
        self.assertTrue(bv._has_extrapolated_bounds)
        self.assertEqual(bv.bounds.name, 'x_bounds')
        self.assertEqual(bv.bounds.ndim, 2)

    def test_write_netcdf(self):
        bv = self.get_boundedvariable()
        dim_x = Dimension('x', 3)
        bv.dimensions = dim_x
        bv.bounds.dimensions = [dim_x, Dimension('bounds', 2)]
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            bv.write_netcdf(ds)
        with self.nc_scope(path, 'r') as ds:
            var = ds.variables[bv.name]
            self.assertEqual(var.bounds, bv.bounds.name)


class TestSourcedVariable(AbstractTestNewInterface):
    def get_sourcedvariable(self, name='tas'):
        data = self.get_request_dataset()
        sv = SourcedVariable(name, request_dataset=data)
        self.assertIsNone(sv._value)
        self.assertIsNone(sv._dimensions)
        self.assertIsNone(sv._dtype)
        self.assertIsNone(sv._fill_value)
        return sv

    def test_init(self):
        sv = self.get_sourcedvariable()
        self.assertIsInstance(sv._request_dataset, RequestDataset)
        self.assertEqual(sv.units, 'K')

        sv = self.get_sourcedvariable(name='time_bnds')
        self.assertIsNone(sv._value)
        sub = sv[5:10, :]
        self.assertIsNone(sub._value)

        # Test initializing with a value.
        sv = SourcedVariable(value=[1, 2, 3], name='foo')
        self.assertEqual(sv.dtype, np.int)
        self.assertEqual(sv.fill_value, 999999)
        self.assertEqual(sv.shape, (3,))
        self.assertIsNotNone(sv.dimensions)
        sv.create_dimensions(names=['time'])
        self.assertIsNotNone(sv.dimensions)

    def test_conform_units_to(self):
        with self.assertRaises(ValueError):
            SourcedVariable(value=[2, 3, 4], conform_units_to='celsius', name='tas')
        sv = SourcedVariable(conform_units_to='celsius', name='tas', request_dataset='foo')
        self.assertTrue(get_are_units_equal((sv.conform_units_to, get_units_object('celsius'))))

    @attr('data')
    def test_conform_units_to_data(self):
        rd = self.get_request_dataset()
        sv = SourcedVariable('tas', request_dataset=rd, conform_units_to='celsius')[5:9, 5, 9]
        self.assertIsNone(sv._value)
        self.assertEqual(sv.units, 'K')
        self.assertLess(sv.value.mean(), 200)
        self.assertEqual(sv.units, 'celsius')
        self.assertIsNone(sv.conform_units_to)

        # Try with a bounded variable.
        bounds = SourcedVariable('lat_bnds', request_dataset=rd)
        sv = BoundedVariable('lat', request_dataset=rd, bounds=bounds, units='celsius', conform_units_to='K')
        self.assertIsNotNone(sv.bounds.conform_units_to)
        self.assertIsNone(sv._value)
        self.assertGreater(sv.value.mean(), 250)
        self.assertEqual(sv.units, 'K')
        self.assertEqual(sv.bounds.units, 'celsius')
        self.assertIsNone(sv.bounds._value)
        self.assertGreater(sv.bounds.value.mean(), 250)
        self.assertIsNone(sv.bounds.conform_units_to)

    def test_getitem(self):
        sv = self.get_sourcedvariable()
        sub = sv[10:20, 5, 6]
        self.assertEqual(sub.shape, (10, 1, 1))
        self.assertIsNone(sub._value)
        self.assertIsNone(sub.dimensions[0].length)
        self.assertEqual(sub.dimensions[0].length_current, 10)

    def test_get_dimensions(self):
        sv = self.get_sourcedvariable()
        self.assertTrue(len(sv.dimensions), 3)

    def test_set_metadata_from_source_(self):
        sv = self.get_sourcedvariable()
        sv._set_metadata_from_source_()
        self.assertEqual(sv.dtype, np.float32)
        self.assertEqual(sv.fill_value, np.float32(1e20))
        dims = sv.dimensions
        self.assertIsNone(dims[0].length)
        self.assertEqual(dims[0].length_current, 3650)
        self.assertEqual(['time', 'lat', 'lon'], [d.name for d in dims])
        for d in dims:
            self.assertIsNone(d.__src_idx__)
        self.assertEqual(sv.attrs['standard_name'], 'air_temperature')

    def test_get_value_from_source_(self):
        sv = self.get_sourcedvariable()
        sub = sv[5:11, 3:6, 5:8]
        res = sub._get_value_from_source_()
        self.assertEqual(res.shape, (6, 3, 3))

        with self.nc_scope(self.get_request_dataset().uri, 'r') as ds:
            var = ds.variables[sv.name]
            actual = var[5:11, 3:6, 5:8]

        self.assertNumpyAll(res, actual)

    def test_value(self):
        sv = self.get_sourcedvariable()
        sub = sv[5:11, 3:6, 5:8]
        self.assertEqual(sub.value.shape, (6, 3, 3))


class TestVariable(AbstractTestNewInterface):
    def get_variable(self, return_original_data=True):
        value = [2, 3, 4, 5, 6, 7]
        time = Dimension('time', length=len(value))
        var = Variable('time_value', value=value, dimensions=time, units='kelvin')
        if return_original_data:
            return time, value, var
        else:
            return var

    def test_init(self):
        time, value, var = self.get_variable()

        self.assertEqual(var.dimensions, (time,))
        self.assertEqual(id(time), id(var.dimensions[0]))
        self.assertEqual(var.name, 'time_value')
        self.assertEqual(var.alias, var.name)
        self.assertEqual(var.shape, (len(value),))
        self.assertNumpyAll(var.value, np.ma.array(value))
        sub = var[2:4]
        self.assertIsInstance(sub, Variable)
        self.assertEqual(sub.shape, (2,))

        dtype = np.float32
        fill_value = 33.0
        var = Variable('foo', value=value, dimensions=time, dtype=dtype, fill_value=fill_value)
        self.assertEqual(var.dtype, dtype)
        self.assertEqual(var.value.dtype, dtype)
        self.assertEqual(var.value.fill_value, fill_value)

        var = Variable('foo')
        self.assertEqual(var.shape, tuple())
        self.assertIsNone(var.dimensions)

        var = Variable('foo', value=[4, 5, 6])
        self.assertEqual(var.shape, (3,))
        self.assertEqual(var.dtype, var.value.dtype)
        self.assertEqual(var.fill_value, var.value.fill_value)
        sub = var[1]
        self.assertEqual(sub.shape, (1,))

        # Test a scalar variable.
        v = Variable(value=2.0, dimensions=[])
        self.assertEqual(v.value, 2.0)

    @attr('cfunits')
    def test_cfunits(self):
        var = self.get_variable(return_original_data=False)
        actual = get_units_object(var.units)
        self.assertTrue(get_are_units_equal((var.cfunits, actual)))

    @attr('cfunits')
    def test_cfunits_conform(self):
        units_kelvin = get_units_object('kelvin')
        original_value = np.array([5, 5, 5])

        # Conversion of celsius units to kelvin.
        attrs = {k: 1 for k in constants.NETCDF_ATTRIBUTES_TO_REMOVE_ON_VALUE_CHANGE}
        var = Variable(name='tas', units='celsius', value=original_value, attrs=attrs)
        self.assertEqual(len(var.attrs), 2)
        var.cfunits_conform(units_kelvin)
        self.assertNumpyAll(var.value, np.ma.array([278.15] * 3))
        self.assertEqual(var.cfunits, units_kelvin)
        self.assertEqual(var.units, 'kelvin')
        self.assertEqual(len(var.attrs), 0)

        # If there are no units associated with a variable, conforming the units should fail.
        var = Variable(name='tas', units=None, value=original_value)
        with self.assertRaises(NoUnitsError):
            var.cfunits_conform(units_kelvin)

        # Conversion should fail for nonequivalent units.
        var = Variable(name='tas', units='kelvin', value=original_value)
        with self.assertRaises(ValueError):
            var.cfunits_conform(get_units_object('grams'))

        # The data type should always be updated to match the output from CF units backend.
        av = Variable(value=np.array([4, 5, 6]), dtype=int, name='what')
        self.assertEqual(av.dtype, np.dtype(int))
        with self.assertRaises(NoUnitsError):
            av.cfunits_conform('K')
        av.units = 'celsius'
        av.cfunits_conform('K')
        self.assertIsNone(av._dtype)
        self.assertEqual(av.dtype, av.value.dtype)

    @attr('cfunits')
    def test_cfunits_conform_masked_array(self):
        # Assert mask is respected by unit conversion.
        value = np.ma.array(data=[5, 5, 5], mask=[False, True, False])
        var = Variable(name='tas', units=get_units_object('celsius'), value=value)
        var.cfunits_conform(get_units_object('kelvin'))
        self.assertNumpyAll(np.ma.array([278.15, 278.15, 278.15], mask=[False, True, False]), var.value)

    def test_copy(self):
        var = self.get_variable(return_original_data=False)
        var2 = var.copy()
        var2.name = 'foobar'
        var2.dimensions[0].name = 'new_time'
        self.assertTrue(np.may_share_memory(var.value, var2.value))
        var2.value[:] = 100
        self.assertNumpyAll(var.value.mean(), var2.value.mean())
        var3 = var2[2:4]
        var3.value[:] = 200
        var3.value.mask[:] = True
        self.assertAlmostEqual(var.value.mean(), 133.33333333)
        self.assertFalse(var.get_mask().any())
        var2.attrs['way'] = 'out'
        self.assertEqual(len(var.attrs), 0)
        self.assertEqual(len(var3.attrs), 0)

    def test_create_dimensions(self):
        var = Variable('tas', value=[4, 5, 6], dtype=float)
        self.assertEqual(var.dimensions[0], Dimension('tas', length=3, ))
        self.assertEqual(len(var.dimensions), 1)
        var.create_dimensions('time')
        self.assertIsNotNone(var.dimensions)
        self.assertEqual(len(var.dimensions), 1)

    def test_dimensions(self):
        # Test with empty.
        var = Variable(name='height')
        self.assertEqual(var.dimensions, ())

        # Test with value.
        var = Variable(name='hat', value=1.)
        self.assertEqual(var.dimensions, ())
        var = Variable(name='hat', value=[1., 2., 3.])
        self.assertEqual(var.shape, (3,))

    def test_shape(self):
        # Test shape with unlimited dimension.
        dim = Dimension('time')
        var = Variable(name='time', value=[4, 5, 6], dimensions=dim)
        self.assertEqual(len(dim), 3)
        self.assertEqual(var.shape, (3,))
        # Copies are made after slicing.
        sub = var[1]
        self.assertEqual(len(dim), 3)
        self.assertEqual(len(sub.dimensions[0]), 1)
        self.assertEqual(sub.shape, (1,))

    def test_write_netcdf(self):
        var = self.get_variable(return_original_data=False)
        var.value.mask[1] = True
        var.attrs['axis'] = 'not_an_ally'
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            var.write_netcdf(ds)
        with self.nc_scope(path) as ds:
            ncvar = ds.variables[var.name]
            self.assertEqual(ncvar.units, 'kelvin')
            self.assertEqual(ncvar.dtype, var.dtype)
            self.assertEqual(ncvar[:].fill_value, var.fill_value)
            self.assertEqual(ncvar.axis, 'not_an_ally')

        # Test writing an unlimited dimension.
        path = self.get_temporary_file_path('foo.nc')
        dim = Dimension('time')
        var = Variable(name='time', value=[4, 5, 6], dimensions=dim)
        self.assertEqual(var.shape, (3,))
        for unlimited_to_fixedsize in [False, True]:
            with self.nc_scope(path, 'w') as ds:
                var.write_netcdf(ds, unlimited_to_fixedsize=unlimited_to_fixedsize)
            # subprocess.check_call(['ncdump', path])
            with self.nc_scope(path) as ds:
                rdim = ds.dimensions['time']
                rvar = ds.variables['time']
                if unlimited_to_fixedsize:
                    self.assertFalse(rdim.isunlimited())
                else:
                    self.assertTrue(rdim.isunlimited())
                # Fill value only present for masked data.
                self.assertNotIn('_FillValue', rvar.__dict__)


class TestVariableCollection(AbstractTestNewInterface):
    def get(self):
        var1 = self.get_variable()
        var2 = self.get_variable(alias='wunderbar')

        var3 = Variable(name='lower', value=[[9, 10, 11], [12, 13, 14]], dtype=np.float32, units='large')
        var3.create_dimensions(names=['y', 'x'])

        var4 = Variable(name='coordinate_system', attrs={'proj4': '+proj=latlon'})

        vc = VariableCollection(variables=[var1, var2, var3, var4], attrs={'foo': 'bar'})
        return vc

    def get_variable(self, name='foo', alias='foobar'):
        dim = Dimension('x', length=3)
        value = [4, 5, 6]
        return Variable(name=name, alias=alias, dimensions=dim, value=value)

    def test_init(self):
        var1 = self.get_variable()

        vc = VariableCollection(variables=var1)
        self.assertEqual(len(vc), 1)

        var2 = self.get_variable()
        with self.assertRaises(VariableInCollectionError):
            VariableCollection(variables=[var1, var2])

        var2 = self.get_variable(alias='wunderbar')
        vc = VariableCollection(variables=[var1, var2])
        self.assertEqual(vc.keys(), ['foobar', 'wunderbar'])

        self.assertEqual(vc.dimensions, {'x': Dimension(name='x', length=3)})

        vc = self.get()
        self.assertEqual(vc.attrs, {'foo': 'bar'})

    def test_write_netcdf_and_read_netcdf(self):
        vc = self.get()
        path = self.get_temporary_file_path('foo.nc')
        vc.write_netcdf(path)
        nvc = VariableCollection.read_netcdf(path)
        path2 = self.get_temporary_file_path('foo2.nc')
        nvc.write_netcdf(path2)
        self.assertNcEqual(path, path2)

    def test_write_netcdf_and_read_netcdf_data(self):
        # Test against a real data file.
        rd = self.get_request_dataset()
        rvc = VariableCollection.read_netcdf(rd.uri)
        self.assertEqual(rvc.dimensions['time'].length_current, 3650)
        for var in rvc.itervalues():
            self.assertIsNone(var._value)
        path3 = self.get_temporary_file_path('foo3.nc')
        rvc.write_netcdf(path3)
        self.assertNcEqual(path3, rd.uri)
