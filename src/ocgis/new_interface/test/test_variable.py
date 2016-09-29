import os
from collections import OrderedDict
from copy import deepcopy
from unittest import SkipTest

import numpy as np
from numpy.core.multiarray import ndarray
from numpy.testing.utils import assert_equal
from shapely.geometry import Point

from ocgis import RequestDataset
from ocgis import constants
from ocgis.api.request.driver.vector import DriverVector
from ocgis.exc import VariableInCollectionError, EmptySubsetError, NoUnitsError, PayloadProtectedError, \
    DimensionsRequiredError
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.mpi import MPI_SIZE, MPI_RANK, OcgMpi, MPI_COMM, hgather
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable, SourcedVariable, VariableCollection, ObjectType, init_from_source
from ocgis.test.base import attr
from ocgis.util.units import get_units_object, get_are_units_equal


class TestVariable(AbstractTestNewInterface):
    def get_boundedvariable_2d(self):
        value = np.array([[2, 2.5],
                          [1, 1.5],
                          [0, 0.5]], dtype=float)
        dims = (Dimension('y', 3), Dimension('x', 2))
        bv = Variable(value=value, name='two_dee', dimensions=dims)
        bv.set_extrapolated_bounds('two_dee_bounds', 'corners')
        return bv

    def get_variable(self, return_original_data=True):
        value = [2, 3, 4, 5, 6, 7]
        time = Dimension('time', size=len(value))
        var = Variable('time_value', value=value, dimensions=time, units='kelvin')
        if return_original_data:
            return time, value, var
        else:
            return var

    def test_init(self):
        # Test an empty variable.
        var = Variable()
        self.assertEqual(var.shape, tuple())
        self.assertEqual(var.dimensions, tuple())
        self.assertEqual(var.value, None)
        self.assertEqual(var.get_mask(), None)
        self.assertIsNotNone(var.parent)

        # Test with a single dimension.
        var = Variable(dimensions=Dimension('five', 5))
        self.assertEqual(var._dimensions, ('five',))
        self.assertEqual(var.shape, (5,))

        # Test with a value and no dimensions.
        var = Variable(value=[[2, 3, 4], [4, 5, 6]])
        self.assertIsNotNone(var.dimensions)
        self.assertEqual(var.shape, (2, 3))
        self.assertEqual(var.shape, tuple([len(dim) for dim in var.dimensions]))
        self.assertEqual(var.ndim, 2)
        # Variable and dimension shapes must be equal.
        with self.assertRaises(ValueError):
            var.set_value(np.random.rand(10, 11))
        with self.assertRaises(ValueError):
            var.set_dimensions(Dimension('a', None))
        var.set_dimensions([Dimension('aa', 2), Dimension('bb')])
        self.assertEqual(var.dimensions[1].size_current, 3)
        self.assertEqual(var.shape, (2, 3))

        # Test a scalar variable.
        v = Variable(value=2.0)
        self.assertEqual(v.value, 2.0)
        self.assertIsInstance(v.value, ndarray)
        self.assertEqual(v.shape, tuple())
        self.assertEqual(v.value.dtype, np.float)
        self.assertEqual(v.dimensions, tuple())
        self.assertEqual(v.ndim, 0)

        # Test a value with no dimensions.
        v = Variable(value=[[1, 2, 3], [4, 5, 6]])
        self.assertEqual(v.shape, (2, 3))
        v.create_dimensions(['one', 'two'])
        self.assertEqual(v.shape, (2, 3))
        self.assertEqual(v.ndim, 2)

        # Test with dimensions only.
        v = Variable(dimensions=[Dimension('a', 3), Dimension('b', 8)], dtype=np.int8, fill_value=2)
        self.assertNumpyAll(v.value, np.zeros((3, 8), dtype=np.int8))

        # Test with an unlimited dimension.
        v = Variable(dimensions=Dimension('unlimited'))
        with self.assertRaises(ValueError):
            v.value

        # Test value converted to dtype and fill_value.
        value = [4.5, 5.5, 6.5]
        desired = np.array(value, dtype=np.int8)
        var = Variable(value=value, dtype=np.int8, fill_value=4)
        self.assertNumpyAll(var.value, desired)
        var._value = None
        var.set_value(np.array(value))
        self.assertNumpyAll(var.value, desired)
        var._value = None
        var.set_value(desired)
        assert_equal(var.get_mask(), [True, False, False])

    def test_tdk(self):
        time, value, var = self.get_variable()

        self.assertEqual(var.dimensions, (time,))
        self.assertEqual(id(time), id(var.dimensions[0]))
        self.assertEqual(var.name, 'time_value')
        self.assertEqual(var.shape, (len(value),))
        self.assertNumpyAll(var.value, np.array(value, dtype=var.dtype))
        sub = var[2:4]
        self.assertIsInstance(sub, Variable)
        self.assertEqual(sub.shape, (2,))

        dtype = np.float32
        fill_value = 33.0
        var = Variable('foo', value=value, dimensions=time, dtype=dtype, fill_value=fill_value)
        self.assertEqual(var.dtype, dtype)
        self.assertEqual(var.value.dtype, dtype)
        self.assertEqual(var.fill_value, fill_value)

        var = Variable('foo', value=[4, 5, 6])
        var.create_dimensions()
        self.assertEqual(var.shape, (3,))
        self.assertEqual(var.dtype, var.value.dtype)
        self.assertEqual(var.fill_value, var.fill_value)
        sub = var[1]
        self.assertEqual(sub.shape, (1,))

        # Test mask is shared.
        value = [1, 2, 3]
        value = np.ma.array(value, mask=[False, True, False], dtype=float)
        var = Variable(value=value, dtype=int)
        self.assertNumpyAll(var.get_mask(), value.mask)
        self.assertEqual(var.value.dtype, int)

        # Test with bounds.
        desired_bounds_value = [[0.5, 1.5], [1.5, 2.5]]
        bounds = Variable(value=desired_bounds_value, name='n_bnds', dimensions=['ens_dims', 'bounds'])
        var = Variable(value=[1, 2], bounds=bounds, name='n', dimensions='ens_dims')
        self.assertNumpyAll(var.bounds.value, np.array(desired_bounds_value))
        self.assertEqual(var.parent.keys(), ['n', 'n_bnds'])
        self.assertEqual(var.attrs['bounds'], bounds.name)

    def test_init_object_array(self):
        value = [[1, 3, 5],
                 [7, 9],
                 [11]]
        v = Variable(value=value, fill_value=4)
        self.assertEqual(v.dtype, ObjectType(object))
        self.assertEqual(v.shape, (3,))
        for idx in range(v.shape[0]):
            actual = v[idx].value[0]
            desired = value[idx]
            self.assertEqual(actual, desired)

        # Test converting object arrays.
        v = Variable(value=value, dtype=ObjectType(float))
        self.assertEqual(v.value[0].dtype, ObjectType(float))

        v = Variable(value=value, name='foo', dtype=ObjectType(np.float32))
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            v.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            desired = ds.variables['foo'][:]
        for idx in np.arange(v.shape[0]):
            self.assertNumpyAll(np.array(v.value[idx]), desired[idx])
        v_actual = SourcedVariable(request_dataset=RequestDataset(uri=path, variable='foo'), name='foo')

        actual = v[1].masked_value[0]
        desired = np.array(value[1], dtype=np.float32)
        self.assertNumpyAll(desired, actual)

        for idx in range(v.shape[0]):
            a = v_actual[idx].value[0]
            d = v[idx].value[0]
            self.assertNumpyAll(a, d)

    def test_system_empty_dimensions(self):
        d1 = Dimension('three', 3)
        d2 = Dimension('is_empty', 0, dist=True)
        d2.convert_to_empty()
        d3 = Dimension('what', 10)

        var = Variable('tester', dimensions=[d1, d2, d3])
        self.assertTrue(var.is_empty)
        self.assertEqual(var.shape, (3, 0, 10))

        self.assertIsNone(var.value)
        self.assertIsNone(var._value)
        self.assertIsNone(var.get_mask())
        self.assertIsNone(var._mask)

    @attr('mpi')
    def test_system_scatter_gather_variable(self):
        """Test a proof-of-concept scatter and gather operation on a simple variable."""

        if MPI_SIZE != 2:
            raise SkipTest('mpi-2 only')

        slices = [slice(0, 3), slice(3, 5)]
        var_value = np.arange(5)

        if MPI_RANK == 0:
            dim = Dimension('five', 5)
            var = Variable('five', value=var_value, dimensions=dim)
            to_scatter = [var[slc] for slc in slices]
        else:
            to_scatter = None

        lvar = MPI_COMM.scatter(to_scatter)

        self.assertNumpyAll(lvar.value, var_value[slices[MPI_RANK]])
        self.assertEqual(lvar.dimensions[0], Dimension('five', 5)[slices[MPI_RANK]])

        gathered = MPI_COMM.gather(lvar)

        if MPI_RANK == 0:
            size = sum([len(v.dimensions[0]) for v in gathered])
            new_dim = Dimension(gathered[0].dimensions[0].name, size)
            new_value = hgather([v.value for v in gathered])
            new_var = Variable(gathered[0].name, value=new_value, dimensions=new_dim)
            self.assertNumpyAll(new_var.value, var.value)
            self.assertEqual(new_var.dimensions[0], dim)

    @attr('mpi')
    def test_system_with_distributed_dimensions_from_file_netcdf(self):
        """Test a distributed read from file."""

        if MPI_RANK == 0:
            path = self.get_temporary_file_path('dist.nc')
            value = np.arange(5 * 3) * 10 + 1
            desired_sum = value.sum()
            value = value.reshape(5, 3)
            var = Variable('has_dist_dim', value=value, dimensions=['major', 'minor'])
            with self.nc_scope(path, 'w') as ds:
                var.write(ds)
        else:
            path = None

        MPI_COMM.Barrier()
        path = MPI_COMM.bcast(path, root=0)
        self.assertTrue(os.path.exists(path))

        # Assert the request dataset distribution is used.
        rd_a = RequestDataset(path, dist='placeholder')
        try:
            rd_a.driver.get_dist()
        except AttributeError as e:
            self.assertEqual(e.message, "'str' object has no attribute 'has_updated_dimensions'")

        use_overloaded_dist = [False, True]
        for u in use_overloaded_dist:
            if u:
                rd_b = RequestDataset(path)
                rd_b.metadata['dimensions']['major']['dist'] = True
                dist = rd_b.driver.get_dist()
            else:
                dist = None

            rd = RequestDataset(uri=path, dist=dist)
            if u:
                fvar = SourcedVariable(name='has_dist_dim', request_dataset=rd)
            else:
                ompi = OcgMpi()
                major = ompi.create_dimension('major', size=5, dist=True, src_idx='auto')
                minor = ompi.create_dimension('minor', size=3, dist=False, src_idx='auto')
                fvar = SourcedVariable(name='has_dist_dim', request_dataset=rd, dimensions=[major, minor])
                ompi.update_dimension_bounds()

            self.assertTrue(fvar.dimensions[0].dist)
            self.assertFalse(fvar.dimensions[1].dist)
            if MPI_RANK <= 4:
                self.assertFalse(fvar.is_empty)
                self.assertIsNotNone(fvar.value)
                self.assertEqual(fvar.shape[1], 3)
                if MPI_SIZE == 2:
                    self.assertLessEqual(fvar.shape[0], 3)
                elif MPI_SIZE == 5:
                    self.assertEqual(fvar.shape[0], 1)
            else:
                self.assertTrue(fvar.is_empty)

            values = MPI_COMM.gather(fvar.value, root=0)
            if MPI_RANK == 0:
                values = [v for v in values if v is not None]
                values = [v.flatten() for v in values]
                values = hgather(values)
                self.assertEqual(values.sum(), desired_sum)
            else:
                self.assertIsNone(values)

        MPI_COMM.Barrier()

    @attr('mpi')
    def test_system_with_distributed_dimensions_from_file_shapefile(self):
        """Test a distributed read from file."""

        path = self.path_state_boundaries

        # These are the desired values.
        rd_desired = RequestDataset(uri=path, driver=DriverVector)
        metadata = rd_desired.metadata
        metadata['dimensions'][constants.NAME_GEOMETRY_DIMENSION]['dist'] = False
        var_desired = SourcedVariable(name='STATE_NAME', request_dataset=rd_desired)
        value_desired = var_desired.value.tolist()

        rd = RequestDataset(uri=path, driver=DriverVector)
        metadata = rd.metadata
        metadata['dimensions'][constants.NAME_GEOMETRY_DIMENSION]['dist'] = True
        fvar = SourcedVariable(name='STATE_NAME', request_dataset=rd)
        self.assertEqual(len(rd.driver.dist.get_group()['dimensions']), 1)

        self.assertTrue(fvar.dimensions[0].dist)
        self.assertIsNotNone(fvar.value)
        if MPI_SIZE > 1:
            self.assertLessEqual(fvar.shape[0], 26)

        values = MPI_COMM.gather(fvar.value)
        if MPI_RANK == 0:
            values = hgather(values)
            self.assertEqual(values.tolist(), value_desired)
        else:
            self.assertIsNone(values)

        MPI_COMM.Barrier()

    @attr('mpi')
    def test_system_with_distributed_dimensions_ndvariable(self):
        """Test multi-dimensional variable behavior with distributed dimensions."""

        d1 = Dimension('d1', size=5, dist=True)
        d2 = Dimension('d2', size=10, dist=False)
        d3 = Dimension('d3', size=3, dist=True)
        dimensions = [d1, d2, d3]
        ompi = OcgMpi()
        for d in dimensions:
            ompi.add_dimension(d)
        ompi.update_dimension_bounds()

        var = Variable('ndist', dimensions=dimensions)

        if MPI_RANK > 4:
            self.assertTrue(var.is_empty)
        else:
            self.assertFalse(var.is_empty)

    def test_system_parents_on_bounds_variable(self):
        extra = self.get_variable(return_original_data=False)
        parent = VariableCollection(variables=extra)
        bounds = Variable(value=[[1, 2], [3, 4]], name='the_bounds', parent=parent, dimensions=['a', 'b'])

        extra2 = Variable(value=7.0, name='remember')
        parent = VariableCollection(variables=extra2)
        var = Variable(name='host', value=[1.5, 3.5], dimensions='a', bounds=bounds, parent=parent)
        # Parents on bounds are not added.
        self.assertEqual(var.parent.keys(), ['remember', 'the_bounds', 'host'])

    def test_bounds(self):
        # Test adding/removing bounds.
        var = Variable('bounded', value=[5], dimensions='one')
        self.assertNotIn('bounds', var.attrs)
        var.bounds = Variable('bds', value=[[6, 7]], dimensions=['one', 'bounds'])
        self.assertEqual(var.attrs['bounds'], 'bds')
        var.bounds = None
        self.assertNotIn('bounds', var.attrs)

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
        var = Variable(name='tas', units='celsius', value=original_value)
        self.assertEqual(len(var.attrs), 1)
        var.cfunits_conform(units_kelvin)
        self.assertNumpyAll(var.masked_value, np.ma.array([278.15] * 3, fill_value=var.fill_value))
        self.assertEqual(var.cfunits, units_kelvin)
        self.assertEqual(var.units, 'kelvin')
        self.assertEqual(len(var.attrs), 1)

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

        # Test with bounds.
        bv = Variable(value=[5., 10., 15.], units='celsius', name='tas', dimensions=['ll'])
        bv.set_extrapolated_bounds('the_bounds', 'bounds')
        self.assertEqual(bv.bounds.units, 'celsius')
        bv.cfunits_conform(get_units_object('kelvin'))
        self.assertEqual(bv.bounds.units, 'kelvin')
        self.assertNumpyAll(bv.bounds.masked_value, np.ma.array([[275.65, 280.65], [280.65, 285.65], [285.65, 290.65]]))

        # Test conforming without bounds.
        bv = Variable(value=[5., 10., 15.], units='celsius', name='tas')
        bv.cfunits_conform('kelvin')
        self.assertNumpyAll(bv.masked_value, np.ma.array([278.15, 283.15, 288.15]))

    @attr('cfunits')
    def test_cfunits_conform_masked_array(self):
        # Assert mask is respected by unit conversion.
        value = np.ma.array(data=[5, 5, 5], mask=[False, True, False])
        var = Variable(name='tas', units=get_units_object('celsius'), value=value)
        var.cfunits_conform(get_units_object('kelvin'))
        desired = np.ma.array([278.15, 278.15, 278.15], mask=[False, True, False], fill_value=var.fill_value)
        self.assertNumpyAll(var.masked_value, desired)

    def test_copy(self):
        var = Variable(value=[5])
        cvar = var.copy()
        self.assertNotEqual(id(var), id(cvar))
        self.assertNotEqual(id(var.parent), id(cvar.parent))
        cvar.set_value([10])
        self.assertNotEqual(var.value[0], cvar.value[0])
        cvar.attrs['new_attr'] = 'new'
        self.assertNotIn('new_attr', var.attrs)
        cvar.set_dimensions(Dimension('overload', 1))
        self.assertNotIn('overload', var.parent.dimensions)

        # Test this is not a deepcopy.
        var = Variable(value=[10, 11, 12], dimensions=Dimension('three', 3, src_idx='auto'))
        cvar = var.copy()
        self.assertNumpyMayShareMemory(var.value, cvar.value)
        self.assertNumpyMayShareMemory(var.dimensions[0]._src_idx, cvar.dimensions[0]._src_idx)

    def test_create_dimensions(self):
        var = Variable('tas', value=[4, 5, 6], dtype=float)
        var.create_dimensions()
        self.assertEqual(var.dimensions[0], Dimension('tas', size=3, ))
        self.assertEqual(len(var.dimensions), 1)
        var.create_dimensions('time')
        self.assertIsNotNone(var.dimensions)
        self.assertEqual(len(var.dimensions), 1)

    def test_deepcopy(self):
        var = Variable(value=[5], dimensions='one', name='o')
        var.bounds = Variable(value=[[4, 6]], dimensions=['one', 'bnds'], name='bounds')
        misc = Variable(value=np.arange(8), dimensions='eight', name='dont_touch_me')
        var.parent.add_variable(misc)

        dvar = var.deepcopy()
        self.assertFalse(np.may_share_memory(var.value, dvar.value))
        self.assertFalse(np.may_share_memory(var.bounds.value, dvar.bounds.value))

    def test_dimensions(self):
        # Test dimensions on bounds variables are updated when the dimensions on the parent variable are updated.
        aa = Dimension('aa', 4, src_idx=np.array([11, 12, 13, 14]))
        var = Variable('bounded', value=[1, 2, 3, 4], dimensions=aa)
        var.set_extrapolated_bounds('aa_bounds', 'aabnds')
        var.dimensions = Dimension('bb', 4)
        self.assertEqual(var.dimensions[0], var.bounds.dimensions[0])

    def test_get_between(self):
        bv = Variable('foo', value=[0])
        with self.assertRaises(EmptySubsetError):
            bv.get_between(100, 200)

        bv = Variable('foo', value=[100, 200, 300, 400])
        vdim_between = bv.get_between(100, 200)
        self.assertEqual(vdim_between.shape[0], 2)

    def test_getitem(self):
        var = Variable(value=[1, 2, 3])
        sub = var[1]
        self.assertEqual(var.shape, (3,))
        self.assertEqual(sub.shape, (1,))

        # Test a dictionary slice.
        var = Variable()
        dslc = {'one': slice(2, 5), 'two': np.array([False, True, False, True]), 'three': 0}
        with self.assertRaises(DimensionsRequiredError):
            var[dslc]
        value = np.ma.arange(5 * 4 * 7 * 10, fill_value=100).reshape(5, 4, 10, 7)
        var = Variable(value=value)
        var.create_dimensions(['one', 'two', 'four', 'three'])
        sub = var[dslc]
        self.assertEqual(sub.shape, (3, 2, 10, 1))
        sub_value = value[2:5, np.array([False, True, False, True], dtype=bool), slice(None), slice(0, 1)]
        self.assertNumpyAll(sub.masked_value, sub_value)

        # Test with a parent.
        var = Variable(name='a', value=[1, 2, 3], dimensions=['one'])
        parent = VariableCollection(variables=[var])
        var2 = Variable(name='b', value=[11, 22, 33], dimensions=['one'], parent=parent)
        self.assertIn('b', var2.parent)
        sub = var2[1]
        self.assertEqual(var2.parent.shapes, OrderedDict([('a', (3,)), ('b', (3,))]))
        self.assertEqual(sub.parent.shapes, OrderedDict([('a', (1,)), ('b', (1,))]))

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
            bounds = Variable('hello_bounds', value=data[key]['bounds'], dimensions=['a', 'b'])
            vdim = Variable('hello', value=data[key]['value'], bounds=bounds, dimensions=['a'])

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
                self.assertNumpyAll(vdim_between.bounds.masked_value, np.ma.array([[2.5, 7.5]]))
            else:
                self.assertNumpyAll(vdim_between.bounds.masked_value, np.ma.array([[7.5, 2.5]]))

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
        bounds = Variable('bounds', bounds, dimensions=['a', 'b'])
        vdim = Variable('foo', value=value, bounds=bounds, dimensions=['a'])
        ret = vdim.get_between(3, 4.5, use_bounds=False)
        self.assertNumpyAll(ret.masked_value, np.ma.array([3.]))
        self.assertNumpyAll(ret.bounds.masked_value, np.ma.array([[2., 4.]]))

    @attr('mpi')
    def test_get_distributed_slice(self):
        if MPI_RANK == 0:
            path = self.get_temporary_file_path('out.nc')
        else:
            path = None
        path = MPI_COMM.bcast(path)

        dist = OcgMpi()
        dim1 = dist.create_dimension('time', 10, dist=False, src_idx='auto')
        dim2 = dist.create_dimension('x', 5, dist=True, src_idx='auto')
        dim3 = dist.create_dimension('y', 4, dist=False, src_idx='auto')
        var = dist.create_variable('var', dimensions=[dim1, dim2, dim3])
        dist.update_dimension_bounds()

        if not var.is_empty:
            var.value[:] = MPI_RANK

        sub = var.get_distributed_slice([slice(None), slice(2, 4), slice(3, 4)])

        vc = VariableCollection(variables=[sub])
        vc.write(path)

        if MPI_RANK == 0:
            rd = RequestDataset(path)
            svar = SourcedVariable('var', request_dataset=rd)
            self.assertEqual(svar.shape, (10, 2, 1))

        MPI_COMM.Barrier()

    @attr('mpi')
    def test_get_distributed_slice_simple(self):
        if MPI_RANK == 0:
            path = self.get_temporary_file_path('out.nc')
        else:
            path = None
        path = MPI_COMM.bcast(path)

        dist = OcgMpi()
        dim = dist.create_dimension('x', 5, dist=True, src_idx='auto')
        var = dist.create_variable('var', dimensions=[dim])
        dist.update_dimension_bounds()

        if not var.is_empty:
            var.value[:] = (MPI_RANK + 1) ** 3 * (np.arange(var.shape[0]) + 1)

        sub = var.get_distributed_slice(slice(2, 4))
        if MPI_SIZE == 3:
            if MPI_RANK != 1:
                self.assertTrue(sub.is_empty)
            else:
                self.assertFalse(sub.is_empty)

        vc = VariableCollection(variables=[sub])
        try:
            vc.write(path)
        except:
            self.log.exception('failed write')
            raise

        if MPI_RANK == 0:
            rd = RequestDataset(path)
            svar = SourcedVariable('var', request_dataset=rd)
            self.assertEqual(svar.shape, (2,))
            if MPI_SIZE == 1:
                self.assertEqual(svar.value.tolist(), [3., 4.])
            elif MPI_SIZE == 2:
                self.assertEqual(svar.value.tolist(), [3., 8.])
            elif MPI_SIZE == 3:
                self.assertEqual(svar.value.tolist(), [8., 16.])
            elif MPI_SIZE == 8:
                self.assertEqual(svar.value.tolist(), [27., 64.])

        MPI_COMM.Barrier()

    def test_get_mask(self):
        var = Variable(value=[1, 2, 3], mask=[False, True, False])
        assert_equal(var.get_mask(), [False, True, False])
        value = np.ma.array([1, 2, 3], mask=[False, True, False])
        var = Variable(value=value)
        assert_equal(var.get_mask(), [False, True, False])

        var = Variable(value=[1, 2, 3])
        self.assertIsNotNone(var.get_mask())
        cpy = var.copy()
        cpy.set_mask([False, True, False])
        cpy.value.fill(10)
        self.assertTrue(np.all(var.value == 10))
        self.assertFalse(var.get_mask().any())

        var = Variable(value=np.random.rand(2, 3, 4), fill_value=200)
        var.value[1, 1] = 200
        self.assertTrue(np.all(var.get_mask()[1, 1]))
        self.assertEqual(var.get_mask().sum(), 4)

        # Test with bounds.
        bv = self.get_boundedvariable()
        bv.set_mask([False, True, False])
        bounds_mask = bv.bounds.get_mask()
        self.assertTrue(np.all(bounds_mask[1, :]))
        self.assertEqual(bounds_mask.sum(), 2)

        # Test with two dimensions.
        bv = self.get_boundedvariable_2d()
        self.assertEqual(bv.bounds.ndim, 3)
        mask = np.array([[False, True],
                         [False, False],
                         [True, False]], dtype=bool)
        bv.set_mask(mask)
        bounds_mask = bv.bounds.get_mask()
        for slc in ((0, 1), (2, 0)):
            self.assertTrue(np.all(bounds_mask[slc]))
        self.assertEqual(bounds_mask.sum(), 8)

    def test_getitem(self):
        bv = self.get_boundedvariable()
        sub = bv[1]
        self.assertEqual(sub.bounds.shape, (1, 2))
        self.assertNumpyAll(sub.bounds.value, bv.bounds[1, :].value)

        # Test with a boolean array.
        var = Variable(value=[1, 2, 3, 4, 5])
        sub = var[[False, True, True, True, False]]
        self.assertEqual(sub.shape, (3,))

    def test_group(self):
        var = Variable()
        self.assertIsNone(var.group)

        vc1 = VariableCollection(name='one')
        vc2 = VariableCollection(name='two', parent=vc1)
        vc3 = VariableCollection(name='three', parent=vc2)
        var = Variable(name='lonely')
        vc3.add_variable(var)
        self.assertEqual(var.group, ['one', 'two', 'three'])

    def test_iter(self):
        var = self.get_variable(return_original_data=False)

        for ctr, (idx, record) in enumerate(var.iter()):
            self.assertEqual(ctr, idx[0])
            self.assertIsInstance(record, OrderedDict)
        self.assertEqual(ctr, 5)

        bv = self.get_boundedvariable()
        for idx, element in bv.iter(add_bounds=True):
            self.assertEqual(len(element), 3)
        self.assertEqual(idx[0], 2)

        # Test with a formatter.
        bv = self.get_boundedvariable()
        for idx, element in bv.iter(formatter=str, add_bounds=True):
            for v in element.values():
                self.assertEqual(type(v), str)

    def test_set_extrapolated_bounds(self):
        bv = self.get_boundedvariable(mask=[False, True, False])
        self.assertIsNotNone(bv.bounds)
        bv.bounds = None
        bv.dimensions = None
        self.assertIsNone(bv.bounds)
        self.assertIsNone(bv.dimensions)
        bv.create_dimensions('x')
        bv.set_extrapolated_bounds('x_bounds', 'bounds')
        self.assertEqual(bv.bounds.name, 'x_bounds')
        self.assertEqual(bv.bounds.ndim, 2)
        bounds_mask = bv.bounds.get_mask()
        self.assertTrue(np.all(bounds_mask[1, :]))
        self.assertEqual(bounds_mask.sum(), 2)

        # Test extrapolating bounds on 2d variable.
        bv = self.get_boundedvariable_2d()
        bv.bounds = None
        self.assertIsNone(bv.bounds)
        bv.set_extrapolated_bounds('two_dee_bounds', 'bounds_dimension')
        bounds_value = bv.bounds.masked_value
        actual = [[[2.25, 2.75, 1.75, 1.25], [2.75, 3.25, 2.25, 1.75]],
                  [[1.25, 1.75, 0.75, 0.25], [1.75, 2.25, 1.25, 0.75]],
                  [[0.25, 0.75, -0.25, -0.75], [0.75, 1.25, 0.25, -0.25]]]
        actual = np.ma.array(actual, mask=False)
        self.assertNumpyAll(actual, bounds_value)
        self.assertEqual(bounds_value.ndim, 3)
        bounds_dimensions = bv.bounds.dimensions
        self.assertEqual(bv.bounds.name, 'two_dee_bounds')
        self.assertEqual(len(bounds_dimensions), 3)
        self.assertEqual(bounds_dimensions[2].name, 'bounds_dimension')

    def test_setitem(self):
        var = Variable(value=[10, 10, 10, 10, 10])
        var2 = Variable(value=[2, 3, 4], mask=[True, True, False])
        var[1:4] = var2
        self.assertEqual(var.value.tolist(), [10, 2, 3, 4, 10])
        self.assertEqual(var.get_mask().tolist(), [False, True, True, False, False])

        # Test 2 dimensions.
        value = np.zeros((3, 4), dtype=int)
        var = Variable(value=value)
        with self.assertRaises(IndexError):
            var[1] = Variable(value=4500)
        var[1, 1:3] = Variable(value=6700)
        self.assertTrue(np.all(var.value[1, 1:3] == 6700))
        self.assertAlmostEqual(var.value.mean(), 1116.66666666)

        # Test with bounds.
        bv = self.get_boundedvariable()
        bounds = Variable(value=[[500, 700]], name='b', dimensions=['a', 'b'])
        bv2 = Variable(value=[600], bounds=bounds, name='c', dimensions=['d'])
        bv[1] = bv2
        self.assertEqual(bv.bounds.value[1, :].tolist(), [500, 700])

    def test_shape(self):
        # Test shape with unlimited dimension.
        dim = Dimension('time')
        var = Variable(name='time', value=[4, 5, 6], dimensions=dim)
        self.assertEqual(var.shape, (3,))
        self.assertEqual(len(dim), 3)
        # Copies are made after slicing.
        sub = var[1]
        self.assertEqual(len(dim), 3)
        self.assertEqual(len(sub.dimensions[0]), 1)
        self.assertEqual(sub.shape, (1,))

    def test_units(self):
        var = Variable()
        self.assertIsNone(var.units)
        self.assertNotIn('units', var.attrs)
        var.units = 'large'
        self.assertEqual(var.attrs['units'], 'large')
        self.assertEqual(var.units, 'large')
        var.units = 'small'
        self.assertEqual(var.attrs['units'], 'small')
        self.assertEqual(var.units, 'small')
        var.units = None
        self.assertEqual(var.units, None)

        var = Variable(units='haze')
        self.assertEqual(var.units, 'haze')

        var = Variable()
        var.units = None
        self.assertEqual(var.attrs['units'], None)

        # Test units behavior with bounds.
        bounds = Variable(value=[[5, 6]], dtype=float, name='bnds', dimensions=['t', 'bnds'], units='celsius')
        var = Variable(name='some', value=[5.5], dimensions='t', bounds=bounds, units='K')
        self.assertEqual(var.bounds.units, 'K')
        var.units = None
        self.assertIsNone(var.bounds.units)

    def test_write_netcdf(self):
        var = self.get_variable(return_original_data=False)
        self.assertIsNone(var.fill_value)
        self.assertIsNone(var._mask)
        new_mask = var.get_mask()
        new_mask[1] = True
        var.set_mask(new_mask)
        self.assertIsNone(var.fill_value)
        var.attrs['axis'] = 'not_an_ally'
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            var.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path) as ds:
            ncvar = ds.variables[var.name]
            self.assertEqual(ncvar.units, 'kelvin')
            self.assertEqual(ncvar.dtype, var.dtype)
            self.assertTrue(ncvar[:].mask[1])
            self.assertEqual(ncvar.axis, 'not_an_ally')

        # Test writing an unlimited dimension.
        path = self.get_temporary_file_path('foo.nc')
        dim = Dimension('time')
        var = Variable(name='time', value=[4, 5, 6], dimensions=dim)
        self.assertEqual(var.shape, (3,))
        for unlimited_to_fixedsize in [False, True]:
            with self.nc_scope(path, 'w') as ds:
                var.write(ds, unlimited_to_fixedsize=unlimited_to_fixedsize)
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

        # Test writing with bounds.
        bv = self.get_boundedvariable()
        dim_x = Dimension('x', 3)
        bv.dimensions = dim_x
        bv.bounds.dimensions = [dim_x, Dimension('bounds', 2)]
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            bv.write(ds)
        # self.ncdump(path)
        with self.nc_scope(path, 'r') as ds:
            var = ds.variables[bv.name]
            self.assertEqual(var.bounds, bv.bounds.name)
            self.assertNumpyAll(ds.variables[bv.bounds.name][:], bv.bounds.value)


class TestSourcedVariable(AbstractTestNewInterface):
    def get_sourcedvariable(self, **kwargs):
        if 'name' not in kwargs:
            kwargs['name'] = 'tas'
        if 'request_dataset' not in kwargs:
            kwargs['request_dataset'] = self.get_request_dataset()
        sv = SourcedVariable(**kwargs)
        self.assertIsNone(sv._value)
        return sv

    def test_init(self):
        sv = self.get_sourcedvariable()
        self.assertIsInstance(sv._request_dataset, RequestDataset)
        self.assertEqual(sv.units, 'K')

        sv = self.get_sourcedvariable(name='time_bnds')
        self.assertIsNone(sv._value)
        self.assertEqual(sv.ndim, 2)
        sub = sv[5:10, :]
        self.assertIsNone(sub._value)

        # Test initializing with a value.
        sv = SourcedVariable(value=[1, 2, 3], name='foo')
        sv.create_dimensions()
        self.assertEqual(sv.dtype, np.int)
        self.assertEqual(sv.masked_value.fill_value, 999999)
        self.assertEqual(sv.shape, (3,))
        self.assertEqual(len(sv.dimensions), 1)
        sv.create_dimensions(names=['time'])
        self.assertIsNotNone(sv.dimensions)

        # Test protecting data.
        sv = self.get_sourcedvariable(protected=True)
        with self.assertRaises(PayloadProtectedError):
            sv.value
        self.assertIsNone(sv._value)

    def test_init_bounds(self):
        bv = self.get_boundedvariable()
        self.assertEqual(bv.shape, (3,))

        # Test loading from source.
        request_dataset = self.get_request_dataset()
        bounds = SourcedVariable(request_dataset=request_dataset, name='time_bnds')
        bv = SourcedVariable(bounds=bounds, name='time', request_dataset=request_dataset)
        self.assertEqual(len(bv.dimensions), 1)
        self.assertEqual(len(bv.bounds.dimensions), 2)
        self.assertEqual(bv.bounds.ndim, 2)
        self.assertEqual(bv.ndim, 1)
        bv = bv[30:50]
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
        y = Variable(value=y_value, bounds=y_bounds, name='y', dimensions=['y', 'x'])
        y.create_dimensions(names=['y', 'x'])
        suby = y[1:3, 1]
        self.assertEqual(suby.bounds.shape, (2, 1, 4))
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            suby.write(ds)

    def test_system_add_offset_and_scale_factor(self):
        path = self.get_temporary_file_path('foo.nc')
        with self.nc_scope(path, 'w') as ds:
            ds.createDimension('four', 4)
            var = ds.createVariable('var', int, dimensions=['four'])
            var[:] = [1, 2, 3, 4]
            var.add_offset = 100.
            var.scale_factor = 0.5
        rd = RequestDataset(uri=path)
        sv = SourcedVariable(name='var', request_dataset=rd)
        self.assertEqual(sv.dtype, np.float)
        self.assertTrue(np.all(sv.value == [100.5, 101., 101.5, 102]))
        self.assertNotIn('add_offset', sv.attrs)
        self.assertNotIn('scale_factor', sv.attrs)

    def test_get_scatter_slices(self):
        sv = self.get_sourcedvariable(protected=True)
        actual = sv.get_scatter_slices((1, 2, 2))
        desired = ((slice(0, 3650, None), slice(0, 32, None), slice(0, 64, None)),
                   (slice(0, 3650, None), slice(0, 32, None), slice(64, 128, None)),
                   (slice(0, 3650, None), slice(32, 64, None), slice(0, 64, None)),
                   (slice(0, 3650, None), slice(32, 64, None), slice(64, 128, None)))
        self.assertEqual(actual, desired)

    @attr('data')
    def test_conform_units_to_data(self):
        rd = self.get_request_dataset(conform_units_to='celsius')
        sv = SourcedVariable('tas', request_dataset=rd)[5:9, 5, 9]
        self.assertIsNone(sv._value)
        self.assertEqual(sv.units, 'K')
        self.assertLess(sv.value.mean(), 200)
        self.assertEqual(sv.units, 'celsius')

    def test_getitem(self):
        sv = self.get_sourcedvariable()
        sub = sv[10:20, 5, 6]
        self.assertEqual(sub.shape, (10, 1, 1))
        self.assertIsNone(sub._value)
        self.assertIsNone(sub.dimensions[0].size)
        self.assertEqual(sub.dimensions[0].size_current, 10)

    def test_get_dimensions(self):
        sv = self.get_sourcedvariable()
        self.assertTrue(len(sv.dimensions), 3)

    def test_init_from_source(self):
        sv = self.get_sourcedvariable()
        init_from_source(sv)
        self.assertEqual(sv.dtype, np.float32)
        self.assertEqual(sv.fill_value, np.float32(1e20))
        dims = sv.dimensions
        self.assertIsNone(dims[0].size)
        self.assertEqual(dims[0].size_current, 3650)
        self.assertEqual(['time', 'lat', 'lon'], [d.name for d in dims])
        for d in dims:
            # Source indices are always created on dimensions loaded from file.
            self.assertIsNotNone(d.__src_idx__)
        self.assertEqual(sv.attrs['standard_name'], 'air_temperature')

    def test_get_value(self):
        sv = self.get_sourcedvariable()
        sub = sv[5:11, 3:6, 5:8]
        self.assertEqual(sub.shape, (6, 3, 3))

        with self.nc_scope(self.get_request_dataset().uri, 'r') as ds:
            var = ds.variables[sv.name]
            actual = var[5:11, 3:6, 5:8]

        self.assertNumpyAll(sub.value, actual)

        # Test value may be set to None and value is not reloaded.
        sv = self.get_sourcedvariable()
        self.assertIsNone(sv._value)
        self.assertIsNotNone(sv.value)
        sv.value = None
        self.assertIsNone(sv.value)

    def test_load(self):
        sv = self.get_sourcedvariable()
        self.assertIsNone(sv._value)
        sv.load(eager=True)
        self.assertIsNotNone(sv._value)

    def test_value(self):
        sv = self.get_sourcedvariable()
        sub = sv[5:11, 3:6, 5:8]
        self.assertTrue(sv.value.mean() > 0)
        self.assertEqual(sub.value.shape, (6, 3, 3))


class TestVariableCollection(AbstractTestNewInterface):
    def get_variablecollection(self, **kwargs):
        var1 = self.get_variable()
        var2 = self.get_variable(name='wunderbar')

        var3 = Variable(name='lower', value=[[9, 10, 11], [12, 13, 14]], dtype=np.float32, units='large')
        var3.create_dimensions(names=['y', 'x'])

        var4 = Variable(name='coordinate_system', attrs={'proj4': '+proj=latlon'})
        var5 = Variable(name='how_far', value=[5000., 6000., 7000., 8000.], dimensions=['loner'])

        kwargs['variables'] = [var1, var2, var3, var4, var5]
        kwargs['attrs'] = {'foo': 'bar'}

        vc = VariableCollection(**kwargs)
        return vc

    def get_variable(self, name='foo'):
        dim = Dimension('x', size=3)
        value = [4, 5, 6]
        return Variable(name=name, dimensions=dim, value=value)

    def test_getitem(self):
        vc = self.get_variablecollection()
        slc = {'y': slice(1, 2)}
        sub = vc[slc]
        self.assertEqual(sub['lower'].shape, (1, 3))
        self.assertNotEqual(sub.shapes, vc.shapes)
        self.assertTrue(np.may_share_memory(vc['lower'].value, sub['lower'].value))

    def test_init(self):
        var1 = self.get_variable()

        vc = VariableCollection(variables=var1)
        self.assertEqual(len(vc), 1)

        var2 = self.get_variable()
        with self.assertRaises(VariableInCollectionError):
            VariableCollection(variables=[var1, var2])

        var2 = self.get_variable()
        var2._name = 'wunderbar'
        vc = VariableCollection(variables=[var1, var2])
        self.assertEqual(vc.keys(), ['foo', 'wunderbar'])

        self.assertEqual(vc.dimensions, {'x': Dimension(name='x', size=3)})

        vc = self.get_variablecollection()
        self.assertEqual(vc.attrs, {'foo': 'bar'})

    def test_system_as_spatial_collection(self):
        """Test creating a variable collection that is similar to a spatial collection."""

        uid = Variable(name='the_uid', value=[4], dimensions='geom')
        geom = GeometryVariable(name='the_geom', value=[Point(1, 2)], dimensions='geom')
        container = OcgField(name=uid.value[0], variables=[uid, geom], dimension_map={'geom': {'variable': 'the_geom'}})
        geom.set_uid(uid)
        coll = VariableCollection()
        coll.add_child(container)
        data = Variable(name='tas', value=[4, 5, 6], dimensions='time')
        contained = OcgField(name='tas_data', variables=[data])
        coll.children[container.name].add_child(contained)
        self.assertEqual(data.group, [None, 4, 'tas_data'])

    @attr('data')
    def test_system_cf_netcdf(self):
        rd = self.get_request_dataset()
        vc = VariableCollection.read(rd.uri)
        for v in vc.values():
            self.assertIsNone(v._value)
        slc = {'lat': slice(10, 23), 'time': slice(0, 1), 'lon': slice(5, 10)}
        sub = vc['tas'][slc].parent
        self.assertNotEqual(vc.shapes, sub.shapes)
        for v in vc.values():
            self.assertIsNotNone(v.attrs)
            self.assertIsNone(v._value)
            self.assertIsNotNone(v.value)

    def test_system_nested(self):
        # Test with nested collections.
        vc = self.get_variablecollection()
        nvc = self.get_variablecollection(name='nest')
        desired = Variable(name='desired', value=[101, 103], dimensions=['one'])
        nvc.add_variable(desired)
        vc.add_child(nvc)
        path = self.get_temporary_file_path('foo.nc')
        vc.write(path)
        # RequestDataset(path).inspect()
        rvc = VariableCollection.read(path)
        self.assertIn('nest', rvc.children)
        self.assertNumpyAll(rvc.children['nest']['desired'].value, desired.value)

    def test_system_as_variable_parent(self):
        # Test slicing variables.
        slc1 = {'x': slice(1, 2), 'y': slice(None)}
        slc2 = [slice(None), slice(1, 2)]
        for slc in [slc1, slc2]:
            vc = self.get_variablecollection()
            sub = vc['lower'][slc]
            self.assertNumpyAll(sub.value, np.array([10, 13], dtype=np.float32).reshape(2, 1))
            sub_vc = sub.parent
            self.assertNumpyAll(sub_vc['foo'].value, np.array([5]))
            self.assertNumpyAll(sub_vc['wunderbar'].value, np.array([5]))
            self.assertEqual(sub_vc['how_far'].shape, (4,))
            self.assertNumpyAll(sub_vc['lower'].value, sub.value)
            self.assertIn('coordinate_system', sub_vc)

    def test_copy(self):
        # Test children added to copied variable collection are not added to the copy source.
        vc = VariableCollection()
        vc_copy = vc.copy()
        vc_copy.add_child(VariableCollection(name='child'))
        self.assertEqual(len(vc.children), 0)

    def test_write_netcdf_and_read_netcdf(self):
        vc = self.get_variablecollection()
        path = self.get_temporary_file_path('foo.nc')
        vc.write(path)
        nvc = VariableCollection.read(path)
        path2 = self.get_temporary_file_path('foo2.nc')
        nvc.write(path2)
        self.assertNcEqual(path, path2)

    @attr('data')
    def test_write_netcdf_and_read_netcdf_data(self):
        # Test against a real data file.
        rd = self.get_request_dataset()
        rvc = VariableCollection.read(rd.uri)
        self.assertEqual(rvc.dimensions['time'].size_current, 3650)
        for var in rvc.itervalues():
            self.assertIsNone(var._value)
        path3 = self.get_temporary_file_path('foo3.nc')
        rvc.write(path3, dataset_kwargs={'format': rd.metadata['file_format']})
        self.assertNcEqual(path3, rd.uri)

        # Test creating dimensions when writing to netCDF.
        v = Variable(value=np.arange(2 * 4 * 3).reshape(2, 4, 3), name='hello')
        path4 = self.get_temporary_file_path('foo4.nc')
        with self.nc_scope(path4, 'w') as ds:
            v.write(ds)
        dname = 'dim_ocgis_hello_1'
        with self.nc_scope(path4) as ds:
            self.assertIn(dname, ds.dimensions)
        desired = Dimension(dname, 4)
        self.assertEqual(v.dimensions[1], desired)
        vc = VariableCollection.read(path4)
        actual = vc['hello'].dimensions[1]
        actual = Dimension(actual.name, actual.size)
        self.assertEqual(actual, desired)
        path5 = self.get_temporary_file_path('foo5.nc')
        with self.nc_scope(path5, 'w') as ds:
            vc.write(ds)
