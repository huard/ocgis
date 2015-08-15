import numpy as np

from ocgis.exc import EmptySubsetError

from ocgis.new_interface.adapter import BoundedVariable, AbstractAdapter
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_bounds_from_1d


class TestBoundedVariable(AbstractTestNewInterface):
    def get(self, with_bounds=True):
        value = np.array([4, 5, 6], dtype=float)
        var = Variable('x', value=value)
        if with_bounds:
            value_bounds = get_bounds_from_1d(value)
            bounds = Variable('x_bounds', value=value_bounds)
        else:
            bounds = None
        bv = BoundedVariable(var, bounds=bounds)
        return bv

    def test_bases(self):
        self.assertEqual(BoundedVariable.__bases__, (AbstractAdapter,))

    def test_init(self):
        bv = self.get()
        with self.assertRaises(AttributeError):
            bv.shape

    def test_getitem(self):
        bv = self.get()
        sub = bv[1]
        self.assertNumpyAll(sub.bounds.value, bv.bounds[1, :].value)

    def test_get_between(self):
        # vdim = VectorDimension(value=[0])
        var = Variable('foo', value=[0])
        bv = BoundedVariable(var)
        with self.assertRaises(EmptySubsetError):
            bv.get_between(100, 200)

        # vdim = VectorDimension(value=[100, 200, 300, 400])
        var = Variable('foo', value=[100, 200, 300, 400])
        bv = BoundedVariable(var)
        vdim_between = bv.get_between(100, 200)
        self.assertEqual(len(vdim_between), 2)

    def test_get_between_bounds(self):
        value = [0., 5., 10.]
        bounds = [[-2.5, 2.5], [2.5, 7.5], [7.5, 12.5]]

        # # a reversed copy of these bounds are created here
        value_reverse = deepcopy(value)
        value_reverse.reverse()
        bounds_reverse = deepcopy(bounds)
        bounds_reverse.reverse()
        for ii in range(len(bounds)):
            bounds_reverse[ii].reverse()

        data = {'original': {'value': value, 'bounds': bounds},
                'reversed': {'value': value_reverse, 'bounds': bounds_reverse}}
        for key in ['original', 'reversed']:
            vdim = VectorDimension(value=data[key]['value'],
                                   bounds=data[key]['bounds'])

            vdim_between = vdim.get_between(1, 3)
            self.assertEqual(len(vdim_between), 2)
            if key == 'original':
                self.assertEqual(vdim_between.bounds.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x04\xc0\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x1e@')
            else:
                self.assertEqual(vdim_between.bounds.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x04\xc0')
            self.assertEqual(vdim.resolution, 5.0)

            ## preference is given to the lower bound in the case of "ties" where
            ## the value could be assumed part of the lower or upper cell
            vdim_between = vdim.get_between(2.5, 2.5)
            self.assertEqual(len(vdim_between), 1)
            if key == 'original':
                self.assertNumpyAll(vdim_between.bounds, np.array([[2.5, 7.5]]))
            else:
                self.assertNumpyAll(vdim_between.bounds, np.array([[7.5, 2.5]]))

            ## if the interval is closed and the subset range falls only on bounds
            ## value then the subset will be empty
            with self.assertRaises(EmptySubsetError):
                vdim.get_between(2.5, 2.5, closed=True)

            vdim_between = vdim.get_between(2.5, 7.5)
            if key == 'original':
                self.assertEqual(vdim_between.bounds.tostring(),
                                 '\x00\x00\x00\x00\x00\x00\x04@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00)@')
            else:
                self.assertEqual(vdim_between.bounds.tostring(),
                                 '\x00\x00\x00\x00\x00\x00)@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x1e@\x00\x00\x00\x00\x00\x00\x04@')

    def test_get_between_use_bounds(self):
        value = [3., 5.]
        bounds = [[2., 4.], [4., 6.]]
        vdim = VectorDimension(value=value, bounds=bounds)
        ret = vdim.get_between(3, 4.5, use_bounds=False)
        self.assertNumpyAll(ret.value, np.array([3.]))
        self.assertNumpyAll(ret.bounds, np.array([[2., 4.]]))

    def test_set_extrapolated_bounds(self):
        bv = self.get(with_bounds=False)
        self.assertIsNone(bv.bounds)
        self.assertFalse(bv._has_extrapolated_bounds)
        bv.set_extrapolated_bounds()
        self.assertTrue(bv._has_extrapolated_bounds)
        self.assertEqual(bv.bounds.name, 'x_bounds')
        self.assertEqual(bv.bounds.ndim, 2)

    def test_write_netcdf(self):
        bv = self.get()
        dim_x = Dimension('x', 3)
        bv.variable.dimensions = dim_x
        bv.bounds.dimensions = [dim_x, Dimension('bounds', 2)]
        path = self.get_temporary_file_path('out.nc')
        with self.nc_scope(path, 'w') as ds:
            bv.write_netcdf(ds)
        with self.nc_scope(path, 'r') as ds:
            var = ds.variables[bv.variable.name]
            self.assertEqual(var.bounds, bv.bounds.name)
