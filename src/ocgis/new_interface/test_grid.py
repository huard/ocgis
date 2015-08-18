from numpy.ma import MaskedArray
import numpy as np

from ocgis.new_interface.grid import Grid
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestGrid(AbstractTestNewInterface):
    def get(self, name=None, with_value=False, with_z=True):
        x = [101, 102, 103]
        y = [40, 41, 42, 43]
        z = [100, 200]

        args = []
        kwds = {}

        if name is not None:
            args.append(name)

        if with_value:
            mx, my, mz = np.meshgrid(x, y, z)
            value = np.zeros([3] + list(mx.shape))
            value[0, ...] = mx
            value[1, ...] = my
            value[2, ...] = mz
            kwds.update(dict(value=value))
        else:
            vx = Variable('x', value=x, dtype=float)
            vy = Variable('y', value=y, dtype=float)
            kwds.update(dict(x=vx, y=vy))
            if with_z:
                vz = Variable('z', value=z, dtype=float)
                kwds.update(dict(z=vz))

        grid = Grid(*args, **kwds)
        return grid

    def test_bases(self):
        self.assertEqual(Grid.__bases__, (Variable,))

    def test_init(self):
        grid = self.get()
        self.assertIsNone(grid.name)

        grid = self.get(name='home')
        self.assertEqual(grid.name, 'home')

        grid = self.get(with_value=True)
        self.assertIsInstance(grid.value, MaskedArray)
        self.assertIsNone(grid.x)

    def test_getitem(self):
        grid = self.get()
        sub = grid[2, 1, 0]
        self.assertEqual(sub.x.value, 102.)
        self.assertEqual(sub.y.value, 42.)
        self.assertEqual(sub.z.value, 100.)
        self.assertIsNone(sub._value)

    def test_get_value(self):
        grid = self.get()
        res = grid._get_value_()
        self.assertIsInstance(res, MaskedArray)
        self.assertEqual(res.shape, (3, 4, 3, 2))
        print res

        # Test with no z coordinate.
        grid = self.get(with_z=False)
        self.assertIsNone(grid.z)
        self.assertEqual(grid.ndim, 2)
        res = grid._get_value_()
        self.assertEqual(res.shape, (2, 4, 3))

    def test_shape(self):
        grid = self.get()
        self.assertEqual(grid.shape, (4, 3, 2))
        self.assertEqual(grid.ndim, 3)
        self.assertIsNone(grid._value)

        grid = self.get(with_value=True)
        self.assertIsInstance(grid.value, MaskedArray)
        self.assertIsNone(grid.x)
        self.assertEqual(grid.value.shape, (3, 4, 3, 2))
        self.assertEqual(grid.value.shape[1:], grid.shape)
