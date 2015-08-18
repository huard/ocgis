from ocgis.new_interface.base import AbstractInterfaceObject

from ocgis.new_interface.grid import Grid
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestGrid(AbstractTestNewInterface):
    def get(self, with_z=True):
        x = [101, 102, 103]
        y = [40, 41, 42, 43]
        z = [100, 200]

        kwds = {}

        vx = Variable('x', value=x, dtype=float)
        vy = Variable('y', value=y, dtype=float)
        kwds.update(dict(x=vx, y=vy))
        if with_z:
            vz = Variable('z', value=z, dtype=float)
            kwds.update(dict(z=vz))

        grid = Grid(**kwds)
        return grid

    def test_bases(self):
        self.assertEqual(Grid.__bases__, (AbstractInterfaceObject,))

    def test_init(self):
        grid = self.get()
        self.assertIsInstance(grid, Grid)

        x = Variable('x', value=[1])
        with self.assertRaises(ValueError):
            Grid(x=x)

    def test_getitem(self):
        grid = self.get()
        sub = grid[2, 1, 0]
        self.assertEqual(sub.x.value, 102.)
        self.assertEqual(sub.y.value, 42.)
        self.assertEqual(sub.z.value, 100.)

    def test_resolution(self):
        grid = self.get()
        self.assertEqual(grid.resolution, 1.)

    def test_shape(self):
        grid = self.get()
        self.assertEqual(grid.shape, (3, 4, 2))
        self.assertEqual(grid.ndim, 3)

        grid = self.get(with_z=False)
        self.assertEqual(grid.ndim, 2)
