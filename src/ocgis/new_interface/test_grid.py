from ocgis.new_interface.grid import Grid
from ocgis.new_interface.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestGrid(AbstractTestNewInterface):
    def get(self, name=None):
        x = [101, 102, 103]
        y = [40, 41, 42, 43]
        z = [100, 200]
        vx = Variable('x', value=x, dtype=float)
        vy = Variable('y', value=y, dtype=float)
        vz = Variable('z', value=z, dtype=float)
        if name is None:
            grid = Grid(x=vx, y=vy, z=vz)
        else:
            grid = Grid(name, x=vx, y=vy, z=vz)
        return grid

    def test_bases(self):
        self.assertEqual(Grid.__bases__, (Variable,))

    def test_init(self):
        grid = self.get()
        self.assertIsNone(grid.name)

        grid = self.get(name='home')
        self.assertEqual(grid.name, 'home')

    def test_getitem(self):
        grid = self.get()
        sub = grid[2, 1, 0]
        self.assertEqual(sub.x.value, 102.)
        self.assertEqual(sub.y.value, 42.)
        self.assertEqual(sub.z.value, 100.)
        self.assertIsNone(sub._value)

    def test_shape(self):
        grid = self.get()
        self.assertEqual(grid.shape, (4, 3, 2))
        self.assertEqual(grid.ndim, 3)
