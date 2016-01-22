import numpy as np

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable
from ocgis.util.helpers import get_iter


class Test(AbstractTestNewInterface):
    def test(self):
        # tdk: test no dimension i.e. len(dimension) == 0

        def create_slices(lengths, n):
            ret = []
            for length in get_iter(lengths, dtype=int):
                slices = [None] * n
                start = 0
                remaining = length
                nlocal = n
                for ii in range(n):
                    step = int(np.ceil(float(remaining) / nlocal))
                    stop = start + step
                    if stop > length:
                        stop = length
                    index_element = slice(start, stop)
                    slices[ii] = index_element
                    remaining -= (stop - start)
                    start = stop
                    nlocal -= 1
                ret.append(slices)
            return ret

        def update_map_slices(map_slices, key, target):
            for idx, slc in enumerate(slices):
                map_slices[idx][key] = target.__getitem__(slc)
            return map_slices

        t = Dimension('t', 35)
        l = Dimension('z', 11)
        y = Dimension('y', 7)
        x = Dimension('x', 5)
        dimensions = [t, l, y, x]
        variable = Variable(name='distribute_me', dimensions=dimensions)
        n_elements = reduce(lambda i, j: i * j, variable.shape)
        value = np.arange(n_elements).reshape(variable.shape)
        variable.value = value
        # print variable.value

        n = 5
        actual = create_slices([len(d) for d in variable.dimensions[2:]], n)

        vscatter = []
        for idx in range(n):
            y_slice = actual[0][idx]
            x_slice = actual[1][idx]
            vscatter.append(variable.__getitem__({'y': y_slice, 'x': x_slice}))

        for v in vscatter:
            print v.shape
        print actual

        thh

        key_dimension = 'dimension'
        map_slices = {idx: {'slice': slc} for idx, slc in enumerate(slices)}
        update_map_slices(map_slices, key_dimension, dimension)
        actual_length = 0
        for idx, sc in enumerate(map_slices.values()):
            actual_length += len(sc[key_dimension])
            self.assertIn('slice', sc)
        self.assertEqual(actual_length, len(dimension))

        var = Variable(value=value)
        update_map_slices(map_slices, 'variable', var)
        actual = []
        for v in map_slices.values():
            actual += v['variable'].value.tolist()
        self.assertEqual(np.mean(actual), value.mean())
        thh
