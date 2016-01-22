import numpy as np

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class Test(AbstractTestNewInterface):
    def test(self):
        # tdk: test no dimension i.e. len(dimension) == 0

        def create_slices(length, n):
            step = int(np.ceil(float(length) / n))
            slices = [None] * n
            start = 0
            for ii in range(n):
                stop = start + step
                if stop > length:
                    stop = length
                index_element = slice(start, stop)
                slices[ii] = index_element
                start = stop
            return slices

        def update_map_slices(map_slices, key, target):
            for idx, slc in enumerate(slices):
                map_slices[idx][key] = target.__getitem__(slc)
            return map_slices

        value = np.arange(100)
        self.assertEqual(len(value), 100)
        dimension = Dimension('t', 100)
        n = 6
        slices = create_slices(len(dimension), n)
        self.assertEqual(len(slices), n)
        actual = 0
        for slc in slices:
            actual += value[slc].sum()
        self.assertEqual(actual, value.sum())

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
