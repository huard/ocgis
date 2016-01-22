import numpy as np

from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface


class Test(AbstractTestNewInterface):
    def test(self):

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

        def get_slices_dimension(slices, dimension):
            ret = {}
            for idx, slc in enumerate(slices):
                ret[idx] = {'slice': slc, 'dimension': dimension.__getitem__(slc)}
            return ret

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

        sliced_dimensions = get_slices_dimension(slices, dimension)

        actual_length = 0
        for idx, sc in enumerate(sliced_dimensions.values()):
            actual_length += len(sc['dimension'])
        self.assertEqual(actual_length, len(dimension))

        thh
