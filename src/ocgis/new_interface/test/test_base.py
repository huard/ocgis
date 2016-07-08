from copy import deepcopy

from ocgis.new_interface.base import renamed_dimensions
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface


class Test(AbstractTestNewInterface):
    def test_renamed_dimensions(self):
        d = [Dimension('a', 5), Dimension('b', 6)]
        desired_after = deepcopy(d)
        name_mapping = {'time': ['b']}
        desired = [Dimension('a', 5), Dimension('time', 6)]
        with renamed_dimensions(d, name_mapping):
            self.assertEqual(d, desired)
        self.assertEqual(desired_after, d)
