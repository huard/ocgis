from ocgis.new_interface.field import FieldBundle
from ocgis.new_interface.geom import SpatialContainer
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface


class TestFieldBundle(AbstractTestNewInterface):
    def get_fieldbundle(self, **kwargs):
        kwargs['name'] = kwargs.get('name', 'fb1')
        if 'spatial' not in kwargs:
            grid = self.get_gridxy()
            kwargs['spatial'] = SpatialContainer(grid=grid)
        return FieldBundle(**kwargs)

    def test_init(self):
        fb = self.get_fieldbundle()
        print fb
        print fb.spatial.point.value
