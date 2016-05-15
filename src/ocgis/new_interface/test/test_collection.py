from copy import deepcopy

import numpy as np
from shapely.geometry import Point

from ocgis.api.request.driver.vector import DriverVector
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable


class TestSpatialCollection(AbstractTestNewInterface):
    def get_exact_field(self, longitude, latitude):
        longitude = deepcopy(longitude)
        select = longitude < 0
        longitude[select] += 360.
        longitude_radians = longitude * 0.0174533
        latitude_radians = latitude * 0.0174533
        exact = 2.0 + np.cos(latitude_radians) ** 2 + np.cos(2.0 * longitude_radians)
        return exact

    def test(self):
        # Create exact field 1.
        gridxy = self.get_gridxy(with_xy_bounds=True)
        exact = self.get_exact_field(gridxy.x.value, gridxy.y.value)
        exact = Variable(name='exact1', value=exact, dimensions=gridxy.dimensions)
        dimension_map = {'x': {'variable': 'x', 'bounds': 'xbounds'}, 'y': {'variable': 'y', 'bounds': 'ybounds'}}
        field1 = OcgField.from_variable_collection(gridxy.parent, dimension_map=dimension_map)
        field1.add_variable(exact)

        # Create exact field 2.
        field2 = deepcopy(field1)
        field2['exact1'].value += 2
        field2['exact2'] = field2.pop('exact1')

        # These are the subset geometries.
        geoms = GeometryVariable(name='geoms', value=[Point(100.972, 41.941), Point(102.898, 40.978)],
                                 dimensions='ngeom')
        gridcode = Variable('gridcode', [110101, 12103], dimensions='ngeom')
        description = Variable('description', ['high point', 'low point'], dimensions='ngeom')
        dimension_map = {'geom': {'variable': 'geoms'}}
        poi = OcgField(variables=[geoms, gridcode, description], dimension_map=dimension_map)

        path = self.get_temporary_file_path('grid.shp')
        path2 = self.get_temporary_file_path('poi.shp')
        field2.write(path, driver=DriverVector)
        poi.write(path2, driver=DriverVector)
        self.fail()
