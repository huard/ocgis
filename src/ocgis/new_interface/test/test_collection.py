from collections import OrderedDict
from copy import deepcopy

import numpy as np
from shapely.geometry import Point

from ocgis.api.request.driver.vector import DriverVector
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.new_interface.collection import SpatialCollection
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.test.test_new_interface import AbstractTestNewInterface
from ocgis.new_interface.variable import Variable, VariableCollection


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
        # tdk: RESUME: this is the main test for the spatial collection
        # Create exact field 1.
        gridxy = self.get_gridxy(with_xy_bounds=True)
        exact = self.get_exact_field(gridxy.x.value, gridxy.y.value)
        exact = Variable(name='exact1', value=exact, dimensions=gridxy.dimensions)
        dimension_map = {'x': {'variable': 'x', 'bounds': 'xbounds'}, 'y': {'variable': 'y', 'bounds': 'ybounds'}}
        field1 = OcgField.from_variable_collection(gridxy.parent, dimension_map=dimension_map)
        field1.add_variable(exact)
        field1.name = 'exact1'

        # Create exact field 2.
        field2 = deepcopy(field1)
        field2['exact1'].value += 2
        field2['exact2'] = field2.pop('exact1')
        field2.name = 'exact2'

        # These are the subset geometries.
        crs = CoordinateReferenceSystem(epsg=2136)
        geoms = GeometryVariable(name='geoms', value=[Point(100.972, 41.941), Point(102.898, 40.978)],
                                 dimensions='ngeom', crs=crs)
        gridcode = Variable('gridcode', [110101, 12103], dimensions='ngeom')
        geoms.set_uid(gridcode)
        description = Variable('description', ['high point', 'low point'], dimensions='ngeom')
        dimension_map = {'geom': {'variable': 'geoms'}}
        poi = OcgField(variables=[geoms, gridcode, description], dimension_map=dimension_map)
        self.assertEqual(geoms.crs, crs)
        self.assertEqual(poi.crs, crs)

        # Execute a subset for each geometry and add to the collection.
        # sc = SpatialCollection(variables=poi.values())
        sc = SpatialCollection(properties=VariableCollection(variables=[gridcode, description]),
                               geometry_variable=geoms)
        for ii in range(poi.geom.shape[0]):
            subset = poi.geom[ii]
            subset_geom = subset.value[0]
            print subset, subset_geom
            for field in [field1, field2]:
                subset_field = field.geom.get_intersects(subset_geom).parent
                uid = subset.parent['gridcode'].value[0]
                if uid not in sc:
                    sc[uid] = OrderedDict()
                sc[uid][subset_field.name] = subset_field

        self.assertTrue(sc[110101]['exact1'].geom.value[0, 0].intersects(Point(100.972, 41.941)))
        self.assertTrue(sc[110101]['exact2'].geom.value[0, 0].intersects(Point(100.972, 41.941)))

        self.assertTrue(sc[12103]['exact1'].geom.value[0, 0].intersects(Point(102.898, 40.978)))
        self.assertTrue(sc[12103]['exact2'].geom.value[0, 0].intersects(Point(102.898, 40.978)))

        self.assertEqual(len(sc.properties), 2)
        self.assertIsInstance(sc.geometry_variable, GeometryVariable)
        self.assertEqual(sc.crs, crs)

        # print sc.properties

        # for cname, c in poi.children.items():
        #     out_path = self.get_temporary_file_path(str(cname) + '.shp')
        #     c.write(out_path, driver=DriverVector)

        path = self.get_temporary_file_path('grid.shp')
        path2 = self.get_temporary_file_path('poi.shp')
        field2.write(path, driver=DriverVector)
        poi.write(path2, driver=DriverVector)
        self.fail()
