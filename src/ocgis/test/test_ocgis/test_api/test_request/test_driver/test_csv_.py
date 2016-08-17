import csv

from shapely.geometry import Point

from ocgis import RequestDataset
from ocgis.api.request.driver.csv_ import DriverCSV
from ocgis.new_interface.field import OcgField
from ocgis.test.base import TestBase


class TestDriverCSV(TestBase):
    def test(self):
        headers = ['ONE', 'two', 'THREE', 'x', 'y']
        record1 = [1, 'number', 4.5, 10.3, 12.4]
        record2 = [2, 'letter', 5.5, 11.3, 13.4]
        path = self.get_temporary_file_path('foo.csv')
        path_out = self.get_temporary_file_path('foo_out.csv')
        with open(path, 'w') as out:
            writer = csv.writer(out)
            for row in [headers, record1, record2]:
                writer.writerow(row)

        rd = RequestDataset(uri=path)
        vc = rd.get_variable_collection()

        for v in vc.values():
            self.assertIsNotNone(v.value)

        field = rd.get()
        self.assertIsInstance(field, OcgField)

        vc.write(path_out, driver=DriverCSV)

        with open(path) as one:
            lines1 = one.readlines()
        with open(path_out) as two:
            lines2 = two.readlines()
        self.assertEqual(lines1, lines2)

    def test_system_field_creation(self):
        """Test creating a field with coordinates."""

        headers = ['ONE', 'two', 'THREE', 'x', 'y']
        record1 = [1, 'number', 4.5, 10.3, 12.4]
        record2 = [2, 'letter', 5.5, 11.3, 13.4]
        path = self.get_temporary_file_path('foo.csv')
        path_shp = self.get_temporary_file_path('foo.shp')
        with open(path, 'w') as out:
            writer = csv.writer(out)
            for row in [headers, record1, record2]:
                writer.writerow(row)

        metadata = RequestDataset(uri=path).metadata
        metadata['variables']['x']['dtype'] = float
        metadata['variables']['y']['dtype'] = float
        rd = RequestDataset(uri=path, dimension_map={'x': {'variable': 'x'}, 'y': {'variable': 'y'}}, metadata=metadata)

        field = rd.get()
        print field.geom
        self.assertIsInstance(field.grid.point.value[0, 0], Point)
        # field.write(path_shp, driver=DriverVector)
