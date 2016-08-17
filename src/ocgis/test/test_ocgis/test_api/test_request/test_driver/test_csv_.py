import csv

from ocgis import RequestDataset
from ocgis.new_interface.field import OcgField
from ocgis.test.base import TestBase


class TestDriverCSV(TestBase):
    def test(self):
        headers = ['ONE', 'two', 'THREE', 'x', 'y']
        record1 = [1, 'number', 4.5, 10.3, 12.4]
        record2 = [2, 'letter', 5.5, 11.3, 13.4]
        path = self.get_temporary_file_path('foo.csv')
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

        self.fail('test writing variable colleciton to file and asserting the load is the same as the test file')
