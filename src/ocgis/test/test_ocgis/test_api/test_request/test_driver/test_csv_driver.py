import csv

from ocgis import RequestDataset
from ocgis.api.request.driver.csv_ import DriverCSV
from ocgis.new_interface.field import OcgField
from ocgis.new_interface.mpi import MPI_RANK, MPI_COMM
from ocgis.test.base import TestBase, attr


class TestDriverCSV(TestBase):
    def assertCSVFilesEqual(self, path1, path2):
        with open(path1) as one:
            lines1 = one.readlines()
        with open(path2) as two:
            lines2 = two.readlines()
        self.assertEqual(lines1, lines2)

    def get_path_to_template_csv(self):
        headers = ['ONE', 'two', 'THREE', 'x', 'y']
        record1 = [1, 'number', 4.5, 10.3, 12.4]
        record2 = [2, 'letter', 5.5, 11.3, 13.4]
        path = self.get_temporary_file_path('foo.csv')
        with open(path, 'w') as out:
            writer = csv.writer(out)
            for row in [headers, record1, record2]:
                writer.writerow(row)
        return path

    def test(self):
        path = self.get_path_to_template_csv()
        path_out = self.get_temporary_file_path('foo_out.csv')

        rd = RequestDataset(uri=path)
        vc = rd.get_variable_collection()

        for v in vc.values():
            self.assertIsNotNone(v.value)

        field = rd.get()
        self.assertIsInstance(field, OcgField)

        vc.write(path_out, driver=DriverCSV)

        self.assertCSVFilesEqual(path, path_out)

    @attr('mpi')
    def test_system_parallel_write(self):
        if MPI_RANK == 0:
            in_path = self.get_path_to_template_csv()
            out_path = self.get_temporary_file_path('foo_out.csv')
        else:
            in_path, out_path = [None] * 2

        in_path = MPI_COMM.bcast(in_path)
        out_path = MPI_COMM.bcast(out_path)

        rd = RequestDataset(in_path)
        rd.metadata['dimensions'].values()[0]['dist'] = True
        vc = rd.get_variable_collection()
        vc.write(out_path, driver=DriverCSV)

        if MPI_RANK == 0:
            self.assertCSVFilesEqual(in_path, out_path)

        MPI_COMM.Barrier()
