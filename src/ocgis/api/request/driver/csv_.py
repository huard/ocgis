import csv
from collections import OrderedDict

from ocgis.api.request.driver.base import AbstractDriver
from ocgis.new_interface.dimension import Dimension
from ocgis.new_interface.mpi import MPI_COMM


class DriverCSV(AbstractDriver):
    extensions = ('.*\.csv',)
    key = 'csv'
    output_formats = 'all'

    def get_dump_report(self):
        lines = ['URI: {}'.format(self.rd.uri)]
        return lines

    def get_metadata(self):
        f = self.open()
        try:
            meta = {}
            # Get variable names.
            reader = csv.reader(f)
            variable_names = reader.next()
            meta['variables'] = OrderedDict()
            meta['dimensions'] = OrderedDict()
            for varname in variable_names:
                meta['variables'][varname] = {'name': varname, 'dtype': object, 'dimensions': ('n_records',)}
            meta['dimensions']['n_records'] = {'name': 'n_records', 'size': sum(1 for _ in f)}
        finally:
            self.close(f)
        return meta

    def get_variable_value(self, variable):
        # For CSV files, it makes sense to load all variables from source simultaneously.
        if variable.parent is None:
            to_load = [variable]
        else:
            to_load = variable.parent.values()

        f = self.open()
        try:
            reader = csv.DictReader(f)
            bounds_local = variable.dimensions[0].bounds_local
            for idx, row in enumerate(reader):
                if idx < bounds_local[0]:
                    continue
                else:
                    if idx >= bounds_local[1]:
                        break
                for tl in to_load:
                    if tl._value is None:
                        tl.allocate_value()
                    tl.value[idx - bounds_local[0]] = row[tl.name]
            return variable.value
        finally:
            self.close(f)

    @staticmethod
    def write_gridxy(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def write_variable(*args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def write_variable_collection(vc, file_or_path, **kwargs):
        comm = kwargs.pop('comm', MPI_COMM)
        rank = comm.Get_rank()
        size = comm.Get_size()

        if size > 1:
            assert isinstance(file_or_path, basestring)

        fieldnames = [v.name for v in vc.iter_data_variables()]

        if rank == 0:
            opened = open(file_or_path, mode='w')
            try:
                writer = csv.DictWriter(opened, fieldnames)
                writer.writeheader()
            finally:
                opened.close()

        for current_rank_write in range(size):
            if rank == current_rank_write:
                if isinstance(file_or_path, basestring):
                    should_close = True
                    file_or_path = open(file_or_path, mode='a')
                else:
                    should_close = False

                try:
                    writer = csv.DictWriter(file_or_path, fieldnames)
                    for idx in range(vc[fieldnames[0]].shape[0]):
                        row = {fn: vc[fn].value[idx] for fn in fieldnames}
                        writer.writerow(row)
                finally:
                    if should_close:
                        file_or_path.close()
            comm.Barrier()

    def _close_(self, obj):
        obj.close()

    def _get_dimensions_main_(self, group_metadata):
        dref = group_metadata['dimensions'].values()[0]
        dim = Dimension(dref['name'], size=dref['size'], src_idx='auto')
        return tuple([dim])

    def _init_variable_from_source_main_(self, *args, **kwargs):
        pass

    def _open_(self, mode='r'):
        uri = self.rd.uri
        ret = open(uri, mode=mode)
        return ret
