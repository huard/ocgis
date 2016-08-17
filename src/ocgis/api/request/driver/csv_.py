import csv
from collections import OrderedDict

from ocgis.api.request.driver.base import AbstractDriver
from ocgis.new_interface.dimension import Dimension


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
                    tl.value[idx] = row[tl.name]
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
    def write_variable_collection(vc, dataset_or_path, **kwargs):
        raise NotImplementedError

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
