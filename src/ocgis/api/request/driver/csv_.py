import csv
from collections import OrderedDict

from ocgis.api.request.driver.base import AbstractDriver, driver_scope


class DriverCSV(AbstractDriver):
    extensions = ('.*\.csv',)
    key = 'csv'
    output_formats = 'all'

    def get_metadata(self):
        with driver_scope(self) as f:
            meta = {}
            # Get variable names assuming headers are always on the first row.
            reader = csv.reader(f)
            variable_names = reader.next()

            # Fill in variable and dimension metadata.
            meta['variables'] = OrderedDict()
            meta['dimensions'] = OrderedDict()
            for varname in variable_names:
                meta['variables'][varname] = {'name': varname, 'dtype': object, 'dimensions': ('n_records',)}
            meta['dimensions']['n_records'] = {'name': 'n_records', 'size': sum(1 for _ in f)}
        return meta

    def get_variable_value(self, variable):
        # For CSV files, it makes sense to load all variables from source simultaneously.
        if variable.parent is None:
            to_load = [variable]
        else:
            to_load = variable.parent.values()

        with driver_scope(self) as f:
            reader = csv.DictReader(f)
            bounds_local = variable.dimensions[0].bounds_local
            for idx, row in enumerate(reader):
                if idx < bounds_local[0]:
                    continue
                else:
                    if idx >= bounds_local[1]:
                        break
                for tl in to_load:
                    if not tl.has_allocated_value:
                        tl.allocate_value()
                    tl.value[idx - bounds_local[0]] = row[tl.name]
        return variable.value

    @classmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, comm, rank, size, **kwargs):
        fieldnames = [v.name for v in vc.iter_data_variables()]
        if rank == 0:
            with driver_scope(cls, opened_or_path, mode='w') as opened:
                writer = csv.DictWriter(opened, fieldnames)
                writer.writeheader()
        for current_rank_write in range(size):
            if rank == current_rank_write:
                with driver_scope(cls, opened_or_path, mode='a') as opened:
                    writer = csv.DictWriter(opened, fieldnames)
                    for idx in range(vc[fieldnames[0]].shape[0]):
                        row = {fn: cls.get_variable_write_value(vc[fn])[idx] for fn in fieldnames}
                        writer.writerow(row)
            comm.Barrier()

    def _init_variable_from_source_main_(self, *args, **kwargs):
        pass
