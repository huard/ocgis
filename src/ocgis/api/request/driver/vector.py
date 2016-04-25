import datetime
from collections import OrderedDict
from copy import deepcopy
from types import NoneType

import fiona
import numpy as np
from shapely.geometry import mapping

from ocgis import constants
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.interface.base.crs import CoordinateReferenceSystem

# tdk: clean-up
from ocgis.new_interface.dimension import SourcedDimension
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.variable import SourcedVariable, VariableCollection


class DriverVector(AbstractDriver):
    extensions = ('.*\.shp',)
    key = 'vector'
    output_formats = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH,
                      constants.OUTPUT_FORMAT_SHAPEFILE]

    def allocate_variable_value(self, variable):
        value = self.get_variable_value(variable)
        for k, v in value.items():
            variable.parent[k]._set_value_(v)
        # Conform the units if requested.
        conform_units_to = self.rd.metadata['variables'][variable.name].get('conform_units_to')
        if conform_units_to is not None:
            variable.cfunits_conform(conform_units_to)

    def _close_(self, obj):
        pass

    def get_crs(self):
        return CoordinateReferenceSystem(value=self.metadata['crs'])

    def get_dimension_map(self, metadata):
        ret = {'geom': {'variable': constants.NAME_GEOMETRY_DIMENSION},
               'crs': {'variable': self.get_crs().name}}
        return ret

    def get_dimensioned_variables(self):
        return self.rd.metadata['variables'].keys()

    def get_dump_report(self):
        """
        :returns: A sequence of strings suitable for printing or writing to file.
        :rtype: [str, ...]
        """

        from ocgis import CoordinateReferenceSystem

        meta = self.rd.source_metadata
        try:
            ds = self.open()
            n = len(ds)
        finally:
            self.close(ds)
        lines = []
        lines.append('')
        lines.append('URI = {0}'.format(self.rd.uri))
        lines.append('')
        lines.append('Geometry Type: {0}'.format(meta['schema']['geometry']))
        lines.append('Geometry Count: {0}'.format(n))
        lines.append('CRS: {0}'.format(CoordinateReferenceSystem(value=meta['crs']).value))
        lines.append('Properties:')
        for k, v in meta['schema']['properties'].iteritems():
            lines.append(' {0} {1}'.format(v, k))
        lines.append('')

        return lines

    def get_metadata(self):
        data = self.open()
        try:
            m = data.sc.get_meta(path=self.rd.uri)
            m['dimensions'] = {constants.NAME_GEOMETRY_DIMENSION: {'length': len(data),
                                                                   'name': constants.NAME_GEOMETRY_DIMENSION}}
            m['variables'] = OrderedDict()
            for p, d in m['schema']['properties'].items():
                d = get_dtype_from_fiona_type(d)
                m['variables'][p] = {'dimensions': (constants.NAME_GEOMETRY_DIMENSION,), 'dtype': d, 'name': p,
                                     'attributes': OrderedDict()}
            m[constants.NAME_GEOMETRY_DIMENSION] = {'dimensions': (constants.NAME_GEOMETRY_DIMENSION,),
                                                    'dtype': object,
                                                    'name': constants.NAME_GEOMETRY_DIMENSION,
                                                                 'attributes': OrderedDict()}
            return m
        finally:
            self.close(data)

    def get_source_metadata_as_json(self):
        # tdk: test on vector and netcdf
        raise NotImplementedError

    def allocate_variable_without_value(self, variable):
        # tdk: set the variable's data type from the fiona datatype
        m = self.rd.metadata
        if isinstance(variable, GeometryVariable):
            mv = m[constants.NAME_GEOMETRY_DIMENSION]
        else:
            mv = m['variables'][variable.name]

        if variable._dimensions is None:
            desired_dimension = mv['dimensions'][0]
            desired_dimension = m['dimensions'][desired_dimension]
            new_dimension = SourcedDimension(name=desired_dimension['name'], length=desired_dimension['length'])
            super(SourcedVariable, variable)._set_dimensions_(new_dimension)

        if variable._dtype is None:
            variable.dtype = mv['dtype']

        variable_attrs = variable._attrs
        for k, v in mv['attributes'].items():
            if k not in variable_attrs:
                variable_attrs[k] = deepcopy(v)

        variable._allocated = True

    def _open_(self):
        from ocgis import GeomCabinetIterator
        return GeomCabinetIterator(path=self.rd.uri)

    def get_variable_collection(self):
        parent = VariableCollection(name=self.rd.name)
        for n, v in self.metadata['variables'].items():
            SourcedVariable(name=n, request_dataset=self.rd, parent=parent)
        GeometryVariable(name=constants.NAME_GEOMETRY_DIMENSION, request_dataset=self.rd, parent=parent)
        parent.add_variable(self.get_crs())
        return parent

    def get_variable_value(self, variable):
        # tdk: test conforming units!
        # For vector formats based on loading via iteration, it makes sense to load all values with a single pass.
        if variable.parent is None:
            raise NotImplementedError
        else:
            ret = {}
            for v in variable.parent.values():
                if not isinstance(v, CoordinateReferenceSystem):
                    ret[v.name] = np.zeros(v.shape, dtype=v.dtype)
            g = self.open()
            for idx, row in enumerate(g):
                for v in self.get_dimensioned_variables():
                    ret[v][idx] = row['properties'][v]
                ret[constants.NAME_GEOMETRY_DIMENSION][idx] = row['geom']
            self.close(g)
        return ret

    def write_gridxy(self, *args, **kwargs):
        raise NotImplementedError

    def write_variable(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def write_variable_collection(field, fiona_or_path, **kwargs):
        # tdk: test passing an open fiona file object
        # tdk: test with a time dimension
        # tdk: test writing cf netcdf data

        if isinstance(fiona_or_path, basestring):
            fiona_driver = kwargs.pop('fiona_driver', 'ESRI Shapefile')
            fiona_crs = get_fiona_crs(field, kwargs.get('crs'))
            fiona_schema = get_fiona_schema(field, kwargs.get('schema'))
            sink = fiona.open(fiona_or_path, 'w', driver=fiona_driver, crs=fiona_crs, schema=fiona_schema)
            should_close = True
        else:
            sink = fiona_or_path
            should_close = False

        try:
            iterators = {k: (ii for ii in range(len(v))) for k, v in field.dimensions.items()}
            properties = fiona_schema['properties'].keys()
            while True:
                try:
                    dslice = {k: v.next() for k, v in iterators.items()}
                except StopIteration:
                    break
                else:
                    sub = field[dslice]
                    record = OrderedDict()
                    for p in properties:
                        record[p] = sub[p].value[0]
                    record = {'properties': record}
                    record['geometry'] = mapping(sub.geom.value[0])
                    sink.write(record)
        finally:
            if should_close:
                sink.close()


def get_dtype_from_fiona_type(ftype):
    if ftype.startswith('int'):
        ret = np.int
    elif ftype.startswith('str'):
        ret = object
    elif ftype.startswith('float'):
        ret = np.float
    else:
        raise NotImplementedError(ftype)
    return ret


def get_fiona_type_from_variable(variable):
    # tdk: test with all masked data
    m = {datetime.date: 'str',
         datetime.datetime: 'str',
         np.int64: 'int',
         NoneType: None,
         np.int32: 'int',
         np.float64: 'float',
         np.float32: 'float',
         np.float16: 'float',
         np.int16: 'int',
         np.int32: 'int',
         str: 'str',
         np.dtype('int32'): 'int',
         np.dtype('int64'): 'int',
         np.dtype('float32'): 'float',
         np.dtype('float64'): 'float',
         int: 'int',
         float: 'float',
         object: 'str'}
    dtype = variable.dtype
    ftype = m[dtype]
    return ftype


def get_fiona_crs(vc_or_field, default):
    ret = default
    if ret is None:
        try:
            ret = vc_or_field.crs.value
        except AttributeError:
            ret = None
    return ret


def get_fiona_string_width(value):
    pass


def get_fiona_schema(vc_or_field, default):
    ret = default
    if ret is None:
        ret = {}
        ret['geometry'] = get_fiona_geometry_type(vc_or_field)
        ret['properties'] = OrderedDict()
        p = ret['properties']
        for v in vc_or_field.values():
            if not isinstance(v, (GeometryVariable, CoordinateReferenceSystem)):
                p[v.name] = get_fiona_type_from_variable(v)
        for k, v in p.items():
            if v == 'str':
                p[k] = get_fiona_string_width(vc_or_field[k].masked_value.compressed())
    return ret


def get_fiona_string_width(arr):
    ret = 0
    for ii in arr.flat:
        if len(ii) > ret:
            ret = len(ii)
    ret = 'str:{}'.format(ret)
    return ret


def get_fiona_geometry_type(vc_or_field):
    try:
        ret = vc_or_field.geom.geom_type
    except AttributeError:
        for v in vc_or_field.values:
            try:
                ret = v.geom_type
                break
            except AttributeError:
                continue
    return ret
