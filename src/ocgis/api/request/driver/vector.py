import datetime
import itertools
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
from ocgis.util.logging_ocgis import ocgis_lh


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
        for k in value.keys():
            v = variable.parent[k]
            try:
                conform_units_to = self.rd.metadata['variables'][v.name].get('conform_units_to')
            except KeyError:
                # This is okay if the target variable is a geometry variable.
                if isinstance(v, GeometryVariable):
                    conform_units_to = None
                else:
                    raise
            if conform_units_to is not None:
                v.cfunits_conform(conform_units_to)

    def _close_(self, obj):
        pass

    def get_crs(self):
        crs = self.metadata['crs']
        if len(crs) == 0:
            ret = None
        else:
            ret = CoordinateReferenceSystem(value=self.metadata['crs'])
        return ret

    def get_dimension_map(self, metadata):
        ret = {'geom': {'variable': constants.NAME_GEOMETRY_DIMENSION}}
        crs = self.get_crs()
        if crs is not None:
            ret['crs'] = {'variable': self.get_crs().name}
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

    def _open_(self, slc=None):
        from ocgis import GeomCabinetIterator
        return GeomCabinetIterator(path=self.rd.uri, slc=slc)

    def get_variable_collection(self):
        parent = VariableCollection(name=self.rd.name)
        for n, v in self.metadata['variables'].items():
            SourcedVariable(name=n, request_dataset=self.rd, parent=parent)
        GeometryVariable(name=constants.NAME_GEOMETRY_DIMENSION, request_dataset=self.rd, parent=parent)
        crs = self.get_crs()
        if crs is not None:
            parent.add_variable(self.get_crs())
        return parent

    def get_variable_value(self, variable):
        # For vector formats based on loading via iteration, it makes sense to load all values with a single pass.
        if variable.parent is None:
            raise NotImplementedError
        else:
            ret = {}
            for v in variable.parent.values():
                if not isinstance(v, CoordinateReferenceSystem):
                    ret[v.name] = np.zeros(v.shape, dtype=v.dtype)
            g = self.open(slc=variable.dimensions[0]._src_idx)
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
        # tdk: test conforming units!

        crs = kwargs.get('crs')
        fiona_schema = kwargs.get('fiona_schema')
        fiona_driver = kwargs.get('fiona_driver', 'ESRI Shapefile')
        variable_names = kwargs.get('variable_names', [])

        # Find the geometry variable.
        geom = None
        try:
            geom = field.geom
        except AttributeError:
            for v in field.values():
                if isinstance(v, GeometryVariable):
                    geom = v
        if geom is None:
            exc = ValueError('A geometry variable is required.')
            ocgis_lh(exc=exc)

        # Identify the dimensions to use for variable iteration. These are found by accumulating variables having the
        # same dimensions as the geometry variable. Any additional dimensions are appended to this list. Also collect
        # the variables that have these dimensions as they are the variables we are writing to file.
        dimensions_to_iterate = list(geom.dimensions_names)
        if len(variable_names) > 0:
            if geom.name not in variable_names:
                variable_names.append(geom.name)
            variables_to_write = [field[v] for v in variable_names]
            for v in variables_to_write:
                dimensions_to_iterate += list(v.dimensions_names)
        else:
            variables_to_write = []
            for v in field.values():
                if len(set(dimensions_to_iterate).intersection(v.dimensions_names)) > 0:
                    dimensions_to_iterate += list(v.dimensions_names)
                    variables_to_write.append(v)
        dimensions_to_iterate = set(dimensions_to_iterate)

        # Open the output Fiona object using overloaded values or values determined at call-time.
        if isinstance(fiona_or_path, basestring):
            fiona_driver = kwargs.pop('fiona_driver', 'ESRI Shapefile')
            fiona_crs = get_fiona_crs(field, crs)
            fiona_schema = get_fiona_schema(field, geom, variables_to_write, fiona_schema)
            sink = fiona.open(fiona_or_path, 'w', driver=fiona_driver, crs=fiona_crs, schema=fiona_schema)
            should_close = True
        else:
            sink = fiona_or_path
            fiona_schema = sink.schema
            should_close = False

        try:
            iter_dslices = iter_field_slices_for_records(field, dimensions_to_iterate,
                                                         [v.name for v in variables_to_write])
            properties = fiona_schema['properties'].keys()
            for sub in iter_dslices:
                record = OrderedDict()
                for p in properties:
                    # Attempt to pull a datetime object if they are available.
                    try:
                        indexed_value = sub[p].value_datetime.flatten()[0]
                    except AttributeError:
                        indexed_value = sub[p].value.flatten()[0]
                    # Convert object to string if this is its data type. This is important for things like datetime
                    # objects which require this conversion before writing.
                    if indexed_value is not None:
                        if fiona_schema['properties'][p].startswith('str'):
                            indexed_value = str(indexed_value)
                        else:
                            # Attempt to convert the data to a Python-native data type.
                            try:
                                indexed_value = indexed_value.tolist()
                            except AttributeError:
                                pass
                    record[p] = indexed_value
                # tdk: OPTIMIZE: this mapping should not repeat for each geometry. there should be a cache.
                record = {'properties': record, 'geometry': mapping(sub[geom.name].value.flatten()[0])}
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
    m = {datetime.date: 'str',
         datetime.datetime: 'str',
         np.int64: 'int',
         NoneType: None,
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
    dtype = variable._get_vector_dtype_()
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


def get_fiona_schema(vc_or_field, geometry_variable, variables_to_write, default):
    ret = default
    if variables_to_write is not None:
        variable_names_to_write = [v.name for v in variables_to_write]
    else:
        variable_names_to_write = [v.name for v in vc_or_field.values()]
    if ret is None:
        ret = {}
        ret['geometry'] = geometry_variable.geom_type
        ret['properties'] = OrderedDict()
        p = ret['properties']
        for v in vc_or_field.values():
            if v.name not in variable_names_to_write:
                continue
            if not isinstance(v, (GeometryVariable, CoordinateReferenceSystem)):
                p[v.name] = get_fiona_type_from_variable(v)
        for k, v in p.items():
            if v == 'str':
                p[k] = get_fiona_string_width(vc_or_field[k].masked_value.compressed())
    return ret


def get_fiona_string_width(arr):
    ret = 0
    for ii in arr.flat:
        ii = str(ii)
        if len(ii) > ret:
            ret = len(ii)
    ret = 'str:{}'.format(ret)
    return ret


def iter_field_slices_for_records(vc_like, dimension_names, variable_names):
    dimensions = [vc_like.dimensions[d] for d in dimension_names]
    target = vc_like.copy()
    to_pop = []
    for v in target.values():
        if v.name not in variable_names:
            to_pop.append(v.name)
    for tp in to_pop:
        target.pop(tp)

    # tdk: RESUME: there should be a high-level method to load all values from source
    for v in target.values():
        assert v.value is not None

    iterators = [range(len(d)) for d in dimensions]
    for indices in itertools.product(*iterators):
        dslice = {d.name: indices[idx] for idx, d in enumerate(dimensions)}
        yield target[dslice]