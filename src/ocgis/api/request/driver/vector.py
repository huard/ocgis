import datetime
import itertools
from collections import OrderedDict
from copy import deepcopy
from types import NoneType

import fiona
import numpy as np
from shapely.geometry import mapping

from ocgis import constants
from ocgis.api.request.driver.base import AbstractDriver, driver_scope
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.variable import SourcedVariable, VariableCollection
from ocgis.util.logging_ocgis import ocgis_lh


class DriverVector(AbstractDriver):
    extensions = ('.*\.shp',)
    key = 'vector'
    output_formats = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH,
                      constants.OUTPUT_FORMAT_SHAPEFILE]

    def init_variable_value(self, variable):
        value = self.get_variable_value(variable)

        if variable.parent is None:
            variable._set_value_(value.values()[0])
        else:
            for k, v in value.items():
                variable.parent[k]._set_value_(v)

        # Conform the units if requested.
        for k in value.keys():
            if variable.parent is None:
                v = variable
            else:
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

    def get_metadata(self):
        with driver_scope(self) as data:
            m = data.sc.get_meta(path=self.rd.uri)
            m['dimensions'] = {constants.NAME_GEOMETRY_DIMENSION: {'size': len(data),
                                                                   'name': constants.NAME_GEOMETRY_DIMENSION}}
            m['variables'] = OrderedDict()

            # Groups are not currently supported in vector formats but metadata expects groups.
            m['groups'] = OrderedDict()

            for p, d in m['schema']['properties'].items():
                d = get_dtype_from_fiona_type(d)
                m['variables'][p] = {'dimensions': (constants.NAME_GEOMETRY_DIMENSION,), 'dtype': d, 'name': p,
                                     'attributes': OrderedDict()}

            m[constants.NAME_GEOMETRY_DIMENSION] = {'dimensions': (constants.NAME_GEOMETRY_DIMENSION,),
                                                    'dtype': object,
                                                    'name': constants.NAME_GEOMETRY_DIMENSION,
                                                    'attributes': OrderedDict()}
            return m

    def get_source_metadata_as_json(self):
        # tdk: test on vector and netcdf
        raise NotImplementedError

    def get_variable_collection(self):
        parent = VariableCollection()
        for n, v in self.metadata['variables'].items():
            SourcedVariable(name=n, request_dataset=self.rd, parent=parent)
        GeometryVariable(name=constants.NAME_GEOMETRY_DIMENSION, request_dataset=self.rd, parent=parent)
        crs = self.get_crs()
        if crs is not None:
            parent.add_variable(self.get_crs())
        return parent

    def get_variable_metadata(self, variable_object):
        if isinstance(variable_object, GeometryVariable):
            # Geometry variables are located in a different metadata section.
            ret = self.metadata[variable_object.name]
        else:
            ret = super(DriverVector, self).get_variable_metadata(variable_object)
        return ret

    def get_variable_value(self, variable):
        # Iteration is always based on source indices. Generate them if they are not available on the variable.
        iteration_dimension = variable.dimensions[0]
        if iteration_dimension._src_idx is None:
            raise ValueError("Iteration dimension must have a source index.")
        else:
            src_idx = iteration_dimension._src_idx

        # For vector formats based on loading via iteration, it makes sense to load all values with a single pass.
        with driver_scope(self, slc=src_idx) as g:
            ret = {}
            if variable.parent is None:
                ret[variable.name] = np.zeros(variable.shape, dtype=variable.dtype)
                for idx, row in enumerate(g):
                    ret[variable.name][idx] = row['properties'][variable.name]
            else:
                ret = {}
                # Initialize the variable data as zero arrays.
                for v in variable.parent.values():
                    if not isinstance(v, CoordinateReferenceSystem):
                        ret[v.name] = np.zeros(v.shape, dtype=v.dtype)
                # Fill those arrays.
                for idx, row in enumerate(g):
                    for dv in self.get_dimensioned_variables():
                        ret[dv][idx] = row['properties'][dv]
                    ret[constants.NAME_GEOMETRY_DIMENSION][idx] = row['geom']
        return ret

    @staticmethod
    def _close_(obj):
        from ocgis.util.geom_cabinet import GeomCabinetIterator
        if isinstance(obj, GeomCabinetIterator):
            # Geometry iterators have no close methods.
            pass
        else:
            obj.close()

    def _init_variable_from_source_main_(self, variable_object, variable_metadata):
        if variable_object._dtype is None:
            variable_object.dtype = variable_metadata['dtype']

        variable_attrs = variable_object._attrs
        for k, v in variable_metadata['attributes'].items():
            if k not in variable_attrs:
                variable_attrs[k] = deepcopy(v)

    @staticmethod
    def _open_(uri, mode='r', **kwargs):
        if mode == 'r':
            from ocgis import GeomCabinetIterator
            return GeomCabinetIterator(path=uri, **kwargs)
        elif mode == 'w':
            ret = fiona.open(uri, mode='w', **kwargs)
        else:
            raise ValueError('Mode not supported: "{}"'.format(mode))
        return ret

    @classmethod
    def _write_variable_collection_main_(cls, vc, opened_or_path, comm, rank, size, **kwargs):
        # tdk: test conforming units!

        fiona_crs = kwargs.get('crs')
        fiona_schema = kwargs.get('fiona_schema')
        fiona_driver = kwargs.get('fiona_driver', 'ESRI Shapefile')
        variable_names = kwargs.get('variable_names', [])

        geom = get_geometry_variable(vc)

        # Identify the dimensions to use for variable iteration. These are found by accumulating variables having the
        # same dimensions as the geometry variable. Any additional dimensions are appended to this list. Also collect
        # the variables that have these dimensions as they are the variables we are writing to file.
        dimensions_to_iterate = list(geom.dimensions_names)
        if len(variable_names) > 0:
            if geom.name not in variable_names:
                variable_names.append(geom.name)
            variables_to_write = [vc[v] for v in variable_names]
            for v in variables_to_write:
                dimensions_to_iterate += list(v.dimensions_names)
        else:
            variables_to_write = []
            for v in vc.values():
                if len(set(dimensions_to_iterate).intersection(v.dimensions_names)) > 0:
                    dimensions_to_iterate += list(v.dimensions_names)
                    variables_to_write.append(v)
        dimensions_to_iterate = set(dimensions_to_iterate)

        # Open the output Fiona object using overloaded values or values determined at call-time.
        if not cls.inquire_opened_state(opened_or_path):
            if fiona_crs is None:
                fiona_crs = get_fiona_crs(vc)
            fiona_schema = get_fiona_schema(vc, geom, variables_to_write, fiona_schema)
        else:
            fiona_schema = opened_or_path.schema
            fiona_crs = opened_or_path.crs
            fiona_driver = opened_or_path.driver

        with driver_scope(cls, opened_or_path=opened_or_path, mode='w', driver=fiona_driver,
                          crs=fiona_crs, schema=fiona_schema) as sink:
            iter_dslices = iter_field_slices_for_records(vc, dimensions_to_iterate,
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


def get_fiona_crs(vc_or_field):
    try:
        # Attempt to pull the coordinate system from the field-like object. If it is a variable collection, look for
        # CRS variables.
        ret = vc_or_field.crs.value
    except AttributeError:
        ret = None
        for v in vc_or_field.values():
            if isinstance(v, CoordinateReferenceSystem):
                ret = v
                break
    return ret


def get_fiona_schema(vc_or_field, geometry_variable, variables_to_write, default):
    ret = default
    if variables_to_write is not None:
        variable_names_to_write = [v.name for v in variables_to_write]
    else:
        variable_names_to_write = [v.name for v in vc_or_field.values()]
    if ret is None:
        ret = {'geometry': geometry_variable.geom_type, 'properties': OrderedDict()}
        p = ret['properties']
        for v in vc_or_field.iter_data_variables():
            if v.name not in variable_names_to_write:
                continue
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
    try:
        ftype = m[dtype]
    except KeyError:
        # This may be a NumPy string type.
        if str(dtype).startswith('|S'):
            ftype = 'str'
        else:
            raise
    return ftype


def iter_field_slices_for_records(vc_like, dimension_names, variable_names):
    dimensions = [vc_like.dimensions[d] for d in dimension_names]
    target = vc_like.copy()
    to_pop = []
    for v in target.values():
        if v.name not in variable_names:
            to_pop.append(v.name)
    for tp in to_pop:
        target.pop(tp)

    # Load all values from source.
    target.load()

    iterators = [range(len(d)) for d in dimensions]
    for indices in itertools.product(*iterators):
        dslice = {d.name: indices[idx] for idx, d in enumerate(dimensions)}
        yield target[dslice]


def get_geometry_variable(field_like):
    """
    :param field_like: A field or variable collection.
    :return: The geometry variable.
    :rtype: GeometryVariable
    :raises: ValueError
    """

    # Find the geometry variable.
    geom = None
    try:
        # Try to get the geometry assuming it is a field object.
        geom = field_like.geom
    except AttributeError:
        for v in field_like.values():
            if isinstance(v, GeometryVariable):
                geom = v
    if geom is None:
        exc = ValueError('A geometry variable is required.')
        ocgis_lh(exc=exc)

    return geom
