from collections import OrderedDict
from copy import deepcopy

import numpy as np

from ocgis import constants
from ocgis.api.request.driver.base import AbstractDriver
from ocgis.interface.base.crs import CoordinateReferenceSystem
from ocgis.interface.base.field import Field
from ocgis.interface.base.variable import Variable

# tdk: clean-up
from ocgis.new_interface.dimension import SourcedDimension
from ocgis.new_interface.geom import GeometryVariable
from ocgis.new_interface.variable import SourcedVariable, VariableCollection


class DriverVector(AbstractDriver):
    extensions = ('.*\.shp',)
    key = 'vector'
    output_formats = [constants.OUTPUT_FORMAT_NUMPY, constants.OUTPUT_FORMAT_NETCDF_UGRID_2D_FLEXIBLE_MESH,
                      constants.OUTPUT_FORMAT_SHAPEFILE]

    def _close_(self, obj):
        pass

    def get_crs(self):
        return CoordinateReferenceSystem(value=self.metadata['crs'])

    def get_dimension_map(self, metadata):
        ret = {'geom': {'variable': constants.NAME_GEOMETRY_DIMENSION},
               'crs': {'variable': self.get_crs().name}}
        return ret

    def get_dimensioned_variables(self):
        # tdk: the geometry variable should not be part of this?
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
                m['variables'][p] = {'dimensions': (constants.NAME_GEOMETRY_DIMENSION,), 'dtype': d, 'name': p,
                                     'attributes': OrderedDict()}
            m['variables'][constants.NAME_GEOMETRY_DIMENSION] = {'dimensions': (constants.NAME_GEOMETRY_DIMENSION,),
                                                                 'dtype': m['schema']['geometry'], 'name': p,
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

        if variable._dimensions is None:
            desired_dimension = m['variables'][variable.name]['dimensions'][0]
            desired_dimension = m['dimensions'][desired_dimension]
            new_dimension = SourcedDimension(name=desired_dimension['name'], length=desired_dimension['length'])
            super(SourcedVariable, variable)._set_dimensions_(new_dimension)

        variable_attrs = variable._attrs
        for k, v in m['variables'][variable.name]['attributes'].items():
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

    def get_variable_value(self):
        raise NotImplementedError

    def write_gridxy(self, *args, **kwargs):
        raise NotImplementedError

    def write_variable(self, *args, **kwargs):
        raise NotImplementedError

    def write_variable_collection(self, *args, **kwargs):
        raise NotImplementedError

    def _get_field_(self, format_time=None):
        # todo: option to pass select_ugid
        # todo: option for time dimension and time subsetting
        # todo: remove format_time option - there for compatibility with the netCDF driver
        from ocgis import SpatialDimension

        ds = self.open()
        try:
            records = list(ds)
            sdim = SpatialDimension.from_records(records, crs=self.get_crs())
            # do not load the properties - they are transformed to variables in the case of the values put into fields
            sdim.properties = None
            vc = VariableCollection()
            for xx in self.rd:
                value = np.array([yy['properties'][xx['variable']] for yy in records]).reshape(1, 1, 1, 1, -1)
                var = Variable(name=xx['variable'], alias=xx['alias'], units=xx['units'], conform_units_to=xx['units'],
                               value=value)
                vc.add_variable(var, assign_new_uid=True)
            field = Field(spatial=sdim, variables=vc, name=self.rd.name)
            return field
        finally:
            self.close(ds)
