from collections import OrderedDict
from copy import copy

from ocgis.interface.base.attributes import Attributes
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.geom import SpatialContainer
from ocgis.new_interface.temporal import TemporalVariable
from ocgis.new_interface.variable import VariableCollection, Variable
from ocgis.util.helpers import get_formatted_slice

_FIELD_BUNDLE_DIMENSIONS = ('realization', 'time', 'level', 'spatial')
_FIELD_BUNDLE_DIMENSIONS_MAP = dict(zip(_FIELD_BUNDLE_DIMENSIONS, range(3)))


class FieldBundle(AbstractInterfaceObject, Attributes):
    # tdk: retain variables for backwards compatibility
    def __init__(self, **kwargs):
        self._should_sync = False
        self._realization = None
        self._time = None
        self._level = None
        self._spatial = None

        self.name = kwargs.pop('name')
        fields = kwargs.pop('fields', [])
        for f in fields:
            raise NotImplementedError
        self.fields = kwargs.pop('fields', VariableCollection())
        self.realization = kwargs.pop('realization', None)

        self.time = kwargs.pop('time', None)
        # Backwards compatibility.
        self.temporal = kwargs.pop('temporal', None)

        self.level = kwargs.pop('level', None)
        self.spatial = kwargs.pop('spatial', None)
        self.extra = kwargs.pop('extra', VariableCollection())
        self.schemas = {}

        Attributes.__init__(self, **kwargs)

        self.should_sync = True

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, self.ndim)
        ret = copy(self)
        ret.should_sync = False
        slice_idx = 0
        for dname in _FIELD_BUNDLE_DIMENSIONS:
            target = getattr(ret, dname)
            if target is not None:
                if dname != 'spatial':
                    slc_target = slc[slice_idx]
                else:
                    slc_target = slc[slice_idx:]
                setattr(ret, dname, target.__getitem__(slc_target))
                slice_idx += 1
        ret.should_sync = True
        return ret

    @property
    def crs(self):
        try:
            ret = self.spatial.crs
        except AttributeError:
            ret = None
        return ret

    @property
    def dimensions(self):
        for fn in _FIELD_BUNDLE_DIMENSIONS:
            yld = getattr(self, fn)
            if yld is not None:
                yield yld

    @property
    def ndim(self):
        ret = 0
        for d in self.dimensions:
            ret += d.ndim
        return ret

    @property
    def realization(self):
        return self._realization

    @realization.setter
    def realization(self, value):
        self._set_dimension_variable_('_realization', value, 'R')

    @property
    def shape(self):
        ret = []
        for d in self.dimensions:
            try:
                ret += list(d.shape)
            except AttributeError:
                # Assume is None.
                ret.append(None)
        return tuple(ret)

    @property
    def shape_dict(self):
        ret = OrderedDict()
        for name, d in zip(_FIELD_BUNDLE_DIMENSIONS, self.dimensions):
            ret[name] = d.shape
        return ret

    @property
    def should_sync(self):
        return self._should_sync

    @should_sync.setter
    def should_sync(self, value):
        self._should_sync = value
        if value:
            self.sync()

    @property
    def temporal(self):
        return self._time

    @temporal.setter
    def temporal(self, value):
        self._set_time_(value)

    @property
    def time(self):
        return self._time

    @time.setter
    def time(self, value):
        self._set_time_(value)

    def _set_time_(self, value):
        if value is not None:
            assert isinstance(value, TemporalVariable)
        self._set_dimension_variable_('_time', value, 'T')

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, value):
        self._set_dimension_variable_('_level', value, 'L')

    @property
    def spatial(self):
        return self._spatial

    @spatial.setter
    def spatial(self, value):
        if value is None:
            assert isinstance(value, SpatialContainer)
        self._spatial = value
        self.sync()

    def create_field(self, variable, schema=None):
        if schema is None:
            raise NotImplementedError
        variable.attrs = variable.attrs.copy()
        self.schemas[variable.alias] = schema
        self.fields[variable.alias] = variable
        self.sync()

    def sync(self):
        if self.should_sync:
            crs = self.crs
            for f in self.fields.itervalues():
                if crs is not None:
                    attrs = f.attrs
                    crs_name = crs.name
                    if 'grid_mapping_name' not in attrs:
                        attrs['grid_mapping_name'] = crs_name

    def _set_dimension_variable_(self, name, value, axis):
        if value is not None:
            assert isinstance(value, Variable)
            value.attrs['axis'] = value.attrs.pop('axis', axis)
        setattr(self, name, value)
        self.sync()

    def write_netcdf(self, *args, **kwargs):
        # tdk: test temporal
        # tdk: test field
        self.sync()
        vc = VariableCollection(attrs=self.attrs, variables=self.fields.itervalues())
        for v in self.extra.itervalues():
            vc.add_variable(v)
        if self.realization is not None:
            vc.add_variable(self.realization)
        if self.level is not None:
            vc.add_variable(self.level)
        if self.spatial is not None:
            vc.add_variable(self.spatial.grid)
        if self.temporal is not None:
            vc.add_variable(self.temporal)
        vc.write_netcdf(*args, **kwargs)
