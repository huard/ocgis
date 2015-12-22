from collections import OrderedDict
from copy import copy, deepcopy

from ocgis.interface.base.attributes import Attributes
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.geom import SpatialContainer
from ocgis.new_interface.temporal import TemporalVariable
from ocgis.new_interface.variable import VariableCollection, Variable
from ocgis.util.helpers import get_formatted_slice

_FIELDBUNDLE_DIMENSIONS = ('realization', 'time', 'level', 'y', 'x', 'n_geom')
# _FIELDBUNDLE_DIMENSIONS_MAP = dict(zip(_FIELDBUNDLE_DIMENSIONS, range(3)))
# _AXES_MAP = dict(zip(('R', 'T', 'L', 'Y', 'X', 'G'),
#                      ('realization', 'time', 'level', 'grid.y', 'grid.x', 'geom')))
#
# schema = {'T': 'time', 'Y': 'lat', 'X': 'lon'}
_FIELDBUNDLE_DIMENSION_NAMES = {k: [k] for k in _FIELDBUNDLE_DIMENSIONS}


class FieldBundle(AbstractInterfaceObject, Attributes):
    # tdk: retain variables for backwards compatibility
    def __init__(self, **kwargs):
        self._should_sync = False
        self._realization = None
        self._time = None
        self._level = None
        self._spatial = None

        self.inherit_names = kwargs.pop('inherit_names', True)
        self.name = kwargs.pop('name')
        self.fields = kwargs.pop('fields', VariableCollection())
        self.realization = kwargs.pop('realization', None)
        self.time = kwargs.pop('time', None)
        # Backwards compatibility.
        if self._time is None:
            self.temporal = kwargs.pop('temporal', None)
        self._dimension_schema = deepcopy(_FIELDBUNDLE_DIMENSION_NAMES)

        self.level = kwargs.pop('level', None)
        self.spatial = kwargs.pop('spatial', None)
        self.extra = kwargs.pop('extra', VariableCollection())
        self.schemas = kwargs.pop('schemas', OrderedDict())

        Attributes.__init__(self, **kwargs)

        self.should_sync = True

    def __getitem__(self, slc):
        slc = get_formatted_slice(slc, self.ndim)
        ret = copy(self)
        ret.should_sync = False
        ret.fields = VariableCollection()
        for alias, f in self.fields.items():
            ret.fields.add_variable(f[slc])
        slc_fb = {}
        fb_dimensions = []
        for ii in self._dimension_schema.values():
            fb_dimensions += ii
        for idx, d in enumerate(self.dimensions):
            if d.name in fb_dimensions:
                slc_fb[d.name] = slc[idx]
        set_getitem_field_bundle(ret, slc_fb)
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
        var = self.fields.first()
        if var is None:
            ret = None
        else:
            ret = var.dimensions
        return ret

    @property
    def ndim(self):
        return len(self.dimensions)

    @property
    def realization(self):
        return self._realization

    @realization.setter
    def realization(self, value):
        self._set_dimension_variable_('_realization', value, 'R')

    @property
    def shape(self):
        dimensions = self.dimensions
        if dimensions is None:
            ret = None
        else:
            ret = tuple([d.length for d in self.dimensions])
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
        if value is not None:
            assert isinstance(value, SpatialContainer)
            value = value.copy()
        self._spatial = value
        self.sync()

    def create_field(self, variable, schema=None, rename_dimensions=False):
        self.should_sync = False
        if schema is not None:
            for k, v in schema.items():
                if rename_dimensions:
                    for idx, d in enumerate(variable.dimensions):
                        if d.name == v:
                            variable.dimensions[idx].name = k
                            break
                else:
                    if v not in self._dimension_schema[k]:
                        self._dimension_schema[k].append(v)
        variable.attrs = variable.attrs.copy()
        self.schemas[variable.alias] = schema
        self.fields[variable.alias] = variable
        self.should_sync = True

    def sync(self):
        if self.should_sync:
            crs = self.crs
            for f in self.fields.itervalues():
                if crs is not None:
                    attrs = f.attrs
                    crs_name = crs.name
                    if 'grid_mapping_name' not in attrs:
                        attrs['grid_mapping_name'] = crs_name

            if self.inherit_names:
                # Update grid dimension names.
                if self.spatial is not None:
                    try:
                        name_y = self._dimension_schema['y'][1]
                        name_x = self._dimension_schema['x'][1]
                    except IndexError:
                        # No new dimension names for x and y.
                        pass
                    else:
                        self.spatial.grid.name_y = name_y
                        self.spatial.grid.name_x = name_x
                        if self.spatial.grid.is_vectorized:
                            self.spatial.grid.y.dimensions[0].name = name_y
                            self.spatial.grid.y.name = name_y
                            self.spatial.grid.x.dimensions[0].name = name_x
                            self.spatial.grid.x.name = name_x
                        else:
                            for idx, d in enumerate([self.spatial.grid.y, self.spatial.grid.x]):
                                name_tuple = (name_y, name_x)
                                d.name = name_tuple[idx]
                                for dsubname, dsub in zip(name_tuple, d.dimensions):
                                    dsub.name = dsubname

                # Update the other dimension names.
                for name in ['time', 'level', 'realization']:
                    try:
                        new_name = self._dimension_schema[name][1]
                    except IndexError:
                        pass
                    else:
                        target = getattr(self, name)
                        if target is not None:
                            target.dimensions[0].name = new_name
                            target.name = new_name

    def _set_dimension_variable_(self, name, value, axis):
        if value is not None:
            assert isinstance(value, Variable)
            value = copy(value)
            value.attrs = value.attrs.copy()
            value.dimensions = deepcopy(value.dimensions)
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
        if self.time is not None:
            vc.add_variable(self.time)
        vc.write_netcdf(*args, **kwargs)


class DSlice(AbstractInterfaceObject):
    def __init__(self, names):
        self.names = names

    def get_reordered(self, slc, slc_names):
        names = self.names
        slc = get_formatted_slice(slc, len(names))
        mapped = dict(zip(slc_names, slc))
        ret = tuple([mapped[n] for n in names])
        return ret


def set_getitem_field_bundle(fb, slc):
    if fb.spatial is not None:
        spatial_slice = [None] * fb.spatial.ndim
    for k, v in slc.items():
        if k in fb._dimension_schema['y']:
            spatial_slice[0] = v
        elif k in fb._dimension_schema['x']:
            spatial_slice[1] = v
        elif k in fb._dimension_schema['n_geom']:
            spatial_slice[0] = v
        else:
            for d, poss in fb._dimension_schema.items():
                if k in poss:
                    setattr(fb, d, getattr(fb, d)[v])
                    break
    if fb.spatial is not None:
        fb.spatial = fb.spatial[spatial_slice]
