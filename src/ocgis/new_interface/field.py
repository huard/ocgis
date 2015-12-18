from ocgis.interface.base.attributes import Attributes
from ocgis.new_interface.base import AbstractInterfaceObject
from ocgis.new_interface.variable import VariableCollection


class FieldBundle(AbstractInterfaceObject, Attributes):
    # tdk: retain variables for backwards compatibility
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.fields = kwargs.pop('fields', VariableCollection())
        self.realization = kwargs.pop('realization', None)
        self.temporal = kwargs.pop('temporal', None)
        self.level = kwargs.pop('level', None)
        self.spatial = kwargs.pop('spatial', None)
        self.extra = kwargs.pop('extra', VariableCollection())

        Attributes.__init__(self, **kwargs)

    def write_netcdf(self, *args, **kwargs):
        # tdk: test temporal
        # tdk: test field
        vc = VariableCollection(attrs=self.attrs, variables=self.fields.itervalues())
        for v in self.extra.itervalues():
            vc.add_variable(v)
        if self.realization is not None:
            vc.add_variable(self.realization)
        if self.level is not None:
            vc.add_variable(self.level)
        if self.spatial is not None:
            vc.add_variable(self.spatial.grid)
        vc.write_netcdf(*args, **kwargs)
