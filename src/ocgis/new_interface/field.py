from collections import OrderedDict

from ocgis.new_interface.base import AbstractInterfaceObject


class FieldBundle(AbstractInterfaceObject):
    # tdk: retain variables for backwards compatibility
    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.fields = kwargs.pop('fields', OrderedDict())
        self.realization = kwargs.pop('realization', None)
        self.temporal = kwargs.pop('temporal', None)
        self.level = kwargs.pop('level', None)
        self.spatial = kwargs.pop('spatial', None)
        self.extra = kwargs.pop('extra', OrderedDict())
