from abc import ABCMeta

from ocgis.new_interface.base import AbstractInterfaceObject


class AbstractAdapter(AbstractInterfaceObject):
    __metaclass__ = ABCMeta


class SpatialAdapter(AbstractAdapter):
    def __init__(self, crs=None):
        self.crs = crs

    def update_crs(self, to_crs):
        raise NotImplementedError

