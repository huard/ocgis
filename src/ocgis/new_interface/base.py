from abc import ABCMeta, abstractmethod


class AbstractInterfaceObject(object):
    __metaclass__ = ABCMeta


class AbstractSpatialReferencedObject(AbstractInterfaceObject):
    __metaclass__ = ABCMeta

    def __init__(self, crs=None):
        self.crs = crs

    @abstractmethod
    def update_crs(self, to_crs):
        """"""
