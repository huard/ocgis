from collections import OrderedDict

from ocgis.api.collection import AbstractCollection


class SpatialCollection(AbstractCollection):
    def __init__(self, *args, **kwargs):
        self.properties = kwargs.pop('properties', None)
        self.geometry_variable = kwargs.pop('geometry_variable', None)

        super(SpatialCollection, self).__init__(*args, **kwargs)

    @property
    def geoms(self):
        ret = OrderedDict()
        for ii in range(self.geometry_variable.shape[0]):
            ret[self.geometry_variable.uid.value[ii]] = self.geometry_variable.value[ii]
        return ret

    @property
    def crs(self):
        return self.geometry_variable.crs
