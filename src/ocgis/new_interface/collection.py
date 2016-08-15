from collections import OrderedDict

from ocgis.new_interface.variable import VariableCollection


class SpatialCollection(VariableCollection):
    def add_field(self, field, uid):
        if uid not in self.children:
            self.children[uid] = OrderedDict()
        children = self.children[uid]
        children

    @property
    def geoms(self):
        ret = OrderedDict()
        for k, v in self.children.items():
            ret[k] = v.geom.value[0]
        return ret

    @property
    def crs(self):
        for key in self.keys():
            return self[key].geom.crs
