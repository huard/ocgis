from collections import OrderedDict

from ocgis.new_interface.variable import VariableCollection


class SpatialCollection(VariableCollection):
    def add_field(self, field, container):
        uid = container.geom.uid.value[0]
        if uid not in self.children:
            self.children[uid] = container
        container.add_child(field)

    @property
    def geoms(self):
        ret = OrderedDict()
        for k, v in self.children.items():
            ret[k] = v.geom.value[0]
        return ret

    @property
    def crs(self):
        for child in self.children.values():
            return child.crs

    @property
    def properties(self):
        ret = OrderedDict()
        for k, v in self.children.items():
            ret[k] = OrderedDict()
            for variable in v.iter_data_variables():
                ret[k][variable.name] = variable.value[0]
        return ret
