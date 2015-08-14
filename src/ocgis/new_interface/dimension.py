from ocgis.new_interface.base import AbstractInterfaceObject


class Dimension(AbstractInterfaceObject):
    # tdk:doc

    def __init__(self, name, length=None):
        # tdk:test
        self.name = name
        self.length = length

    def __eq__(self, other):
        if other.__dict__ == self.__dict__:
            ret = True
        else:
            ret = False
        return ret

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        msg = '{0}(name={1}, length={2})'.format(self.__class__.__name__, self.name, self.length)
        return msg
