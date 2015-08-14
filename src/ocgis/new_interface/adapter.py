from ocgis.new_interface.base import AbstractInterfaceObject


class AbstractAdapter(AbstractInterfaceObject):
    # tdk:doc
    # tdk:test

    def __init__(self, vc):
        self.vc = vc
