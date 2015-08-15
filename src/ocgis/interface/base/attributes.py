from collections import OrderedDict


class Attributes(object):
    """
    Adds an ``attrs`` attribute and writes to an open netCDF object.

    :param dict attrs: A dictionary of arbitrary attributes to write to a netCDF object.
    """

    def __init__(self, attrs=None):
        self._attrs = None

        self.attrs = attrs

    @property
    def attrs(self):
        if self._attrs is None:
            self._attrs = OrderedDict()
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        if value is not None:
            self._attrs = OrderedDict(value)

    def write_attributes_to_netcdf_object(self, target):
        """
        :param target: A netCDF data object to write attributes to.
        :type target: :class:`netCDF4.Variable` or :class:`netCDF4.Dataset`
        """

        for k, v in self.attrs.iteritems():
            if k == 'axis' and isinstance(v, basestring):
                target.axis = str(v)
            else:
                target.setncattr(k, v)
