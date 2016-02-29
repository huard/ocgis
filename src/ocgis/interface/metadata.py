from warnings import warn


# tdk: remove this module, etc.

class NcMetadata(object):
    """
    :param rootgrp: An open NetCDF4 dataset object.
    :type rootgrp: :class:`netCDF4.Dataset`
    """

    def get_lines(self):
        lines = ['dimensions:']
        template = '    {0} = {1} ;{2}'
        for key, value in self['dimensions'].iteritems():
            if value['isunlimited']:
                one = 'ISUNLIMITED'
                two = ' // {0} currently'.format(value['len'])
            else:
                one = value['len']
                two = ''
            lines.append(template.format(key, one, two))

        lines.append('')
        lines.append('variables:')
        var_template = '    {0} {1}({2}) ;'
        attr_template = '      {0}:{1} = "{2}" ;'
        for key, value in self['variables'].iteritems():
            dims = [str(d) for d in value['dimensions']]
            dims = ', '.join(dims)
            lines.append(var_template.format(value['dtype'], key, dims))
            for key2, value2 in value['attrs'].iteritems():
                lines.append(attr_template.format(key, key2, value2))

        lines.append('')
        lines.append('// global attributes:')
        template = '    :{0} = {1} ;'
        for key, value in self['dataset'].iteritems():
            try:
                lines.append(template.format(key, value))
            except UnicodeEncodeError:
                # for a unicode string, if "\u" is in the string and an inappropriate unicode character is used, then
                # template formatting will break.
                msg = 'Unable to encode attribute "{0}". Skipping printing of attribute value.'.format(key)
                warn(msg)

        return lines

