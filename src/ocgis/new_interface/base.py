from abc import ABCMeta
from contextlib import contextmanager
from copy import copy, deepcopy

from ocgis.new_interface.ocgis_logging import log


class AbstractInterfaceObject(object):
    __metaclass__ = ABCMeta

    @property
    def log(self):
        return log

    def copy(self):
        """Return a shallow copy of self."""
        return copy(self)

    def deepcopy(self):
        """Return a deep copy of self."""
        return deepcopy(self)


def get_keyword_arguments_from_template_keys(kwargs, keys, ignore_self=True, pop=False):
    ret = {}
    for key in keys:
        if ignore_self and key == 'self':
            continue
        try:
            if pop:
                ret[key] = kwargs.pop(key)
            else:
                ret[key] = kwargs[key]
        except KeyError:
            # Pass on key errors to allow classes to overload default keyword argument values.
            pass
    return ret


@contextmanager
def orphaned(target):
    original_parent = target.parent
    target.parent = None
    try:
        yield target
    finally:
        target.parent = original_parent


@contextmanager
def renamed_dimensions(dimensions, name_mapping):
    original_names = [d.name for d in dimensions]
    try:
        items = name_mapping.items()
        for d in dimensions:
            for k, v in items:
                if d.name in v:
                    d._name = k
                    break
        yield dimensions
    finally:
        for d, o in zip(dimensions, original_names):
            d._name = o


@contextmanager
def renamed_dimensions_on_variables(vc, name_mapping):
    variables = vc.values()
    original_names = {v.name: [d.name for d in v.dimensions] for v in variables}
    try:
        items = name_mapping.items()
        for v in variables:
            for d in v.dimensions:
                for desired_name, possible_names in items:
                    if d.name in possible_names:
                        d._name = desired_name
                        break
        yield vc
    finally:
        for v in variables:
            for d, o in zip(v.dimensions, original_names[v.name]):
                d._name = o
