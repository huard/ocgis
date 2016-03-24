import functools
import sys
from logging import ERROR

from logbook import Logger, StreamHandler, FileHandler

from ocgis.new_interface.mpi import MPI_RANK

sh = StreamHandler(sys.stdout, bubble=True)
sh.format_string += ' (rank={})'.format(MPI_RANK)
sh.push_application()

fh = FileHandler('/home/benkoziol/htmp/ocgis.log', bubble=True)
fh.format_string += ' (rank={})'.format(MPI_RANK)
fh.push_application()

log = Logger('ocgis', level=ERROR)

# log.handlers.append(fh)
# # log.handlers.append(sh)


class log_entry_exit(object):
    def __init__(self, f):
        self.f = f

    def __call__(self, *args, **kwargs):
        log.debug("entering {0})".format(self.f.__name__))
        try:
            return self.f(*args, **kwargs)
        finally:
            log.debug("exited {0})".format(self.f.__name__))

    def __get__(self, obj, _):
        """Support instance methods."""

        return functools.partial(self.__call__, obj)
