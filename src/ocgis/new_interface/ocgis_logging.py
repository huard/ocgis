import functools
import sys
import time
from logging import DEBUG

from logbook import Logger, StreamHandler, FileHandler

from ocgis.new_interface.mpi import MPI_RANK


def formatter(record, handler):
    msg = '[{} {}]: {} (rank={}, time={}): {}'.format(record.channel, record.time, record.level_name, MPI_RANK,
                                                      time.time(), record.message)
    if record.level_name == 'ERROR':
        msg += '\n' + record.formatted_exception
    return msg


sh = StreamHandler(sys.stdout, bubble=True)
# sh.format_string += ' (rank={})'.format(MPI_RANK)
sh.formatter = formatter
# sh.push_application()

fh = FileHandler('/home/benkoziol/htmp/ocgis-rank-{}.log'.format(MPI_RANK), bubble=True, mode='w')
# fh.format_string += ' (rank={})'.format(MPI_RANK)
fh.formatter = formatter
# fh.push_application()

log = Logger('ocgis', level=DEBUG)

log.handlers.append(fh)
log.handlers.append(sh)


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
