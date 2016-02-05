import sys

from logbook import Logger, StreamHandler, FileHandler

from ocgis.new_interface.mpi import MPI_RANK

sh = StreamHandler(sys.stdout, bubble=True)
sh.format_string += ' (rank={})'.format(MPI_RANK)
sh.push_application()

fh = FileHandler('/home/benkoziol/htmp/ocgis.log', bubble=True)
fh.format_string += ' (rank={})'.format(MPI_RANK)
fh.push_application()

log = Logger('ocgis')  # , level='DEBUG')

# log.handlers.append(fh)
# # log.handlers.append(sh)
