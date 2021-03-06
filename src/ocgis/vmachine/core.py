from contextlib import contextmanager

import numpy as np
from ocgis.base import AbstractOcgisObject
from ocgis.constants import MPIOps
from ocgis.exc import SubcommNotFoundError, SubcommAlreadyCreatedError
from ocgis.vmachine.mpi import MPI_COMM, get_nonempty_ranks, MPI_SIZE, MPI_RANK, COMM_NULL, MPI_TYPE_MAPPING, \
    DummyMPIComm


class OcgVM(AbstractOcgisObject):
    """
    Manages communicators for parallel execution. Provides access to a dummy communicator when running in serial.
    
    :param comm: The default communicator.
    :type comm: MPI Communicator or :class:`~ocgis.vmachine.mpi.DummyMPIComm`
    """

    def __init__(self, comm=None):
        self._subcomms = {}
        self._current_comm_name = None

        if comm is None:
            comm = MPI_COMM
        self._comm = comm
        self._original_comm = comm

        if isinstance(comm, DummyMPIComm):
            is_dummy = True
        else:
            is_dummy = False
        self._is_dummy = is_dummy

    def __del__(self):
        try:
            self.finalize()
        except:
            pass

    @property
    def comm(self):
        if self.current_comm_name is None:
            ret = self._comm
        else:
            ret = self.get_subcomm(self.current_comm_name)
        return ret

    @property
    def comm_world(self):
        return MPI_COMM

    @property
    def current_comm_name(self):
        return self._current_comm_name

    @property
    def is_null(self):
        return self.comm == COMM_NULL

    @property
    def rank(self):
        if self.is_null:
            ret = None
        else:
            ret = self.comm.Get_rank()
        return ret

    @property
    def rank_global(self):
        return MPI_RANK

    @property
    def ranks(self):
        return range(self.size)

    @property
    def size(self):
        if self.is_null:
            ret = None
        else:
            ret = self.comm.Get_size()
        return ret

    @property
    def size_global(self):
        return MPI_SIZE

    def barrier(self):
        self.comm.Barrier()

    def Barrier(self):
        self.barrier()

    def bcast(self, *args, **kwargs):
        return self.comm.bcast(*args, **kwargs)

    def create_subcomm(self, name, ranks, is_current=False, clobber=False):
        if self._is_dummy:
            self._subcomms[name] = self._comm
        else:
            if len(ranks) == 0:
                self._subcomms[name] = COMM_NULL
            else:
                the_pool = self.comm.Get_group()
                sub_group = the_pool.Incl(ranks)
                try:
                    ret = self.comm.Create(sub_group)
                    if name in self._subcomms:
                        if clobber:
                            vm.free_subcomm(name=name)
                        else:
                            raise SubcommAlreadyCreatedError(name)
                    self._subcomms[name] = ret
                finally:
                    sub_group.Free()
        if is_current:
            self._current_comm_name = name
        return name

    def create_subcomm_by_emptyable(self, name, emptyable, **kwargs):
        live_ranks = self.get_live_ranks_from_object(emptyable)
        name = self.create_subcomm(name, live_ranks, **kwargs)
        return name, live_ranks

    def free_subcomm(self, subcomm=None, name=None):
        if not self._is_dummy:
            if subcomm is None:
                if name not in self._subcomms:
                    raise SubcommNotFoundError(name)
                subcomm = self._subcomms.pop(name)
            if subcomm != COMM_NULL:
                subcomm.Free()

    def finalize(self):
        for v in self._subcomms.values():
            self.free_subcomm(subcomm=v)
        self._subcomms = {}
        self._current_comm_name = None
        self._comm = self._original_comm

    def gather(self, *args, **kwargs):
        return self.comm.gather(*args, **kwargs)

    def get_live_ranks_from_object(self, target):
        return get_nonempty_ranks(target, self)

    @staticmethod
    def get_mpi_type(other):
        other = np.dtype(other)
        if other == np.int32:
            other = np.int32
        elif other == np.int64:
            other = np.int64

        if MPI_TYPE_MAPPING is None:
            ret = None
        else:
            ret = MPI_TYPE_MAPPING[other]
        return ret

    def scatter(self, *args, **kwargs):
        return self.comm.scatter(*args, **kwargs)

    @staticmethod
    def barrier_print(*args, **kwargs):
        from ocgis.vmachine.mpi import barrier_print
        barrier_print(*args, **kwargs)

    def get_subcomm(self, name):
        try:
            return self._subcomms[name]
        except KeyError:
            raise SubcommNotFoundError(name)

    @staticmethod
    def rank_print(*args, **kwargs):
        from ocgis.vmachine.mpi import rank_print
        rank_print(*args, **kwargs)

    def reduce(self, target, op, root=0):
        if self._is_dummy:
            ret = target
        else:
            ret = vm.comm.reduce(target, MPIOps.get_op(op), root=root)
        return ret

    def set_comm(self, name=None):
        self._current_comm_name = name

    def scoped(self, *args, **kwargs):
        return vm_scope(self, *args, **kwargs)

    def scoped_by_emptyable(self, name, emptyable):
        live_ranks = self.get_live_ranks_from_object(emptyable)
        return self.scoped(name, live_ranks)

    def scoped_by_name(self, name):
        return vm_scoped_by_name(self, name)


@contextmanager
def vm_scope(vm_obj, name, ranks):
    original = vm_obj.current_comm_name
    vm_obj.create_subcomm(name, ranks, is_current=True)
    try:
        yield vm_obj
    finally:
        vm_obj.free_subcomm(name=name)
        vm_obj.set_comm(name=original)


@contextmanager
def vm_scoped_by_name(vm_obj, name):
    original_name = vm_obj.current_comm_name
    vm_obj.set_comm(name=name)
    try:
        yield vm_obj
    finally:
        vm_obj.set_comm(name=original_name)


vm = OcgVM()
