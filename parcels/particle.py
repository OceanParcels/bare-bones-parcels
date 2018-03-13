import numpy as np
from grids import GridIndexSet
from ctypes import pointer, c_void_p, cast, byref
from mpi4py import MPI

lastID = 0

class Variable(object):
    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

    def __get__(self, instance, cls):
        return  instance._cptr.__getitem__(self.name)


    def __set__(self, instance, value):
        instance._cptr.__setitem__(self.name, value)

    def __repr__(self):
            return "PVar<%s|%s>" % (self.name, self.dtype)

    def is64bit(self):
        if self.name is 'CGridIndexSet': return True
        return True if self.dtype == np.float64 or self.dtype == np.int64 else False

class ParticleType(object):

    def __init__(self, pclass):
        self.name = pclass.__name__
        self.variables = [v for v in pclass.__dict__.values() if isinstance(v, Variable)]
        self.variables = [v for v in self.variables if v.is64bit()] + \
                         [v for v in self.variables if not v.is64bit()]
 
    def __repr__(self):
        return "PType<%s>::%s" % (self.name, self.variables)

    @property
    def dtype(self):
        """Numpy.dtype object that defines the C struct"""
        type_list = [(v.name, v.dtype) for v in self.variables]
        for v in self.variables:
            if v.dtype not in self.supported_dtypes:
                raise RuntimeError(str(v.dtype) + " variables are not implemented in JIT mode")
        if self.size % 8 > 0:
            # Add padding to be 64-bit aligned
            type_list += [('pad', np.float32)]
        return np.dtype(type_list)

    @property
    def size(self):
        """Size of the underlying particle struct in bytes"""
        return sum([8 if v.is64bit() else 4 for v in self.variables])

    @property
    def supported_dtypes(self):
        return [np.int32, np.int64, np.float32, np.float64, c_void_p]


class JITParticle(object):
    xi = Variable('xi', dtype=np.int32)
    yi = Variable('yi', dtype=np.int32)
    id = Variable('id', dtype=np.int32)
    rank0 = Variable('rank0', dtype=np.int32)
    time = Variable('time', dtype=np.float64)
    CGridIndexSet = Variable('CGridIndexSet', dtype=np.dtype(c_void_p))

    def __init__(self, xi, yi, time, *args, **kwargs):
        global lastID
        self._cptr = kwargs.pop('cptr', None)
        self.time = time
        self.xi = xi 
        self.yi = yi
        self.id = lastID
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        self.rank0 = rank
        lastID +=1
        self.gridIndexSet = GridIndexSet(self.id)
        self.CGridIndexSetptr = cast(pointer(self.gridIndexSet.ctypes_struct), c_void_p)
        self.CGridIndexSet = self.CGridIndexSetptr.value

    @classmethod
    def getPType(cls):
        return ParticleType(cls)
