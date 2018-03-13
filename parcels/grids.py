import numpy as np
from ctypes import Structure, c_int, c_void_p, c_char_p, POINTER, cast, pointer
from ctypes import byref

class GVariable(object):
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, cls):
        return instance._cptr.__getitem__(self.name)

    def __set__(self, instance, value):
        instance._cptr.__setitem__(self.name, value)

class GridIndex(object):
    name = GVariable('name')
    xi = GVariable('xi')
    yi = GVariable('yi')
    def __init__(self, name, *args, **kwargs):
        self._cptr = kwargs.pop('cptr', None)
        self.nameP = name
        ptr = cast(self.nameP, (c_void_p))
        self.name = ptr.value
        self.xi = 0
        self.yi = 0

    @classmethod
    def dtype(cls):
        type_list = [('name',c_void_p),('xi',np.int32), ('yi',np.int32)]
        return np.dtype(type_list)

from ctypes import addressof

class GridIndexSet(object):
    def __init__(self, id):
        size = 2
        self.size = c_int(size)

        self.gridindicess = np.empty(size, GridIndex)
        self._gridindices_data = np.empty(size, GridIndex.dtype())

        def cptr(i):
            return self._gridindices_data[i]

        self.gridindicess[0] = GridIndex('grid0', cptr=cptr(0))
        self.gridindicess[1] = GridIndex('grid1', cptr=cptr(1))

    @property
    def ctypes_struct(self):
        class CGridIndexSet(Structure):
            _fields_ = [('size', c_int),
                        ('grids', POINTER(c_void_p))]
        cstruct = CGridIndexSet(self.size,
                        self._gridindices_data.ctypes.data_as(POINTER(c_void_p)))
        return cstruct
