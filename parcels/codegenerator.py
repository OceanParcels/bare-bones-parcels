import cgen as c
import numpy as np
from ctypes import Structure, c_int, byref, POINTER, pointer, c_void_p

def code_generate(pset, src_file, lib_file, log_file):
    
    ccode = '#include "c_include.h"\n'
    ccode += '#include <mpi.h>\n\n'

    vdecl=[]
    for v in pset.ptype.variables:
        if v.name is 'CGridIndexSet':
            vdecl.append(c.Pointer(c.POD(np.void, v.name))) 
        else:
            vdecl.append(c.POD(v.dtype, v.name))
    ccode += str(c.Typedef(c.Struct("", vdecl, declname='JITParticle')))
    ccode += "\n\n"

    ccode += 'int mainFunc(int n, JITParticle* pset)\n'
    ccode += '{\n'
    ccode += 'int rank, size;\n'
    ccode += 'MPI_Comm_size(MPI_COMM_WORLD, &size);\n'
    ccode += 'MPI_Comm_rank(MPI_COMM_WORLD, &rank);\n'
    ccode += 'int i;\n'
    ccode += 'for(i=0;i<n;++i){\n'
    ccode += '  JITParticle* p = &pset[i];\n'
    ccode += '  p->xi += 1;\n'
    ccode += '  if (p->xi > 15)\n'
    ccode += '    p->xi = 0;\n'
    ccode += '  printf("rank:%d P[%d %d %d]: %d\\n", rank, i, p->rank0, p->id, p->xi);\n'
    ccode += '}\n'
    ccode += 'return 0;\n}'
    
    with open(src_file, 'w') as f:
        f.write(ccode)
    
