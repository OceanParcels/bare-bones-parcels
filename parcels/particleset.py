import numpy as np
import compiler
from ctypes import Structure, c_int, byref, POINTER, pointer, c_void_p, cast
from particle import JITParticle
from codegenerator import code_generate
from collections import Iterable
from mpi4py import MPI
import time


class ParticleSet(object):
    def __init__(self, lons, lats):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        
        particles_per_processor = len(lons) // size
        remainder = len(lons) % size

        if rank < remainder:
            self.size = particles_per_processor + 1
        else:
            self.size = particles_per_processor
        self.particles = np.empty(self.size, dtype=JITParticle)
        self.ptype = JITParticle.getPType()
        dtype = self.ptype.dtype
        self._particle_data = np.empty(self.size, dtype=dtype)

        def cptr(i):
            return self._particle_data[i]
        
        first_particle = rank * particles_per_processor + remainder
        if rank < remainder:
            first_particle -= remainder - rank
        
        last_particle = (rank + 1) * particles_per_processor + remainder
        if rank + 1 < remainder:
            last_particle -= remainder - rank + 1

        for i in range(first_particle, last_particle):
            self.particles[i - first_particle] = JITParticle(lons[i], lats[i], 0, cptr=cptr(i - first_particle))

    def add(self, particles):
        """Method to add particles to the ParticleSet"""
        if isinstance(particles, ParticleSet):
            particles = particles.particles
        if not isinstance(particles, Iterable):
            particles = [particles]
        self.particles = np.append(self.particles, particles)
        if True:#self.ptype.uses_jit:
            particles_data = [p._cptr for p in particles]
            self._particle_data = np.append(self._particle_data, particles_data)
            # Update C-pointer on particles
            for p, pdata in zip(self.particles, self._particle_data):
                p._cptr = pdata

    def remove(self, indices):
        """Method to remove particles from the ParticleSet, based on their `indices`"""
        if isinstance(indices, Iterable):
            particles = [self.particles[i] for i in indices]
        else:
            particles = self.particles[indices]
        self.particles = np.delete(self.particles, indices)
        if True:  # self.ptype.uses_jit:
            self._particle_data = np.delete(self._particle_data, indices)
            # Update C-pointer on particles
            for p, pdata in zip(self.particles, self._particle_data):
                p._cptr = pdata
        return particles


    def check_particles(self):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        prem = None
        toRem = []
        for i in range(len(self.particles)):
            p = self.particles[i]
            if (rank == 0 and p.xi > 5) or (rank == 1 and p.xi < 6):
                toRem.append(i)
        if len(toRem) > 0:
            prem = self.remove(toRem)
            prem = prem[0]
            prem.CGridIndexSetptr = 0
            comm.isend(prem, (rank+1)%2, 17)
            print 'p sent'
        req2 = comm.irecv(source=(rank+1)%2, tag=17)
        comm.Barrier()
        p2 = req2.test()
        if not p2[0]:
            req2.Cancel()
        else:
            p = p2[1]
            p.CGridIndexSetptr = cast(pointer(p.gridIndexSet.ctypes_struct), c_void_p)
            p.CGridIndexSet = p.CGridIndexSetptr.value
            self.add(p)
        self.size = len(self.particles)
                

pset = ParticleSet([1, 4, 8], [2, 3, 4])

src_file = 'c_code.c'
lib_file = 'c_code.so'
log_file = 'c_code.log'

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
if rank == 0:
    code_generate(pset, src_file, lib_file, log_file)
    compiler = compiler.GNUCompiler()
    compiler.compile(src_file, lib_file, log_file)
    print("Compiled %s ==> %s\n" % (src_file, lib_file))
comm.Barrier()

import numpy.ctypeslib as npct
lib = npct.load_library(lib_file, '.')
function = lib.mainFunc


for iter in range(17):
    #if rank == 0:
    print('ITER %d' % iter)
    particle_data = pset._particle_data.ctypes.data_as(c_void_p)
    function(c_int(pset.size), particle_data)
    pset.check_particles()
    time.sleep(.5)

time.sleep(200)
