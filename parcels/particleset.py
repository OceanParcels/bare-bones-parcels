import numpy as np
import compiler
from ctypes import Structure, c_int, byref, POINTER, pointer, c_void_p, cast
from particle import JITParticle
from codegenerator import code_generate
from collections import Iterable
from mpi4py import MPI
import time
import random


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


def determine_partition(pset, subset_size):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    partition = []
    
    if rank != 0:
        # Sample
        subset = random.sample(range(len(pset.particles), subset_size))
        # Send sample to processor 0
        psend = []
        for i in subset:
            psend.append([i, pset.particles[i].xi, pset.particles[i].yi])
        comm.isend(psend, 0)
    if rank == 0:
        sample = []
        
        # Determine own sample
        subset = random.sample(range(len(pset.particles), subset_size))
        for i in subset:
            sample.append([i, pset.particles[i].xi, pset.particles[i].yi])
        
        # Gather samples
        messages = [[] for x in size]
        for i in range(1, size):
            messages[i] = comm.irecv(source=i)
        
        comm.Barrier()
        
        for i in range(1, size):
            sample += messages[i]
        
        # Assign all particles a unique id
        for i in range(len(sample)):
            sample[i][0] = i
        
        # Start recursive paritioning
        partition = recursive_partition(size, sample, sample, 'x')
            # Say, we have $p$ processors, so we want to end up with $p$ partitions.
            # Our first cut will result in floor(p/2)/p particles to be on one side, and ceil(p/2)/p particles on the other. Recurse with p1 = floor(p/2) and p2 = ceil(p/1) until px = 1.
            # The cut is defined as \leq, so if dir = x and cut = 4, to the left are all particles with x <= 4.
        
        print(partition)
        
        # Communicate cuts
        for i in range(1, size):
            comm.isend(partition, i)
    if rank != 0:
        # Recieve cut-information
        partition = comm.Recv(source=0)
        
    # Send particles to other processors
    # Receive particles from other processors
    
    
    def recursive_partition(no_proc, dict, sub, dir):
        # Base case
        if no_proc == 1:
            return {"dir": 'l', "cut": -1, "left": [], "right": [], "sub": sub}
        
        new_dir = 'x'
        cut = -1
        no_proc_l = ceil(no_proc / 2)
        no_proc_r = floor(no_proc / 2)
        no_part_l = ceil(no_proc_l / no_proc * len(sub))
        
        if dir == 'x':
            new_dir = 'y'
            # Cut in the x-direction
            sub.sort(key = lambda x: x[1])
            cut = sub[no_part_l - 1][1]
        elif dir == 'y':
            # Cut in the y-direction
            sub.sort(key = lambda x: x[2])
            cut = sub[no_part_l - 1][2]
        else:
            raise ValueError('A cut in a unknown dimension was requested during partitioning.')
        
        sub_l = sub[:no_part_l]
        sub_r = sub[no_part_l:]
        
        left_partition = recursive_partition(no_proc_l, dict, sub_l, new_dir)
        right_partition = recursive_partition(no_proc_r, dict, sub_r, new_dir)
        
        return {"dir": dir, "cut": cut, "left": left_partition, "right": right_partition, "sub": sub}
    

lons = [1, 4, 8]
lats = [2, 3, 4]

pset = ParticleSet(lons, lats)

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


# Initial particle distribution
subset_size = 1 # placeholder value
determine_partition(pset, subset_size)

for iter in range(17):
    #if rank == 0:
    if iter % 5 == 0:
        determine_partition(subset_size)
    print('ITER %d' % iter)
    particle_data = pset._particle_data.ctypes.data_as(c_void_p)
    function(c_int(pset.size), particle_data)
    pset.check_particles()
    time.sleep(.5)

time.sleep(200)