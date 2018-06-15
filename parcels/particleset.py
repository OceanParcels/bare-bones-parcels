from __future__ import division # fixes division issue in Python 2
import numpy as np
import compiler
from ctypes import Structure, c_int, byref, POINTER, pointer, c_void_p, cast
from particle import JITParticle
from codegenerator import code_generate
from collections import Iterable
from mpi4py import MPI
import time
import random
import math


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


    def check_particles(self, area):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        prem = None
        toRem = []
        for i in range(len(self.particles)):
            p = self.particles[i]
            if p.xi <= area[0] or p.xi > area[1] or p.yi <= area[2] or p.yi > area[3]:
                toRem.append(i)
        if len(toRem) > 0:
            prem = self.remove(toRem)
            # TODO: split prem according to the processors to which they have to be sent
            for p in prem:
                p.CGridIndexSetptr = 0
            comm.isend(prem, (rank+1)%2)
            print 'p sent'
        req2 = comm.irecv(source=(rank+1)%2)
        comm.Barrier()
        p2 = req2.test()
        if not p2[0]:
            req2.Cancel()
        else:
            p = p2[1]
            for padd in p:
                padd.CGridIndexSetptr = cast(pointer(padd.gridIndexSet.ctypes_struct), c_void_p)
                padd.CGridIndexSet = padd.CGridIndexSetptr.value
                self.add(padd)
        self.size = len(self.particles)


def determine_partition(pset, subset_size):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    partition = []
    area = [float('-inf'), float('inf'), float('-inf'), float('inf')]
    
    if rank != 0:
        # Sample
        subset= []
        if (len(pset.particles) > subset_size):
            subset = random.sample(range(len(pset.particles)), subset_size)
        else:
            subset = range(len(pset.particles))
        
        # Send sample to processor 0
        psend = []
        for i in subset:
            psend.append([pset.particles[i].xi, pset.particles[i].yi])
        comm.isend(psend, 0)
    if rank == 0:
        sample = []
        
        # Determine own sample
        subset= []
        if (len(pset.particles) > subset_size):
            subset = random.sample(range(len(pset.particles)), subset_size)
        else:
            subset = range(len(pset.particles))
            
        for i in subset:
            sample.append([pset.particles[i].xi, pset.particles[i].yi])
        
        # Gather samples
        messages = []
        reqs = []
        for i in range(1, size):
            reqs.append(comm.irecv(source=i))
        
        for i in range(1, size):
            messages.append(reqs[i - 1].wait())
        
        for i in range(size - 1):
            sample += messages[i]
        
        # Start recursive paritioning
        partition = recursive_partition(range(size), sample, 'x')
            # Say, we have $p$ processors, so we want to end up with $p$ partitions.
            # Our first cut will result in floor(p/2)/p particles to be on one side, and ceil(p/2)/p particles on the other. Recurse with p1 = floor(p/2) and p2 = ceil(p/1) until px = 1.
            # The cut is defined as \leq, so if dir = x and cut = 4, to the left are all particles with x <= 4.
        
        # Determine areas of responsibility
        areas = determine_area(partition, area)
        area = areas[0]
        
        # Communicate cuts
        for i in range(1, size):
            comm.isend(partition, i)
            comm.isend(areas[i], i)
    if rank != 0:
        # Recieve cut-information, relies on MPI messages to be order-preserving
        partition = comm.recv(source=0)
        area = comm.recv(source=0)

    # Send particles to other processors
    to_send = [[] for x in range(size)]
    indices = []
    procs = []

    for i in range(len(pset.particles)):
        branch = partition
        while branch["dir"] != 'l':
            if branch["dir"] == 'x':
                if pset.particles[i].xi <= branch["cut"]:
                    branch = branch["left"]
                else:
                    branch = branch["right"]
            elif branch["dir"] == 'y':
                if pset.particles[i].yi <= branch["cut"]:
                    branch = branch["left"]
                else:
                    branch = branch["right"]
            else:
                raise ValueError('Unknown cut direction encountered')
        if branch["proc"][0] != rank:
            # Note, this approach relies on both Python list and the remove method to be order preserving
            indices.append(i)
            procs.append(branch["proc"][0])
    
    prem = pset.remove(indices)
    
    for i in range(len(prem)):
        prem[i].CGridIndexSetptr = 0
        to_send[procs[i]].append(prem[i])
    
    for i in range(size):
        if i != rank:
            comm.isend(to_send[i], i)

    # Receive particles from other processors
    reqs = []
    for i in range(size):
        if i != rank:
            reqs.append(comm.irecv(source=i))
    
    for i in range(size - 1):
        res = reqs[i].wait()
        for p in res:
            p.CGridIndexSetptr = cast(pointer(p.gridIndexSet.ctypes_struct), c_void_p)
            p.CGridIndexSet = p.CGridIndexSetptr.value
            pset.add(p)
    pset.size = len(pset.particles)
    
    print("Area of processor " + str(rank) + ": " + str(area))
    return area
    
def recursive_partition(proc, sub, dir):
    # If the sample is smaller than the number of processors, this has strange results (it will try to partition a single particle)
    # Base case
    if len(proc) == 1:
        return {"dir": 'l', "cut": -1, "left": [], "right": [], "proc": proc}

    new_dir = 'x'
    cut = -1
    proc_l = proc[:int(math.ceil(len(proc) / 2))]
    proc_r = proc[int(math.ceil(len(proc) / 2)):]
    no_part_l = int(math.ceil(len(proc_l) / len(proc) * len(sub)))
    
    if dir == 'x':
        new_dir = 'y'
        # Cut in the x-direction
        sub.sort(key = lambda x: x[0])
        cut = sub[no_part_l - 1][0]
    elif dir == 'y':
        # Cut in the y-direction
        sub.sort(key = lambda x: x[1])
        cut = sub[no_part_l - 1][1]
    else:
        raise ValueError('A cut in a unknown dimension was requested during partitioning.')

    sub_l = sub[:no_part_l]
    sub_r = sub[no_part_l:]

    left_partition = recursive_partition(proc_l, sub_l, new_dir)
    right_partition = recursive_partition(proc_r, sub_r, new_dir)

    return {"dir": dir, "cut": cut, "left": left_partition, "right": right_partition, "proc": proc}


def determine_area(partition, region):
    if len(partition["proc"]) == 1:
        return { partition["proc"][0]: region }
    if partition["dir"] == 'x':
        left_region = list(region)
        left_region[1] = partition["cut"]
        left_res = determine_area(partition["left"], left_region)
        right_region = list(region)
        right_region[0] = partition["cut"]
        right_res = determine_area(partition["right"], right_region)
        left_res.update(right_res)
        return left_res
    elif partition["dir"] == 'y':
        left_region = list(region)
        left_region[3] = partition["cut"]
        left_res = determine_area(partition["left"], left_region)
        right_region = list(region)
        right_region[2] = partition["cut"]
        right_res = determine_area(partition["right"], right_region)
        left_res.update(right_res)
        return left_res
    else:
        raise ValueError('A cut in a unknown dimension was encountered.')


lons = [1, 3, 4, 8, 10]
lats = [2, 3, 4, 5, 6]

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
area = [float('-inf'), float('inf'), float('-inf'), float('inf')]
#determine_partition(pset, subset_size)

for iter in range(17):
    #if rank == 0:
    if iter % 5 == 0:
        area = determine_partition(pset, subset_size)
    print('ITER %d' % iter)
    particle_data = pset._particle_data.ctypes.data_as(c_void_p)
    function(c_int(pset.size), particle_data)
    pset.check_particles(area)
    time.sleep(.5)

time.sleep(200)
