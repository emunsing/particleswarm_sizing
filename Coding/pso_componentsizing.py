import RadioSimulator
import datetime
import sys, time, copy, os, random, math, pickle
from multiprocessing import Pool

import numpy as np
import pandas as pd

errFile = 'psoErrorLog.txt'

try:
	os.remove(errFile)
except OSError:
	pass
sys.stderr = open(errFile, 'w')

class Particle:
    def __init__(self, minx, maxx, seed, initPosition=None, initCost = None):
        self.sim = RadioSimulator.RadioSimulator(radioFile = '../Data/PowerMEMS_Sample_Data_em_20160707.csv')
        self.minx = minx
        self.maxx = maxx
        self.rnd = random.Random(seed)
        dim = len(minx)
        self.position = np.zeros(dim)
        self.velocity = np.zeros(dim)

        # If we dictate an initial position, we will accept that as our position. 
        #  Otherwise, we will randomly generate one. 
        if initPosition is not None:
            self.position = initPosition
            self.velocity = [((maxx[i] - minx[i]) * self.rnd.random() + minx[i])  for i in range(dim)]
            self.cost = initCost
        else:    
            for i in range(dim):
                self.position[i] = ((maxx[i] - minx[i]) * self.rnd.random() + minx[i])
                self.velocity[i] = ((maxx[i] - minx[i]) * self.rnd.random() + minx[i])
            self.cost = evaluateCost(self)

        self.best_part_cost = self.cost # best error
        self.best_part_pos = copy.copy(self.position)
        
        # Placeholders for later updating
        self.best_swarm_cost = 0.0
        self.best_swarm_pos = copy.copy(self.position)

    def set_position(self, newPos, cost=None):
        self.position = newPos
        if cost is not None:
            self.cost = cost
        else:
            self.cost = evaluateCost(self)
            
        self.best_part_pos = copy.copy(self.position) 
        self.best_part_cost = self.cost # best error

def evaluateCost(myParticle):
    a = myParticle.position
    initVariables = {'TEGserial':a[0], 'TEGparallel':a[1], 'batts':a[2], 'caps':a[3], 'SOC':a[4], 'V_b':a[5], 'V_c':a[6]}
    return myParticle.sim.computeCost(initVariables)
    err = 0.0
    for i in range(len(myParticle.position)):
        xi = myParticle.position[i]
        err += (xi * xi) - (10 * math.cos(2 * math.pi * xi)) + 10
    return err        
        
def stepForward(myParticle):
    ## Initialization
    w = 0.729    # inertia
    c1 = 1.49445 # cognitive (particle)
    c2 = 1.49445 # social (swarm)
    dim = len(myParticle.position)
    rnd = random.Random(0)
    
    # compute new velocity of curr particle, in each dimension
    for k in range(dim): 
        r1 = rnd.random()    # randomizations
        r2 = rnd.random()

        # New velocity = w * inertia + c1 * own best + c2 * swarm best
        myParticle.velocity[k] = ( (w * myParticle.velocity[k]) + 
                                 (c1 * r1 * (myParticle.best_part_pos[k] - myParticle.position[k])) +  
                                 (c2 * r2 * (myParticle.best_swarm_pos[k] - myParticle.position[k])) )  

        # Make sure that the particles stay within the (minx, maxx) bounds in each dimension
        if (maxx[k] - myParticle.position[k]) < myParticle.velocity[k]:
              myParticle.velocity[k] = maxx[k] - myParticle.position[k]
        elif (minx[k] - myParticle.position[k]) > myParticle.velocity[k]:
              myParticle.velocity[k] = minx[k] - myParticle.position[k]

    # compute new position using new velocity
    myParticle.position += myParticle.velocity

    # compute error of new position
    myParticle.cost = evaluateCost(myParticle)

    # is new position a new best for the particle?
    if myParticle.cost < myParticle.best_part_cost:
        myParticle.best_part_cost = myParticle.cost
        myParticle.best_part_pos = copy.copy(myParticle.position)

    return (myParticle.cost, myParticle)
        
def Solve(max_epochs, n, minx, maxx, initValues=None, initCostList=None):
    # max_epochs: Number of simulation epochs, i.e. flight time steps
    # n : Number of particles. If initial values are used, make sure n<=initValues
    # dim: dimensionality of Rastriggin's function
    # minx, maxx: Assuming that the simulation is in a hypercube defined by the range (minx, maxx) in each dimension
    # initValues: A Numpy array, with columns of position variables and each 
    
    ## Create Swarm
    if initValues is not None:
        swarm = [Particle(minx, maxx, i, initValues[i], initCostList[i]) for i in range(n)]
    else: 
        swarm = [Particle(minx, maxx, i) for i in range(n)]
            
    ## Identify the best cost in the initial batch
    best_swarm_cost = float('inf') # High initial value    
    for i in range(n): # See what the actual best position is so far
        if swarm[i].cost < best_swarm_cost:
            best_swarm_cost = swarm[i].cost
            best_swarm_pos = copy.copy(swarm[i].position) 

    # Now that we've identified the best position, broadcast that to all the particles
    for i in range(n):
        swarm[i].best_swarm_cost = best_swarm_cost
        swarm[i].best_swarm_pos = best_swarm_pos
        
    ## Done with initialization of the swarm- now move on to the actual work!
        
    rnd = random.Random(0)
    myPool = Pool()
    dim = len(minx)
    epoch = 0
    while epoch < max_epochs:

        # Map: Parallelize computation of the new position for each particle
        # - Return a tuple with (error, particle)
        mapResults = myPool.map_async(stepForward,swarm)
        
        # Reduce:
        #  - Compute which error is minimal
        #  - if that error is an improvement
        #     - identify the position at which it occured
        #     - Broadcast that position and error to all the particles in the swarm

        (costs, swarm) = tuple(zip(*mapResults.get() ))
        minCost = min(costs)
        # Check whether the error has improved
        if minCost < best_swarm_cost:
            best_swarm_cost = minCost
            minIdx = costs.index(minCost)
            best_swarm_pos = copy.copy(swarm[minIdx].position)
        # If the error has improved, update each particle with that information.
        for i in range(n):
            swarm[i].best_swarm_cost = best_swarm_cost
            swarm[i].best_swarm_pos = best_swarm_pos

        epoch += 1
        if epoch % 1 == 0:
        	epochStr = "Epoch = %s with best error of %.1f at %s"%(str(epoch),best_swarm_cost, datetime.datetime.now())
        	print(epochStr)
        	sys.stderr.write(epochStr+'\n')
        	pickle.dump(swarm, open( "swarm.pkl", "wb" ))
    
    return best_swarm_pos

##### MAIN EXECUTION FLOW ####
max_epochs = 600

#  x = [  p,   s ,  b ,   c, SOC, V_b, V_c]
minx = [  1,   1 ,  0 ,   0, 0.2,  0 ,  0 ]
maxx = [100,  100, 100, 100, 0.8, 2.6, 3.6]

# Load a set of points with which to initialize some of the particles.  These will have coordinates in a Numpy array.
# gridsearchResults = pd.read_csv("../Results/gridSearchSuccesses_2016-07-08_10_07.csv", index_col=0)
# initPoints = gridsearchSuccesses.sort_values(by='cost')[0:initial_successes]

gridSearchResults = pd.read_csv("../Results/gridSearchAllResults_2016-07-10.csv", index_col=0)
initPoints = gridSearchResults[gridSearchResults['cost']<float('inf')]
num_particles = initPoints.shape[0]

#num_particles = 10

initVariables = initPoints[['TEGparallel','TEGserial','batts','caps','SOC','V_b','V_c']].values  # Get the values in the right order
initCosts = initPoints['cost'].values

print("Started at %s"%datetime.datetime.now())

sys.stdout.flush()
starttime = time.time()
best_position = Solve(max_epochs, num_particles, minx, maxx, initValues=initVariables, initCostList=initCosts)

finishStr = "Done execution in %.2f seconds with best solution %s "%(time.time()-starttime,best_position)
print(finishStr)
sys.stderr.write(finishStr+'\n')
