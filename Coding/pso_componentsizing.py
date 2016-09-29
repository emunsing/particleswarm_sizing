import sys, time, copy, os, random, math, pickle, datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
sys.path.insert(0,'../Coding')
import RadioSimulator

errFile = '../Results/psoErrorLog.log'

try:
	os.remove(errFile)
except OSError:
	pass
sys.stderr = open(errFile, 'w')

class Particle:
    def __init__(self, minx, maxx, seed, initPosition=None, initCost = None):
        self.sim = RadioSimulator.RadioSimulator(radioFile = '../Data/PowerMEMS_Sample_Data_em_20160928.csv')
        self.minx = minx
        self.maxx = maxx
        np.random.seed(seed)  # Set the seed for random number generation; ensures that this is set for the multiprocessing instance
        self.rnd = random.Random(seed)
        dim = len(minx)

        # If an initial position is dictated, we will accept that as our position. 
        #  Otherwise, we will randomly generate one. 
        if initPosition is None:
            self.position = (maxx - minx) * np.random.rand(dim) + minx
            self.cost = evaluateCost(self)
        else:
            self.position = initPosition
            self.cost = initCost

        self.velocity_scaler = 1  # This dictates what portion of the total space we can cover in one step
        self.velocity = (maxx - minx) * (np.random.rand(dim)-0.5) * self.velocity_scaler

        self.best_part_cost = self.cost # best error that we know of in the swarm
        self.best_part_pos = copy.copy(self.position)
        
        # Placeholders for later updating
        self.best_swarm_cost = 0.0
        self.best_swarm_pos = copy.copy(self.position)

def evaluateCost(myParticle):
    # Run the simulator and compute the cost for the particle's current position
    a = myParticle.position
    initVariables = {'TEGparallel':a[0], 'TEGserial':a[1], 'batts':a[2], 'caps':a[3], 'SOC':a[4], 'V_b':a[5], 'V_c':a[6]}
    return myParticle.sim.computeCost(initVariables)
       
def stepForward(myParticle):
    ## Initialization
    w = 0.729    # inertia
    c1 = 1.49445 # cognitive (particle)
    c2 = 1.49445 # social (swarm)
    dim = len(myParticle.position)
    
    # compute new velocity of the particle, in each dimension
    r1 = np.random.rand(dim)    # randomizations in the range 0-1
    r2 = np.random.rand(dim)
    # New velocity = w * inertia + c1 * own best + c2 * swarm best
    myParticle.velocity = ( (w * myParticle.velocity) + 
                            (c1 * r1 * (myParticle.best_part_pos - myParticle.position)) +  
                            (c2 * r2 * (myParticle.best_swarm_pos - myParticle.position)) )  

    # Make sure that the particles stay within the (minx, maxx) bounds in each dimension
    myParticle.velocity = np.minimum( myParticle.velocity, maxx - myParticle.position)
    myParticle.velocity = np.maximum( myParticle.velocity, minx - myParticle.position)

    # compute new position using new velocity
    myParticle.position += myParticle.velocity
    
    # compute cost of new position
    myParticle.cost = evaluateCost(myParticle)

    # is new position a new best for the particle?
    if myParticle.cost < myParticle.best_part_cost:
        myParticle.best_part_cost = myParticle.cost
        myParticle.best_part_pos = copy.copy(myParticle.position)

    return (myParticle.cost, myParticle)
        
def Solve(max_epochs, minx, maxx, n=None, initValues=None, initCostList=None):
    # max_epochs: Number of simulation epochs, i.e. flight time steps
    # n : Number of particles. If initial values are used, make sure n<=initValues
    # minx, maxx: Assuming that the simulation is in a hypercube defined by the range (minx, maxx) in each dimension
    # initValues: A Numpy array, with columns of position variables and rows of samples
    # initCostList: costs corresponding to the costs of each position
    if n is None:  n = initValues.shape[0]

    ## Create Swarm
    if initValues is None:
        swarm = [Particle(minx, maxx, i) for i in range(n)]
    else: 
        swarm = [Particle(minx, maxx, i, initValues[i], initCostList[i]) for i in range(n)]        
            
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
    myPool = Pool(10)
    epoch = 0
    while epoch < max_epochs:

        # Map: Parallelize computation of the new position for each particle
        # - Return a tuple with (error, particle)

        mapResults = myPool.map_async(stepForward,swarm)
        (costs, swarm) = tuple(zip(*mapResults.get() ))
        
        # mapResults = [stepForward(p) for p in swarm]
        # (costs, swarm) = tuple(zip(*mapResults))

        # Reduce:
        #  - Compute which error is minimal
        #  - if that error is an improvement
        #     - identify the position at which it occured
        #     - Broadcast that position and error to all the particles in the swarm

        minCost = min(costs)
        # Check whether the error has improved
        if minCost < best_swarm_cost:
            msg= ("Swarm Cost Improved to %0.1f on epoch %s !"%(minCost,epoch))
            print(msg)
            sys.stderr.write(msg+'\n')
            sys.stderr.flush()
            best_swarm_cost = minCost
            minIdx = costs.index(minCost)
            best_swarm_pos = copy.copy(swarm[minIdx].position)
            # If the error has improved, update each particle with that information.
            for i in range(n):
                swarm[i].best_swarm_cost = best_swarm_cost
                swarm[i].best_swarm_pos = best_swarm_pos

        epoch += 1
        if epoch % 10 == 0:
            epochStr = "Epoch = %s with best error of %.1f at %s"%(str(epoch),best_swarm_cost, datetime.datetime.now())
            print(epochStr)
            sys.stderr.write(epochStr+'\n')
            pickle.dump(swarm, open( "../Results/swarm.pkl", "wb" ))
    
    return best_swarm_pos

##### MAIN EXECUTION FLOW ####
max_epochs = 1000

#  x =           [  p,   s ,  b ,   c, SOC, V_b, V_c]
minx = np.array( [  1,   1 ,  0 ,   0, 0.2,  0 ,  0 ])
maxx = np.array( [100,  100, 100, 100, 0.8, 2.6, 3.6])
# Load a set of points with which to initialize some of the particles.  These will have coordinates in a Numpy array.
gridSearchResults = pd.read_csv("../Results/gridSearchAllResults_2016-09-28_13_56.csv", index_col=0)
initPoints = gridSearchResults.sort_values(by='cost',ascending=True)

num_particles = 300  # Comment this out to use all points in initPoints

initVariables = initPoints[['TEGparallel','TEGserial','batts','caps','SOC','V_b','V_c']].values  # Get the values in the right order
initCosts = initPoints['cost'].values

sys.stderr.write("Started at %s \n"%datetime.datetime.now())

sys.stdout.flush()
starttime = time.time()
best_position = Solve(max_epochs, minx, maxx, initValues=initVariables, initCostList=initCosts, n= num_particles)

finishStr = "Done execution in %.2f seconds with best solution %s \n"%(time.time()-starttime,best_position)
print(finishStr)
sys.stderr.write(finishStr+'\n')
