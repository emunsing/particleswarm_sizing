import RadioSimulator
import datetime
import sys, os, time, copy

import numpy as np
import pandas as pd

errFile = 'gridsearchErrorLog.txt'

try:
	os.remove(errFile)
except OSError:
	pass
sys.stderr = open(errFile, 'w')

## Define the grid
TEGserialSeries   = np.arange(10,36,5)
TEGparallelSeries = np.arange(1,22,5)
battSeries        = np.arange(1,22,5)
capSeries         = np.arange(1,22,5)
SOCseries         = np.arange(0.2,0.81,0.2)
V_bSeries         = np.arange(0, 1.9, 0.4)
V_cSeries         = np.arange(1.8, 3.5, 0.4)

#TEGserialSeries   = [25]
#TEGparallelSeries = [5]
#battSeries        = np.arange(5,26,10)
#capSeries         = np.arange(5,26,10)
#SOCseries         = [0.2,0.3,0.4,0.5,0.6,0.7]
#V_bSeries         = np.arange(0,0.9,0.4)
#V_cSeries         = [2]

mySim = RadioSimulator.RadioSimulator(radioFile = '../Data/PowerMEMS_Sample_Data_em_20160707.csv')
# mySim = RadioSimulator.RadioSimulator(radioFile = '../Data/50step_downsampled_toy.csv')
outfile = '../Results/gridSearchSuccesses_'+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")+'.csv'

#### Parallelized Grid Search ####
import itertools
from multiprocessing import Pool

## The following section preps a list of scenarios for a parallelized grid search. 
#    This could be replaced by nested loops if parallelization is not desired.

# Steps: 
#  - Create a list of lists with all the variables.
#  - Create all combinations of these variable values (combinatorial combinations; same as nested loops
#  - Pack each of these into the initVariables dictionary form
#  - map all of these to a multiprocessing pool
#  - take the output of the pool and pack it into a dataframe
#  - Assign the costs to the dataframe
#  - Save the dataframe
def tupleToDict(a):
    return {'TEGserial':a[0], 'TEGparallel':a[1], 'batts':a[2], 'caps':a[3], 'SOC':a[4], 'V_b':a[5], 'V_c':a[6]}

def processTupleSim(myTuple):
    (initVariable, mySim) = myTuple
    return mySim.computeCost(initVariable)

varList = [TEGserialSeries,TEGparallelSeries, battSeries, capSeries, SOCseries, V_bSeries, V_cSeries]

scenarioVarList = list(itertools.product(*varList) )   # Create a list of all scenario combinations

# scenarioList = [tupleToDict(myTuple) for myTuple in scenarioList]  # Pack them each into dictionaries
# scenarioSimTupleList = [(initDict, mySim) for initDict in scenarioList]

print("Number of scenarios: %s"%len(scenarioVarList))
sys.stdout.flush()

### Prepping for parallel execution
results = pd.DataFrame()  # This currently does not have the cost data; that will be added later
success = pd.DataFrame()
myPool = Pool()

#batches = 3
#batchSize = int(len(scenarioVarList)/batches)
startAt = 18900
batchSize = 300


## Block the problem into batches, so that we can save progress between batches
for j in np.arange(startAt,len(scenarioVarList),batchSize):
	scenarioDictList = [tupleToDict(myTuple) for myTuple in scenarioVarList[j:j+batchSize-1] ]  # Pack them each into dictionaries
	scenarioSimTupleList = [(initDict, mySim )  for initDict in scenarioDictList]

	scenarioResults = pd.DataFrame(scenarioDictList)

	print("Started batch for scenarios %s to %s at %s"%(j, j+batchSize-1, datetime.datetime.now()))
	sys.stderr.write("Started batch for scenarios %s to %s at %s \n"%(j, j+batchSize-1, datetime.datetime.now()))
	sys.stdout.flush()

	starttime = time.time()
	scenarioResults['cost'] = myPool.map(processTupleSim, scenarioSimTupleList, chunksize=2)
	print("Finished batch in %.2f seconds"%(time.time()-starttime))
	sys.stderr.write("Finished batch in %.2f seconds \n"%(time.time()-starttime))

	scenarioSuccess =  scenarioResults[ scenarioResults['cost']<float('inf')]
	success = success.append(scenarioSuccess, ignore_index=True)
	success.to_csv(outfile)

print("Done")

sys.stderr.close()
sys.stderr = sys.__stderr__
