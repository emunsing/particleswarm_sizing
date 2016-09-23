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

# TEGserialSeries   = [25]
# TEGparallelSeries = [5]
# battSeries        = np.arange(5,26,10)
# capSeries         = np.arange(5,26,10)
# SOCseries         = [0.4] #[0.2,0.4,0.5]
# V_bSeries         = np.arange(0,0.9,0.4)
# V_cSeries         = [2]

mySim = RadioSimulator.RadioSimulator(radioFile = '../Data/PowerMEMS_Sample_Data_em_20160707.csv')
# mySim = RadioSimulator.RadioSimulator(radioFile = '../Data/50step_downsampled_toy.csv')
resultfile = '../Results/gridSearchAllResults_'+datetime.datetime.now().strftime("%Y-%m-%d_%H_%M")+'.csv'


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
scenarioDictList = [tupleToDict(myTuple) for myTuple in scenarioVarList ]

# scenarioList = [tupleToDict(myTuple) for myTuple in scenarioList]  # Pack them each into dictionaries
# scenarioSimTupleList = [(initDict, mySim) for initDict in scenarioList]

print("Number of scenarios: %s"%len(scenarioVarList))
sys.stdout.flush()

### Prepping for parallel execution
results = pd.DataFrame(scenarioDictList)  # This currently does not have the cost data; that will be added later
results['cost'] = float('NaN')
success = pd.DataFrame()
myPool = Pool()


#batches = 3
#batchSize = int(len(scenarioVarList)/batches)
batchSize = 300
startAt = 62100

## Block the problem into batches, so that we can save progress between batches
for j in np.arange(startAt,len(scenarioVarList),batchSize):
	scenarioSimTupleList = [(initDict, mySim )  for initDict in scenarioDictList[j:j+batchSize] ]
	# scenarioResults = pd.DataFrame(scenarioDictList)

	initStr = "Started batch for scenarios %s to %s at %s"%(j, j+batchSize-1, datetime.datetime.now())
	print(initStr)
	sys.stderr.write(initStr+'\n')
	sys.stderr.flush()
	starttime = time.time()

	results.loc[j:j+batchSize-1, 'cost'] = myPool.map(processTupleSim, scenarioSimTupleList)
	results.to_csv(resultfile)

	finishStr = "Finished batch in %.2f seconds with %s successes"%(time.time()-starttime, 
		sum(results.loc[j:j+batchSize-1, 'cost']<float('inf')) ) 
	print(finishStr)
	sys.stderr.write(finishStr+'\n')
	sys.stderr.flush()

print("Done")

sys.stderr.close()
sys.stderr = sys.__stderr__
