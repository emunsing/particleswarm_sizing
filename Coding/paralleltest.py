import time
from multiprocessing import Pool

def sleepawhile(x):
	time.sleep(0.01)
	return x*x

myPool = Pool(1)

starttime = time.time()

i = range(100)
results = myPool.map_async(sleepawhile,i)
results.get()
print("Finished in %.2f seconds"%(time.time() - starttime) )
