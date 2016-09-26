import time
from multiprocessing import Pool

def sleepawhile(x):
	time.sleep(1)
	return x*x

myPool = Pool()

starttime = time.time()

i = range(5)
results = myPool.map_async(sleepawhile,i)
print results.get()
print("Finished in %.2f seconds"%(time.time() - starttime) )
