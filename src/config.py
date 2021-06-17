import logging
import random

seed = 45
random.seed(seed)

expname = 'data'
thresh = 35

addition = '-' + expname + '-' + str(thresh)
outputDir = 'output/'  # All files in this folder will be deleted
outputDirRaw = outputDir + 'raw/'
outputDirDist = outputDir + 'dist/'
outputDirStats = 'stats/'
outputDirFigs = outputDir + 'figs' + addition

sequentialDistanceCacheName = f'data/sequentialDistance-{thresh}-{seed}.pkl'
statisticalDistanceCacheName = f'data/statisticalDistance-{thresh}-{seed}.pkl'
propertiesCacheName = f'data/normalizedProperties-{thresh}-{seed}.pkl'
pklCache = f'data/meta-{thresh}-{seed}.pkl'
mappingCache = f'data/mapping-{thresh}-{seed}.pkl'
totalLabelsCache = f'data/totalLabels-{thresh}-{seed}.pkl'

logLevel = logging.INFO
generateTSNEGraphs = False
generateAllGraphs = True
