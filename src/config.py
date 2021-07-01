import dataclasses
import logging
import random


@dataclasses.dataclass()
class Config:
    expname = 'data'
    _thresh: int
    _seed: int

    outputDir = 'output/'  # All files in this folder will be deleted
    outputDirRaw = outputDir + 'raw/'
    outputDirDist = outputDir + 'dist/'
    outputDirStats = 'stats/'

    logLevel = logging.INFO

    generateRaw = False
    generateDist = False
    saveDistanceCache = True
    generateTSNEGraphs = False
    generateAllGraphs = True

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, v):
        logging.info(f"Setting random seed to {v}")
        random.seed(v)
        self._seed = v

    @property
    def thresh(self):
        return self._thresh

    @thresh.setter
    def thresh(self, v):
        logging.info(f"Setting thresh to {v}")
        self._thresh = v

    @property
    def addition(self):
        return f'-{self.expname}-{self.thresh}-{self.seed}'

    @property
    def outputDirFigs(self):
        return self.outputDir + 'figs' + self.addition

    @property
    def sequentialDistanceCacheName(self):
        return f'cache/sequentialDistance-{self.thresh}-{self.seed}.pkl'

    @property
    def statisticalDistanceCacheName(self):
        return f'cache/statisticalDistance-{self.thresh}-{self.seed}.pkl'

    @property
    def statisticalPropertiesCacheName(self):
        return f'cache/statisticalNormalizedProperties-{self.thresh}-{self.seed}.pkl'

    @property
    def pklCache(self):
        return f'cache/meta-{self.thresh}-{self.seed}.pkl'

    @property
    def mappingCache(self):
        return f'cache/mapping-{self.thresh}-{self.seed}.pkl'

    @property
    def totalLabelsCache(self):
        return f'cache/totalLabels-{self.thresh}-{self.seed}.pkl'
