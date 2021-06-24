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
    saveDistanceCache = False
    generateTSNEGraphs = False
    generateAllGraphs = False

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
        return f'data/sequentialDistance-{self.thresh}-{self.seed}.pkl'

    @property
    def statisticalDistanceCacheName(self):
        return f'data/statisticalDistance-{self.thresh}-{self.seed}.pkl'

    @property
    def propertiesCacheName(self):
        return f'data/normalizedProperties-{self.thresh}-{self.seed}.pkl'

    @property
    def pklCache(self):
        return f'data/meta-{self.thresh}-{self.seed}.pkl'

    @property
    def mappingCache(self):
        return f'data/mapping-{self.thresh}-{self.seed}.pkl'

    @property
    def totalLabelsCache(self):
        return f'data/totalLabels-{self.thresh}-{self.seed}.pkl'
