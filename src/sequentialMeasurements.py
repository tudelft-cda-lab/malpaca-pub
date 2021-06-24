import logging
import os
import pickle

import numpy as np
from numba.core import types
from numba.typed import Dict, List

from fastDistances import dtwDistance, ngramDistance
from models import PackageInfo


def getSequentialNormalizedDistanceMeasurement(values, config):
    if os.path.exists(config.sequentialDistanceCacheName):
        logging.debug("Using cache for sequentialDistance")
        with open(config.sequentialDistanceCacheName, 'rb') as file:
            ndm = pickle.load(file)
    else:
        ndmBytes = normalizedByteDistance(values, config)
        ndmGaps = normalizedGapsDistance(values, config)
        ndmSourcePort = normalizedSourcePortDistance(values, config)
        ndmDestinationPort = normalizedDestinationPortDistance(values, config)
        ndm = normalizedDistanceMeasurement(ndmBytes, ndmGaps, ndmSourcePort, ndmDestinationPort)

        if config.saveDistanceCache:
            with open(config.sequentialDistanceCacheName, 'wb') as file:
                pickle.dump(ndm, file)

    return ndm


def normalizedDistanceMeasurement(*args):
    return sum(args) / len(args)


def normalizedByteDistance(values: list[list[PackageInfo]], config):
    filename = 'bytesDist' + config.addition + '.txt'

    bytesDistances = np.zeros((len(values), config.thresh))

    for i, value in enumerate(values):
        bytesDistances[i] = [x.bytes for x in value]

    distm = dtwDistance(bytesDistances)

    if config.generateDist:
        with open(config.outputDirDist + filename, 'w') as outfile:
            for a in range(len(distm)):
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm / distm.max()


def normalizedGapsDistance(values: list[list[PackageInfo]], config):
    filename = 'gapsDist' + config.addition + '.txt'

    gapsDistances = np.zeros((len(values), config.thresh))

    for i, value in enumerate(values):
        gapsDistances[i] = [x.gap for x in value]

    distm = dtwDistance(gapsDistances)

    if config.generateDist:
        with open(config.outputDirDist + filename, 'w') as outfile:
            for a in range(len(distm)):
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm / distm.max()


def normalizedSourcePortDistance(values: list[list[PackageInfo]], config):
    filename = 'sportDist' + config.addition + '.txt'

    ngrams = generateNGrams(PackageInfo.sourcePort.__name__, values)

    return generateCosineDistanceFromNGramsAndSave(filename, ngrams, config)


def normalizedDestinationPortDistance(values: list[list[PackageInfo]], config):
    filename = 'dportDist' + config.addition + '.txt'

    ngrams = generateNGrams(PackageInfo.destinationPort.__name__, values)

    return generateCosineDistanceFromNGramsAndSave(filename, ngrams, config)


def generateCosineDistanceFromNGramsAndSave(filename, ngrams, config):
    distm = ngramDistance(ngrams)

    if config.generateDist:
        with open(config.outputDirDist + filename, 'w') as outfile:
            for a in range(len(distm)):
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm


def generateNGrams(attribute, values: list[list[PackageInfo]]):
    ngrams = List()
    for value in values:
        profile = Dict.empty(types.int64, types.int64)

        dat = [getattr(x, attribute) for x in value]

        li = zip(dat, dat[1:], dat[2:])

        for b in li:
            key = hash(b)
            if key not in profile:
                profile[key] = 0

            profile[key] += 1

        ngrams.append(profile)
    return ngrams
