import numpy as np
from numba.core import types
from numba.typed import Dict, List

import config
from fastDistances import dtwDistance, ngramDistance
from models import PackageInfo


def getSequentialNormalizedDistanceMeasurement(values):
    ndmBytes = normalizedByteDistance(values)
    ndmGaps = normalizedGapsDistance(values)
    ndmSourcePort = normalizedSourcePortDistance(values)
    ndmDestinationPort = normalizedDestinationPortDistance(values)

    return [], normalizedDistanceMeasurement(ndmBytes, ndmGaps, ndmSourcePort, ndmDestinationPort)


def normalizedDistanceMeasurement(*args):
    return sum(args) / len(args)


def normalizedByteDistance(values: list[list[PackageInfo]]):
    filename = 'bytesDist' + config.addition + '.txt'

    bytesDistances = np.zeros((len(values), config.thresh))

    for i, value in enumerate(values):
        bytesDistances[i] = [x.bytes for x in value]

    distm = dtwDistance(bytesDistances)

    with open(config.outputDirDist + filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm / distm.max()


def normalizedGapsDistance(values: list[list[PackageInfo]]):
    filename = 'gapsDist' + config.addition + '.txt'

    gapsDistances = np.zeros((len(values), config.thresh))

    for i, value in enumerate(values):
        gapsDistances[i] = [x.gap for x in value]

    distm = dtwDistance(gapsDistances)

    with open(config.outputDirDist + filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm / distm.max()


def normalizedSourcePortDistance(values: list[list[PackageInfo]]):
    filename = 'sportDist' + config.addition + '.txt'

    ngrams = generateNGrams(PackageInfo.sourcePort.__name__, values)

    return generateCosineDistanceFromNGramsAndSave(filename, ngrams)


def normalizedDestinationPortDistance(values: list[list[PackageInfo]]):
    filename = 'dportDist' + config.addition + '.txt'

    ngrams = generateNGrams(PackageInfo.destinationPort.__name__, values)

    return generateCosineDistanceFromNGramsAndSave(filename, ngrams)


def generateCosineDistanceFromNGramsAndSave(filename, ngrams):
    distm = ngramDistance(ngrams)

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
