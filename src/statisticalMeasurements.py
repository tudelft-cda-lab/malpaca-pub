import numpy as np
from fastdist import fastdist

from models import PackageInfo


expname = 'data'
thresh = 20
addition = '-' + expname + '-' + str(thresh)
outputDir = 'output/'  # All files in this folder will be deleted
outputDirRaw = outputDir + 'raw/'
outputDirDist = outputDir + 'dist/'
outputDirFigs = outputDir + 'figs' + addition


def getStatisticalNormalizedDistanceMeasurement(values):
    ndmBytesAverage = normalizedStatisticalByteDistanceAverage(values)
    ndmGaps = normalizedStatisticalGapsDistance(values)

    return normalizedDistanceMeasurement(ndmGaps, ndmBytesAverage)


def normalizedStatisticalByteDistanceAverage(values: list[list[PackageInfo]]):
    bytesDistances = np.zeros((len(values), 3))

    for i, value in enumerate(values):
        bytesDistances[i][0] = np.mean([x.bytes for x in value])
        bytesDistances[i][1] = np.min([x.bytes for x in value])
        bytesDistances[i][2] = np.max([x.bytes for x in value])

    distm = fastdist.matrix_pairwise_distance(bytesDistances, fastdist.euclidean, "euclidean", return_matrix=True)

    return distm / distm.max()


def normalizedStatisticalGapsDistance(values: list[list[PackageInfo]]):
    gapsDistances = np.zeros((len(values), 3))

    for i, value in enumerate(values):
        gapsDistances[i][0] = np.mean([x.gap for x in value])
        gapsDistances[i][1] = np.min([x.gap for x in value])
        gapsDistances[i][2] = np.max([x.gap for x in value])

    distm = fastdist.matrix_pairwise_distance(gapsDistances, fastdist.euclidean, "euclidean", return_matrix=True)

    return distm / distm.max()


def normalizedDistanceMeasurement(*args):
    return sum(args) / len(args)