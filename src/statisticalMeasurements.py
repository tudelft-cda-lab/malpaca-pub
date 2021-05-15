import numpy as np
from fastdist import fastdist

from models import PackageInfo
from sequentialMeasurements import normalizedSourcePortDistance, normalizedDestinationPortDistance

smallPacket = range(63, 400 + 1)
uSToMs = 1 / 1e3
uSToS = 1 / 1e6

def getStatisticalNormalizedDistanceMeasurement(values):
    ndm = normalizedStatisticalDistance(values)
    ndmSourcePort = normalizedSourcePortDistance(values)
    ndmDestinationPort = normalizedDestinationPortDistance(values)

    return 0.8 * ndm + 0.1 * ndmSourcePort + 0.1 * ndmDestinationPort


def normalizedStatisticalDistance(values: list[list[PackageInfo]]):
    distances = np.zeros((len(values), 11))

    for i, value in enumerate(values):
        bytesValues = [x.bytes for x in value]
        gapsValues = [x.gap for x in value]
        destinationPortValues = [x.destinationPort for x in value]
        sourcePortValues = [x.sourcePort for x in value]
        distances[i][0] = np.sum([byte in smallPacket for byte in bytesValues])  # NSP
        distances[i][1] = np.average(gapsValues) * uSToMs  # AIT
        distances[i][2] = np.sum(bytesValues)  # TBT
        distances[i][3] = np.average(bytesValues)  # APL
        distances[i][4] = np.std(bytesValues)  # PV
        distances[i][5] = len(set(bytesValues))  # DPL
        distances[i][6] = bytesValues.count(distances[i][2])  # MP
        distances[i][7] = np.sum(gapsValues) * uSToS  # PPS
        distances[i][8] = distances[i][2] / distances[i][7]  # BPS
        distances[i][9] = len(set(destinationPortValues))  # uniqueDestinationPortCount
        distances[i][10] = len(set(sourcePortValues))  # uniqueSourcePortCount

    distm = fastdist.matrix_pairwise_distance(distances, fastdist.euclidean, "euclidean", return_matrix=True)

    return distm / distm.max()