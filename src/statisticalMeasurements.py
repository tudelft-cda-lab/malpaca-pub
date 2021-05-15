import numpy as np
from fastdist import fastdist
from sklearn.preprocessing import normalize

from models import PackageInfo, StatisticalAnalysisProperties
from sequentialMeasurements import normalizedSourcePortDistance, normalizedDestinationPortDistance

smallPacket = range(63, 400 + 1)
uSToS = 1 / 1e6


def getStatisticalNormalizedDistanceMeasurement(values):
    normalizedProperties = getPropertiesFromValues(values)
    ndm = normalizedStatisticalDistance(normalizedProperties)
    ndmSourcePort = normalizedSourcePortDistance(values)
    ndmDestinationPort = normalizedDestinationPortDistance(values)

    # return ndm
    return normalizedProperties, np.average([ndm, ndmSourcePort, ndmDestinationPort], weights=[len(StatisticalAnalysisProperties.__slots__), 1, 1], axis=0)


def normalizedStatisticalDistance(normalizedProperties):
    distm = fastdist.matrix_pairwise_distance(normalizedProperties, fastdist.sqeuclidean, "sqeuclidean", return_matrix=True)

    return distm / distm.max()


def getPropertiesFromValues(values: list[list[PackageInfo]]):
    properties = []

    for i, packets in enumerate(values):
        pBytes = []
        pGaps = []
        destinationPorts = []
        sourcePorts = []

        for packet in packets:
            pBytes.append(packet.bytes)
            pGaps.append(packet.gap)
            destinationPorts.append(packet.destinationPort)
            sourcePorts.append(packet.sourcePort)

        TBT = np.sum(pBytes)
        MX = np.max(pBytes)
        PPS = min(max(0.01, np.sum(pGaps) * uSToS), 1000)

        properties.append(StatisticalAnalysisProperties(
            NSP=np.sum([byte in smallPacket for byte in pBytes]),
            AIT=np.average(pGaps) * uSToS,
            TBT=TBT,
            APL=np.average(pBytes),
            PV=np.std(pBytes),
            DPL=len(set(pBytes)),
            MX=MX,
            MP=pBytes.count(MX),
            PPS=PPS,
            BPS=min(max(0.1, TBT / PPS), 10000),
            USP=len(set(sourcePorts)),
            UDP=len(set(destinationPorts))
        ))

    ## Save before outputting?

    distances = np.zeros((len(properties), len(StatisticalAnalysisProperties.__slots__)))

    for i, pProperty in enumerate(properties):
        distances[i] = pProperty.__array__()

    return normalize(distances, axis=0, norm='max')