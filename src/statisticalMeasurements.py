import logging
import os
import pickle

import numpy as np
from fastdist import fastdist
from sklearn.preprocessing import normalize

import config
from models import PackageInfo, StatisticalAnalysisProperties
from sequentialMeasurements import normalizedSourcePortDistance, normalizedDestinationPortDistance

smallPacket = range(63, 400 + 1)
commonPorts = [25, 53, 80, 119, 123, 143, 161, 443, 5353]
uSToS = 1 / 1e6


def getStatisticalNormalizedDistanceMeasurement(values, useCache=False):
    if os.path.exists('data/statisticalDistance.pkl') and os.path.exists('data/normalizedProperties.pkl') and useCache:
        logging.debug("Using cache for statisticalDistance")
        with open('data/statisticalDistance.pkl', 'rb') as file:
            ndm = pickle.load(file)
        with open('data/normalizedProperties.pkl', 'rb') as file:
            normalizedProperties = pickle.load(file)
    else:
        normalizedProperties = getPropertiesFromValues(values)
        ndm = normalizedStatisticalDistance(normalizedProperties)
        # ndmSourcePort = normalizedSourcePortDistance(values)
        # ndmDestinationPort = normalizedDestinationPortDistance(values)
        #
        # ndm = np.average([ndm, ndmSourcePort, ndmDestinationPort], weights=[len(StatisticalAnalysisProperties.__slots__), 1, 1], axis=0)

        with open('data/statisticalDistance.pkl', 'wb') as file:
            pickle.dump(ndm, file)
        with open('data/normalizedProperties.pkl', 'wb') as file:
            pickle.dump(normalizedProperties, file)

    return normalizedProperties, ndm


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
            NSP=np.sum([byte in smallPacket for byte in pBytes], dtype=np.int16),
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
            UDP=len(set(destinationPorts)),
            CP=sum([1 if port in commonPorts else 0 for port in sourcePorts]) + sum([1 if port in commonPorts else 0 for port in destinationPorts])
        ))

    distances = np.zeros((len(properties), len(StatisticalAnalysisProperties.__slots__)))

    for i, pProperty in enumerate(properties):
        distances[i] = pProperty.__array__()

    filename = 'statisticalProperties' + config.addition + '.txt'

    with open(config.outputDirDist + filename, 'w') as outfile:
        for i, pProperty in enumerate(properties):
            outfile.write(pProperty.__str__() + "\n")

    return normalize(distances, axis=0, norm='max')
