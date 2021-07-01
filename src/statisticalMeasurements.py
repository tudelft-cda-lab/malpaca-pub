import logging
import os
import pickle

import numpy as np
import pandas as pd
from fastdist import fastdist
from sklearn.preprocessing import normalize

from models import PackageInfo, StatisticalAnalysisProperties

smallPacket = range(63, 400 + 1)
commonPorts = [25, 53, 80, 119, 123, 143, 161, 443, 5353]
microSecondsToSeconds = 1 / 1e6


def getStatisticalNormalizedDistanceMeasurement(values, config):
    if os.path.exists(config.statisticalDistanceCacheName) and os.path.exists(config.statisticalPropertiesCacheName):
        logging.debug("Using cache for statisticalDistance")
        with open(config.statisticalDistanceCacheName, 'rb') as file:
            ndm = pickle.load(file)
        with open(config.statisticalPropertiesCacheName, 'rb') as file:
            normalizedProperties = pickle.load(file)
    else:
        normalizedProperties = getPropertiesFromValues(values, config)
        ndm = normalizedStatisticalDistance(normalizedProperties)

        if config.saveDistanceCache:
            with open(config.statisticalDistanceCacheName, 'wb') as file:
                pickle.dump(ndm, file)
            with open(config.statisticalPropertiesCacheName, 'wb') as file:
                pickle.dump(normalizedProperties, file)

    return normalizedProperties, ndm


def normalizedStatisticalDistance(normalizedProperties):
    distm = fastdist.matrix_pairwise_distance(normalizedProperties, fastdist.euclidean, "euclidean", return_matrix=True)

    return distm / distm.max()


def getPropertiesFromValues(values: list[list[PackageInfo]], config):
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
        PPS = np.sum(pGaps) * microSecondsToSeconds

        properties.append(StatisticalAnalysisProperties(
            NSP=np.sum([byte in smallPacket for byte in pBytes], dtype=np.int16),
            AIT=np.average(pGaps) * microSecondsToSeconds,
            TBT=TBT,
            APL=np.average(pBytes),
            PV=np.std(pBytes),
            DPL=len(set(pBytes)),
            MX=MX,
            MP=pBytes.count(MX),
            PPS=PPS,
            BPS=TBT / PPS,
            USP=len(set(sourcePorts)),
            UDP=len(set(destinationPorts)),
            CP=sum([1 if port in commonPorts else 0 for port in sourcePorts]) + sum([1 if port in commonPorts else 0 for port in destinationPorts])
        ))

    distances = np.zeros((len(properties), len(StatisticalAnalysisProperties.__slots__)))

    for i, pProperty in enumerate(properties):
        distances[i] = pProperty.__array__()

    filename = 'statisticalProperties' + config.addition + '.txt'

    if config.generateDist:
        with open(config.outputDirDist + filename, 'w') as outfile:
            for i, pProperty in enumerate(properties):
                outfile.write(pProperty.__str__() + "\n")

    preClip = pd.DataFrame(distances)
    clippedDistances = preClip.clip(lower=preClip.quantile(0.025), upper=preClip.quantile(0.975), axis=1)

    return normalize(clippedDistances.to_numpy(), axis=0, norm='max')
