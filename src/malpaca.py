#!/usr/bin/python3
import csv
import glob
import os
import pickle
# import shutil
import sys
import warnings
import random
from collections import deque, defaultdict
from dataclasses import fields
import logging

import hdbscan
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from tqdm import tqdm

from config import Config
from helpers import timeFunction
from models import PackageInfo, ConnectionKey, StatisticalAnalysisProperties
from processPCAP import readPCAP
from sequentialMeasurements import getSequentialNormalizedDistanceMeasurement
from statisticalMeasurements import getStatisticalNormalizedDistanceMeasurement

numba_logger = logging.getLogger('numba')
matplotlib_logger = logging.getLogger('matplotlib')
numba_logger.setLevel(logging.WARNING)
matplotlib_logger.setLevel(logging.WARNING)

config = Config(_thresh=24, _seed=0)
results = defaultdict(dict)

logging.basicConfig(level=config.logLevel, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')

plt.rcParams.update({'figure.max_open_warning': 0})
warnings.filterwarnings("ignore", message="Attempting to set identical left == right")


def connlevel_sequence(metadata: dict[ConnectionKey, list[PackageInfo]], mapping):
    inv_mapping: dict[int, ConnectionKey] = {v: k for k, v in mapping.items()}

    values = list(metadata.values())

    generateOutputFolders()

    if config.generateRaw:
        storeRawData(values)

    sequentialProperties, normalizeDistanceMeasurementStatistical = timeFunction(
        getStatisticalNormalizedDistanceMeasurement.__name__,
        lambda: getStatisticalNormalizedDistanceMeasurement(values, config)
    )

    normalizeDistanceMeasurementSequential = timeFunction(
        getSequentialNormalizedDistanceMeasurement.__name__,
        lambda: getSequentialNormalizedDistanceMeasurement(values, config)
    )

    finalClustersStatistical, heatmapClusterStatistical = processMeasurements(normalizeDistanceMeasurementStatistical, mapping, inv_mapping, 'Statistical')
    finalClustersSequential, heatmapClusterSequential = processMeasurements(normalizeDistanceMeasurementSequential, mapping, inv_mapping, 'Sequential')

    compareFinalClusters(finalClustersSequential, finalClustersStatistical)

    if config.generateAllGraphs:
        # clusterAmount = len(finalClusters)
        # generateDag(dagClusters, clusterAmount)
        timeFunction(generateGraphs.__name__, lambda: generateGraphs('Statistical', heatmapClusterStatistical, values, sequentialProperties))
        timeFunction(generateGraphs.__name__, lambda: generateGraphs('Sequential', heatmapClusterSequential, values, []))


def processMeasurements(normalizeDistanceMeasurement, mapping, inv_mapping, name):
    if os.path.exists(f"{config.outputDirStats}{name}{config.addition}.txt"):
        os.remove(f"{config.outputDirStats}{name}{config.addition}.txt")

    clu, projection = timeFunction(f'[{name}] {generateClusters.__name__}', lambda: generateClusters(normalizeDistanceMeasurement, name))

    if config.generateTSNEGraphs:
        timeFunction(generateClusterGraph.__name__, lambda: generateClusterGraph(clu, projection, name))

    finalClusters, dagClusters, heatmapCluster = saveClustersToCsv(clu, mapping, inv_mapping, name)

    finalClusterSummary(finalClusters, inv_mapping, name)

    return finalClusters, heatmapCluster


def generateOutputFolders():
    if not os.path.exists(config.outputDir):
        # shutil.rmtree(config.outputDir)
        os.mkdir(config.outputDir)
        os.mkdir(config.outputDirRaw)
        os.mkdir(config.outputDirDist)

    if not os.path.exists(config.outputDirFigs) and config.generateAllGraphs:
        os.mkdir(config.outputDirFigs)
        os.mkdir(config.outputDirFigs + '/bytes')
        os.mkdir(config.outputDirFigs + '/gap')
        os.mkdir(config.outputDirFigs + '/sourcePort')
        os.mkdir(config.outputDirFigs + '/destinationPort')
        os.mkdir(config.outputDirFigs + '/statistics')


def storeRawData(values):
    for field in fields(PackageInfo):
        feat = field.name
        with open(config.outputDirRaw + feat + '-features' + config.addition, 'w') as f:
            for val in values:
                vi = [str(x.__getattribute__(feat)) for x in val]
                f.write(','.join(vi))
                f.write("\n")


def finalClusterSummary(finalClusters, inv_mapping: dict[int, ConnectionKey], extraName):
    averageClusterPurity = []
    averageClusterMaliciousPurity = []

    for n, cluster in finalClusters.items():
        if n == -1:
            continue
        connectionKeys = []

        for connectionNumber in cluster:
            connectionKeys.append(inv_mapping[connectionNumber])

        summary = labelSummary(connectionKeys)
        percentage = summary['percentage']

        if percentage > 0:
            clusterPurity = abs(percentage - 50) / 50
            averageClusterPurity.append(clusterPurity)

            if percentage > 60:
                averageClusterMaliciousPurity.append(calculateMaliciousPurity(connectionKeys, summary))

            logging.debug(f"[{extraName}] Cluster {n} is {round(percentage, 2)}% malicious, contains following labels: {','.join(summary['labels'])}, connections: {len(cluster)}, clusterPurity: {clusterPurity}")
        else:
            averageClusterPurity.append(1)
            logging.debug(f"[{extraName}] Cluster {n} does not contain any malicious packages, connections: {len(cluster)}")

    appendStatsToOutputFile(extraName, "Cluster purity", round(np.average(averageClusterPurity), 3))
    appendStatsToOutputFile(extraName, "Cluster malicious purity", round(np.average(averageClusterMaliciousPurity), 3))


def calculateMaliciousPurity(connectionKeys, summary):
    sumOfMaliciousItems = 0
    labelCount = {}
    for key in summary['labels']:
        maliciousConnections = sum(1 if clusterKey.connectionLabel == key else 0 for clusterKey in connectionKeys)
        sumOfMaliciousItems += maliciousConnections
        labelCount[key] = maliciousConnections
    labelWithMaxCount = max(labelCount, key=labelCount.get)
    return labelCount[labelWithMaxCount] / sumOfMaliciousItems


def labelSummary(connectionKeys: list[ConnectionKey]):
    summary = {'labels': set(), 'total': len(connectionKeys), 'malicious': 0, 'benign': 0}

    for key in connectionKeys:
        if key.connectionLabel != '-':
            summary['malicious'] += 1
            summary['labels'].add(key.connectionLabel)
        else:
            summary['benign'] += 1

    summary.update({'percentage': summary['malicious'] / summary['total'] * 100})

    return summary


def generateClusterGraph(clusters, projection, extraName):
    labels = clusters.labels_

    pal = sns.color_palette("colorblind", 10)
    pal.extend(sns.color_palette('Set1', 10))
    pal.extend(sns.color_palette('Set2', 10))
    pal.extend(sns.color_palette('Set3', 10))
    pal.extend(sns.color_palette('Paired', 9))

    col = [pal[x] for x in labels]

    plt.figure(figsize=(10, 10))

    for i, txt in enumerate(labels):
        alpha = 0.8
        if txt == -1:
            alpha = 0.05

        plt.scatter(projection.T[0][i], projection.T[1][i], color=col[i], alpha=alpha, label=txt)

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(sorted(zip(labels, handles), key=lambda tuples: int(tuples[0])))
    plt.legend(by_label.values(), by_label.keys(), title='Clusters', bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='xx-small')

    plt.draw()
    plt.savefig(f'{config.outputDir}clustering-result-{extraName}{config.addition}')
    plt.clf()


def saveClustersToCsv(clu, mapping, inv_mapping: dict[int, ConnectionKey], extraName):
    csv_file = f'clusters-{extraName}{config.addition}.csv'
    labels = list(range(len(mapping)))

    final_clusters = {}
    dagClusters = {}
    heatmapClusters = {}
    final_probs = {}

    for lab in set(clu.labels_):
        occ = [i for i, x in enumerate(clu.labels_) if x == lab]
        final_probs[lab] = [x for i, x in zip(clu.labels_, clu.probabilities_) if i == lab]
        final_clusters[lab] = [labels[x] for x in occ]

    with open(config.outputDir + csv_file, 'w') as outfile:
        outfile.write("clusnum,connnum,probability,class,filename,srcip,dstip\n")
        for n, cluster in final_clusters.items():
            heatmapClusters[n] = []
            for idx, connectionKey in enumerate([inv_mapping[x] for x in cluster]):
                className = connectionKey.name()
                outfile.write(
                    f"{n},{mapping[connectionKey]},{final_probs[n][idx]},{className},{connectionKey.filename},{connectionKey.sourceIp},{connectionKey.destinationIp}\n")

                if connectionKey.filename not in dagClusters:
                    dagClusters[connectionKey.filename] = []

                dagClusters[connectionKey.filename].append((className, n))
                heatmapClusters[n].append((mapping[connectionKey], className))

    return final_clusters, dagClusters, heatmapClusters


def generateClusters(normalizeDistanceMeasurement, extraName):
    RS = 3072018
    projection = TSNE(random_state=RS).fit_transform(normalizeDistanceMeasurement)
    plt.figure(figsize=(10, 10))
    plt.scatter(*projection.T)
    plt.savefig(f'{config.outputDir}tsne-result-{extraName}{config.addition}')
    plt.clf()

    clusterSize = len(normalizeDistanceMeasurement) // 150
    model = hdbscan.HDBSCAN(min_cluster_size=clusterSize, min_samples=clusterSize, cluster_selection_method='leaf',
                            metric='precomputed')
    clu = model.fit(normalizeDistanceMeasurement)

    amountOfClusters = len(set(clu.labels_)) - 1

    silhouetteScores = silhouette_samples(normalizeDistanceMeasurement, clu.labels_, metric='precomputed')

    avgClusterSize = 0
    avgSilhouetteScore = 0

    for clusterNumber in list(set(clu.labels_)):
        if clusterNumber == -1:
            continue
        scoresForSpecificCluster = silhouetteScores[clu.labels_ == clusterNumber]
        silhouetteAverageForCluster = np.average(scoresForSpecificCluster)
        avgSilhouetteScore += silhouetteAverageForCluster
        avgClusterSize += np.count_nonzero(clu.labels_ == clusterNumber)

    appendStatsToOutputFile(extraName, "Clustering size", clusterSize)
    appendStatsToOutputFile(extraName, "Number of clusters", amountOfClusters)
    appendStatsToOutputFile(extraName, "Average cluster size", round((avgClusterSize/amountOfClusters)/len(normalizeDistanceMeasurement) * 100, 2))
    appendStatsToOutputFile(extraName, "Average Silhouette score", round(avgSilhouetteScore / amountOfClusters, 3))
    appendStatsToOutputFile(extraName, "Samples not in noise", round((len(normalizeDistanceMeasurement) - np.count_nonzero(clu.labels_ == -1)) / len(normalizeDistanceMeasurement), 3))

    return clu, projection


def generateScatterPlot(extraName, clusterNumber, clusters, properties: list[StatisticalAnalysisProperties], name, propertyName):
    labels = []
    clusterMapData = []

    for cluster in clusters:
        labels.append(cluster[1])
        valueIndex = cluster[0]
        clusterMapData.append(properties[valueIndex])

    clusterMapDf = pd.DataFrame(clusterMapData, index=labels, columns=StatisticalAnalysisProperties.__slots__)

    plt.figure(figsize=(10, 10))
    plt.suptitle(f"Exp: {config.expname} | Cluster: {clusterNumber} | Feature: {name}")
    sns.boxplot(data=clusterMapDf)
    plt.ylim(-0.1, 1.1)
    plt.draw()
    plt.savefig(f'{config.outputDirFigs}/{propertyName}/{extraName}-{clusterNumber}')
    plt.clf()


def generateGraphs(extraName, clusterInfo, values: list[list[PackageInfo]], properties: list[StatisticalAnalysisProperties]):
    sns.set(font_scale=0.9)
    matplotlib.rcParams.update({'font.size': 10})

    wantedFeatures = [
        ("Packet sizes", PackageInfo.bytes.__name__),
        # ("Interval", PackageInfo.gap.__name__),
        # ("Source Port", PackageInfo.sourcePort.__name__),
        # ("Dest. Port", PackageInfo.destinationPort.__name__),
        ("Statistics", "statistics")
    ]

    with tqdm(total=(len(wantedFeatures) * len(clusterInfo)), unit='graphs') as t:
        for name, propertyName in wantedFeatures:
            for clusterNumber, cluster in clusterInfo.items():
                t.set_description_str(f"Working on {name}, cluster #{clusterNumber}")
                if propertyName == 'statistics':
                    if len(properties) == 0:
                        t.update(1)
                        continue
                    generateScatterPlot(extraName, clusterNumber, cluster, properties, name, propertyName)
                else:
                    generateGraph(extraName, clusterNumber, cluster, values, properties, name, propertyName)
                t.update(1)
        t.set_description_str(f"Done generating graphs")


def generateGraph(extraName, clusterNumber, clusters, values: list[list[PackageInfo]], properties: list[StatisticalAnalysisProperties], name, propertyName):
    labels = []
    clusterMapData = []

    for cluster in clusters:
        labels.append(cluster[1])
        valueIndex = cluster[0]
        if propertyName == 'statistics':
            clusterMapData.append(properties[valueIndex])
        else:
            connection = values[valueIndex]
            clusterMapData.append([package.__getattribute__(propertyName) for package in connection])

    clusterMapDf = pd.DataFrame(clusterMapData, index=labels)
    clusterMap = sns.clustermap(clusterMapDf, xticklabels=False, col_cluster=False)

    labelsReordered = []
    heatmapData = []

    for it in clusterMap.dendrogram_row.reordered_ind:
        labelsReordered.append(labels[it])
        heatmapData.append(clusterMapData[it])

    if len(clusters) <= 50:
        fig = plt.figure(figsize=(15.0, 9.0))
    elif len(clusters) <= 100:
        fig = plt.figure(figsize=(15.0, 18.0))
    else:
        fig = plt.figure(figsize=(20.0, 27.0))

    plt.suptitle(f"Exp: {config.expname} | Cluster: {clusterNumber} | Feature: {name}")

    if propertyName == 'statistics':
        heatmapDf = pd.DataFrame(heatmapData, index=labelsReordered, columns=StatisticalAnalysisProperties.__slots__)
        heatmap = sns.heatmap(heatmapDf, annot=True if clusterNumber != -1 else False, vmin=0, vmax=1, fmt=".3f")
    else:
        heatmapDf = pd.DataFrame(heatmapData, index=labelsReordered)
        heatmap = sns.heatmap(heatmapDf, xticklabels=False)

    plt.setp(heatmap.get_yticklabels(), rotation=0)

    if len(clusters) <= 50:
        fig.subplots_adjust(top=0.93, bottom=0.06, left=0.25, right=1.05)
    elif len(clusters) <= 100:
        fig.subplots_adjust(top=0.95, bottom=0.04, left=0.225, right=1.025)
    else:
        fig.subplots_adjust(top=0.97, bottom=0.02, left=0.2, right=1)

    plt.draw()
    plt.savefig(f'{config.outputDirFigs}/{propertyName}/{extraName}-{clusterNumber}', transparent=True)
    plt.clf()


def generateDag(dagClusters, clusterAmount):
    logging.info('Producing DAG with relationships between pcaps')

    array = [x for x in range(-1, clusterAmount - 1)]
    treeprep = dict()
    for filename, val in dagClusters.items():
        arr = [0] * clusterAmount
        for fam, clus in val:
            ind = array.index(clus)
            arr[ind] = 1
        mas = ''.join([str(x) for x in arr[:-1]])
        famname = fam
        logging.info(filename + "\t" + fam + "\t" + ''.join([str(x) for x in arr[:-1]]))
        if mas not in treeprep.keys():
            treeprep[mas] = dict()
        if famname not in treeprep[mas].keys():
            treeprep[mas][famname] = set()
        treeprep[mas][famname].add(str(filename))

    with open(config.outputDir + 'mas-details' + config.addition + '.csv', 'w') as f2:
        for k, v in treeprep.items():
            for kv, vv in v.items():
                f2.write(str(k) + ';' + str(kv) + ';' + str(len(vv)) + '\n')

    graph = {}
    names = {}
    with open(config.outputDir + 'mas-details' + config.addition + '.csv', 'r') as f3:
        csv_reader = csv.reader(f3, delimiter=';')

        for line in csv_reader:
            graph[line[0]] = set()
            if line[0] not in names.keys():
                names[line[0]] = []
            names[line[0]].append(line[1] + "(" + line[2] + ")")

    zeros = ''.join(['0'] * (clusterAmount - 1))
    if zeros not in graph.keys():
        graph[zeros] = set()

    ulist = graph.keys()
    covered = set()
    next = deque()

    next.append(zeros)

    while len(next) > 0:
        l1 = next.popleft()
        covered.add(l1)
        for l2 in ulist:
            if l2 not in covered and difference(l1, l2) == 1:
                graph[l1].add(l2)

                if l2 not in next:
                    next.append(l2)

    val = set()
    for v in graph.values():
        val.update(v)

    notmain = [x for x in ulist if x not in val]
    notmain.remove(zeros)
    nums = [sum([int(y) for y in x]) for x in notmain]
    notmain = [x for _, x in sorted(zip(nums, notmain))]

    extras = set()

    for nm in notmain:
        comp = set()
        comp.update(val)
        comp.update(extras)

        mindist = 1000

        for line in comp:
            if nm != line:
                diff = difference(nm, line)
                if diff < mindist:
                    mindist = diff
                    minli = line

        diffbase = difference(nm, zeros)

        if diffbase <= mindist:
            minli = zeros

        num1 = sum([int(s) for s in nm])
        num2 = sum([int(s) for s in minli])
        if num1 < num2:
            graph[nm].add(minli)
        else:
            graph[minli].add(nm)

        extras.add(nm)

    val = set()
    for v in graph.values():
        val.update(v)
        with open(config.outputDir + 'relation-tree' + config.addition + '.dot', 'w') as f2:
            f2.write("digraph dag {\n")
            f2.write("rankdir=LR;\n")
            for idx, li in names.items():
                name = str(idx) + '\n'

                for line in li:
                    name += line + ',\n'
                if idx not in notmain:
                    text = str(idx) + " [label=\"" + name + "\" , shape=box;]"
                else:  # treat in a special way. For now, leaving intact
                    text = str(idx) + " [shape=box label=\"" + name + "\"]"

                f2.write(text)
                f2.write('\n')
            for k, v in graph.items():
                for vi in v:
                    f2.write(str(k) + "->" + str(vi))
                    f2.write('\n')
            f2.write("}")

    # Rendering DAG
    logging.info('Rendering DAG -- using graphviz dot')
    try:
        os.system(f'dot -Tpng {config.outputDir}relation-tree{config.addition}.dot -o {config.outputDir}DAG{config.addition}.png')
    except Exception:
        pass


def difference(str1, str2):
    assert len(str1) == len(str2)
    return sum([str1[x] != str2[x] for x in range(len(str1))])


def compareFinalClusters(finalClustersSequential, finalClustersStatistical):
    similarityArray = np.zeros((len(finalClustersSequential), len(finalClustersStatistical)))

    for number, cluster in finalClustersSequential.items():
        for number2, cluster2 in finalClustersStatistical.items():
            setOne = set(cluster)
            setTwo = set(cluster2)
            similarityArray[number+1][number2+1] = len(setOne & setTwo)

    logging.info('------------------------ Sequential to Statistical')

    maxesColumns = np.argmax(similarityArray, axis=1)
    for index in range(len(finalClustersSequential)):
        maxIndex = maxesColumns[index]
        maxValue = round(similarityArray[index][maxIndex])

        sequentialIndex = index-1
        statisticalIndex = maxIndex-1

        packetsInSequentialCluster = len(finalClustersSequential[sequentialIndex])
        packetsInStatisticalCluster = len(finalClustersStatistical[statisticalIndex])
        totalUnique = packetsInSequentialCluster + packetsInSequentialCluster - maxValue

        if sequentialIndex == -1 and statisticalIndex == -1:
            logging.info(f'There was an overlap of {round(maxValue/totalUnique * 100, 2)}% of the non-clustered data of {totalUnique} packages')
        elif statisticalIndex == -1:
            logging.info(f'{sequentialIndex} (Sequential) containing {packetsInSequentialCluster} packages, was mostly not clustered by Statistical')
        elif maxValue == packetsInStatisticalCluster:
            logging.info(f'{statisticalIndex} (Sequential) cluster is fully contained in {sequentialIndex} (Statistical) cluster')
        elif maxValue == packetsInSequentialCluster:
            logging.info(f'{sequentialIndex} (Statistical) cluster is fully contained in {statisticalIndex} (Sequential) cluster')
        else:
            logging.info(f'Overlap between {sequentialIndex} (Sequential) and {statisticalIndex} (Statistical) is {round(maxValue/totalUnique * 100, 2)}% of {totalUnique} packages')

    logging.info('------------------------ Statistical to Sequential')

    maxesRows = np.argmax(similarityArray, axis=0)
    for index in range(len(finalClustersStatistical)):
        maxIndex = maxesRows[index]
        maxValue = round(similarityArray[maxIndex][index])

        statisticalIndex = index-1
        sequentialIndex = maxIndex-1

        packetsInStatisticalCluster = len(finalClustersStatistical[statisticalIndex])
        packetsInSequentialCluster = len(finalClustersSequential[sequentialIndex])
        totalUnique = packetsInSequentialCluster + packetsInSequentialCluster - maxValue

        if statisticalIndex == -1 and sequentialIndex == -1:
            logging.info(f'There was an overlap of {round(maxValue/totalUnique * 100, 2)}% of the non-clustered data of {totalUnique} packages')
        elif sequentialIndex == -1:
            logging.info(f'{statisticalIndex} (Statistical) containing {packetsInStatisticalCluster} packages, was mostly not clustered by Sequential')
        elif maxValue == packetsInStatisticalCluster:
            logging.info(f'{statisticalIndex} (Sequential) cluster is fully contained in {sequentialIndex} (Statistical) cluster')
        elif maxValue == packetsInSequentialCluster:
            logging.info(f'{sequentialIndex} (Statistical) cluster is fully contained in {statisticalIndex} (Sequential) cluster')
        else:
            logging.info(f'Overlap between {statisticalIndex} (Statistical) and {sequentialIndex} (Sequential) is {round(maxValue/totalUnique * 100, 2)}% of {totalUnique} packages')

    logging.info('------------------------')


def readFolderWithPCAPs(useFileCache=False, forceFileCacheUse=True):
    meta = {}
    mapping = {}
    totalLabels = defaultdict(int)
    mappingIndex = 0
    if forceFileCacheUse:
        files = glob.glob(sys.argv[2] + "/*.pcap.pkl")
    else:
        files = glob.glob(sys.argv[2] + "/**/*.pcap")
    logging.info(f'About to read pcap... from {len(files)} files')

    if os.path.exists(config.pklCache) and os.path.exists(config.mappingCache) and os.path.exists(config.totalLabelsCache):
        with open(config.pklCache, 'rb') as file:
            meta = pickle.load(file)
        with open(config.mappingCache, 'rb') as file:
            mapping = pickle.load(file)
        with open(config.totalLabelsCache, 'rb') as file:
            totalLabels = pickle.load(file)
    else:
        for f in files:
            cacheKey = os.path.basename(f)
            cacheName = f'data/{cacheKey}.pkl'
            if os.path.exists(cacheName) and useFileCache:
                logging.debug(f'Using cache: {cacheKey}')
                with open(cacheName, 'rb') as file:
                    connections = pickle.load(file)
            elif os.path.exists(f) and forceFileCacheUse:
                logging.debug(f'Using cache: {cacheKey}')
                with open(f, 'rb') as file:
                    connections = pickle.load(file)
            elif not forceFileCacheUse:
                logging.info(f'Reading file: {cacheKey}')
                connections = timeFunction(readPCAP.__name__, lambda: readPCAP(f, config))

                if len(connections.items()) < 1:
                    continue

                with open(cacheName, 'wb') as file:
                    pickle.dump(connections, file)
            else:
                logging.info(f'Skipping {f} because it has no cache file: {cacheName}')
                continue

            connectionItems: list[(ConnectionKey, list[PackageInfo])] = list(connections.items())
            random.shuffle(connectionItems)
            selectedLabelsPerFile = defaultdict(int)

            for i, v in connectionItems:
                wantedWindow = getWantedWindow(v)

                for window in wantedWindow:
                    selection: list[PackageInfo] = v[config.thresh * window:config.thresh * (window + 1)]
                    labels = set()
                    for package in selection:
                        labels.add(package.connectionLabel)

                    if len(labels) != 1:
                        continue

                    label = labels.pop()

                    # if selectedLabelsPerFile[label] >= 200:
                    #     continue

                    key = ConnectionKey(cacheKey, i[0], i[1], window, selection[0].connectionLabel)

                    selectedLabelsPerFile[label] += 1
                    mapping[key] = mappingIndex
                    mappingIndex += 1
                    meta[key] = selection

            # connectionSummary(connections, selectedLabelsPerFile)
            for k, v in selectedLabelsPerFile.items():
                totalLabels[k] += v

        with open(config.pklCache, 'wb') as file:
            pickle.dump(meta, file)
        with open(config.mappingCache, 'wb') as file:
            pickle.dump(mapping, file)
        with open(config.totalLabelsCache, 'wb') as file:
            pickle.dump(totalLabels, file)

    logging.info(f'Collective surviving connections {len(meta)}')
    connectionSummary(meta, totalLabels)

    if len(meta) < 50:
        logging.error('Too little connections to create clustering')
        raise Exception

    return meta, mapping


def getWantedWindow(v):
    amountOfPackages = len(v)
    windowRange = list(range(amountOfPackages // config.thresh))
    possibleWindows = len(windowRange)

    if possibleWindows == 0:
        return []
    if possibleWindows == 1:
        return [0]
    elif possibleWindows == 2:
        return [0, 1]
    else:
        wantedWindow = windowRange[:1] + windowRange[-1:]
        wantedWindow += random.sample(windowRange[1:-1], min(len(windowRange) - 2, 8))
        return wantedWindow


def connectionSummary(connections, selectedLabelsPerFile):
    connectionLengths = [len(x) for i, x in connections.items()]
    logging.debug(f"Different connections: {len(connections)}")
    logging.debug(f"Average conn length: {np.mean(connectionLengths)}")
    logging.debug(f"Minimum conn length: {np.min(connectionLengths)}")
    logging.debug(f"Maximum conn length: {np.max(connectionLengths)}")
    if selectedLabelsPerFile:
        with open(f"{config.outputDirStats}Labels{config.addition}.txt", 'w') as f:
            for label in sorted(selectedLabelsPerFile.items()):
                name = label[0].replace('&', '\&')
                if label[0] == '-':
                    name = 'Benign'
                f.write(f"{name} & {label[1]}\n")
            f.write(f"Total & {len(connections)}\n")
        logging.info(', '.join(map(lambda x: f'{x[0]}: {x[1]}' if x[0] != '-' else f'Benign: {x[1]}', sorted(selectedLabelsPerFile.items()))))


def appendStatsToOutputFile(extraName, stat, value):
    results[f'{extraName}{config.addition}']['type'] = extraName
    results[f'{extraName}{config.addition}']['threshold'] = config.thresh
    results[f'{extraName}{config.addition}']['seed'] = config.seed
    results[f'{extraName}{config.addition}'][stat] = value
    logging.info(f"[{extraName}] {stat}: {value}")
    with open(f"{config.outputDirStats}{extraName}{config.addition}.txt", 'a') as f:
        f.write(f"{stat}    &   {value}\n")


def execute():
    for i in range(1):
        config.seed = i
        meta, mapping = timeFunction(readFolderWithPCAPs.__name__, lambda: readFolderWithPCAPs())
        timeFunction(connlevel_sequence.__name__, lambda: connlevel_sequence(meta, mapping))

        resultsDf = pd.DataFrame.from_dict(results).T.rename_axis('Name')
        resultsDf.to_csv(f'{config.outputDirStats}stats-{config.thresh}.csv')


def main():
    if len(sys.argv) < 2:
        logging.error('incomplete command')
    elif sys.argv[1] == 'folder':
        timeFunction("totalRuntime", lambda: execute())
    else:
        logging.error('incomplete command')
