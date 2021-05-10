#!/usr/bin/python3
import csv
import datetime
import glob
import os
import pickle
import shutil
import socket
import sys
import time
import warnings
import random
from collections import deque, defaultdict
from dataclasses import fields
from typing import TypeVar, Callable

import dpkt
import hdbscan
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from fastdist import fastdist
from sklearn.manifold import TSNE
from tqdm import tqdm

from models import PackageInfo, ConnectionKey
from fast_dtw import dtw_distance
from statisticalMeasurements import getStatisticalNormalizedDistanceMeasurement

T = TypeVar('T')

random.seed(42)
plt.rcParams.update({'figure.max_open_warning': 0})
warnings.filterwarnings("ignore", message="Attempting to set identical left == right")

expname = 'exp'
if len(sys.argv) > 3:
    expname = sys.argv[3]

thresh = 20
if len(sys.argv) > 4:
    thresh = int(sys.argv[4])

addition = '-' + expname + '-' + str(thresh)
outputDir = 'output/'  # All files in this folder will be deleted
outputDirRaw = outputDir + 'raw/'
outputDirDist = outputDir + 'dist/'
outputDirFigs = outputDir + 'figs' + addition

minClusterSize = 7


# @profile
def connlevel_sequence(metadata: dict[ConnectionKey, list[PackageInfo]], mapping, generateGraph=False):
    inv_mapping: dict[int, ConnectionKey] = {v: k for k, v in mapping.items()}

    values = list(metadata.values())

    generateOutputFolders()

    storeRawData(values)

    normalizeDistanceMeasurement = timeFunction(getSequentialNormalizedDistanceMeasurement.__name__, lambda: getSequentialNormalizedDistanceMeasurement(values))

    clu, projection = timeFunction(generateClusters.__name__, lambda: generateClusters(normalizeDistanceMeasurement))

    generateClusterGraph(clu.labels_, projection)

    finalClusters, dagClusters, heatmapCluster = saveClustersToCsv(clu, mapping, inv_mapping)

    finalClusterSummary(finalClusters, values)

    if generateGraph:
        clusterAmount = len(finalClusters)
        generateDag(dagClusters, clusterAmount)
        timeFunction(generateGraphs.__name__, lambda: generateGraphs(heatmapCluster, values))


def getSequentialNormalizedDistanceMeasurement(values):
    ndmBytes = normalizedByteDistance(values)
    ndmGaps = normalizedGapsDistance(values)
    ndmSourcePort = normalizedSourcePortDistance(values)
    ndmDestinationPort = normalizedDestinationPortDistance(values)

    return normalizedDistanceMeasurement(ndmBytes, ndmGaps, ndmSourcePort, ndmDestinationPort)


def storeRawData(values):
    for field in fields(PackageInfo):
        feat = field.name
        with open(outputDirRaw + feat + '-features' + addition, 'w') as f:
            for val in values:
                vi = [str(x.__getattribute__(feat)) for x in val]
                f.write(','.join(vi))
                f.write("\n")


def generateOutputFolders():
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.mkdir(outputDir)
    os.mkdir(outputDirRaw)
    os.mkdir(outputDirDist)
    os.mkdir(outputDirFigs)
    os.mkdir(outputDirFigs + '/bytes')
    os.mkdir(outputDirFigs + '/gap')
    os.mkdir(outputDirFigs + '/sourcePort')
    os.mkdir(outputDirFigs + '/destinationPort')


def saveClustersToCsv(clu, mapping, inv_mapping: dict[int, ConnectionKey]):
    csv_file = 'clusters' + addition + '.csv'
    labels = list(range(len(mapping)))

    final_clusters = {}
    dagClusters = {}
    heatmapClusters = {}
    final_probs = {}

    for lab in set(clu.labels_):
        occ = [i for i, x in enumerate(clu.labels_) if x == lab]
        final_probs[lab] = [x for i, x in zip(clu.labels_, clu.probabilities_) if i == lab]
        final_clusters[lab] = [labels[x] for x in occ]

    with open(outputDir + csv_file, 'w') as outfile:
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


def finalClusterSummary(finalClusters, values):
    for n, cluster in finalClusters.items():
        packages = []

        for connectionNumber in cluster:
            packages += values[connectionNumber]

        summary = labelSummary(packages)
        percentage = summary['percentage']
        if percentage > 0:
            print(f"cluster {n} is {round(percentage, 2)}% malicious, contains following labels: {','.join(summary['labels'])}, connections: {len(cluster)}")
        else:
            print(f"cluster {n} does not contain any malicious packages, connections: {len(cluster)}")


def labelSummary(packages: list[PackageInfo]):
    summary = {'labels': set(), 'total': len(packages), 'malicious': 0, 'benign': 0}

    for package in packages:
        if package.connectionLabel:
            if package.connectionLabel != '-':
                summary['malicious'] += 1
                summary['labels'].add(package.connectionLabel)
            else:
                summary['benign'] += 1

    summary.update({'percentage': summary['malicious'] / summary['total'] * 100})

    return summary


def generateClusterGraph(labels, projection):
    colors = ['royalblue', 'red', 'darksalmon', 'sienna', 'mediumpurple', 'palevioletred', 'plum', 'darkgreen',
              'lightseagreen', 'mediumvioletred', 'gold', 'navy', 'sandybrown', 'darkorchid', 'olivedrab', 'rosybrown',
              'maroon', 'deepskyblue', 'silver']
    pal = sns.color_palette(colors)
    extra_cols = len(set(labels)) - 18
    pal_extra = sns.color_palette('Paired', extra_cols)
    pal.extend(pal_extra)

    col = [pal[x] for x in labels]

    plt.scatter(*projection.T, s=50, linewidth=0, c=col, alpha=0.2)
    for i, txt in enumerate(labels):
        plt.scatter(projection.T[0][i], projection.T[1][i], color=col[i], alpha=0.6)
        if txt == -1:
            continue

        plt.annotate(txt, (projection.T[0][i], projection.T[1][i]), color=col[i], alpha=0.6)
    plt.savefig(outputDir + "clustering-result" + addition)


def generateClusters(normalizeDistanceMeasurement):
    RS = 3072018
    projection = TSNE(random_state=RS).fit_transform(normalizeDistanceMeasurement)
    plt.scatter(*projection.T)
    plt.savefig(outputDir + "tsne-result" + addition)
    plt.close()

    model = hdbscan.HDBSCAN(min_cluster_size=minClusterSize, min_samples=minClusterSize, cluster_selection_method='leaf',
                            metric='precomputed')
    clu = model.fit(np.array([np.array(x) for x in normalizeDistanceMeasurement]))  # final for citadel and dridex

    print("num clusters: " + str(len(set(clu.labels_)) - 1))
    avg = 0.0
    for line in list(set(clu.labels_)):
        if line != -1:
            avg += sum([(1 if x == line else 0) for x in clu.labels_])
    print("average size of cluster:" + str(float(avg) / float(len(set(clu.labels_)) - 1)))
    print("samples in noise: " + str(sum([(1 if x == -1 else 0) for x in clu.labels_])))

    return clu, projection


def generateGraphs(clusterInfo, values: list[list[PackageInfo]]):
    sns.set(font_scale=0.9)
    matplotlib.rcParams.update({'font.size': 10})
    actualLabels = list(range(len(values)))
    with tqdm(total=(4 * len(clusterInfo)), unit='graphs') as t:
        for name, propertyName in [("Packet sizes", "bytes"), ("Interval", "gap"), ("Source Port", "sourcePort"), ("Dest. Port", "destinationPort")]:
            for clusterNumber, cluster in clusterInfo.items():
                t.set_description_str(f"Working on {name}, cluster #{clusterNumber}")
                generateTheGraph(clusterNumber, cluster, actualLabels, values, name, propertyName)
                t.update(1)


def generateTheGraph(clusterNumber, cluster, actlabels, values: list[list[PackageInfo]], name, propertyName):
    labels = [x[1] for x in cluster]

    acha = [actlabels.index(int(x[0])) for x in cluster]

    blah = [values[a] for a in acha]

    dataf = []

    for b in blah:
        dataf.append([x.__getattribute__(propertyName) for x in b])

    df = pd.DataFrame(dataf, index=labels)

    g = sns.clustermap(df, xticklabels=False, col_cluster=False)

    if df.shape[0] <= 50:
        plt.figure(figsize=(10.0, 9.0))
    elif df.shape[0] <= 100:
        plt.figure(figsize=(15.0, 18.0))
    else:
        plt.figure(figsize=(20.0, 27.0))

    plt.suptitle(f"Exp: {expname} | Cluster: {clusterNumber} | Feature: {name}")

    labelsnew = []
    lol = []
    for it in g.dendrogram_row.reordered_ind:
        labelsnew.append(labels[it])

        lol.append(cluster[[x[1] for x in cluster].index(labels[it])][0])

    acha = [actlabels.index(int(x)) for x in lol]

    blah = [values[a] for a in acha]

    dataf = []

    for b in blah:
        dataf.append([x.__getattribute__(propertyName) for x in b][:20])

    df = pd.DataFrame(dataf, index=labelsnew)
    g = sns.heatmap(df, xticklabels=False)
    plt.setp(g.get_yticklabels(), rotation=0)
    plt.subplots_adjust(top=0.92, bottom=0.02, left=0.25, right=1, hspace=0.94)
    plt.savefig(outputDirFigs + "/" + propertyName + "/" + str(clusterNumber))
    plt.clf()


def generateDag(dagClusters, clusterAmount):
    print('Producing DAG with relationships between pcaps')

    array = [x for x in range(-1, clusterAmount - 1)]
    treeprep = dict()
    for filename, val in dagClusters.items():
        arr = [0] * clusterAmount
        for fam, clus in val:
            ind = array.index(clus)
            arr[ind] = 1
        mas = ''.join([str(x) for x in arr[:-1]])
        famname = fam
        print(filename + "\t" + fam + "\t" + ''.join([str(x) for x in arr[:-1]]))
        if mas not in treeprep.keys():
            treeprep[mas] = dict()
        if famname not in treeprep[mas].keys():
            treeprep[mas][famname] = set()
        treeprep[mas][famname].add(str(filename))

    with open(outputDir + 'mas-details' + addition + '.csv', 'w') as f2:
        for k, v in treeprep.items():
            for kv, vv in v.items():
                f2.write(str(k) + ';' + str(kv) + ';' + str(len(vv)) + '\n')

    graph = {}
    names = {}
    with open(outputDir + 'mas-details' + addition + '.csv', 'r') as f3:
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
        with open(outputDir + 'relation-tree' + addition + '.dot', 'w') as f2:
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
    print('Rendering DAG -- using graphviz dot')
    try:
        os.system(f'dot -Tpng {outputDir}relation-tree{addition}.dot -o {outputDir}DAG{addition}.png')
    except:
        pass


def normalizedDistanceMeasurement(*args):
    return sum(args) / len(args)


def normalizedByteDistance(values: list[list[PackageInfo]]):
    filename = 'bytesDist' + addition + '.txt'

    lenth = len(values)
    bytesDistances = np.zeros((lenth, thresh))

    for i, value in enumerate(values):
        bytesDistances[i] = [x.bytes for x in value]

    distm = dtw_distance(bytesDistances)

    # for i in tqdm(range(lenth)):
    #     for j in range(i, lenth):
    #         if i == j:
    #             continue
    #         distance = _dtw_distance(bytesDistances[i], bytesDistances[j])
    #         distm[i][j] = distance
    #         distm[j][i] = distance

    # distm = fastdist.matrix_pairwise_distance(bytesDistances, fastdist.euclidean, "euclidean", return_matrix=True)

    with open(outputDirDist + filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm / distm.max()


def normalizedGapsDistance(values: list[list[PackageInfo]]):
    filename = 'gapsDist' + addition + '.txt'

    lenth = len(values)

    gapsDistances = np.zeros((lenth, thresh))

    for i, value in enumerate(values):
        gapsDistances[i] = [x.gap for x in value]

    distm = dtw_distance(gapsDistances)

    # distm = np.zeros((lenth, lenth))
    #
    # for i in tqdm(range(lenth)):
    #     for j in range(i, lenth):
    #         if i == j:
    #             continue
    #         distance = _dtw_distance(gapsDistances[i], gapsDistances[j])
    #         distm[i][j] = distance
    #         distm[j][i] = distance

    # distm = fastdist.matrix_pairwise_distance(gapsDistances, fastdist.euclidean, "euclidean", return_matrix=True)

    with open(outputDirDist + filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm / distm.max()


def normalizedSourcePortDistance(values: list[list[PackageInfo]]):
    filename = 'sportDist' + addition + '.txt'

    ngrams = generateNGrams('sourcePort', values)

    return generateCosineDistanceFromNGramsAndSave(filename, ngrams)


def normalizedDestinationPortDistance(values: list[list[PackageInfo]]):
    filename = 'dportDist' + addition + '.txt'

    ngrams = generateNGrams('destinationPort', values)

    return generateCosineDistanceFromNGramsAndSave(filename, ngrams)


def generateCosineDistanceFromNGramsAndSave(filename, ngrams):
    dataValuesLength = len(ngrams)

    distm = np.zeros((dataValuesLength, dataValuesLength))

    for a in tqdm(range(dataValuesLength)):
        for b in range(a, dataValuesLength):
            if a == b:
                continue

            i = ngrams[a]
            j = ngrams[b]

            ngram_all = list(set(i.keys()) | set(j.keys()))
            i_vec = np.array([(i[item] if item in i.keys() else 0) for item in ngram_all])
            j_vec = np.array([(j[item] if item in j.keys() else 0) for item in ngram_all])

            dist = 1 - fastdist.cosine(i_vec, j_vec)

            distm[a][b] = dist
            distm[b][a] = dist

    with open(outputDirDist + filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm


def generateNGrams(attribute, values: list[list[PackageInfo]]):
    ngrams = []
    for value in values:
        profile = defaultdict(int)

        dat = [getattr(x, attribute) for x in value]

        li = zip(dat, dat[1:], dat[2:])
        for b in li:
            profile[b] += 1
        ngrams.append(profile)
    return ngrams


def difference(str1, str2):
    assert len(str1) == len(str2)
    return sum([str1[x] != str2[x] for x in range(len(str1))])


def inet_to_str(inet: bytes) -> str:
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def readLabeled(filename) -> (dict[int, str], int):
    labelsFilename = filename.replace("pcap", "labeled")
    if not os.path.exists(labelsFilename):
        print(f"Label file for {filename} doesn't exist")
        return {}

    connectionLabels = {}

    line_count = 0
    with open(labelsFilename, 'r') as f:
        for _ in f:
            line_count += 1

    with open(labelsFilename, 'r') as f:
        for line in tqdm(f, total=line_count, unit='lines', unit_scale=True, postfix=labelsFilename, mininterval=0.5):
            labelFields = line.split("\x09")

            if len(labelFields) != 21:
                continue

            sourceIp = labelFields[2]
            sourcePort = int(labelFields[3])
            destIp = labelFields[4]
            destPort = int(labelFields[5])
            labeling = labelFields[20].strip().split("   ")

            key = hash((sourceIp, destIp, sourcePort, destPort))

            connectionLabels[key] = labeling[2]

    print(f'Done reading {len(connectionLabels)} labels...')

    return connectionLabels, line_count


def readPCAP(filename, labels, count=0) -> dict[tuple[str, str], list[PackageInfo]]:
    preProcessed = defaultdict(list)

    with open(filename, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, pkt in tqdm(pcap, total=count, unit='packages', unit_scale=True, postfix=filename, mininterval=0.5):
            try:
                eth = dpkt.ethernet.Ethernet(pkt)
            except:
                continue

            level3 = eth.data

            if type(level3) is not dpkt.ip.IP:
                continue

            key = hash((level3.src, level3.dst))

            preProcessed[key].append((ts, pkt))

    print(f'Before cleanup: {len(preProcessed)} connections.')

    flattened = []
    for values in preProcessed.values():
        if len(values) < thresh:
            continue
        flattened.extend(values)
    del preProcessed

    print(f'After cleanup: {len(flattened)} packages.')

    connections = defaultdict(list)
    previousTimestamp = {}
    count = 0

    for ts, pkt in tqdm(flattened, unit='packages', unit_scale=True, postfix=filename, mininterval=0.5):
        eth = dpkt.ethernet.Ethernet(pkt)

        count += 1
        level3 = eth.data

        level4 = level3.data

        src_ip = inet_to_str(level3.src)
        dst_ip = inet_to_str(level3.dst)

        key = (src_ip, dst_ip)
        timestamp = datetime.datetime.utcfromtimestamp(ts)

        if key in previousTimestamp:
            gap = round((timestamp - previousTimestamp[key]).microseconds)
        else:
            gap = 0

        previousTimestamp[key] = timestamp

        if type(level4) is dpkt.tcp.TCP:
            source_port = level4.sport
            destination_port = level4.dport
        elif type(level4) is dpkt.udp.UDP:
            source_port = level4.sport
            destination_port = level4.dport
        else:
            continue

        label = labels.get(hash((src_ip, dst_ip, source_port, destination_port))) or labels.get(hash((dst_ip, src_ip, destination_port, source_port)))

        flow_data = PackageInfo(gap, level3.len, source_port, destination_port, label)

        connections[key].append(flow_data)

    return {key: value for (key, value) in connections.items() if len(value) >= thresh}


def readFolderWithPCAPs(perLabelThreshold=2000, useCache=False, useFileCache=True):
    meta = {}
    mapping = {}
    selectedLabels = defaultdict(int)
    mappingIndex = 0
    files = glob.glob(sys.argv[2] + "/*.pcap")
    print('About to read pcap...')

    if os.path.exists('data/meta.pkl') and os.path.exists('data/mapping.pkl') and useCache:
        with open('data/meta.pkl', 'rb') as file:
            meta = pickle.load(file)
        with open('data/mapping.pkl', 'rb') as file:
            mapping = pickle.load(file)
    else:
        for f in files:
            cacheKey = os.path.basename(f)
            cacheName = f'data/{cacheKey}.pkl'
            if os.path.exists(cacheName) and useFileCache:
                print(f'Using cache: {cacheKey}')
                with open(cacheName, 'rb') as file:
                    connections = pickle.load(file)
            else:
                print(f'Reading file: {cacheKey}')
                labels, lineCount = timeFunction(readLabeled.__name__, lambda: readLabeled(f))
                connections = timeFunction(readPCAP.__name__, lambda: readPCAP(f, labels, lineCount))

                if len(connections.items()) < 1:
                    continue

                with open(cacheName, 'wb') as file:
                    pickle.dump(connections, file)

            # slidingWindow = maxConnections // len(connections)
            # print(f"Using slidingWindow {slidingWindow} for {len(connections)} connections")

            for i, v in connections.items():
                amountOfPackages = len(v)
                windowRange = list(range(amountOfPackages // thresh))

                if len(windowRange) < 11:
                    continue

                wantedWindow = windowRange[:1] + windowRange[-1:]
                wantedWindow += random.sample(windowRange[2:-1], 8)

                for window in wantedWindow:
                    key = ConnectionKey(cacheKey, i[0], i[1], window)
                    selection = v[thresh * window:thresh * (window + 1)]
                    labels = set()
                    for package in selection:
                        labels.add(package.connectionLabel)

                    if len(labels) != 1:
                        # print("Got set with multiple labels", labels)
                        continue

                    label = labels.pop()

                    if label == '-':
                        if selectedLabels[label] >= 100:
                            continue
                    elif selectedLabels[label] >= perLabelThreshold:
                        continue

                    selectedLabels[label] += 1
                    mapping[key] = mappingIndex
                    mappingIndex += 1
                    meta[key] = selection

            connectionSummary(connections)
            selectedLabels['-'] = 0
            print(selectedLabels)

        with open('data/meta.pkl', 'wb') as file:
            pickle.dump(meta, file)
        with open('data/mapping.pkl', 'wb') as file:
            pickle.dump(mapping, file)

    print('Done reading pcaps...')
    print('Collective surviving connections ', len(meta))

    if len(meta) < 50:
        print('Too little connections to create clustering')
        raise ValueError

    return meta, mapping


def timeFunction(name, fun: Callable[[], T]) -> T:
    print(f"Started {name}...")
    startTime = time.perf_counter()
    value = fun()
    endTime = time.perf_counter()
    print(f"Completed {name} in {endTime - startTime:0.4f} seconds")
    return value


def connectionSummary(connections):
    connectionLengths = [len(x) for i, x in connections.items()]
    print("Average conn length: ", np.mean(connectionLengths))
    print("Minimum conn length: ", np.min(connectionLengths))
    print("Maximum conn length: ", np.max(connectionLengths))
    print('----------------')


def execute():
    meta, mapping = timeFunction(readFolderWithPCAPs.__name__, lambda: readFolderWithPCAPs())
    timeFunction(connlevel_sequence.__name__, lambda: connlevel_sequence(meta, mapping))


def main():
    if len(sys.argv) < 2:
        print('incomplete command')
    elif sys.argv[1] == 'folder':
        timeFunction("totalRuntime", lambda: execute())
    else:
        print('incomplete command')
