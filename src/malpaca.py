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
from collections import deque
from dataclasses import dataclass, fields
from typing import TypeVar, Callable, Optional

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

T = TypeVar('T')

plt.rcParams.update({'figure.max_open_warning': 0})

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


@dataclass(frozen=True)
class ConnectionLabel:
    __slots__ = ['isMalicious', 'label']
    isMalicious: bool
    label: str

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            object.__setattr__(self, slot, value)


@dataclass(frozen=True)
class ConnectionKey:
    __slots__ = ['filename', 'sourceIp', 'destinationIp', 'slice']
    filename: str
    sourceIp: str
    destinationIp: str
    slice: int

    def name(self):
        return f"{self.sourceIp}->{self.destinationIp}:{self.slice}"

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            object.__setattr__(self, slot, value)


@dataclass(frozen=True)
class LabelKey:
    __slots__ = ['sourceIp', 'destinationIp', 'sourcePort', 'destinationPort']
    sourceIp: str
    destinationIp: str
    sourcePort: int
    destinationPort: int

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            object.__setattr__(self, slot, value)


@dataclass()
class PackageInfo:
    __slots__ = ['gap', 'bytes', 'sourcePort', 'destinationPort', 'connectionLabel']
    gap: int
    bytes: int
    sourcePort: int
    destinationPort: int
    connectionLabel: Optional[ConnectionLabel]

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            object.__setattr__(self, slot, value)


# @profile
def connlevel_sequence(metadata: dict[ConnectionKey, list[PackageInfo]], mapping):
    inv_mapping: dict[int, ConnectionKey] = {v: k for k, v in mapping.items()}

    keys = list(metadata.keys())
    values = list(metadata.values())

    # save intermediate results
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
    # ----- start porting -------

    for field in fields(PackageInfo):
        feat = field.name
        with open(outputDirRaw + feat + '-features' + addition, 'w') as f:
            for val in values:
                vi = [str(x.__getattribute__(feat)) for x in val]
                f.write(','.join(vi))
                f.write("\n")

    normalizeDistanceMeasurementBytes = timeFunction(normalizedByteDistance.__name__, lambda: normalizedByteDistance(mapping, inv_mapping, keys, values))

    normalizeDistanceMeasurementGaps = timeFunction(normalizedGapsDistance.__name__, lambda: normalizedGapsDistance(values))

    normalizeDistanceMeasurementSourcePort = timeFunction(normalizedSourcePortDistance.__name__, lambda: normalizedSourcePortDistance(values))

    normalizeDistanceMeasurementDestinationPort = timeFunction(normalizedDestinationPortDistance.__name__, lambda: normalizedDestinationPortDistance(values))

    normalizeDistanceMeasurement = timeFunction(normalizedDistanceMeasurement.__name__,
                                                lambda: normalizedDistanceMeasurement(normalizeDistanceMeasurementBytes,
                                                                                      normalizeDistanceMeasurementDestinationPort,
                                                                                      normalizeDistanceMeasurementGaps,
                                                                                      normalizeDistanceMeasurementSourcePort))

    clu, projection = timeFunction(generateClusters.__name__, lambda: generateClusters(normalizeDistanceMeasurement))

    generateClusterGraph(clu.labels_, projection)

    finalClusters, dagClusters, heatmapCluster = timeFunction(saveClustersToCsv.__name__, lambda: saveClustersToCsv(clu, mapping, inv_mapping))

    clusterAmount = len(finalClusters)

    finalClusterSummary(finalClusters, values)

    generateDag(dagClusters, clusterAmount)

    timeFunction(generateGraphs.__name__, lambda: generateGraphs(heatmapCluster, values))


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
            print(f"cluster {n} does not contain any (known) malicious packages, connections: {len(cluster)}")


def labelSummary(packages: list[PackageInfo]):
    summary = {'labels': set(), 'total': len(packages), 'malicious': 0, 'benign': 0}

    for package in packages:
        if package.connectionLabel:
            if package.connectionLabel.isMalicious:
                summary['malicious'] += 1
                summary['labels'].add(package.connectionLabel.label)
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
    size = 7
    sample = 7
    model = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=sample, cluster_selection_method='leaf',
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

    plt.suptitle("Exp: " + expname + " | Cluster: " + str(clusterNumber) + " | Feature: " + name)

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

    specials = notmain

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
                if idx not in specials:
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


def normalizedDistanceMeasurement(ndistmB, ndistmD, ndistmG, ndistmS):
    return (ndistmB + ndistmD + ndistmG + ndistmS) / 4


def normalizedByteDistance(mapping, inv_mapping, keys, values: list[list[PackageInfo]]):
    dataValuesLength = len(values)
    filename = 'bytesDist' + addition + '.txt'

    ipmapping = []
    bytesDistances = np.zeros((dataValuesLength, thresh))

    for a in range(dataValuesLength):
        ipmapping.append((mapping[keys[a]], inv_mapping[mapping[keys[a]]]))
        bytesDistances[a] = [x.bytes for x in values[a]]

    distm = fastdist.matrix_pairwise_distance(bytesDistances, fastdist.euclidean, "euclidean", return_matrix=True)

    with open(outputDirDist + filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    with open(outputDir + 'mapping' + addition + '.txt', 'w') as outfile:
        outfile.write(' '.join([str(l) for l in ipmapping]) + '\n')

    return distm / distm.max()


def normalizedGapsDistance(values: list[list[PackageInfo]]):
    dataValuesLength = len(values)
    filename = 'gapsDist' + addition + '.txt'

    gapsDistances = np.zeros((dataValuesLength, thresh))

    for a in range(dataValuesLength):
        gapsDistances[a] = [x.gap for x in values[a]]

    distm = fastdist.matrix_pairwise_distance(gapsDistances, fastdist.euclidean, "euclidean", return_matrix=True)

    with open(outputDirDist + filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return distm / distm.max()


def normalizedSourcePortDistance(values: list[list[PackageInfo]]):
    dataValuesLength = len(values)
    filename = 'sportDist' + addition + '.txt'

    ngrams = generateNGrams('sourcePort', values)

    return generateCosineDistanceFromNGramsAndSave(filename, ngrams, dataValuesLength)


def normalizedDestinationPortDistance(values: list[list[PackageInfo]]):
    dataValuesLength = len(values)
    filename = 'dportDist' + addition + '.txt'

    ngrams = generateNGrams('destinationPort', values)

    return generateCosineDistanceFromNGramsAndSave(filename, ngrams, dataValuesLength)


def generateCosineDistanceFromNGramsAndSave(filename, ngrams, dataValuesLength):
    assert len(ngrams) == dataValuesLength

    distm = np.zeros((dataValuesLength, dataValuesLength))

    for a in range(dataValuesLength):
        for b in range(a, dataValuesLength):
            if a == b:
                distm[a][b] = 0.0
            else:
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
    for a in range(len(values)):
        profile = dict()

        dat = [getattr(x, attribute) for x in values[a]]

        li = zip(dat, dat[1:], dat[2:])
        for b in li:
            if b not in profile.keys():
                profile[b] = 0
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


def readLabeled(filename) -> dict[int, ConnectionLabel]:
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

            labeling = labelFields[20].replace("(empty)", "").replace("-", "").strip(" ").strip(" \n").split("   ")

            if labeling[0] == "Benign":
                isMalicious = False
                label = None
            else:
                isMalicious = True
                label = labeling[1]

            key = LabelKey(sourceIp, destIp, sourcePort, destPort).__hash__()

            connectionLabels[key] = ConnectionLabel(isMalicious, label)

    print(f'Done reading {len(connectionLabels)} labels...')

    return connectionLabels


def readFolderWithLabels(useCache=True, useFileCache=True):
    connsLabeled = {}
    files = glob.glob(sys.argv[2] + "/*.labeled")
    print('About to read labels...')

    if os.path.exists('data/connsLabels.pkl') and useCache:
        with open('data/connsLabels.pkl', 'rb') as file:
            connsLabeled = pickle.load(file)
    else:
        for f in files:
            cacheKey = os.path.basename(f)
            cacheName = f'data/bro/{cacheKey}.pkl'
            if os.path.exists(cacheName) and useFileCache:
                print(f'Using cache: {cacheKey}')
                with open(cacheName, 'rb') as file:
                    fileLabels = pickle.load(file)
            else:
                print(f'Reading file: {cacheKey}')
                fileLabels = timeFunction(readLabeled.__name__, lambda: readLabeled(f))

                if len(fileLabels.items()) < 1:
                    continue

                with open(cacheName, 'wb') as file:
                    pickle.dump(fileLabels, file)

        with open('data/connsLabels.pkl', 'wb') as file:
            pickle.dump(connsLabeled, file)

    print(f'Done reading {len(connsLabeled)} labels...')

    return connsLabeled


def readPCAP(filename, labels) -> dict[tuple[str, str], list[PackageInfo]]:
    connections = {}
    previousTimestamp = {}

    count = 0
    with open(filename, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, pkt in tqdm(pcap, unit='packages', unit_scale=True, postfix=filename, mininterval=0.5):
            try:
                eth = dpkt.ethernet.Ethernet(pkt)
            except:
                continue

            count += 1
            level3 = eth.data

            if type(level3) is not dpkt.ip.IP:
                continue

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

            labelHash = LabelKey(src_ip, dst_ip, source_port, destination_port).__hash__()
            labelHashAlt = LabelKey(dst_ip, src_ip, destination_port, source_port).__hash__()

            flow_data = PackageInfo(gap, level3.len, source_port, destination_port, labels.get(labelHash) or labels.get(labelHashAlt))

            if not connections.get(key):
                connections[key] = []

            connections[key].append(flow_data)

    print('Before cleanup: Total packets: ', len(connections), ' connections.')

    return {key: value for (key, value) in connections.items() if len(value) >= thresh}


def readFolderWithPCAPs(maxConnections=2000, useCache=True, useFileCache=True):
    meta = {}
    mapping = {}
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
                labels = timeFunction(readLabeled.__name__, lambda: readLabeled(f))
                connections = timeFunction(readPCAP.__name__, lambda: readPCAP(f, labels))

                if len(connections.items()) < 1:
                    continue

                with open(cacheName, 'wb') as file:
                    pickle.dump(connections, file)

            fno = 0

            slidingWindow = maxConnections // len(connections)
            print(f"Using slidingWindow {slidingWindow} for {len(connections)} connections")

            for i, v in connections.items():
                if fno >= maxConnections:
                    break

                amountOfPackages = len(v)
                for window in range(amountOfPackages // thresh):
                    if window >= slidingWindow:
                        break
                    key = ConnectionKey(cacheKey, i[0], i[1], window)
                    mapping[key] = fno
                    fno += 1
                    meta[key] = v[thresh * window:thresh * (window + 1)]

            connectionSummary(connections)

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
