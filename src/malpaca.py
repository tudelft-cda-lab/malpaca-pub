#!/usr/bin/python3

import csv
import datetime
import glob
import os
import pickle
import socket
import sys
import time
from collections import deque
from fastdist import fastdist

import dpkt
import hdbscan
import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.spatial.distance import cosine
from sklearn.manifold import TSNE

plt.rcParams.update({'figure.max_open_warning': 0})

expname = 'exp'
if len(sys.argv) > 3:
    expname = sys.argv[3]

thresh = 20
if len(sys.argv) > 4:
    thresh = int(sys.argv[4])

addition = '-' + expname + '-' + str(thresh)


# @profile
def connlevel_sequence(metadata, mapping):
    data = metadata
    inv_mapping = {v: k for k, v in mapping.items()}

    values = list(data.values())
    keys = list(data.keys())

    # save intermediate results

    # ----- start porting -------

    for n, feat in [(1, 'bytes'), (0, 'gaps'), (2, 'sport'), (3, 'dport')]:
        with open(feat + '-features' + addition, 'w') as f:
            for val in values:
                vi = [str(x[n]) for x in val]
                f.write(','.join(vi))
                f.write("\n")

    labels, ndistmB = timeFunction(normalizedByteDistance.__name__, lambda: normalizedByteDistance(mapping, inv_mapping, keys, values))

    ndistmG = timeFunction(normalizedGapsDistance.__name__, lambda: normalizedGapsDistance(values))

    ndistmS = timeFunction(normalizedSourcePortDistance.__name__, lambda: normalizedSourcePortDistance(values))

    ndistmD = timeFunction(normalizedDestinationPortDistance.__name__, lambda: normalizedDestinationPortDistance(values))

    ndistm = timeFunction(normalizedDistanceMeasurement.__name__, lambda: normalizedDistanceMeasurement(ndistmB, ndistmD, ndistmG, ndistmS))

    clu, projection = generateClusters(ndistm)

    generateClusterGraph(clu.labels_, ndistm, projection)

    csv_file = 'clusters' + addition + '.csv'

    timeFunction(saveClustersToCsv.__name__, lambda: saveClustersToCsv(clu, csv_file, labels, mapping, inv_mapping))

    generateDag(clu.labels_, csv_file)

    generateGraphs(csv_file, mapping, keys, values)


def saveClustersToCsv(clu, csv_file, labels, mapping, inv_mapping):
    final_clusters = {}
    final_probs = {}

    cluster_string = "clusters: "
    for lab in set(clu.labels_):
        occ = [i for i, x in enumerate(clu.labels_) if x == lab]
        final_probs[lab] = [x for i, x in zip(clu.labels_, clu.probabilities_) if i == lab]
        cluster_string += str(lab) + ":" + str(len([labels[x] for x in occ])) + " items, "
        final_clusters[lab] = [labels[x] for x in occ]
    print(cluster_string)

    with open(csv_file, 'w') as outfile:
        outfile.write("clusnum,connnum,probability,class,filename,srcip,dstip\n")
        for n, clus in final_clusters.items():

            for idx, el in enumerate([inv_mapping[x] for x in clus]):

                ip = el.split('->')
                if '-' in ip[0]:
                    classname = el.split('-')[1]
                else:
                    classname = el.split('.pcap')[0]

                filename = el.split('.pcap')[0]

                outfile.write(
                    str(n) + "," + str(mapping[el]) + "," + str(final_probs[n][idx]) + "," + str(classname) + "," + str(
                        filename) + "," + ip[0] + "," + ip[1] + "\n")


def generateClusterGraph(labels, ndistm, projection):
    colors = ['royalblue', 'red', 'darksalmon', 'sienna', 'mediumpurple', 'palevioletred', 'plum', 'darkgreen',
              'lightseagreen', 'mediumvioletred', 'gold', 'navy', 'sandybrown', 'darkorchid', 'olivedrab', 'rosybrown',
              'maroon', 'deepskyblue', 'silver']
    pal = sns.color_palette(colors)
    extra_cols = len(set(labels)) - 18
    pal_extra = sns.color_palette('Paired', extra_cols)
    pal.extend(pal_extra)

    col = [pal[x] for x in labels]

    assert len(labels) == len(ndistm)
    plt.scatter(*projection.T, s=50, linewidth=0, c=col, alpha=0.2)
    for i, txt in enumerate(labels):
        plt.scatter(projection.T[0][i], projection.T[1][i], color=col[i], alpha=0.6)
        if txt == -1:
            continue

        plt.annotate(txt, (projection.T[0][i], projection.T[1][i]), color=col[i], alpha=0.6)
    plt.savefig("clustering-result" + addition)


def generateClusters(ndistm):
    RS = 3072018
    projection = TSNE(random_state=RS).fit_transform(ndistm)
    plt.scatter(*projection.T)
    plt.savefig("tsne-result" + addition)
    plt.close()
    size = 7
    sample = 7
    model = hdbscan.HDBSCAN(min_cluster_size=size, min_samples=sample, cluster_selection_method='leaf',
                            metric='precomputed')
    clu = model.fit(np.array([np.array(x) for x in ndistm]))  # final for citadel and dridex
    joblib.dump(clu, 'model' + addition + '.pkl')
    print("num clusters: " + str(len(set(clu.labels_)) - 1))
    avg = 0.0
    for line in list(set(clu.labels_)):
        if line != -1:
            avg += sum([(1 if x == line else 0) for x in clu.labels_])
    print("avergae size of cluster:" + str(float(avg) / float(len(set(clu.labels_)) - 1)))
    print("samples in noise: " + str(sum([(1 if x == -1 else 0) for x in clu.labels_])))
    return clu, projection


def generateGraphs(csv_file, mapping, keys, values):
    print("writing temporal heatmaps")
    if not os.path.exists('figs' + addition + '/'):
        os.mkdir('figs' + addition + '/')
        os.mkdir('figs' + addition + '/bytes')
        os.mkdir('figs' + addition + '/gaps')
        os.mkdir('figs' + addition + '/sport')
        os.mkdir('figs' + addition + '/dport')
    actlabels = []
    for a in range(len(values)):
        actlabels.append(mapping[keys[a]])
    clusterinfo = {}
    seqclufile = csv_file
    lines = open(seqclufile).readlines()[1:]
    for line in lines:
        li = line.split(",")  # clusnum, connnum, prob, srcip, dstip
        srcip = li[5]
        dstip = li[6][:-1]
        has = int(li[1])

        name = str('%12s->%12s' % (srcip, dstip))
        if li[0] not in clusterinfo.keys():
            clusterinfo[li[0]] = []
        clusterinfo[li[0]].append((has, name))
    print("rendering ... ")
    sns.set(font_scale=0.9)
    matplotlib.rcParams.update({'font.size': 10})
    for names, sname, q in [("Packet sizes", "bytes", 1), ("Interval", "gaps", 0), ("Source Port", "sport", 2),
                            ("Dest. Port", "dport", 3)]:
        for clusnum, cluster in clusterinfo.items():
            labels = [x[1] for x in cluster]

            acha = [actlabels.index(int(x[0])) for x in cluster]

            blah = [values[a] for a in acha]

            dataf = []

            for b in blah:
                dataf.append([x[q] for x in b])

            df = pd.DataFrame(dataf, index=labels)

            g = sns.clustermap(df, xticklabels=False, col_cluster=False)
            ind = g.dendrogram_row.reordered_ind
            fig = plt.figure(figsize=(10.0, 9.0))
            plt.suptitle("Exp: " + expname + " | Cluster: " + clusnum + " | Feature: " + names)
            labelsnew = []
            lol = []
            for it in ind:
                labelsnew.append(labels[it])

                lol.append(cluster[[x[1] for x in cluster].index(labels[it])][0])

            acha = [actlabels.index(int(x)) for x in lol]

            blah = [values[a] for a in acha]

            dataf = []

            for b in blah:
                dataf.append([x[q] for x in b][:20])

            df = pd.DataFrame(dataf, index=labelsnew)
            g = sns.heatmap(df, xticklabels=False)
            plt.setp(g.get_yticklabels(), rotation=0)
            plt.subplots_adjust(top=0.92, bottom=0.02, left=0.25, right=1, hspace=0.94)
            plt.savefig("figs" + addition + "/" + sname + "/" + clusnum)


def generateDag(labels, csv_file):
    print('Producing DAG with relationships between pcaps')
    clusters = {}
    numclus = len(set(labels))
    with open(csv_file, 'r') as f1:
        reader = csv.reader(f1, delimiter=',')
        for i, line in enumerate(reader):
            if i > 0:
                if line[4] not in clusters.keys():
                    clusters[line[4]] = []
                clusters[line[4]].append((line[3], line[0]))  # classname, cluster#
    f1.close()
    array = [str(x) for x in range(numclus - 1)]
    array.append("-1")
    treeprep = dict()
    for filename, val in clusters.items():
        arr = [0] * numclus
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
    with open('mas-details' + addition + '.csv', 'w') as f2:
        for k, v in treeprep.items():
            for kv, vv in v.items():
                f2.write(str(k) + ';' + str(kv) + ';' + str(len(vv)) + '\n')

    graph = {}
    names = {}
    with open('mas-details' + addition + '.csv', 'r') as f3:
        csv_reader = csv.reader(f3, delimiter=';')

        for line in csv_reader:
            graph[line[0]] = set()
            if line[0] not in names.keys():
                names[line[0]] = []
            names[line[0]].append(line[1] + "(" + line[2] + ")")

    zeros = ''.join(['0'] * (numclus - 1))
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
        with open('relation-tree' + addition + '.dot', 'w') as f2:
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
    print('Rendering DAG -- needs graphviz dot')
    try:
        os.system('dot -Tpng relation-tree' + addition + '.dot -o DAG' + addition + '.png')
    except:
        pass


def normalizedDistanceMeasurement(ndistmB, ndistmD, ndistmG, ndistmS):
    ndistm = []

    for a in range(len(ndistmS)):
        ndistm.append([])
        for b in range(len(ndistmS)):
            ndistm[a].append((ndistmB[a][b] + ndistmG[a][b] + ndistmD[a][b] + ndistmS[a][b]) / 4.0)

    return ndistm


def normalizedByteDistance(mapping, inv_mapping, keys, values):
    dataValuesLength = len(values)
    filename = 'bytesDist' + addition + '.txt'

    labels = []
    ipmapping = []
    bytesDistances = np.zeros((dataValuesLength, thresh))

    for a in range(dataValuesLength):
        labels.append(mapping[keys[a]])
        ipmapping.append((mapping[keys[a]], inv_mapping[mapping[keys[a]]]))
        bytesDistances[a] = [x[1] for x in values[a]]

    distm = fastdist.matrix_pairwise_distance(bytesDistances, fastdist.euclidean, "euclidean", return_matrix=True)

    with open(filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    with open('labels' + addition + '.txt', 'w') as outfile:
        outfile.write(' '.join([str(l) for l in labels]) + '\n')

    with open('mapping' + addition + '.txt', 'w') as outfile:
        outfile.write(' '.join([str(l) for l in ipmapping]) + '\n')

    return labels, normalize2dArray(distm)


def normalizedGapsDistance(values):
    dataValuesLength = len(values)
    filename = 'gapsDist' + addition + '.txt'

    gapsDistances = np.zeros((dataValuesLength, thresh))

    for a in range(dataValuesLength):
        gapsDistances[a] = [x[0] for x in values[a]]

    distm = fastdist.matrix_pairwise_distance(gapsDistances, fastdist.euclidean, "euclidean", return_matrix=True)

    with open(filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    return normalize2dArray(distm)


def normalize2dArray(distm):
    normalized_distm = []

    mini = distm.min()
    maxi = distm.max()
    subtracted = maxi - mini

    for a in range(len(distm)):
        normalized_distm.append([])
        for b in range(len(distm)):
            normed = (distm[a][b] - mini) / subtracted
            normalized_distm[a].append(normed)

    return normalized_distm


def normalizedSourcePortDistance(values):
    ndistmS = []

    dataValuesLength = len(values)
    filename = 'sportDist' + addition + '.txt'
    distm = [x[:] for x in [[-1] * dataValuesLength] * dataValuesLength]

    ngrams = []
    for a in range(len(values)):
        profile = dict()

        dat = [x[3] for x in values[a]]

        li = zip(dat, dat[1:], dat[2:])
        for b in li:
            if b not in profile.keys():
                profile[b] = 0

            profile[b] += 1

        ngrams.append(profile)
    assert len(ngrams) == len(values)
    for a in range(len(ngrams)):
        for b in range(a, len(ngrams)):
            if a == b:
                distm[a][b] = 0.0

            i = ngrams[a]
            j = ngrams[b]
            ngram_all = list(set(i.keys()) | set(j.keys()))
            i_vec = [(i[item] if item in i.keys() else 0) for item in ngram_all]
            j_vec = [(j[item] if item in j.keys() else 0) for item in ngram_all]

            dist = cosine(i_vec, j_vec)
            distm[a][b] = dist
            distm[b][a] = dist
    with open(filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    for a in range(len(distm)):
        ndistmS.append([])
        for b in range(len(distm)):
            ndistmS[a].append(distm[a][b])

    return ndistmS


def normalizedDestinationPortDistance(values):
    ndistmD = []

    dataValuesLength = len(values)
    filename = 'dportDist' + addition + '.txt'
    distm = [x[:] for x in [[-1] * dataValuesLength] * dataValuesLength]

    ngrams = []
    for a in range(dataValuesLength):
        profile = dict()
        dat = [x[4] for x in values[a]]

        li = zip(dat, dat[1:], dat[2:])

        for b in li:
            if b not in profile.keys():
                profile[b] = 0
            profile[b] += 1
        ngrams.append(profile)

    assert len(ngrams) == dataValuesLength
    for a in range(dataValuesLength):
        for b in range(a, dataValuesLength):
            if a == b:
                distm[a][b] = 0.0
            else:
                i = ngrams[a]
                j = ngrams[b]
                ngram_all = list(set(i.keys()) | set(j.keys()))
                i_vec = [(i[item] if item in i.keys() else 0) for item in ngram_all]
                j_vec = [(j[item] if item in j.keys() else 0) for item in ngram_all]
                dist = round(cosine(i_vec, j_vec), 8)
                distm[a][b] = dist
                distm[b][a] = dist

    with open(filename, 'w') as outfile:
        for a in range(len(distm)):
            outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    for a in range(len(distm)):
        ndistmD.append([])
        for b in range(len(distm)):
            ndistmD[a].append(distm[a][b])

    return ndistmD


def difference(str1, str2):
    assert len(str1) == len(str2)
    return sum([str1[x] != str2[x] for x in range(len(str1))])


def inet_to_str(inet: bytes) -> str:
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)


def readpcap(filename):
    print("Reading", os.path.basename(filename))

    counter = 0
    connections = {}
    previousTimestamp = {}

    with open(filename, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, pkt in pcap:
            counter += 1
            try:
                eth = dpkt.ethernet.Ethernet(pkt)
            except:
                continue

            level3 = eth.data

            if type(level3) is not dpkt.ip.IP:
                continue

            level4 = level3.data

            src_ip = inet_to_str(level3.src)
            dst_ip = inet_to_str(level3.dst)

            key = (src_ip, dst_ip)

            timestamp = datetime.datetime.utcfromtimestamp(ts)

            if key in previousTimestamp:
                gap = (timestamp - previousTimestamp[key]).microseconds / 1000
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
                source_port = 0
                destination_port = 0

            flow_data = (gap, level3.len, level3.p, source_port, destination_port)

            if connections.get(key):
                connections[key].append(flow_data)
            else:
                connections[key] = [flow_data]

        print(os.path.basename(filename), " num connections: ", len(connections))
        print('Before cleanup: Total packets: ', len(connections), ' connections.')
        for k in list(connections.keys()):  # clean it up
            if len(connections[k]) < thresh:
                connections.pop(k)

        print("Remaining connections after clean up ", len(connections))

    return connections


def readfolder(maxConnections=500, useCache=False):
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
            connections = timeFunction(readpcap.__name__, lambda: readpcap(f))

            if len(connections.items()) < 1:
                continue

            key = os.path.basename(f)
            fno = 0

            for i, v in connections.items():
                if fno >= maxConnections:
                    break

                amountOfPackages = len(v)
                for window in range(amountOfPackages // thresh):
                    if window >= 3:
                        break
                    name = key + i[0] + "->" + i[1] + ":" + str(window)
                    mapping[name] = fno
                    fno += 1
                    meta[name] = v[thresh * window:thresh * (window + 1)]

            connectionSummary(connections)

        with open('data/meta.pkl', 'wb') as file:
            pickle.dump(meta, file)
        with open('data/mapping.pkl', 'wb') as file:
            pickle.dump(mapping, file)

    print('Done reading pcaps...')
    print('Collective surviving connections ', len(meta))

    timeFunction(connlevel_sequence.__name__, lambda: connlevel_sequence(meta, mapping))


def timeFunction(name, fun):
    print(f"Started {name}...")
    startf = time.perf_counter()
    value = fun()
    endf = time.perf_counter()
    print(f"Completed {name} in {endf - startf:0.4f} seconds")
    return value


def readfile(f):
    print('About to read pcap...')
    connections = timeFunction(readpcap.__name__, lambda: readpcap(f))
    print('Done reading pcaps...')

    if len(connections.items()) < 1:
        return

    meta = {}
    mapping = {}

    fno = 0
    for i, v in connections.items():
        name = i[0] + "->" + i[1]
        mapping[name] = fno
        fno += 1
        meta[name] = v[:thresh]

    connectionSummary(connections)

    timeFunction(connlevel_sequence.__name__, lambda: connlevel_sequence(meta, mapping))


def connectionSummary(connections):
    connectionLengths = [len(x) for i, x in connections.items()]
    print("Average conn length: ", np.mean(connectionLengths))
    print("Minimum conn length: ", np.min(connectionLengths))
    print("Maximum conn length: ", np.max(connectionLengths))
    print('----------------')


def main():
    if len(sys.argv) < 2:
        print('incomplete command')
    elif sys.argv[1] == 'file':
        readfile(sys.argv[2])
    elif sys.argv[1] == 'folder':
        readfolder()
    else:
        print('incomplete command')