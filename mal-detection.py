#!/usr/bin/python3

import sys, dpkt, datetime, glob, os, operator, subprocess, csv
import socket
import matplotlib
from collections import deque
import copy
from itertools import permutations
from dtw import dtw
from fastdtw import fastdtw
from math import log
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist, pdist, cosine, euclidean,cityblock
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import json
from sklearn.manifold import TSNE
from pandas import Series
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.spatial.distance as ssd
import scipy
from itertools import groupby
import itertools
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
import hdbscan
import time

def difference(str1, str2):
    return sum([str1[x]!=str2[x] for x in range(len(str1))])

totalconn = 0

expname = 'exp'
if len(sys.argv) > 4:
    expname = sys.argv[4]

thresh = 20
if len(sys.argv) > 5:
    thresh = int(sys.argv[5])


def computeDTW(old_data, new_data, f, thresh):
    print("starting dtw dist computation")
    
    new_dist = dict()
    print(len(old_data), len(new_data))
    for a in range(len(new_data)):
        for b in range(len(old_data)):
            i = [x[f] for x in new_data[a]][:thresh]
            j = old_data[b][:thresh]
            if len(i) == 0 or len(j) == 0: continue             
            dist,_= fastdtw(i,j,dist=euclidean)
            if a not in new_dist.keys():
                new_dist[a] = dict()
            if b not in new_dist[a].keys():
                new_dist[a][b] = dist
                
    new_new_dist = dict()

    for a in range(len(new_data)):
        for b in range(len(new_data)):
            i = [x[f] for x in new_data[a]][:thresh]
            j = [x[f] for x in new_data[b]][:thresh]
            if len(i) == 0 or len(j) == 0: continue             
            dist,_= fastdtw(i,j,dist=euclidean)
            if a not in new_new_dist.keys():
                new_new_dist[a] = dict()
            if b not in new_new_dist[a].keys():
                new_new_dist[a][b] = dist
    return (new_dist, new_new_dist)
    
def computeNgram(old_data, new_data, f, thresh):
    print("starting ngram dist computation")
    
    
    print(len(old_data), len(new_data))
    
    old_ngrams = []
    for a in range(len(old_data)):
        profile = dict()
        dat =  old_data[a][:thresh]

        li = zip(dat, dat[1:], dat[2:])
        for b in li:
            if b not in profile.keys():
                profile[b] = 0
            profile[b] += 1  
        old_ngrams.append(profile)
    
    new_ngrams = []    
    for a in range(len(new_data)):
        profile = dict()
        dat =  [x[f] for x in new_data[a]][:thresh]

        li = zip(dat, dat[1:], dat[2:])
        for b in li:
            if b not in profile.keys():
                profile[b] = 0
            profile[b] += 1  
        new_ngrams.append(profile)
    
    new_dist = dict()
    for a in range(len(new_ngrams)):
        for b in range(len(old_ngrams)):

            i = new_ngrams[a]
            j = old_ngrams[b]
            ngram_all = list(set(i.keys()) | set(j.keys()))
            i_vec = [(i[item] if item in i.keys() else 0) for item in ngram_all]
            j_vec = [(j[item] if item in j.keys() else 0) for item in ngram_all]
                                         
            dist = cosine(i_vec, j_vec)
            
            if a not in new_dist.keys():
                new_dist[a] = dict()
            if b not in new_dist[a].keys():
                new_dist[a][b] = dist 
                
    new_new_dist = dict()
    for a in range(len(new_ngrams)):
        for b in range(len(new_ngrams)):

            i = new_ngrams[a]
            j = new_ngrams[b]
            ngram_all = list(set(i.keys()) | set(j.keys()))
            i_vec = [(i[item] if item in i.keys() else 0) for item in ngram_all]
            j_vec = [(j[item] if item in j.keys() else 0) for item in ngram_all]
                                         
            dist = cosine(i_vec, j_vec)
            
            if a not in new_new_dist.keys():
                new_new_dist[a] = dict()
            if b not in new_new_dist[a].keys():
                new_new_dist[a][b] = dist 
    return (new_dist, new_new_dist)

                

def compositeDist(old_data, new_data, old_dist, f, thresh, method):
    
    new_dist, new_new_dist = None, None
    if method == 'DTW': 
        new_dist, new_new_dist  = computeDTW(old_data, new_data, f, thresh)
    else:
        new_dist, new_new_dist  = computeNgram(old_data, new_data, f, thresh)
        
    # make a full dist matrix
    comp = []
    for i in range(len(old_data)+len(new_data)):
        c = []
        for j in range(len(old_data)+len(new_data)):
            #print(i,j, len(old_data), len(new_data))
            if i < len(old_data) and j < len(old_data):
                c.append(old_dist[i][j])
                #print('-- ', old_dist[i][j])
            elif j >= len(old_data) and i < len(old_data):
                c.append(new_dist[j-len(old_data)][i])
                #print('-- ', new_dist[j-len(old_data)][i])
            elif i >= len(old_data) and j < len(old_data):
                c.append(new_dist[i-len(old_data)][j])
                #print('-- ', new_dist[i-len(old_data)][j])
            else:
                c.append(new_new_dist[j-len(old_data)][i-len(old_data)])
        print(c)
        comp.append(c)
    
    return comp





def readdatafile(filename):
    data = []
    for line in open(filename,'r').readlines():
        content = line[:-1].split(',')
        data.append([float(x) for x in content])
    return copy.deepcopy(data)


def readdistfile(filename):
    distm = []
    linecount = 0
    for line in open(filename,'r').readlines():
        distm.append([])
        ele = line.split(" ")
        for e in ele:
            distm[linecount].append(float(e))
        linecount+=1
    
    
    return copy.deepcopy(distm)


    


def connlevel_sequence(metadata, mapping):

    inv_mapping = {v:k for k,v in mapping.items()}
    data = metadata
    timing= {}

    values = list(data.values())
    keys = list(data.keys())
    ipmapping = []


    addition = '-'+expname+'-'+str(thresh)

    past_exp = sys.argv[1].replace('model-', '').replace('.pkl', '')
    
    addition_past = '-'+past_exp
    
    # ---- Reloading old traces ---- #
    filename = 'bytes-features'+addition_past
    dataB = readdatafile(filename)
    print( "loaded bytes data")
    filename = 'gaps-features'+addition_past
    dataG = readdatafile(filename)
    print( "loaded gaps data")
    filename = 'sport-features'+addition_past
    dataS = readdatafile(filename)
    print( "loaded sport data")
    filename = 'dport-features'+addition_past
    dataD = readdatafile(filename)
    print( "loaded dport data")
    
    # ----- Reloading old distance matrices for tsne plot ---- #
    labels = []
    for line in open('labels'+addition_past+'.txt','r').readlines():
        labels = [int(e) for e in line.split(' ')]
        
    filename = 'bytesDist'+addition_past+'.txt'
    ndistmB = readdistfile(filename)
    print( "loaded bytes dist")
    filename = 'gapsDist'+addition_past+'.txt'
    ndistmG = readdistfile(filename)
    print( "loaded gaps dist")
    filename = 'sportDist'+addition_past+'.txt'
    ndistmS = readdistfile(filename)
    print( "loaded sport dist")
    filename = 'dportDist'+addition_past+'.txt'
    ndistmD = readdistfile(filename)
    print( "loaded dport dist")

    ndistm = []

    for a in range(len(ndistmS)):#len(data.values())): #range(10):
        ndistm.append([])
        for b in range(len(ndistmS)):
            ndistm[a].append((ndistmB[a][b]+ndistmG[a][b]+ndistmD[a][b]+ndistmS[a][b])/4.0)

    print("done reloading everything")
    print(len(ndistm))
    print(len(ndistm[0]))
    print(len(labels))
    #print "effective number of connections: " + str(len(dist))

    plot_kwds = {'alpha': 0.5, 's' : 80, 'linewidths': 0}
    RS=3072018
    projection = TSNE(random_state=RS).fit_transform(ndistm)
    plt.scatter(*projection.T)
    
    # plot new points here
    #old_data, new_data, old_dist, f, thresh
    # old_data, new_data, thresh
    distB = compositeDist(dataB, values, ndistmB, 1 , thresh, 'DTW')
    distG = compositeDist(dataG, values, ndistmG, 0 , thresh, 'DTW')
    distS = compositeDist(dataS, values, ndistmS, 2 , thresh, 'Ngram')
    distD = compositeDist(dataD, values, ndistmD, 3 , thresh, 'Ngram')
    
    
    plt.savefig("tsne-result"+addition)
    #plt.show()

    clu = joblib.load(sys.argv[1])

    print('reloaded clustering model')
    
    sys.exit()
    clu.predict()

    print( "num clusters: " + str(len(set(clu.labels_))-1))

    avg = 0.0
    for l in list(set(clu.labels_)):
        if l !=-1:
            avg+= sum([(1 if x==l else 0) for x in clu.labels_])
    print( "avergae size of cluster:" + str(float(avg)/float(len(set(clu.labels_))-1)))
    print( "samples in noise: " + str(sum([(1 if x==-1 else 0) for x in clu.labels_])))
    #clu.single_linkage_tree_.plot(cmap='viridis', colorbar=True)
    #plt.show()
    #clu.condensed_tree_.plot(select_clusters=True, selection_palette=sns.color_palette())
    #plt.show()

    cols = ['royalblue', 'red', 'darksalmon', 'sienna', 'mediumpurple', 'palevioletred', 'plum', 'darkgreen', 'lightseagreen', 'mediumvioletred', 'gold', 'navy', 'sandybrown', 'darkorchid', 'olivedrab', 'rosybrown', 'maroon' ,'deepskyblue', 'silver']
    pal = sns.color_palette(cols)#

    extra_cols =  len(set(clu.labels_)) - 18

    pal_extra = sns.color_palette('Paired', extra_cols)
    pal.extend(pal_extra)
    col = [pal[x] for x in clu.labels_]
    assert len(clu.labels_) == len(ndistm)


    mem_col = [sns.desaturate(x,p) for x,p in zip(col,clu.probabilities_)]

    plt.scatter(*projection.T, s=50, linewidth=0, c=col, alpha=0.2)
    for i,txt in enumerate(clu.labels_):#mapping.keys()): #zip([x[:1] for x in mapping.keys()],clu.labels_)):
        #if txt == -1:
        #   continue
        plt.scatter(projection.T[0][i],projection.T[1][i], color=col[i], alpha=0.6)
        plt.annotate(txt, (projection.T[0][i],projection.T[1][i]), color=col[i], alpha=0.6)

    plt.savefig("clustering-result"+addition)
    #plt.show()



    # writing csv file
    print("writing csv file")
    final_clusters = {}
    final_probs = {}
    for lab in set(clu.labels_):
        occ = [i for i,x in enumerate(clu.labels_) if x == lab]
        final_probs[lab] = [x for i,x in zip(clu.labels_, clu.probabilities_) if i == lab]
        print( "cluster: " + str(lab)+ " num items: "+ str(len([labels[x] for x in occ])))
        final_clusters[lab] = [labels[x] for x in occ]

    csv_file = 'clusters'+addition+'.csv'
    outfile = open(csv_file, 'w')
    outfile.write("clusnum,connnum,probability,class,filename,srcip,dstip\n")


    for n,clus in final_clusters.items():
        #print "cluster numbeR: " + str(n)

        for idx,el in  enumerate([inv_mapping[x] for x in clus]):
            print(el)
            ip = el.split('->')
            if '-' in ip[0]:
                classname = el.split('-')[0]
            else:
                classname = el.split('.pcap')[0]

            filename = el.split('.pcap')[0]
            #print(str(n)+","+ip[0]+","+ip[1]+","+str(final_probs[n][idx])+","+str(mapping[el])+"\n")
            outfile.write(str(n)+","+str(mapping[el])+","+str(final_probs[n][idx])+","+str(classname)+","+str(filename)+","+ip[0]+","+ip[1]+"\n")
    outfile.close()
    # Making tree
    print('Producing DAG with relationships between pcaps')
    clusters = {}
    numclus = len(set(clu.labels_))
    with open(csv_file, 'r') as f1:
        reader = csv.reader(f1, delimiter = ',')
        for i,line in enumerate(reader):#f1.readlines()[1:]:
            if i > 0:
                if line[4] not in clusters.keys():
                    clusters[line[4]] = []
                clusters[line[4]].append((line[3],line[0])) # classname, cluster#
    print(clusters)
    f1.close()
    array = [str(x) for x in range(numclus-1)]
    array.append("-1")
    treeprep = dict()
    for filename,val in clusters.items():
        for fam, clus in val:

            arr = [0]*numclus

            ind = array.index(clus)

            arr[ind] = 1

            mas = ''.join([str(x) for x in arr[:-1]])
            famname = fam
            print(filename + "\t"+ fam+"\t"+''.join([str(x) for x in arr[:-1]]))
            if mas not in treeprep.keys():
                treeprep[mas] = dict()
            if famname not in treeprep[mas].keys():
                treeprep[mas][famname] = set()
            treeprep[mas][famname].add(str(filename))

    f2 = open('mas-details'+addition+'.csv', 'w')
    for k,v in treeprep.items():
        for kv,vv in v.items():
            print(k, str(kv), (vv))
            f2.write(str(k)+';'+str(kv)+';'+str(len(vv))+'\n')
    f2.close()

    with open('mas-details'+addition+'.csv', 'rU') as f3:
        csv_reader = csv.reader(f3, delimiter=';')

        graph = {}
        names ={}
        for line in csv_reader:
            graph[line[0]] = set()
            if line[0] not in names.keys():
                names[line[0]] = []
            names[line[0]].append(line[1]+"("+line[2]+")")

        ulist = graph.keys()
        print(len(ulist))
        covered = set()
        next = deque()

        zeros = ''.join(['0']*(numclus-1))

        specials  = []

        next.append(zeros)
        while(len(next)>0):
            l1 = next.popleft()
            covered.add(l1)
            for l2 in ulist:
                if l2 not in covered and difference(l1,l2) == 1:
                    graph[l1].add(l2)

                    if l2 not in next:
                        next.append(l2)

        #keys = graph.keys()
        val = set()
        for v in graph.values():
            val.update(v)

        notmain = [x for x in ulist if x not in val]
        notmain.remove(zeros)
        nums = [sum([int(y) for y in x]) for x in notmain]
        notmain = [x for _,x in sorted(zip(nums,notmain))]

        specials = notmain
        print(notmain)
        print(len(notmain))



        extras = set()

        for nm in notmain:
            comp = set()
            comp.update(val)
            comp.update(extras)

            mindist = 1000
            minli1, minli2 = None, None
            for l in comp:
                if nm != l:
                    diff  = difference(nm,l)
                    if diff < mindist:
                        mindist = diff
                        minli = l

            diffbase = difference(nm,zeros)
            #print('diffs', nm, 'extra', mindist, 'with root', diffbase)
            if diffbase <= mindist:
                mindist = diffbase
                minli = zeros
                #print('replaced')



            num1 = sum([int(s) for s in nm])
            num2 = sum([int(s) for s in minli])
            if num1 < num2:
                graph[nm].add(minli)
            else:
                graph[minli].add(nm)


            extras.add(nm)


        #keys = graph.keys()
        val = set()
        for v in graph.values():
            val.update(v)
            f2 = open('relation-tree'+addition+'.dot', 'w')
            f2.write("digraph dag {\n")
            f2.write("rankdir=LR;\n")
            num = 0
            for idx,li in names.items():
                text = ''
                #print(idx)
                name = str(idx)+'\n'

                for l in li:
                    name+=l+',\n'
                #print(str(idx) + " [label=\""+str(num)+"\"]")
                if idx not in specials:
                    print(str(idx) + " [label=\""+name+"\"]")
                    text = str(idx) + " [label=\""+name+"\" , shape=box;]"
                else:
                    print(str(idx) + " [style=\"filled\" fillcolor=\"red\" label=\""+name+"\"]")
                    text = str(idx) + " [style=\"filled\" shape=box, fillcolor=\"red\" label=\""+name+"\"]"

                f2.write(text)
                f2.write('\n')
            for k,v in graph.items():
                for vi in v:
                    f2.write(str(k)+"->"+str(vi))
                    f2.write('\n')
                    print(k+"->"+vi)
            f2.write("}")
            f2.close()
        # Rendering DAG
        print('Rendering DAG -- needs graphviz dot')
        try:
            os.system('dot -Tpng relation-tree'+addition+'.dot -o DAG'+addition+'.png')
            print('Done')
        except:
            print('Failed')
            pass


    # temporal heatmaps start

    '''print("writing temporal heatmaps")
    #print("prob: ", clu.probabilities_)
    if not os.path.exists('figs'+addition+'/'):
        os.mkdir('figs'+addition+'/')
        os.mkdir('figs'+addition+'/bytes')
        os.mkdir('figs'+addition+'/gaps')
        os.mkdir('figs'+addition+'/sport')
        os.mkdir('figs'+addition+'/dport')


    actlabels = []
    for a in range(len(values)): #range(10):
        actlabels.append(mapping[keys[a]])


    clusterinfo = {}
    seqclufile = csv_file
    lines = []
    lines = open(seqclufile).readlines()[1:]

    for line in lines:
        li = line.split(",")   # clusnum, connnum, prob, srcip, dstip
        #if li[0] == '-1':
        #    continue

        srcip = li[3]
        dstip = li[4][:-1]
        has = int(li[1])

        name = str('%12s->%12s' % (srcip,dstip))
        if li[0] not in clusterinfo.keys():
            clusterinfo[li[0]] = []
        clusterinfo[li[0]].append((has,name))
    print("rendering ... ")

    sns.set(font_scale=0.9)
    matplotlib.rcParams.update({'font.size':10})
    for names,sname,q in [("Packet sizes","bytes",1),("Interval","gaps",0),("Source Port","sport",2),("Dest. Port","dport",3)]:
        for clusnum,cluster in clusterinfo.items():
            items = [int(x[0]) for x in cluster]
            labels = [x[1] for x in cluster]

            acha = [actlabels.index(int(x[0])) for x in cluster]

            blah =  [values[a] for a in acha]

            dataf = []

            for b in blah:

                    dataf.append([x[q] for x in b][:thresh])

            df = pd.DataFrame(dataf, index=labels)

            g = sns.clustermap(df, xticklabels=False, col_cluster=False)#, vmin= minb, vmax=maxb)
            ind = g.dendrogram_row.reordered_ind
            fig = plt.figure(figsize=(10.0,9.0))
            plt.suptitle("Exp: " + expname + " | Cluster: " + clusnum + " | Feature: "+ names)
            ax = fig.add_subplot(111)
            datanew = []
            labelsnew = []
            lol = []
            for it in ind:
                labelsnew.append(labels[it])
                #print labels[it]

                #print cluster[[x[1] for x in cluster].index(labels[it])][0]
                lol.append(cluster[[x[1] for x in cluster].index(labels[it])][0])
            #print len(labelsnew)
            #print len(lol)
            acha = [actlabels.index(int(x)) for x in lol]
            #print acha
            blah =  [values[a] for a in acha]

            dataf = []

            for b in blah:
                    dataf.append([x[q] for x in b][:20])
            df = pd.DataFrame(dataf, index=labelsnew)
            g = sns.heatmap(df, xticklabels=False)
            plt.setp(g.get_yticklabels(),rotation=0)
            plt.subplots_adjust(top=0.92,bottom=0.02,left=0.25,right=1,hspace=0.94)
            plt.savefig("figs"+addition+"/"+sname+"/"+clusnum)'''


def inet_to_str(inet):
    """Convert inet object to a string
        Args:
            inet (inet struct): inet network address
        Returns:
            str: Printable/readable IP address
    """
    # First try ipv4 and then ipv6
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)

src_set , dst_set, gap_set, proto_set, bytes_set, events_set, ip_set, dns_set, port_set = set(), set(), set(), set(), set(), set(), set(), set(), set()
src_dict , dst_dict, proto_dict, events_dict, dns_dict, port_dict = {}, {}, {}, {}, {}, {}
bytes, gap_list = [], []


def readpcap(filename):
    mal = 0
    ben = 0
    tot = 0
    counter=0
    ipcounter=0
    tcpcounter=0
    udpcounter=0

    data = []
    connections = {}
    packetspersecond=[]
    bytesperhost = {}
    count = 0
    prev = -1
    bytespersec = 0
    gaps = []
    incoming = []
    outgoing = []
    period = 0
    bla =0
    f = open(filename, 'rb')
    pcap = dpkt.pcap.Reader(f)
    for ts, pkt in pcap:
                #try:
                timestamp = (datetime.datetime.utcfromtimestamp(ts))
                gap = 0.0 if prev==-1 else round(float((timestamp-prev).microseconds)/float(1000),3)
                #print gap
                if prev == -1:
                    period = timestamp

                prev = timestamp
                counter+=1
                eth= None
                bla += 1
                try:
                        eth=dpkt.ethernet.Ethernet(pkt)
                except:
                        continue

                if eth.type!=dpkt.ethernet.ETH_TYPE_IP:
                    continue

                ip=eth.data


                tupple = (gap, ip.len, ip.p)

                gaps.append(tupple)


                src_ip= inet_to_str(ip.src)
                dst_ip = inet_to_str(ip.dst)
                #print(src_ip, dst_ip)
                sport = 0
                dport = 0
                try:
                    if ip.p==dpkt.ip.IP_PROTO_TCP or ip.p==dpkt.ip.IP_PROTO_UDP:
                        sport = ip.data.sport
                        dport = ip.data.dport
                except:
                    continue

                if (src_ip, dst_ip) not in connections.keys():
                    connections[(src_ip, dst_ip)] = []
                connections[(src_ip,dst_ip)].append((gap, ip.len, ip.p, sport, dport))



    print(os.path.basename(filename), " num connections: ", len(connections))

    values = []
    todel = []
    print('Before cleanup: Total packets: ', len(gaps), ' in ', len(connections), ' connections.' )
    for i,v in connections.items(): # clean it up
        if len(v) < thresh:

            todel.append(i)


    for item in todel:
        del connections[item]


    print("Remaining connections after clean up ", len(connections))

    return (gaps,connections)


def readfolder():
    fno = 0
    meta = {}
    mapping= {}
    files = glob.glob(sys.argv[3]+"/*.pcap")
    print('About to read pcap...')
    for f in files:
        key = os.path.basename(f)#[:-5].split('-')

        data,connections = (readpcap(f))
        if len(connections.items()) < 1:
            continue

        for i,v in connections.items():
            name = key+ i[0] + "->" + i[1]
            print (name)
            #name = meta[key[len(key)-1]]['threat']+"|" +key[len(key)-1][:5]+"|"+i[0]+"->"+i[1]
            mapping[name] = fno
            fno += 1
            meta[name] = v

        print("Average conn length: ", np.mean([len(x) for i,x in connections.items()]))
        print("Minimum conn length: ", np.min([len(x) for i,x in connections.items()]))
        print("Maximum conn length: ", np.max([len(x) for i,x in connections.items()]))
        print ('----------------')

    print('Done reading pcaps...')
    print('Collective surviving connections ', len(meta))


    connlevel_sequence(meta, mapping)

def readfile():
    startf = time.time()
    mapping= {}
    print('About to read pcap...')
    data, connections = readpcap(sys.argv[3])
    print('Done reading pcaps...')
    if len(connections.items()) < 1:
        return


    endf = time.time()
    print('file reading ', (endf-startf))
    fno = 0
    meta = {}
    nconnections = {}
    print("Average conn length: ", np.mean([len(x) for i,x in connections.items()]))
    print("Minimum conn length: ", np.min([len(x) for i,x in connections.items()]))
    print("Maximum conn length: ", np.max([len(x) for i,x in connections.items()]))
    #print("num connections survived ", len(connections))
    #print(sum([1 for i,x in connections.items() if len(x)>=50]))
    for i, v in connections.items():
        name = i[0] + "->" + i[1]
        mapping[name] = fno
        fno += 1
        meta[name] = v

        '''fig = plt.figure()
        plt.title(''+name)
        plt.plot([x[0] for x in v], 'r')
        plt.plot([x[0] for x in v], 'r.')
        plt.savefig('figs/'+str(mapping[name])+'.png')'''
    print('Surviving connections ', len(meta))
    startc = time.time()
    connlevel_sequence(meta, mapping)
    endc = time.time()
    print('Total time ', (endc-startc))

if sys.argv[2] == 'file':
    readfile()
elif sys.argv[2] == 'folder':
    readfolder()
else:
    print('incomplete command')
