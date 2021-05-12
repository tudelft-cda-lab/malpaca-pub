#!/usr/bin/python3

import sys, dpkt, datetime, glob, os, operator, subprocess, csv
import socket
import matplotlib
from collections import deque
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
if len(sys.argv) > 3:
    expname = sys.argv[3]

thresh = 20 
if len(sys.argv) > 4:
    thresh = int(sys.argv[4])
win = 2

RPY2 = False
if len(sys.argv) > 5:
    RPY2 = bool(sys.argv[5])

if RPY2 == True:
    import rpy2.robjects as robjects
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.vectors import StrVector
    from rpy2.robjects.packages import importr
    from rpy2.robjects import r
    from rpy2.robjects import ListVector

#@profile
def connlevel_sequence(metadata, mapping):
    
    inv_mapping = {v:k for k,v in mapping.items()}
    data = metadata
    timing= {}


    values = list(data.values())
    keys = list(data.keys())
    distm = []
    labels = []
    ipmapping = []
    '''for i,v in data.items():
        fig = plt.figure(figsize=(10.0,9.0))
        ax = fig.add_subplot(111)
        ax.set_title(i)
        plt.plot([x[1] for x in v][:75], 'b')
        plt.plot([x[1] for x in v][:75], 'b.')
        cid = keys.index(i)
        plt.savefig('unzipped/malevol/data/connections/'+str(cid)+'.png')'''

    # save intermediate results
    
    addition = '-'+expname+'-'+str(thresh)

    # ----- start porting -------

    utils, r = None, None
    if RPY2:

        packageNames = ('IncDTW', 'rx2')
        utils = rpackages.importr('utils')
        base = rpackages.importr('base')
        utils.chooseCRANmirror(ind=1)
        to_install = [x for x in packageNames if not rpackages.isinstalled(x)]
        for x in to_install:
            utils.install_packages(x)
        dtw = rpackages.importr('IncDTW')
        r  = robjects.r

    for n,feat in [(1, 'bytes'), (0, 'gaps'), (3, 'sport'), (4, 'dport')]:
        f = open(feat+'-features'+addition, 'w')
        for val in values:
                vi = [str(x[n]) for x in val]
                f.write(','.join(vi))
                f.write("\n")
        f.close()



    startb = time.time()

    filename = 'bytesDist'+addition+'.txt'
    if os.path.exists(filename):
        distm = []
        linecount = 0
        for line in open(filename,'r').readlines():
            distm.append([])
            ele = line.split(" ")
            for e in ele:
                distm[linecount].append(float(e))
            linecount+=1

        for line in open('labels'+addition+'.txt','r').readlines():
            labels = [int(e) for e in line.split(' ')]
        
        print( "found bytes.txt")

    else:
        if RPY2:
            bytes = []
            for i in range(len(values)):
                
                labels.append(mapping[keys[i]])
                ipmapping.append((mapping[keys[i]], inv_mapping[mapping[keys[i]]]))

                #f.write("%s\n" % [x[1] for x in values[i]][:thresh])
                #arr = [x[1] for x in values[i]][:thresh]
                #arr = [int(x) for x in arr]
                bytes.append("%s\n" % [x[1] for x in values[i]][:thresh])


            #data = r['readLines'](orig_filename)
            bytes = r['gsub']("\\[|\\]", "", bytes)
            fun = r('function(x) as.numeric(unlist(strsplit(x, "\\\\,")))')
            bytes = r['lapply'](bytes, fun)
            #print(bytes)
            args = {'lot': bytes, 'dist_method' : 'norm2', 'step_pattern' : 'symmetric1', 'normalize' : False, 'ws' : robjects.NULL, 'threshold' : robjects.NULL, 'return_matrix' : True, 'ncores' : robjects.NULL, 'useRcppParallel' : True}
            bytesdistance = dtw.dtw_dismat(**args)
            #print(bytesdistance.rx2('dismat'))
            bytes = bytesdistance.rx2('dismat')
           
            distm = []
            vals = []
            for i in range(len(bytes)):
                vals.append(bytes[i])
                if (i+1)%len(values) == 0:
                    distm.append(vals)
                    vals = []
                #print(vals, distm)
                
            

        else:
            print("starting bytes dist")

            distm = [-1] * len(data.values())
            distm = [[-1]*len(data.values()) for i in distm]

            for a in range(len(data.values())): #range(10):


                labels.append(mapping[keys[a]])
                ipmapping.append((mapping[keys[a]], inv_mapping[mapping[keys[a]]]))
                for b in range(a+1):
                   

                    i = [x[1] for x in values[a]][:thresh]
                    j = [x[1] for x in values[b]][:thresh]
                    if len(i) == 0 or len(j) == 0: continue

                    if a==b:
                        distm[a][b] = 0.0
                    else:                            
                        dist,_= fastdtw(i,j,dist=euclidean)
                        distm[a][b] = dist
                        distm[b][a] = dist

        with open(filename, 'w') as outfile:
            for a in range(len(distm)):#len(data.values())): #range(10):
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")
        with open('labels'+addition+'.txt', 'w') as outfile:
            outfile.write(' '.join([str(l) for l in labels]) + '\n')
        with open('mapping'+addition+'.txt', 'w') as outfile:
           outfile.write(' '.join([str(l) for l in ipmapping]) + '\n')
    endb = time.time()
    print('+++++++ TIME: bytes ', (endb-startb))
    ndistmB = []
    mini = min(min(distm))
    maxi = max(max(distm))
   
    
    for a in range(len(distm)):
        ndistmB.append([])
        for b in range(len(distm)):
            normed = (distm[a][b] - mini) / (maxi-mini)
            ndistmB[a].append(normed)


    startg = time.time()
    distm = []
    

    filename = 'gapsDist'+addition+'.txt'
    if os.path.exists(filename):

        linecount = 0
        for line in open(filename,'r').readlines():
            distm.append([])
            ele = line.split(" ")
            for e in ele:
                try:
                    distm[linecount].append(float(e))
                except:
                    print( "error on: " + e)
            linecount+=1


        #print distm
        print( "found gaps.txt")
    else:    
        if RPY2:
            gaps = []
            for i in range(len(values)):
                gaps.append("%s\n" % [x[0] for x in values[i]][:thresh])

            gaps = r['gsub']("\\[|\\]", "", gaps)
            fun = r('function(x) as.numeric(unlist(strsplit(x, "\\\\,")))')
            gaps = r['lapply'](gaps, fun)
            #print(gaps)
            args = {'lot': gaps, 'dist_method' : 'norm2', 'step_pattern' : 'symmetric1', 'normalize' : False, 'ws' : robjects.NULL, 'threshold' : robjects.NULL, 'return_matrix' : True, 'ncores' : robjects.NULL, 'useRcppParallel' : True}
            gapsdistance = dtw.dtw_dismat(**args)
            #print(gapsdistance.rx2('dismat'))
            gaps = gapsdistance.rx2('dismat')
            distm = []
            vals = []
            for i in range(len(gaps)):
                vals.append(gaps[i])
                if (i+1)%len(values) == 0:
                    distm.append(vals)
                    vals = []
                #print(vals, distm)
        else:
            print("starting gaps dist")
            distm = [-1] * len(data.values())
            distm = [[-1]*len(data.values()) for i in distm]

            for a in range(len(data.values())): #range(10):

                for b in range(a+1):

                    i = [x[0] for x in values[a]][:thresh]
                    j = [x[0] for x in values[b]][:thresh]

                    if len(i) == 0 or len(j) == 0: continue

                    if a==b:
                        distm[a][b] = 0.0
                    else:                                
                        dist,_= fastdtw(i,j,dist=euclidean)
                        distm[a][b] = dist
                        distm[b][a] = dist

                    

        with open(filename, 'w') as outfile:
            for a in range(len(distm)):#len(data.values())): #range(10):
                #print distm[a]
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    endg = time.time()
    print('+++++++ TIME: gaps ', (endg-startg))
    ndistmG = []
    mini = min(min(distm))
    maxi = max(max(distm))

    for a in range(len(distm)):#len(data.values())): #range(10):
        ndistmG.append([])
        for b in range(len(distm)):
            normed = (distm[a][b] - mini) / (maxi-mini)
            ndistmG[a].append(normed)

    
    # source port
    ndistmS= []
    distm = []


    starts = time.time()

    filename = 'sportDist'+addition+'.txt'
    same , diff = set(), set()
    if os.path.exists(filename):

        linecount = 0
        for line in open(filename,'r').readlines():
            distm.append([])
            ele = line.split(" ")
            for e in ele:
                try:
                    distm[linecount].append(float(e))
                except:
                    print( "error on: " + e)
            linecount+=1


        #print distm
        print( "found sport.txt")
    else:
        print("starting sport dist")
        distm = [-1] * len(data.values())
        distm = [[-1]*len(data.values()) for i in distm]


        ngrams = []
        for a in range(len(values)):
            profile = dict()

            dat =  [x[3] for x in values[a]][:thresh]


            li = zip(dat, dat[1:], dat[2:])
            for b in li:
                if b not in profile.keys():
                    profile[b] = 0
            
                profile[b] += 1

                    
            ngrams.append(profile)



        profiles = []
        # update for arrays

        
        assert len(ngrams) == len(values)
        for a in range(len(ngrams)):
            # distm.append([])
            #labels.append(mapping[keys[a]])
            for b in range(a+1):
                if a==b:
                    distm[a][b] = 0.0
                else:                                
                    i = ngrams[a]
                    j = ngrams[b]
                    ngram_all = list(set(i.keys()) | set(j.keys()))
                    i_vec = [(i[item] if item in i.keys() else 0) for item in ngram_all]
                    j_vec = [(j[item] if item in j.keys() else 0) for item in ngram_all]
                    dist = cosine(i_vec, j_vec)
                    distm[a][b] = dist
                    distm[b][a] = dist
                


        with open(filename, 'w') as outfile:
            for a in range(len(distm)):#len(data.values())): #range(10):
                #print distm[a]
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")

    ends = time.time()
    print('+++++++ TIME: sport ', (ends-starts))

    #mini = min(min(distm))
    #maxi = max(max(distm))
    #print mini
    #print maxi
    #print "effective connections " + str(len(distm[0]))
    #print "effective connections  " + str(len(distm))


    for a in range(len(distm)):#len(data.values())): #range(10):
        ndistmS.append([])
        for b in range(len(distm)):
            #normed = (distm[a][b] - mini) / (maxi-mini)
            ndistmS[a].append(distm[a][b])

            


    # dest port
    ndistmD= []
    distm = []

    startd = time.time()

    filename = 'dportDist'+addition+'.txt'
    if os.path.exists(filename):

        linecount = 0
        for line in open(filename,'r').readlines():
            distm.append([])
            ele = line.split(" ")
            for e in ele:
                try:
                    distm[linecount].append(float(e))
                except:
                    print( "error on: " + e)
            linecount+=1


        #print distm
        print( "found dport.txt")
    else:
        print("starting dport dist")
        distm = [-1] * len(data.values())
        distm = [[-1]*len(data.values()) for i in distm]

        ngrams = []
        for a in range(len(values)):
        
            profile = dict()
            dat =  [x[4] for x in values[a]][:thresh]

            li = zip(dat, dat[1:], dat[2:])
            
            for b in li:
                if b not in profile.keys():
                    profile[b] = 0 
                profile[b] += 1      
            ngrams.append(profile)


        
        assert len(ngrams) == len(values)
        for a in range(len(ngrams)):
            for b in range(a+1):
                if a==b:
                    distm[a][b] = 0.0
                else:   
                    i = ngrams[a]
                    j = ngrams[b]
                    ngram_all = list(set(i.keys()) | set(j.keys()))
                    i_vec = [(i[item] if item in i.keys() else 0) for item in ngram_all]
                    j_vec = [(j[item] if item in j.keys() else 0) for item in ngram_all]                             
                    dist = round(cosine(i_vec, j_vec),8)
                    distm[a][b] = dist
                    distm[b][a] = dist

        with open(filename, 'w') as outfile:
            for a in range(len(distm)):#len(data.values())): #range(10):
                #print distm[a]
                outfile.write(' '.join([str(e) for e in distm[a]]) + "\n")
        
        

    endd = time.time()
    print('+++++++ TIME: dport ', (endd-startd))
    mini = min(min(distm))
    maxi = max(max(distm))
    #print mini
    #print maxi
    for a in range(len(distm)):#len(data.values())): #range(10):
        ndistmD.append([])
        for b in range(len(distm)):
            #normed = (distm[a][b] - mini) / (maxi-mini)
            ndistmD[a].append(distm[a][b])

    ndistm = []

    for a in range(len(ndistmS)):#len(data.values())): #range(10):

        ndistm.append([])
        for b in range(len(ndistmS)):
            ndistm[a].append((ndistmB[a][b]+ndistmG[a][b]+ndistmD[a][b]+ndistmS[a][b])/4.0)
        #print "a: " + str(mapping[keys[a]]) + " b: " + str(mapping[keys[b]]) + " dist: " + str(ndistm[a][b])


    #print ndistm[len(ndistm)-1]
    

    print("done distance meaurement")
    print(len(ndistm))
    print(len(ndistm[0]))
    #print "effective number of connections: " + str(len(dist))

    

    plot_kwds = {'alpha': 0.5, 's' : 80, 'linewidths': 0}
    RS=3072018
    projection = TSNE(random_state=RS).fit_transform(ndistm)
    plt.scatter(*projection.T)
    plt.savefig("tsne-result"+addition)
    #plt.show()
    



    size = 20
    sample= 15

    model = hdbscan.HDBSCAN(min_cluster_size = size,  min_samples = sample, cluster_selection_method='leaf', metric='precomputed')
    clu = model.fit(np.array([np.array(x) for x in ndistm])) # final for citadel and dridex
    joblib.dump(clu, 'model'+addition+'.pkl')
    #print "size: " + str(size) + "sample: " + str(sample)+ " silhouette: " +  str(silhouette_score(ndistm, clu.labels_, metric='precomputed'))

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
    
    #classes = ['Alexa', 'Hue', 'Somfy', 'malware']
    #print([(x, col[i]) for i,x in enumerate(classes)])
    for i,txt in enumerate(clu.labels_):#mapping.keys()): #zip([x[:1] for x in mapping.keys()],clu.labels_)):

        realind = labels[i]
        name = inv_mapping[realind]
        '''thiscol = None
        thislab = None
        for cdx, cc in enumerate(classes):
            if cc in name:
                thiscol = col[cdx]
                thislab = cc
                break'''
        plt.scatter(projection.T[0][i],projection.T[1][i], color=col[i], alpha=0.6)
        if txt == -1:
           continue

        plt.annotate(txt, (projection.T[0][i],projection.T[1][i]), color=col[i], alpha=0.6)
        #plt.scatter(projection.T[0][i],projection.T[1][i], color=col[i], alpha=0.6)
        #plt.annotate(thislab, (projection.T[0][i],projection.T[1][i]), color=thiscol, alpha=0.2)
    
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
            #print(el)

            ip = el.split('->')
            if '-' in ip[0]:
                classname = el.split('-')[1]
            else: 
                classname = el.split('.pcap')[0]
            ## TODO: Overriding this FOR NOW!!!
            '''for cdx, cc in enumerate(classes):
                if cc in el:
                    classname = cc
                    break'''
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
    #print(clusters)
    f1.close()
    array = [str(x) for x in range(numclus-1)]
    array.append("-1")
    
    treeprep = dict()
    for filename,val in clusters.items():
        arr = [0]*numclus
        for fam, clus in val:
            ind = array.index(clus)
            arr[ind] = 1
        #print(filename, )    
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
            #print(k, str(kv), (vv))
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

        zeros = ''.join(['0']*(numclus-1))
        if zeros not in graph.keys():
            graph[zeros] = set()
		
        ulist = graph.keys()
        #print(len(ulist), ulist)
        covered = set()
        next = deque()

        

        specials  = []
    
        next.append(zeros)
        
        while(len(next)>0):
            #print(graph)
            l1 = next.popleft()
            covered.add(l1)
            for l2 in ulist:
                #print(l1, l2, difference(l1,l2))
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
        #print(notmain)
        #print(len(notmain))



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
                    #print(str(idx) + " [label=\""+name+"\"]")
                    text = str(idx) + " [label=\""+name+"\" , shape=box;]"
                else: # treat in a special way. For now, leaving intact
                    #print(str(idx) + " [style=\"filled\" fillcolor=\"red\" label=\""+name+"\"]")
                    text = str(idx) + " [shape=box label=\""+name+"\"]"

                f2.write(text)
                f2.write('\n')
            for k,v in graph.items():
                for vi in v:
                    f2.write(str(k)+"->"+str(vi))
                    f2.write('\n')
                    #print(k+"->"+vi)
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

    print("writing temporal heatmaps")
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
        
        srcip = li[5]
        dstip = li[6][:-1]
        has = int(li[1])
        
        name = str('%12s->%12s' % (srcip,dstip))
        if li[0] not in clusterinfo.keys():
            clusterinfo[li[0]] = []
        clusterinfo[li[0]].append((has,name))
    print("rendering ... ")

    sns.set(font_scale=0.9)
    matplotlib.rcParams.update({'font.size':10})
    for names,sname,q in [("Packet sizes","bytes",1),("Interval","gaps",0),("Source Port","sport",3),("Dest. Port","dport",4)]:
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
            plt.savefig("figs"+addition+"/"+sname+"/"+clusnum)


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
    print("Reading", os.path.basename(filename))
    mal = 0
    ben = 0 
    tot = 0
    counter=0
    ipcounter=0
    tcpcounter=0
    udpcounter=0

    windows = 2
    data = []
    connections = {}
    packetspersecond=[]
    bytesperhost = {}
    count = 0
    previousTimestamp = {}
    bytespersec = 0
    stats = []
    incoming = []
    outgoing = []
    period = 0
    bla =0 
    f = open(filename, 'rb')
    pcap = dpkt.pcap.Reader(f)
    for ts, pkt in pcap:
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

                src_ip = inet_to_str(ip.src)
                dst_ip = inet_to_str(ip.dst)

                key = (src_ip, dst_ip)

                timestamp = datetime.datetime.utcfromtimestamp(ts)

                if key in previousTimestamp:
                    gap = (timestamp - previousTimestamp[key]).microseconds / 1000 # milliseconds
                else:
                    gap = 0

                previousTimestamp[key] = timestamp

                

                stats.append(0)

                sport = 0
                dport = 0

                try:
                    if ip.p==dpkt.ip.IP_PROTO_TCP or ip.p==dpkt.ip.IP_PROTO_UDP:
                        sport = ip.data.sport
                        dport = ip.data.dport
                except:
                    continue

                if key not in connections.keys():
                    connections[key] = []
                connections[key].append((gap, ip.len, ip.p, sport, dport))
                

                
    print(os.path.basename(filename), " num connections: ", len(connections))
    
    values = []
    todel = []
    print('Before cleanup: Total packets: ', len(stats), ' in ', len(connections), ' connections.' )
    print("Average conn length (before): ", np.mean([len(x) for i,x in connections.items()]))
    print("Minimum conn length (before): ", np.min([len(x) for i,x in connections.items()]))
    print("Maximum conn length (before): ", np.max([len(x) for i,x in connections.items()]))
    for i,v in connections.items(): # clean it up
        if len(v) < thresh:
            todel.append(i)
    
    for item in todel:
        del connections[item]

    
    print("Remaining connections after clean up ", len(connections))
    
    ## Breaking longer connections into multiple windows
    windowed = dict()
    for key,val in connections.items():
        i = 0
        start = i*thresh
        end = start + thresh
        while True:
	        if len(val) <= end or i == win:
		        break
	        part = val[start:end]
	        k = (key[0],key[1],i)
	        windowed[k] = part

	        start = end
	        end = start+thresh
	        i+=1 
    print("After splitting in windows, # connections ", len(windowed))
        
    return (windowed)


def readfolder():
    fno = 0
    meta = {}
    mapping= {}
    files = glob.glob(sys.argv[2]+"/*.pcap")
    print('About to read pcap...')
    for f in files:
        key = os.path.basename(f)#[:-5].split('-')
   
        connections = (readpcap(f))
        if len(connections.items()) < 1:
            continue

        for i,v in connections.items():
            name = key+ i[0] + "->" + i[1] + '-' + str(i[2])
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
    connections = readpcap(sys.argv[2])
    print('Done reading pcaps...')
    if len(connections.items()) < 1:
        return


    endf = time.time()
    print('+++++++ TIME: file reading ', (endf-startf))
    fno = 0
    meta = {}
    nconnections = {}
    print("Average conn length: ", np.mean([len(x) for i,x in connections.items()]))
    print("Minimum conn length: ", np.min([len(x) for i,x in connections.items()]))
    print("Maximum conn length: ", np.max([len(x) for i,x in connections.items()]))
    #print("num connections survived ", len(connections))
    #print(sum([1 for i,x in connections.items() if len(x)>=50]))
    for i, v in connections.items():
        name = i[0] + "->" + i[1] + '-' + str(i[2])
        mapping[name] = fno
        fno += 1
        meta[name] = v
    print('Surviving connections ', len(meta))
    startc = time.time()
    connlevel_sequence(meta, mapping)
    endc = time.time()
    print('+++++++ TIME: Total ', (endc-startc))

if sys.argv[1] == 'file':
    readfile()
elif sys.argv[1] == 'folder':
    readfolder()
else:
    print('incomplete command')


