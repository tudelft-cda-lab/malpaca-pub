import os, csv, sys
import numpy as np
import matplotlib.pyplot as plt
from collections import deque

def difference(str1, str2):
    return sum([str1[x]!=str2[x] for x in range(len(str1))])


csv_file = sys.argv[1]
expname = sys.argv[2]
thresh = sys.argv[3]

addition = '-'+expname+'-'+str(thresh)

# Making tree
print('Producing DAG with relationships between pcaps')
clusters = {}
numclus = 0#len(set(clu.labels_))
with open(csv_file, 'r') as f1:
    reader = csv.reader(f1, delimiter = ',')
    for i,line in enumerate(reader):#f1.readlines()[1:]:
        if i > 0:
            if line[4] not in clusters.keys():
                clusters[line[4]] = []
            clusters[line[4]].append((line[5],line[0])) # classname, cluster#
            numclus = int(line[0]) if int(line[0])>numclus else numclus

numclus = numclus+1
f1.close()
array = [str(x) for x in range(numclus)]
array.append("-1")

treeprep = dict()
for filename,val in clusters.items():
    arr = [0]*(numclus+1)
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