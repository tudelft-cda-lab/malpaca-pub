import os, csv, sys
import numpy as np
import matplotlib.pyplot as plt


f = sys.argv[1]


malicious_ips = ['147.32.84.165', '147.32.84.191', '147.32.84.192', '147.32.84.193', '147.32.84.204', '147.32.84.205', \
'147.32.84.206', '147.32.84.207', '147.32.84.208', '147.32.84.209']
benign_ips = ['147.32.84.170', '147.32.84.134', '147.32.84.164', '147.32.87.36', '147.32.80.9', '147.32.87.11']

mal = {key:[set(), 0] for key in malicious_ips}
ben = {key:[set(), 0] for key in benign_ips}


with open(f, newline='') as csvfile:
    # clusnum,connnum,probability,class,filename,srcip,dstip
    spamreader = csv.reader(csvfile, delimiter=',')
    for row in spamreader:
        cluster = row[0]
        prob = row[2]
        connnum = row[1]
        srcip = row[5]
        dstip = row[6]
        
        c = 'N' if cluster == '-1' else cluster
        
        if srcip in malicious_ips:
            (mal[srcip][0]).update(c)
            (mal[srcip][1]) += 1
        elif srcip in benign_ips:
            (ben[srcip][0]).update(c)
            (ben[srcip][1]) += 1
        else:
            pass
            #print('Unknown ip', srcip)

print('Profiles of malicious ips')
print()
print('IP \t Profile \t\t #connections') 
for k,v in mal.items():
    if len(v[0]) > 0:
        print('%s \t %s \t\t %d'%(k, ''.join(sorted(v[0])), v[1]))
     
print()     
print('Profiles of benign ips')
print()
print('IP \t Profile \t\t #connections') 
for k,v in ben.items():
    if len(v[0]) > 0:
        print('%s \t %s \t\t %d'%(k, ''.join(sorted(v[0])), v[1]))
            