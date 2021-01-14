# MalPaCA (Malware Packet-sequence Clustering and Analysis)

This repo contains the implementation of MalPaCA (Malware Packet-sequence Clustering and Analysis). Roughly, it does the following:

1. It takes as input a pcap file or a folder containing pcap files
2. It reads each file and splits pcaps into uni-directional connections
3. Each connection is represented as 4 sequential features, i.e. packet sizes (bytes), inter-arrival-times (ms), source ports, dest ports
4. Then, for each feature, it computes the pairwise distance between all connections and stores them in their respective distance matrix
5. The distance matrices are combined using a simple weighted average, where all features have equal weights
6. The final diatcne matrix goes as input into HDBScan clustering algorithm
7. The final clusters are post-processed and printed out as a .csv and in their respective temporal heatmaps.

## Usage (for clustering):
`python malpaca.py {file|folder} {path/to/file|path/to/folder} {experiment-name} {sequence-length-threshold} {speedup}`

`{file|folder}` : Choose one option for either one pcap or multiple in a folder

`{experiment-name}` : Name of your experiments. It will be used to name the generated artifacts, i.e. distance matrices, tsne plots, clusters.

`{sequence-length-threshold}` : Length of feature sequences

`{speedup}` : Boolean depending on whether to use R's parallel DTW library or not

Required packages:

`pip install dpkt, statsmodels, cython, dtw, fastdtw, hdbscan`

`sudo apt-get install r-base-dev` (If you want to use the `speedup`)

## Usage (for re-clustering):
`python mal-detection.py {path/to/model.pkl} {file|folder} {path/to/file|path/to/folder} {experiment-name} {sequence-length-threshold}`

`{path/to/model.pkl}` : Path to the model file of the prior experiment, to which you want to re-cluster new sequences

`{file|folder}` : (For the new experiment) Choose one option for either one pcap or multiple in a folder

`{path/to/file|path/to/folder}`: (For the new experiment) Path to .pcap file (if `file` selected) or folder containing .pcap files (if `folder` selected) 

`{experiment-name}` : (For the new experiment) Name of your new experiments. It will be used to name the generated artifacts, i.e. distance matrices, tsne plots, clusters.

`{sequence-length-threshold}` : (For the new experiment) Length of feature sequences, i.e. number of packets per sequence to consider for clustering

## Parameters:

1. ngram order : Size of the sliding window for port numbers. Can be found as `ngrams.append(zip(x,x+1,x+2))`. _Default: 3_
2. sequence-length-threshold (_thresh_) : Length of feature sequences. _Default: 20_
3. _min\_cluster\_size_ : Minimum cluster size. The smaller, the better. _Default: 7_
4. _min\_samples_ : Size of radius around each point to consider nearest neighbor. The smaller, the higher the noise samples. _Default: 7_
5. speedup : `True` | `False` depending on whether you want to speed up the code using R's parallel DTW computation library

## Caveats:

- If connection length falls less than 3 packets, this code will fail (due to trigram calculation).
- If number of clusters are 0 for some reason, this code will fail.

**If you use MalPaCA in a scientific work, consider citing the following paper:**

_@article{nadeembeyond,
  title={Beyond Labeling: Using Clustering to Build Network Behavioral Profiles of Malware Families},
  author={Nadeem, Azqa and Hammerschmidt, Christian and Ga{\~n}{\'a}n, Carlos H and Verwer, Sicco},
  journal={Malware Analysis Using Artificial Intelligence and Deep Learning},
  pages={381},
  publisher={Springer}
}_

#### Azqa Nadeem
#### TU Delft
