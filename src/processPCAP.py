import datetime
import logging
import os
import socket
from collections import defaultdict

import dpkt
from tqdm import tqdm

from helpers import timeFunction
from models import PackageInfo


def readPCAP(filename, config, cutOff=5000) -> dict[tuple[str, str], list[PackageInfo]]:
    preProcessed = defaultdict(list)
    reachedSizeLimit = []

    with open(filename, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, pkt in tqdm(pcap, unit='packages', unit_scale=True, postfix=filename, mininterval=0.5):
            try:
                eth = dpkt.ethernet.Ethernet(pkt)
            except Exception:
                continue

            level3 = eth.data

            if type(level3) is not dpkt.ip.IP:
                continue

            key = hash((level3.src, level3.dst))

            if key in reachedSizeLimit:
                continue

            preProcessed[key].append((ts, pkt))

            if len(preProcessed[key]) > cutOff:
                reachedSizeLimit.append(key)

    logging.info(f'Before cleanup: {len(preProcessed)} connections.')

    flattened = []
    for values in preProcessed.values():
        if len(values) < config.thresh:
            continue
        flattened.extend(values)
    del preProcessed

    logging.info(f'After cleanup: {len(flattened)} packages.')

    connections = defaultdict(list)
    previousTimestamp = {}
    count = 0

    labels, lineCount = timeFunction(readLabeled.__name__, lambda: readLabeled(filename))

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

        label = labels.get(hash((src_ip, dst_ip, source_port, destination_port))) or labels.get(hash((dst_ip, src_ip, destination_port, source_port))) or '-'

        flow_data = PackageInfo(gap, level3.len, source_port, destination_port, label)

        connections[key].append(flow_data)

    return {key: value for (key, value) in connections.items() if len(value) >= config.thresh}


def readLabeled(filename) -> (dict[int, str], int):
    labelsFilename = filename.replace("pcap", "labeled")
    if not os.path.exists(labelsFilename):
        logging.info(f"Label file for {filename} doesn't exist")
        return {}, 0

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

    logging.info(f'Done reading {len(connectionLabels)} labels...')

    return connectionLabels, line_count


def inet_to_str(inet: bytes) -> str:
    try:
        return socket.inet_ntop(socket.AF_INET, inet)
    except ValueError:
        return socket.inet_ntop(socket.AF_INET6, inet)

