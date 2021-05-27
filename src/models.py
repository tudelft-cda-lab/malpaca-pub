from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConnectionKey:
    __slots__ = ['filename', 'sourceIp', 'destinationIp', 'slice', 'connectionLabel']
    filename: str
    sourceIp: str
    destinationIp: str
    slice: int
    connectionLabel: str

    def name(self):
        label = 'Benign' if self.connectionLabel == '-' else self.connectionLabel
        return f"{self.sourceIp}->{self.destinationIp}:{self.slice}-{label}"

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
class PackageInfo:
    __slots__ = ['gap', 'bytes', 'sourcePort', 'destinationPort', 'connectionLabel']
    gap: int
    bytes: int
    sourcePort: int
    destinationPort: int
    connectionLabel: str

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
class StatisticalAnalysisProperties:
    """"
    Params:
        NSP (int)& Number of small packets (len of 63 - 400 bytes).
        AIT (float)& Average arrival time of packets.
        TBT (float)& Total number of transmitted bytes.
        APL (float)& Average payload packet length for time interval.
        PV (float)& Standard deviation of payload packet length.
        DPL (int)& The total of number of different packet sizes.
        MX (int)& Size of largest package
        MP (int)& The total number of bytes transmitted by the largest packet.
        PPS (float)& Number of packets per second.
        BPS (float)& Average bits-per-second.
        USP (int)& Total number of unique Source ports.
        UDP (int)& Total number of unique Destination ports.
        CP (int)& Common ports in Source and Destination ports
    """
    __slots__ = ['NSP', 'AIT', 'TBT', 'APL', 'PV', 'DPL', 'MX', 'MP', 'PPS', 'BPS', 'USP', 'UDP', 'CP']
    NSP: int
    AIT: float
    TBT: float
    APL: float
    PV: float
    DPL: int
    MX: int
    MP: int
    PPS: float
    BPS: float
    USP: int
    UDP: int
    CP: int

    def __array__(self) -> np.ndarray:
        return np.array([getattr(self, attribute) for attribute in self.__slots__])

    def __str__(self):
        return ';'.join(str(getattr(self, attribute)) for attribute in self.__slots__)

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            object.__setattr__(self, slot, value)
