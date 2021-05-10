from dataclasses import dataclass
from typing import Optional


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
class PackageInfo:
    __slots__ = ['gap', 'bytes', 'sourcePort', 'destinationPort', 'connectionLabel']
    gap: int
    bytes: int
    sourcePort: int
    destinationPort: int
    connectionLabel: Optional[str]

    def __getstate__(self):
        return dict(
            (slot, getattr(self, slot))
            for slot in self.__slots__
            if hasattr(self, slot)
        )

    def __setstate__(self, state):
        for slot, value in state.items():
            object.__setattr__(self, slot, value)
