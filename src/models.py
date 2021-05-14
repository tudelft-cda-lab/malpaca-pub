from dataclasses import dataclass


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