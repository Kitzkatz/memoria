from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RepositoryRecord:
    name: str
    path: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class DirectoryRecord:
    name: str
    path: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class FileRecord:
    name: str
    path: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class SymbolRecord:
    name: str
    path: str
    text: str
    metadata: dict = field(default_factory=dict)
