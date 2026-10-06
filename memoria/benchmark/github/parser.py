from __future__ import annotations

import ast
from pathlib import Path

from .models import (
    DirectoryRecord,
    FileRecord,
    RepositoryRecord,
    SymbolRecord,
)


class GitHubParser:
    """Parse a local repository into normalized GitHub adapter records."""

    EXCLUDED_DIRS = {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "venv",
        "node_modules",
    }

    def __init__(self, repository_path: str | Path, repository_name: str):
        self.repository_path = Path(repository_path).expanduser().resolve()
        self.repository_name = repository_name

        if not self.repository_path.exists():
            raise FileNotFoundError(
                f"Repository does not exist: {self.repository_path}"
            )

        if not self.repository_path.is_dir():
            raise ValueError(
                f"Repository path is not a directory: {self.repository_path}"
            )

    def parse_repository(self) -> RepositoryRecord:
        """Create a repository-level record."""

        readme = self._find_readme()
        text = readme.read_text(encoding="utf-8") if readme else ""

        return RepositoryRecord(
            name=self.repository_name.rsplit("/", 1)[-1],
            path=".",
            text=text,
            metadata={
                "source": "github",
                "record_type": "repository",
                "repository": self.repository_name,
            },
        )

    def parse_directories(self) -> list[DirectoryRecord]:
        """Create one structural record for each repository directory."""

        records = []

        for path in self._iter_directories():
            relative = path.relative_to(self.repository_path).as_posix()

            records.append(
                DirectoryRecord(
                    name=path.name,
                    path=relative,
                    text=self._directory_text(path),
                    metadata={
                        "source": "github",
                        "record_type": "directory",
                        "repository": self.repository_name,
                        "path": relative,
                    },
                )
            )

        return records

    def parse_files(self) -> list[FileRecord]:
        """Create one record for each supported repository file."""

        records = []

        for path in self._iter_files():
            relative = path.relative_to(self.repository_path).as_posix()

            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue

            records.append(
                FileRecord(
                    name=path.name,
                    path=relative,
                    text=text,
                    metadata={
                        "source": "github",
                        "record_type": "file",
                        "repository": self.repository_name,
                        "path": relative,
                        "extension": path.suffix,
                    },
                )
            )

        return records

    def parse_symbols(self) -> list[SymbolRecord]:
        """Extract classes, functions, and methods from Python files."""

        records = []

        for path in self._iter_files():
            if path.suffix != ".py":
                continue

            try:
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
            except (UnicodeDecodeError, OSError, SyntaxError):
                continue

            relative = path.relative_to(self.repository_path).as_posix()

            def visit(node, scope=(), class_scope=None):
                if isinstance(node, ast.ClassDef):
                    record = self._class_record(
                        node,
                        source,
                        relative,
                        scope,
                    )
                    if record is not None:
                        records.append(record)

                    child_scope = (*scope, node.name)

                    for child in node.body:
                        visit(
                            child,
                            child_scope,
                            child_scope,
                        )

                    return

                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    record = self._function_record(
                        node,
                        source,
                        relative,
                        scope,
                        class_scope,
                    )
                    if record is not None:
                        records.append(record)

                    child_scope = (*scope, node.name)

                    for child in node.body:
                        visit(
                            child,
                            child_scope,
                            class_scope,
                        )

                    return

                for child in ast.iter_child_nodes(node):
                    visit(child, scope, class_scope)

            visit(tree)

        return records

    def _class_record(
        self,
        node: ast.ClassDef,
        source: str,
        relative_path: str,
        scope=(),
    ) -> SymbolRecord | None:
        text = ast.get_source_segment(source, node)

        if text is None:
            return None

        bases = [
            self._node_name(base)
            for base in node.bases
        ]

        qualified_name = ".".join((*scope, node.name))

        return SymbolRecord(
            name=node.name,
            path=relative_path,
            text=text,
            metadata={
                "source": "github",
                "record_type": "symbol",
                "repository": self.repository_name,
                "path": relative_path,
                "symbol_type": "class",
                "name": node.name,
                "qualified_name": qualified_name,
                "line_start": node.lineno,
                "line_end": getattr(node, "end_lineno", node.lineno),
                "bases": bases,
                "parent": scope[-1] if scope else None,
            },
        )

    def _function_record(
        self,
        node: ast.FunctionDef | ast.AsyncFunctionDef,
        source: str,
        relative_path: str,
        scope=(),
        class_scope=None,
    ) -> SymbolRecord | None:
        text = ast.get_source_segment(source, node)

        if text is None:
            return None

        is_async = isinstance(node, ast.AsyncFunctionDef)
        qualified_name = ".".join((*scope, node.name))

        if class_scope is not None and scope == class_scope:
            symbol_type = "async_method" if is_async else "method"
        else:
            symbol_type = "async_function" if is_async else "function"

        parent = scope[-1] if scope else None

        return SymbolRecord(
            name=node.name,
            path=relative_path,
            text=text,
            metadata={
                "source": "github",
                "record_type": "symbol",
                "repository": self.repository_name,
                "path": relative_path,
                "symbol_type": symbol_type,
                "name": node.name,
                "qualified_name": qualified_name,
                "line_start": node.lineno,
                "line_end": getattr(node, "end_lineno", node.lineno),
                "parent": parent,
            },
        )

    

    @staticmethod
    def _node_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id

        if isinstance(node, ast.Attribute):
            parts = []
            current = node

            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value

            if isinstance(current, ast.Name):
                parts.append(current.id)

            return ".".join(reversed(parts))

        return ast.unparse(node)

    def _iter_directories(self):
        for path in self.repository_path.rglob("*"):
            if not path.is_dir():
                continue

            if self._is_excluded(path):
                continue

            yield path

    def _iter_files(self):
        for path in self.repository_path.rglob("*"):
            if not path.is_file():
                continue

            if self._is_excluded(path):
                continue

            yield path

    def _is_excluded(self, path: Path) -> bool:
        try:
            relative_parts = path.relative_to(self.repository_path).parts
        except ValueError:
            return True

        return any(
            part in self.EXCLUDED_DIRS
            for part in relative_parts
        )

    def _find_readme(self) -> Path | None:
        for name in ("README.md", "README.rst", "README.txt", "README"):
            path = self.repository_path / name
            if path.is_file():
                return path

        return None

    def _directory_text(self, path: Path) -> str:
        entries = []

        for child in sorted(path.iterdir()):
            if child.name in self.EXCLUDED_DIRS:
                continue

            kind = "directory" if child.is_dir() else "file"
            entries.append(f"{kind}: {child.name}")

        return "\n".join(entries)
