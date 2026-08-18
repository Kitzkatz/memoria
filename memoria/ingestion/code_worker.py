# ingestion/code_worker.py
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_language, get_parser

from core.logger import debug
from memory.models import MemoryRecord

# Language to file extension mapping
LANG_EXTENSIONS = {
    "python": [".py"],
    "javascript": [".js", ".mjs", ".cjs"],
    "typescript": [".ts", ".tsx"],
    "go": [".go"],
    "rust": [".rs"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cc", ".cxx", ".hpp", ".hxx"],
    "java": [".java"],
    "ruby": [".rb"],
    "php": [".php"],
    "c_sharp": [".cs"],
    "swift": [".swift"],
    "kotlin": [".kt"],
}

# Default ignore patterns (like .gitignore for codebases)
DEFAULT_IGNORE = {
    ".git", ".venv", "venv", "env",
    "__pycache__", "node_modules", "dist", "build",
    ".idea", ".vscode", "*.pyc", "*.pyo", "*.so", "*.dll",
    "*.log", "*.tmp", "*.swp"
}


class CodeWorker:
    def __init__(self, memory_system):
        self.memory = memory_system
        self.parsers = {}
        self._init_parsers()

    def _init_parsers(self):
        """Initialize Tree-sitter parsers for all supported languages."""
        for lang in LANG_EXTENSIONS.keys():
            try:
                parser = get_parser(lang)
                self.parsers[lang] = parser
                debug(f"[CodeWorker] Loaded parser for {lang}")
            except Exception as e:
                debug(f"[CodeWorker] Failed to load parser for {lang}: {e}")

    def _get_language_for_file(self, filepath: str) -> Optional[str]:
        """Determine language from file extension."""
        ext = os.path.splitext(filepath)[1].lower()
        for lang, exts in LANG_EXTENSIONS.items():
            if ext in exts:
                return lang
        return None

    def _should_ignore(self, path: str) -> bool:
        """Check if path should be ignored."""
        path_parts = Path(path).parts
        for part in path_parts:
            if part in DEFAULT_IGNORE:
                return True
            if part.startswith('.'):
                return True
        return False

    def _is_binary(self, filepath: str) -> bool:
        """Quick check if file is likely binary."""
        try:
            with open(filepath, 'rb') as f:
                chunk = f.read(1024)
                # Check for null bytes (indicates binary)
                return b'\0' in chunk
        except Exception:
            return True

    def _extract_symbols_python(self, source: str, filepath: str) -> List[Dict[str, Any]]:
        """Extract symbols from Python AST (fallback if Tree-sitter fails)."""
        import ast
        symbols = []
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            debug(f"[CodeWorker] Python AST parse error in {filepath}: {e}")
            return symbols

        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                # Get function signature
                args = []
                for arg in n.args.args:
                    if arg.arg:
                        args.append(arg.arg)
                args_str = f"({', '.join(args)})"
                symbols.append({
                    "type": "function",
                    "name": n.name,
                    "text": f"def {n.name}{args_str}:",
                    "full_text": source[n.lineno-1:n.end_lineno],
                    "line": n.lineno,
                    "memory_type": "code"
                })
            elif isinstance(n, ast.ClassDef):
                symbols.append({
                    "type": "class",
                    "name": n.name,
                    "text": f"class {n.name}:",
                    "full_text": source[n.lineno-1:n.end_lineno],
                    "line": n.lineno,
                    "memory_type": "code"
                })
            elif isinstance(n, ast.Import):
                for alias in n.names:
                    symbols.append({
                        "type": "import",
                        "name": alias.name,
                        "text": f"import {alias.name}",
                        "full_text": source[n.lineno-1:n.end_lineno],
                        "line": n.lineno,
                        "memory_type": "code"
                    })
            elif isinstance(n, ast.ImportFrom):
                for alias in n.names:
                    symbols.append({
                        "type": "import_from",
                        "name": f"{n.module}.{alias.name}" if n.module else alias.name,
                        "text": f"from {n.module or ''} import {alias.name}",
                        "full_text": source[n.lineno-1:n.end_lineno],
                        "line": n.lineno,
                        "memory_type": "code"
                    })
        return symbols

    def _extract_symbols_tree_sitter(self, parser, source: str, filepath: str) -> List[Dict[str, Any]]:
        """Extract symbols using Tree-sitter."""
        try:
            tree = parser.parse(bytes(source, "utf8"))
            root = tree.root_node
        except Exception as e:
            debug(f"[CodeWorker] Tree-sitter parse error in {filepath}: {e}")
            return []

        symbols = []

        def walk(node):
            if node.type in ("function_definition", "method_definition"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    name = source[name_node.start_byte:name_node.end_byte]
                    text = source[node.start_byte:node.end_byte]
                    # Truncate text if too long
                    if len(text) > 500:
                        text = text[:500] + "..."
                    symbols.append({
                        "type": "function",
                        "name": name,
                        "text": text,
                        "full_text": source[node.start_byte:node.end_byte],
                        "line": node.start_point[0] + 1,
                        "memory_type": "code"
                    })
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append({
                        "type": "class",
                        "name": source[name_node.start_byte:name_node.end_byte],
                        "text": source[node.start_byte:node.end_byte],
                        "full_text": source[node.start_byte:node.end_byte],
                        "line": node.start_point[0] + 1,
                        "memory_type": "code"
                    })
            elif node.type in ("import_statement", "import_declaration", "import_from_statement"):
                symbols.append({
                    "type": "import",
                    "name": source[node.start_byte:node.end_byte][:100],
                    "text": source[node.start_byte:node.end_byte],
                    "full_text": source[node.start_byte:node.end_byte],
                    "line": node.start_point[0] + 1,
                    "memory_type": "code"
                })
            for child in node.children:
                walk(child)

        walk(root)
        return symbols

    def extract_symbols(self, filepath: str) -> List[Dict[str, Any]]:
        """Extract symbols from a file using Tree-sitter (fallback to AST for Python)."""
        if self._is_binary(filepath):
            return []

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception as e:
            debug(f"[CodeWorker] Failed to read {filepath}: {e}")
            return []

        if not source or len(source) > 5 * 1024 * 1024:  # Skip >5MB files
            debug(f"[CodeWorker] Skipping {filepath} (empty or too large)")
            return []

        lang = self._get_language_for_file(filepath)
        if not lang:
            return []

        # Python uses AST for more reliable extraction
        if lang == "python":
            symbols = self._extract_symbols_python(source, filepath)
            if symbols:
                return symbols
            # Fall through to Tree-sitter if AST failed

        parser = self.parsers.get(lang)
        if not parser:
            return []

        try:
            return self._extract_symbols_tree_sitter(parser, source, filepath)
        except Exception as e:
            debug(f"[CodeWorker] Tree-sitter extraction failed for {filepath}: {e}")
            return []

    def ingest_codebase(
        self,
        directory: str,
        max_files: int = 1000,
        batch_size: int = 50,
        max_file_size_mb: int = 5
    ) -> dict:
        """
        Recursively ingest all supported code files in a directory.

        Args:
            directory: Root directory to scan
            max_files: Maximum number of files to process
            batch_size: Number of symbols to batch per store operation
            max_file_size_mb: Skip files larger than this

        Returns:
            dict with ingestion statistics
        """
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            debug(f"[CodeWorker] Directory not found: {directory}")
            return {"files_processed": 0, "symbols_ingested": 0}

        debug(f"[CodeWorker] Scanning {directory}...")

        files_processed = 0
        total_symbols = 0
        symbol_buffer = []

        for root, dirs, files in os.walk(directory):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not self._should_ignore(os.path.join(root, d))]

            for file in files:
                filepath = os.path.join(root, file)

                if self._should_ignore(filepath):
                    continue

                lang = self._get_language_for_file(filepath)
                if lang is None:
                    continue

                if files_processed >= max_files:
                    break

                # Check file size
                try:
                    if os.path.getsize(filepath) > max_file_size_mb * 1024 * 1024:
                        debug(f"[CodeWorker] Skipping {filepath} (>{max_file_size_mb}MB)")
                        continue
                except Exception:
                    continue

                symbols = self.extract_symbols(filepath)
                if symbols:
                    # Add file context to each symbol
                    for sym in symbols:
                        sym["source_file"] = filepath
                        sym["language"] = lang
                        sym["relative_path"] = os.path.relpath(filepath, directory)
                        symbol_buffer.append(sym)

                    files_processed += 1
                    total_symbols += len(symbols)
                    debug(f"[CodeWorker] {filepath}: {len(symbols)} symbols")

                # Flush buffer if batch size reached
                if len(symbol_buffer) >= batch_size:
                    self._store_symbols(symbol_buffer)
                    symbol_buffer = []

                if files_processed >= max_files:
                    break

            if files_processed >= max_files:
                break

        # Flush remaining
        if symbol_buffer:
            self._store_symbols(symbol_buffer)

        debug(f"[CodeWorker] Complete: {files_processed} files, {total_symbols} symbols")
        return {
            "files_processed": files_processed,
            "symbols_ingested": total_symbols
        }

    def _store_symbols(self, symbols: List[Dict[str, Any]]):
        """Store symbols as memories with full metadata."""
        for sym in symbols:
            # Build rich memory text with context
            text = sym.get("text", "")
            if sym.get("type") == "function":
                # Include file context for functions
                display_text = f"[{sym.get('language')}] {sym.get('type')} {sym.get('name')} from {sym.get('relative_path', sym.get('source_file', ''))}: {text}"
            else:
                display_text = f"[{sym.get('language')}] {sym.get('type')} {sym.get('name')}: {text}"

            # Store with full metadata
            self.memory.store(
                display_text,
                memory_type="code",
                metadata={
                    "source_file": sym.get("source_file"),
                    "relative_path": sym.get("relative_path"),
                    "language": sym.get("language"),
                    "symbol_type": sym.get("type"),
                    "symbol_name": sym.get("name"),
                    "line": sym.get("line"),
                    "full_text": sym.get("full_text", text)[:2000],  # Truncate for storage
                    "ingestion_type": "code_symbol"
                }
            )

    def ingest_file(self, filepath: str) -> int:
        """Ingest a single file and return number of symbols."""
        if not os.path.isfile(filepath):
            return 0

        symbols = self.extract_symbols(filepath)
        if symbols:
            lang = self._get_language_for_file(filepath)
            for sym in symbols:
                sym["source_file"] = filepath
                sym["language"] = lang or "unknown"
                sym["relative_path"] = os.path.basename(filepath)

            self._store_symbols(symbols)
            return len(symbols)
        return 0

    def get_supported_languages(self) -> List[str]:
        """Return list of supported languages."""
        return list(LANG_EXTENSIONS.keys())

    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        return [ext for exts in LANG_EXTENSIONS.values() for ext in exts]
