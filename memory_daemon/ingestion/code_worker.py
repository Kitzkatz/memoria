# ingestion/code_worker.py
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from tree_sitter import Language, Parser
from tree_sitter_language_pack import get_language, get_parser

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
            except Exception:
                print(f"[CodeWorker] Failed to load parser for {lang}")

    def _get_language_for_file(self, filepath: str) -> Optional[str]:
        """Determine language from file extension."""
        ext = os.path.splitext(filepath)[1]
        for lang, exts in LANG_EXTENSIONS.items():
            if ext in exts:
                return lang
        return None

    def _extract_symbols_python(self, node, source: str) -> List[Dict[str, Any]]:
        """Extract symbols from Python AST (fallback if Tree-sitter fails)."""
        import ast
        symbols = []
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return symbols
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef):
                symbols.append({
                    "type": "function",
                    "name": n.name,
                    "text": f"def {n.name}(...):",
                    "memory_type": "code"
                })
            elif isinstance(n, ast.ClassDef):
                symbols.append({
                    "type": "class",
                    "name": n.name,
                    "text": f"class {n.name}:",
                    "memory_type": "code"
                })
            elif isinstance(n, ast.Import):
                for alias in n.names:
                    symbols.append({
                        "type": "import",
                        "name": alias.name,
                        "text": f"import {alias.name}",
                        "memory_type": "code"
                    })
        return symbols

    def _extract_symbols_tree_sitter(self, parser, source: str) -> List[Dict[str, Any]]:
        """Extract symbols using Tree-sitter."""
        tree = parser.parse(bytes(source, "utf8"))
        root = tree.root_node
        symbols = []

        def walk(node):
            if node.type == "function_definition" or node.type == "method_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append({
                        "type": "function",
                        "name": source[name_node.start_byte:name_node.end_byte],
                        "text": source[node.start_byte:node.end_byte],
                        "memory_type": "code"
                    })
            elif node.type == "class_definition":
                name_node = node.child_by_field_name("name")
                if name_node:
                    symbols.append({
                        "type": "class",
                        "name": source[name_node.start_byte:name_node.end_byte],
                        "text": source[node.start_byte:node.end_byte],
                        "memory_type": "code"
                    })
            elif node.type == "import_statement" or node.type == "import_declaration":
                symbols.append({
                    "type": "import",
                    "name": source[node.start_byte:node.end_byte][:100],
                    "text": source[node.start_byte:node.end_byte],
                    "memory_type": "code"
                })
            for child in node.children:
                walk(child)

        walk(root)
        return symbols

    def extract_symbols(self, filepath: str) -> List[Dict[str, Any]]:
        """Extract symbols from a file using Tree-sitter (fallback to AST for Python)."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                source = f.read()
        except Exception:
            return []

        lang = self._get_language_for_file(filepath)
        if not lang:
            return []

        # Special case: Python uses AST for more reliable extraction
        if lang == "python":
            return self._extract_symbols_python(source)

        parser = self.parsers.get(lang)
        if not parser:
            return []

        try:
            return self._extract_symbols_tree_sitter(parser, source)
        except Exception:
            return []

    def ingest_codebase(self, directory: str, max_files: int = 1000):
        """Recursively ingest all supported code files in a directory."""
        count = 0
        for root, dirs, files in os.walk(directory):
            for file in files:
                filepath = os.path.join(root, file)
                lang = self._get_language_for_file(filepath)
                if lang is None:
                    continue
                if count >= max_files:
                    break
                symbols = self.extract_symbols(filepath)
                for sym in symbols:
                    self.memory.store(sym["text"])
                count += 1
                if count >= max_files:
                    break
