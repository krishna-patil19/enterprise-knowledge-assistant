# Specialized Parsers for Enterprise Engineering Knowledge Assistant
# File: backend/ingestion/parsers.py

import re
import ast
import os
from typing import List, Dict, Any, Tuple, Optional, Set

# --- Simple Token Estimator ---
def estimate_tokens(text: str) -> int:
    """Estimates the number of tokens in a text block (approx 4 chars per token)."""
    return max(1, len(text) // 4)

# --- 1. SQL Parser ---
class SQLParser:
    """
    Parses SQL scripts, splitting them query-by-query (structure-aware chunking).
    Extracts table names, joins, engine types, and specific ClickHouse/analytic functions.
    """
    @staticmethod
    def parse(content: str, file_name: str) -> List[Dict[str, Any]]:
        # Regex to split on semicolons while avoiding split on semicolons inside quotes
        # For a robust local parser, we can use a simpler line-by-line assembly that respects statements
        raw_queries = []
        current_query = []
        
        for line in content.splitlines():
            # Skip pure comment lines for partitioning queries, but keep them inside query chunks if helpful
            trimmed = line.strip()
            if not trimmed:
                continue
            current_query.append(line)
            if trimmed.endswith(";"):
                raw_queries.append("\n".join(current_query))
                current_query = []
        if current_query:
            raw_queries.append("\n".join(current_query))

        chunks = []
        for idx, query in enumerate(raw_queries):
            query_trimmed = query.strip()
            if not query_trimmed:
                continue
                
            # Extract metadata
            tables = SQLParser.extract_tables(query_trimmed)
            functions = SQLParser.extract_functions(query_trimmed)
            engines = SQLParser.extract_engines(query_trimmed)
            
            chunk_id = f"{file_name}_query_{idx + 1}"
            
            chunks.append({
                "id": chunk_id,
                "content": query_trimmed,
                "type": "sql",
                "token_count": estimate_tokens(query_trimmed),
                "metadata": {
                    "query_index": idx + 1,
                    "tables": list(tables),
                    "functions": list(functions),
                    "engines": list(engines),
                    "file_name": file_name
                }
            })
        return chunks

    @staticmethod
    def extract_tables(sql: str) -> Set:
        # Matches typical patterns: FROM table, JOIN table, TABLE table, INTO table, VIEW table, TO table
        tables = set()
        # Case insensitive regexes
        patterns = [
            r'(?i)\bfrom\s+([a-zA-Z0-9_\.]+)',
            r'(?i)\bjoin\s+([a-zA-Z0-9_\.]+)',
            r'(?i)\binto\s+([a-zA-Z0-9_\.]+)',
            r'(?i)\btable\s+([a-zA-Z0-9_\.]+)',
            r'(?i)\bview\s+([a-zA-Z0-9_\.]+)',
            r'(?i)\bto\s+([a-zA-Z0-9_\.]+)'
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, sql):
                table_name = match.group(1).lower()
                # Clean table name from ClickHouse modifiers (e.g. ON CLUSTER name)
                table_name = table_name.split()[0]
                tables.add(table_name)
        return tables

    @staticmethod
    def extract_functions(sql: str) -> Set:
        # Detect ClickHouse/general SQL functions, especially aggregate states
        functions = set()
        patterns = [
            r'\b([a-zA-Z0-9_]+State)\b', # e.g. sumState, avgState, uniqState
            r'\b([a-zA-Z0-9_]+Merge)\b', # e.g. sumMerge, uniqMerge
            r'\b(toYYYYMM|toDateTime|toStartOfMonth|uniq|uniqCombined|uniqHLL12)\b'
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, sql):
                functions.add(match.group(1))
        return functions

    @staticmethod
    def extract_engines(sql: str) -> Set:
        # ClickHouse table engines
        engines = set()
        pattern = r'(?i)\bengine\s*=\s*([a-zA-Z0-9_]+MergeTree|[a-zA-Z0-9_]+Log|[a-zA-Z0-9_]+Tree)\b'
        for match in re.finditer(pattern, sql):
            engines.add(match.group(1))
        return engines


# --- 2. Python AST Parser ---
class PythonASTVisitor(ast.NodeVisitor):
    """AST Visitor to extract classes, functions, docstrings, imports and embedded tables."""
    def __init__(self, content_lines: List[str]):
        self.content_lines = content_lines
        self.chunks = []
        self.imports = []
        self.current_class = None

    def visit_Import(self, node):
        for name in node.names:
            self.imports.append(name.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module:
            self.imports.append(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node):
        prev_class = self.current_class
        self.current_class = node.name
        
        # Slices class body content
        start_line = node.lineno - 1
        # Find end line (ast has node.end_lineno in Python 3.8+)
        end_line = getattr(node, "end_lineno", len(self.content_lines))
        class_content = "\n".join(self.content_lines[start_line:end_line])
        
        docstring = ast.get_docstring(node) or ""
        
        self.chunks.append({
            "name": node.name,
            "type": "class",
            "content": class_content,
            "start_line": start_line + 1,
            "end_line": end_line,
            "docstring": docstring,
            "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
        })
        
        self.generic_visit(node)
        self.current_class = prev_class

    def visit_FunctionDef(self, node):
        # We only treat it as an independent function chunk if it's top-level
        # Classes are already chunked, but we can capture functions inside classes if we want.
        # Let's chunk top-level functions and class methods independently for modular retrieval!
        start_line = node.lineno - 1
        end_line = getattr(node, "end_lineno", len(self.content_lines))
        func_content = "\n".join(self.content_lines[start_line:end_line])
        
        docstring = ast.get_docstring(node) or ""
        
        self.chunks.append({
            "name": node.name,
            "type": "function",
            "class_parent": self.current_class,
            "content": func_content,
            "start_line": start_line + 1,
            "end_line": end_line,
            "docstring": docstring
        })
        self.generic_visit(node)


class PythonParser:
    """
    Parses Python source code using AST (Abstract Syntax Tree).
    Splits content into class and function-level chunks (structure-aware chunking).
    Extracts imports, docstrings, classes, functions, and string-contained table/query patterns.
    """
    @staticmethod
    def parse(content: str, file_name: str) -> List[Dict[str, Any]]:
        lines = content.splitlines()
        try:
            tree = ast.parse(content)
            visitor = PythonASTVisitor(lines)
            visitor.visit(tree)
            
            chunks = []
            
            # If no classes or functions were found, treat the whole file as a single module chunk
            if not visitor.chunks:
                chunks.append({
                    "id": f"{file_name}_module",
                    "content": content,
                    "type": "python",
                    "token_count": estimate_tokens(content),
                    "metadata": {
                        "name": file_name,
                        "type": "module",
                        "imports": visitor.imports,
                        "tables": list(PythonParser.scan_for_tables(content)),
                        "file_name": file_name
                    }
                })
                return chunks
                
            for idx, item in enumerate(visitor.chunks):
                chunk_id = f"{file_name}_{item['type']}_{item['name']}"
                
                # Scan chunk content for DB table references
                tables = PythonParser.scan_for_tables(item["content"])
                
                metadata = {
                    "name": item["name"],
                    "type": item["type"],
                    "start_line": item["start_line"],
                    "end_line": item["end_line"],
                    "docstring": item["docstring"],
                    "imports": visitor.imports,
                    "tables": list(tables),
                    "file_name": file_name
                }
                
                if item.get("class_parent"):
                    metadata["class_parent"] = item["class_parent"]
                if item.get("methods"):
                    metadata["methods"] = item["methods"]
                    
                chunks.append({
                    "id": chunk_id,
                    "content": item["content"],
                    "type": "python",
                    "token_count": estimate_tokens(item["content"]),
                    "metadata": metadata
                })
                
            return chunks
            
        except SyntaxError:
            # Fallback to general chunking if AST parsing fails
            return PythonParser.fallback_parse(content, file_name)

    @staticmethod
    def scan_for_tables(text: str) -> Set:
        # Scans strings inside python files for ClickHouse or relational schemas
        # e.g., "analytics.raw_sales" or "analytics.sales_aggregates_local"
        tables = set()
        pattern = r'\b(analytics\.[a-zA-Z0-9_]+)\b'
        for match in re.finditer(pattern, text):
            tables.add(match.group(1).lower())
        # Also catch ClickHouse tables without dots if they match common nouns
        common_nouns = ['raw_sales', 'sales_aggregates_local', 'mv_sales_aggregates']
        for noun in common_nouns:
            if noun in text.lower():
                tables.add(f"analytics.{noun}")
        return tables

    @staticmethod
    def fallback_parse(content: str, file_name: str) -> List[Dict[str, Any]]:
        # Splitting by chunks of 30 lines
        lines = content.splitlines()
        chunks = []
        chunk_size = 40
        overlap = 10
        
        for idx in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[idx:idx + chunk_size]
            if not chunk_lines:
                break
            chunk_content = "\n".join(chunk_lines)
            chunk_id = f"{file_name}_fallback_chunk_{idx // (chunk_size - overlap) + 1}"
            chunks.append({
                "id": chunk_id,
                "content": chunk_content,
                "type": "python",
                "token_count": estimate_tokens(chunk_content),
                "metadata": {
                    "file_name": file_name,
                    "fallback": True,
                    "start_line": idx + 1,
                    "end_line": idx + len(chunk_lines),
                    "tables": list(PythonParser.scan_for_tables(chunk_content))
                }
            })
        return chunks


# --- 3. Markdown Parser ---
class MarkdownParser:
    """
    Parses Markdown files structure-aware.
    Splits documents strictly by heading boundaries (#, ##, ###).
    """
    @staticmethod
    def parse(content: str, file_name: str) -> List[Dict[str, Any]]:
        lines = content.splitlines()
        chunks = []
        
        current_header = "Overview"
        current_level = 1
        current_lines = []
        chunk_idx = 1
        
        # Matching #, ##, ###, ####, etc.
        header_pattern = re.compile(r'^(#{1,6})\s+(.+)$')
        
        for line in lines:
            match = header_pattern.match(line)
            if match:
                # If we have collected content under the previous header, write it out
                if current_lines:
                    chunk_content = "\n".join(current_lines).strip()
                    if chunk_content:
                        chunk_id = f"{file_name}_section_{chunk_idx}"
                        chunks.append({
                            "id": chunk_id,
                            "content": chunk_content,
                            "type": "docs",
                            "token_count": estimate_tokens(chunk_content),
                            "metadata": {
                                "header": current_header,
                                "header_level": current_level,
                                "section_index": chunk_idx,
                                "file_name": file_name
                            }
                        })
                        chunk_idx += 1
                
                level = len(match.group(1))
                header_text = match.group(2).strip()
                current_header = header_text
                current_level = level
                current_lines = [line] # Include the header in the chunk
            else:
                current_lines.append(line)
                
        # Write remaining lines
        if current_lines:
            chunk_content = "\n".join(current_lines).strip()
            if chunk_content:
                chunk_id = f"{file_name}_section_{chunk_idx}"
                chunks.append({
                    "id": chunk_id,
                    "content": chunk_content,
                    "type": "docs",
                    "token_count": estimate_tokens(chunk_content),
                    "metadata": {
                        "header": current_header,
                        "header_level": current_level,
                        "section_index": chunk_idx,
                        "file_name": file_name
                    }
                })
                
        return chunks


# --- 4. YAML / JSON Config Parser ---
class ConfigParser:
    """Parses JSON and YAML configuration files, extracting service names, table mappings, etc."""
    @staticmethod
    def parse(content: str, file_name: str) -> List[Dict[str, Any]]:
        # Return a single chunk for the config as it is highly contextual
        # We can extract references and metadata
        tables = set()
        services = set()
        
        # Scan content for clickhouse tables or file pathways
        table_pattern = r'\b(analytics\.[a-zA-Z0-9_]+)\b'
        for match in re.finditer(table_pattern, content):
            tables.add(match.group(1).lower())
            
        file_path_pattern = r'\b([a-zA-Z0-9_]+/[a-zA-Z0-9_]+\.(?:py|sql|md|yaml|json))\b'
        for match in re.finditer(file_path_pattern, content):
            services.add(match.group(1))
            
        metadata = {
            "file_name": file_name,
            "tables": list(tables),
            "linked_assets": list(services)
        }
        
        # If it's YAML, try to load its headers as key-values
        if file_name.endswith((".yaml", ".yml")):
            try:
                import yaml
                data = yaml.safe_load(content)
                if isinstance(data, dict):
                    if "service" in data and isinstance(data["service"], dict):
                        metadata["service_name"] = data["service"].get("name")
                    if "clickhouse" in data and isinstance(data["clickhouse"], dict):
                        metadata["clickhouse_host"] = data["clickhouse"].get("host")
            except Exception:
                pass
                
        # Truncate content to avoid LLM context window explosion for giant JSONs
        safe_content = content[:80000]
                
        return [{
            "id": f"{file_name}_config",
            "content": safe_content,
            "type": "config",
            "token_count": estimate_tokens(safe_content),
            "metadata": metadata
        }]


# --- 5. PDF Parser ---
class PDFParser:
    """Parses PDF files using pypdf, creating paragraph-sized text chunks."""
    @staticmethod
    def parse(file_path: str, file_name: str) -> List[Dict[str, Any]]:
        chunks = []
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            
            # Simple page-by-page chunking
            for idx, page in enumerate(reader.pages):
                text = page.extract_text()
                if not text or not text.strip():
                    continue
                    
                chunk_id = f"{file_name}_page_{idx + 1}"
                chunks.append({
                    "id": chunk_id,
                    "content": text.strip(),
                    "type": "pdf",
                    "token_count": estimate_tokens(text),
                    "metadata": {
                        "page_number": idx + 1,
                        "total_pages": len(reader.pages),
                        "file_name": file_name
                    }
                })
        except Exception as e:
            # Fallback to direct text read if pdf reading fails
            fallback_text = f"PDF Ingestion Error on {file_name}: {str(e)}"
            chunks.append({
                "id": f"{file_name}_error",
                "content": fallback_text,
                "type": "pdf",
                "token_count": estimate_tokens(fallback_text),
                "metadata": {"file_name": file_name, "error": True}
            })
        return chunks


# --- Global Parser Factory ---
def parse_file(file_path: str) -> List[Dict[str, Any]]:
    """Detects file type extension and invokes the appropriate specialized parser."""
    file_name = os.path.basename(file_path)
    ext = os.path.splitext(file_name)[1].lower()
    
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    if ext == ".sql":
        return SQLParser.parse(content, file_name)
    elif ext == ".py":
        return PythonParser.parse(content, file_name)
    elif ext == ".md":
        return MarkdownParser.parse(content, file_name)
    elif ext in [".yaml", ".yml", ".json"]:
        return ConfigParser.parse(content, file_name)
    elif ext == ".pdf":
        # PDFs need the absolute file path, not just string content
        return PDFParser.parse(file_path, file_name)
    else:
        # Fallback text chunker (sliding character window for safety against massive files)
        chunks = []
        chunk_chars = 4000
        overlap = 500
        text = content.strip()
        if not text:
            return []
            
        for idx in range(0, len(text), chunk_chars - overlap):
            chunk_content = text[idx:idx + chunk_chars]
            chunk_id = f"{file_name}_text_{idx // (chunk_chars - overlap) + 1}"
            chunks.append({
                "id": chunk_id,
                "content": chunk_content,
                "type": "docs",
                "token_count": estimate_tokens(chunk_content),
                "metadata": {"file_name": file_name, "fallback": True}
            })
        return chunks
