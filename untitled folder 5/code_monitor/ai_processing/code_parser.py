import logging
import ast
import re
import httpx
from typing import List, Dict, Any, Optional
from code_monitor.config import settings

logger = logging.getLogger(__name__)

class CodeParser:
    """Parse code files to extract functions, classes, methods, etc."""
    
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = settings.OLLAMA_MODEL
        
        # Regex patterns for detecting code structures in different languages
        self.patterns = {
            "python": {
                "function": r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                "class": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\(|\:)",
                "method": r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(self"
            },
            "javascript": {
                "function": r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                "class": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\{|extends)",
                "method": r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*\{"
            },
            "typescript": {
                "function": r"function\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:<.*>)?\s*\(",
                "class": r"class\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:\{|extends|implements)",
                "method": r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\([^)]*\)\s*(?::\s*[a-zA-Z_<>]*)?\s*\{"
            },
            # New patterns for HTML
            "html": {
                "element": r"<([a-zA-Z][a-zA-Z0-9]*)[^>]*>",
                "script": r"<script[^>]*>([^<]*)</script>",
                "style": r"<style[^>]*>([^<]*)</style>"
            },
            # New patterns for CSS
            "css": {
                "rule": r"([a-zA-Z0-9_\-\.#][^{]*)\s*\{",
                "media": r"@media\s+([^{]*)\s*\{",
                "import": r"@import\s+([^;]*);"
            },
            # New patterns for JSON
            "json": {
                "object": r"\"([a-zA-Z_][a-zA-Z0-9_]*)\"\s*:",
            },
            # New patterns for Markdown
            "markdown": {
                "heading": r"^(#{1,6})\s+(.+)$",
                "list": r"^(\*|-|\+|[0-9]+\.)\s+(.+)$",
                "codeblock": r"```([a-zA-Z0-9]*)\n"
            },
            # New patterns for unknown/generic code
            "unknown": {
                "function_like": r"(?:function|def|void|public|private)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
                "block": r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\{",
                "declaration": r"(?:var|let|const|int|float|double|string|bool)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
            }
        }
    
    async def parse_code(self, content: str, language: str) -> List[Dict[str, Any]]:
        """
        Parse code to extract structures like functions, classes, etc.
        
        Args:
            content: The code content to parse
            language: The programming language of the code
            
        Returns:
            A list of dictionaries containing information about each code structure
        """
        if language == "python":
            return self._parse_python(content)
        
        # For other languages, we'll use regex-based parsing
        # In a more complete implementation, you might use language-specific parsers
        # like esprima for JavaScript, TypeScript compiler API for TypeScript, etc.
        return self._parse_with_regex(content, language)
    
    def _parse_python(self, content: str) -> List[Dict[str, Any]]:
        """Parse Python code using the ast module"""
        structures = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's a method (inside a class)
                    is_method = False
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef) and node in parent.body:
                            is_method = True
                            break
                    
                    structure_type = "method" if is_method else "function"
                    
                    # Get the code for this function
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    function_lines = content.split('\n')[start_line:end_line]
                    function_code = '\n'.join(function_lines)
                    
                    structures.append({
                        "type": structure_type,
                        "name": node.name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": function_code
                    })
                    
                elif isinstance(node, ast.ClassDef):
                    # Get the code for this class
                    start_line = node.lineno - 1
                    end_line = node.end_lineno
                    class_lines = content.split('\n')[start_line:end_line]
                    class_code = '\n'.join(class_lines)
                    
                    structures.append({
                        "type": "class",
                        "name": node.name,
                        "start_line": start_line,
                        "end_line": end_line,
                        "content": class_code
                    })
        
        except SyntaxError as e:
            logger.error(f"Syntax error parsing Python code: {str(e)}")
            # If AST parsing fails, fall back to regex
            return self._parse_with_regex(content, "python")
        
        return structures
    
    def _parse_with_regex(self, content: str, language: str) -> List[Dict[str, Any]]:
        """Parse code using regex patterns"""
        structures = []
        lines = content.split('\n')
        
        if language not in self.patterns:
            logger.debug(f"No regex patterns for language: {language}")
            return []
        
        patterns = self.patterns[language]
        
        for structure_type, pattern in patterns.items():
            # Find all matches in the content
            matches = re.finditer(pattern, content)
            
            for match in matches:
                name = match.group(1)
                start_pos = match.start()
                
                # Find the start line
                start_line = content[:start_pos].count('\n')
                
                # Heuristic to find the end of this code block
                # This is a simplification - a real parser would be more accurate
                # We'll try to find the matching closing brace or the next structure
                end_line = self._find_structure_end(lines, start_line)
                
                # Extract the code content
                structure_lines = lines[start_line:end_line]
                structure_content = '\n'.join(structure_lines)
                
                structures.append({
                    "type": structure_type,
                    "name": name,
                    "start_line": start_line,
                    "end_line": end_line,
                    "content": structure_content
                })
        
        return structures
    
    def _find_structure_end(self, lines: List[str], start_line: int) -> int:
        """Find the end line of a code block using simple indentation heuristic"""
        if start_line >= len(lines):
            return start_line + 1
            
        # Get the indentation of the first line
        start_indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        
        # Look for a line with the same or less indentation
        for i in range(start_line + 1, len(lines)):
            # Skip empty lines
            if not lines[i].strip():
                continue
                
            line_indent = len(lines[i]) - len(lines[i].lstrip())
            if line_indent <= start_indent:
                return i
        
        return len(lines)
    
    async def extract_code_features(self, content: str, language: str) -> Dict[str, Any]:
        """Enhanced code feature extraction using Ollama"""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.base_url}/api/chat",
                    json={
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": "You are an expert code analyst focusing on detailed code understanding and documentation."},
                            {"role": "user", "content": f"Analyze this {language} code and extract key features, return ONLY valid JSON:\n\n{content}"}
                        ],
                        "temperature": 0.1,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json().get("message", {}).get("content", "")
                    if not result:
                        return {"error": "Empty response from code analysis"}
                        
                    # Clean up the response
                    result = result.strip()
                    
                    # Remove any markdown formatting
                    if "```json" in result:
                        result = result.split("```json")[1].split("```")[0]
                    elif "```" in result:
                        result = result.split("```")[1].split("```")[0]
                    
                    # Remove any non-JSON text before/after
                    result = re.sub(r'^[^{]*', '', result)
                    result = re.sub(r'[^}]*$', '', result)
                    result = result.strip()
                    
                    try:
                        # First try normal parsing
                        features = json.loads(result)
                        if not isinstance(features, dict):
                            raise ValueError("Parsed result is not a dictionary")
                        return features
                    except (json.JSONDecodeError, ValueError) as e:
                        logger.error(f"JSON parse error: {str(e)}")
                        logger.debug(f"Problem JSON: {result}")
                        
                        try:
                            # Try to recover by removing problematic characters
                            cleaned = ''.join(char for char in result if char.isprintable()).strip()
                            features = json.loads(cleaned)
                            if not isinstance(features, dict):
                                raise ValueError("Cleaned result is not a dictionary")
                            return features
                        except (json.JSONDecodeError, ValueError):
                            logger.error("Failed to recover JSON after cleaning")
                            return {
                                "error": "Failed to parse code analysis result",
                                "raw_content": result[:200]  # Include start of raw content for debugging
                            }
                else:
                    return {
                        "error": f"API error: {response.status_code}",
                        "message": response.text
                    }
                    
        except Exception as e:
            logger.error(f"Error in code feature extraction: {str(e)}")
            return {
                "error": f"Code analysis failed: {str(e)}",
                "language": language
            }