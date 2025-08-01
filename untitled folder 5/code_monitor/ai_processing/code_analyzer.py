import json
import logging
import numpy as np
import asyncio
import re
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy import select, func, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import or_

from code_monitor.db.models import (
    Repository, RepositoryFile, Commit, File, CodeSnippet,
    QAGRepositoryFile, QAGCodeSnippet, QAGFile, QAGRepository
)
from code_monitor.ai_processing.embedding import EmbeddingGenerator
from code_monitor.ai_processing.code_parser import CodeParser
from code_monitor.config import settings
from code_monitor.utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

class CodeAnalyzer:
    """
    Analyze code repositories using semantic understanding of the code via embeddings.
    This class uses the vector embeddings to understand code structure, dependencies,
    and relationships between different parts of the codebase.
    """
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.code_parser = CodeParser()
        self.ollama_client = None
        # Use CodeLlama for code-specific tasks if available, otherwise fall back to default model
        self.code_llama_model = settings.OLLAMA_MODEL
        # Check if a specific Code Llama model is defined in environment
        if hasattr(settings, 'CODELLAMA_MODEL') and settings.CODELLAMA_MODEL:
            self.code_llama_model = settings.CODELLAMA_MODEL
        
    async def initialize(self):
        """Initialize async components with robust error handling"""
        try:
            logger.info("Initializing CodeAnalyzer components")
            # Initialize Ollama client with retries
            retry_count = 0
            max_retries = 3
            while retry_count < max_retries:
                try:
                    self.ollama_client = await OllamaClient.create()
                    logger.info(f"Successfully initialized Ollama client with model: {self.code_llama_model}")
                    break
                except Exception as e:
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to initialize Ollama client after {max_retries} attempts: {str(e)}")
                        # Create a basic client without initialization that will attempt to reconnect on use
                        self.ollama_client = OllamaClient(self.code_llama_model)
                    else:
                        logger.warning(f"Ollama client initialization attempt {retry_count} failed: {str(e)}. Retrying...")
                        await asyncio.sleep(1)  # Wait a bit before retrying
            
            # Initialize other components as needed
            await self.embedding_generator.initialize()
            await self.code_parser.initialize()
            
        except Exception as e:
            logger.error(f"Error during CodeAnalyzer initialization: {str(e)}")
            # Set up a basic client that can be retried later
            if not self.ollama_client:
                self.ollama_client = OllamaClient(self.code_llama_model)
    
    async def analyze_repository(self, db: AsyncSession, repo_id: int, is_qag: bool = False) -> Dict[str, Any]:
        """
        Analyze an entire repository using code embeddings
        
        Args:
            db: Database session
            repo_id: Repository ID
            is_qag: Whether to use QAG-specific models
            
        Returns:
            Dictionary with repository analysis results
        """
        # Select the appropriate model classes based on repository type
        repo_file_model = QAGRepositoryFile if is_qag else RepositoryFile
        
        # Get repository info
        query = select(Repository).where(Repository.id == repo_id)
        result = await db.execute(query)
        repo = result.scalars().first()
        
        if not repo:
            logger.error(f"Repository with ID {repo_id} not found")
            return {"error": f"Repository with ID {repo_id} not found"}
        
        # Get all files with embeddings
        files_query = select(repo_file_model).where(
            repo_file_model.repository_id == repo_id,
            repo_file_model.embedding.is_not(None)
        )
        files_result = await db.execute(files_query)
        files = files_result.scalars().all()
        
        if not files:
            logger.warning(f"No files with embeddings found for repository {repo.name} (ID: {repo_id})")
            return {
                "repository": repo.name,
                "error": "No files with embeddings found. Please generate embeddings first."
            }
        
        # File statistics
        file_count = len(files)
        language_counts = self._count_languages(files)
        
        # Find key concepts in the codebase by clustering embeddings
        key_concepts = await self._identify_key_concepts(files, top_n=5)
        
        # Find core modules by analyzing dependencies between files
        core_modules = await self._identify_core_modules(db, files)
        
        # Generate a high-level summary of the repository using Code Llama
        repo_summary = await self._generate_repository_summary(db, repo, files)
        
        # Analyze application features with Code Llama enhanced analysis
        application_features = await self.analyze_application_features(db, repo_id, files, is_qag)
        
        # Advanced code architecture analysis using Code Llama
        architecture_analysis = await self.analyze_code_architecture(files)
        
        return {
            "repository_name": repo.name,
            "repository_id": repo_id,
            "file_count": file_count,
            "languages": language_counts,
            "key_concepts": key_concepts,
            "core_modules": core_modules,
            "application_features": application_features,
            "summary": repo_summary,
            "architecture_analysis": architecture_analysis
        }
    
    async def analyze_application_features(
        self, 
        db: AsyncSession, 
        repo_id: int, 
        files: List[Any] = None,
        is_qag: bool = False
    ) -> Dict[str, Any]:
        """
        Analyze the application features implemented in the repository with enhanced UI feature detection
        
        Args:
            db: Database session
            repo_id: Repository ID
            files: Optional list of files (to avoid another query if already loaded)
            is_qag: Whether to use QAG-specific models
            
        Returns:
            Dictionary with application features analysis including detailed UI functionality
        """
        # Select the appropriate model classes
        repo_file_model = QAGRepositoryFile if is_qag else RepositoryFile
        code_snippet_model = QAGCodeSnippet if is_qag else CodeSnippet
        
        # Get files if not provided
        if not files:
            files_query = select(repo_file_model).where(
                repo_file_model.repository_id == repo_id,
                repo_file_model.embedding.is_not(None)
            )
            files_result = await db.execute(files_query)
            files = files_result.scalars().all()
            
        if not files:
            return {"error": "No files found for feature analysis"}
            
        # Get code snippets
        snippets_query = (
            select(code_snippet_model)
            .join(repo_file_model, code_snippet_model.file_id == repo_file_model.id)
            .where(
                repo_file_model.repository_id == repo_id,
                code_snippet_model.embedding.is_not(None)
            )
        )
        snippets_result = await db.execute(snippets_query)
        snippets = snippets_result.scalars().all()
        
        # If no code snippets with embeddings are found, extract snippets directly from files
        extracted_snippets = []
        if not snippets:
            logger.info(f"No code snippets with embeddings found for repository ID {repo_id}. Extracting snippets directly from files.")
            for file in files:
                if file.content:
                    try:
                        language = file.language if file.language else self._detect_language_from_path(file.file_path)
                        parsed_snippets = await self.code_parser.parse_code(file.content, language)
                        for snippet in parsed_snippets:
                            extracted_snippets.append({
                                "id": f"extract_{file.id}_{snippet['name']}",  # Generate temporary ID for extracted snippets
                                "file_id": file.id,
                                "snippet_type": snippet["type"],
                                "name": snippet["name"],
                                "content": snippet["content"],
                                "start_line": snippet["start_line"],
                                "end_line": snippet["end_line"]
                            })
                    except Exception as e:
                        logger.warning(f"Error extracting snippets from file {file.file_path}: {str(e)}")
        
        # Core feature categories with enhanced UI features
        feature_categories = {
            "api": [],
            "database": [],
            "authentication": [],
            "user_interface": [],
            "ui_components": [],  # New category for specific UI components
            "ui_functionality": [],  # New category for UI interactivity features
            "data_processing": [],
            "integration": [],
            "configuration": [],
            "utility": []
        }
        
        # Enhanced feature indicators with more UI specific patterns
        feature_indicators = {
            "api": [
                "api", "endpoint", "route", "rest", "graphql", "request", "response",
                "@app.route", "router", "fastapi", "express", "fetch", "axios"
            ],
            "database": [
                "database", "db", "sql", "query", "model", "schema", "repository", 
                "migration", "orm", "entity", "sqlalchemy", "sequelize", "mongoose",
                "persist", "storage", "save", "retrieve"
            ],
            "authentication": [
                "auth", "login", "logout", "password", "token", "jwt", "oauth",
                "permission", "role", "user", "session", "credential", "secure",
                "accesstoken", "refreshtoken", "verify", "validate"
            ],
            "user_interface": [
                "ui", "component", "view", "template", "render", "display", "html",
                "css", "style", "frontend", "react", "vue", "angular", "streamlit",
                "interface", "layout", "responsive", "mobile"
            ],
            "ui_components": [
                "button", "input", "form", "modal", "dialog", "menu", "nav", "navbar",
                "sidebar", "tab", "panel", "card", "grid", "table", "list", "dropdown",
                "select", "checkbox", "radio", "toggle", "slider", "chart", "graph",
                "dashboard", "widget", "container", "header", "footer", "section"
            ],
            "ui_functionality": [
                "click", "submit", "change", "select", "toggle", "drag", "drop", "hover",
                "scroll", "resize", "filter", "sort", "search", "pagination", "animation", 
                "transition", "validation", "feedback", "notification", "alert", "toast",
                "datepicker", "colorpicker", "upload", "download", "print", "export"
            ],
            "data_processing": [
                "process", "analyze", "transform", "parse", "extract", "filter",
                "map", "reduce", "compute", "calculate", "algorithm", "format",
                "convert", "normalize", "aggregate", "summarize", "visualize"
            ],
            "integration": [
                "webhook", "callback", "client", "service", "integration", "connector",
                "api_key", "third-party", "external", "github", "provider", "jira",
                "slack", "twitter", "facebook", "google", "aws", "azure", "firebase"
            ],
            "configuration": [
                "config", "setting", "environment", "env", "option", "parameter",
                "constant", "variable", "setup", "initialize", ".env", "dotenv",
                "profile", "preference", "mode", "feature flag", "toggle"
            ],
            "utility": [
                "util", "helper", "common", "shared", "formatter", "converter",
                "validator", "logger", "decorator", "middleware", "mixin", "hook",
                "plugin", "extension", "library", "toolkit", "framework"
            ]
        }
        
        # Extract README content for repository purpose/description
        readme_content = None
        repository_purpose = None
        repository_description = None
        
        for file in files:
            if file.file_path.lower() == "readme.md" or file.file_path.lower().endswith("/readme.md"):
                readme_content = file.content
                
                # Extract purpose from README
                if readme_content:
                    # Look for common patterns that describe purpose
                    purpose_patterns = [
                        r"#\s*([^\n#]+)",  # First heading
                        r"## (?:About|Overview|Introduction|Purpose|Description)\s*\n\s*([^\n#]+)",  # About/Overview section
                        r"This (?:project|app|application|tool) (?:is|provides|offers|allows)(.*?)\.",  # Purpose statement
                    ]
                    
                    for pattern in purpose_patterns:
                        matches = re.findall(pattern, readme_content, re.IGNORECASE)
                        if matches:
                            repository_purpose = matches[0].strip()
                            break
                    
                    # Extract a more detailed description
                    desc_patterns = [
                        r"## (?:About|Overview|Introduction|Description)\s*\n(.*?)(?:\n##|\Z)",  # About/Overview section content
                        r"## Features\s*\n(.*?)(?:\n##|\Z)",  # Features section
                    ]
                    
                    for pattern in desc_patterns:
                        matches = re.findall(pattern, readme_content, re.DOTALL)
                        if matches:
                            repository_description = matches[0].strip()
                            break
        
        # Look for specific UI frameworks to determine UI capabilities
        ui_frameworks = {
            "react": False,
            "vue": False,
            "angular": False,
            "svelte": False,
            "bootstrap": False,
            "material-ui": False,
            "tailwind": False,
            "chakra-ui": False,
            "ant-design": False
        }
        
        # Deep scan for UI frameworks in package.json
        for file in files:
            if file.file_path.lower().endswith("package.json") and file.content:
                try:
                    package_data = self._parse_package_json(file.content)
                    dependencies = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
                    
                    # Check for UI frameworks
                    if "react" in dependencies:
                        ui_frameworks["react"] = True
                    if "vue" in dependencies:
                        ui_frameworks["vue"] = True
                    if "angular" in dependencies or "@angular/core" in dependencies:
                        ui_frameworks["angular"] = True
                    if "svelte" in dependencies:
                        ui_frameworks["svelte"] = True
                    if "bootstrap" in dependencies:
                        ui_frameworks["bootstrap"] = True
                    if "@material-ui/core" in dependencies or "@mui/material" in dependencies:
                        ui_frameworks["material-ui"] = True
                    if "tailwindcss" in dependencies:
                        ui_frameworks["tailwind"] = True
                    if "@chakra-ui/react" in dependencies:
                        ui_frameworks["chakra-ui"] = True
                    if "antd" in dependencies:
                        ui_frameworks["ant-design"] = True
                        
                except (json.JSONDecodeError, AttributeError):
                    logger.warning(f"Failed to parse package.json in {file.file_path}")
        
        # Extract file features first from file paths and basic content
        for file in files:
            file_path = file.file_path.lower()
            file_content_sample = file.content[:10000] if file.content else ""
            
            # Look for React component files
            if file_path.endswith(".jsx") or file_path.endswith(".tsx"):
                if "export" in file_content_sample and ("function" in file_content_sample or "class" in file_content_sample):
                    # This is likely a React component
                    component_name = file_path.split("/")[-1].split(".")[0]
                    
                    # Extract UI component features with more details
                    component_details = self._extract_ui_component_details(file_path, file_content_sample)
                    
                    feature_categories["ui_components"].append({
                        "type": "component",
                        "id": file.id,
                        "name": component_name,
                        "file_path": file.file_path,
                        "relevance": "high",
                        "details": component_details
                    })
            
            # Look for Vue components
            if file_path.endswith(".vue"):
                component_name = file_path.split("/")[-1].split(".")[0]
                
                # Extract Vue component details
                component_details = self._extract_ui_component_details(file_path, file_content_sample)
                
                feature_categories["ui_components"].append({
                    "type": "component",
                    "id": file.id,
                    "name": component_name,
                    "file_path": file.file_path,
                    "relevance": "high",
                    "details": component_details
                })
            
            # Look for CSS/SCSS files for UI styling information
            if file_path.endswith((".css", ".scss", ".sass", ".less")):
                styling_features = self._extract_styling_features(file_content_sample)
                
                if styling_features:
                    feature_categories["ui_functionality"].append({
                        "type": "styling",
                        "id": file.id,
                        "name": file.file_path,
                        "relevance": "medium",
                        "details": styling_features
                    })
            
            # Check file path patterns for all feature categories
            for category, indicators in feature_indicators.items():
                # Check path first
                if any(ind in file_path for ind in indicators):
                    feature_categories[category].append({
                        "type": "file", 
                        "id": file.id,
                        "name": file.file_path,
                        "relevance": "high" if any(ind in file_path.split("/")[-1] for ind in indicators) else "medium"
                    })
                    continue
                
                # Check content for strong indicators
                if file_content_sample and any(f" {ind} " in f" {file_content_sample.lower()} " for ind in indicators):
                    feature_categories[category].append({
                        "type": "file",
                        "id": file.id,
                        "name": file.file_path,
                        "relevance": "medium"
                    })
        
        # Analyze all snippets (either from database or extracted)
        snippets_to_analyze = snippets if snippets else extracted_snippets
        
        # Extract features from snippets (more specific functionality)
        for snippet in snippets_to_analyze:
            snippet_name = snippet.name.lower() if hasattr(snippet, "name") else snippet["name"].lower()
            snippet_content = snippet.content if hasattr(snippet, "content") else snippet["content"]
            snippet_id = snippet.id if hasattr(snippet, "id") else snippet["id"]
            snippet_file_id = snippet.file_id if hasattr(snippet, "file_id") else snippet["file_id"]
            snippet_type = snippet.snippet_type if hasattr(snippet, "snippet_type") else snippet["snippet_type"]
            
            if not snippet_content:
                continue
            
            # For functions/methods, check if they represent UI event handlers
            if snippet_type in ("function", "method") and any(handler in snippet_name for handler in ["handle", "on", "click", "submit", "change", "select", "toggle"]):
                # This is likely a UI event handler
                feature_categories["ui_functionality"].append({
                    "type": "event_handler",
                    "id": snippet_id,
                    "name": f"{snippet_type}: {snippet_name}",
                    "file_id": snippet_file_id,
                    "relevance": "high",
                    "details": self._extract_ui_handler_details(snippet_content)
                })
            
            # For each standard feature category
            for category, indicators in feature_indicators.items():
                # Check name first (higher confidence)
                if any(ind in snippet_name for ind in indicators):
                    # If not already added with high relevance
                    existing = next((item for item in feature_categories[category] 
                                    if item["type"] == "snippet" and item["id"] == snippet_id), None)
                    if existing:
                        if existing["relevance"] != "high":
                            existing["relevance"] = "high"
                    else:
                        feature_categories[category].append({
                            "type": "snippet",
                            "id": snippet_id,
                            "name": f"{snippet_type}: {snippet_name}",
                            "file_id": snippet_file_id,
                            "relevance": "high"
                        })
                    continue
                
                # Check content for strong indicators with context awareness
                if any(f" {ind} " in f" {snippet_content.lower()} " for ind in indicators):
                    # Count occurrences for relevance score
                    occurrences = sum(snippet_content.lower().count(ind) for ind in indicators)
                    relevance = "high" if occurrences > 2 else "medium"
                    
                    # If not already added with equal or higher relevance
                    existing = next((item for item in feature_categories[category] 
                                    if item["type"] == "snippet" and item["id"] == snippet_id), None)
                    if existing:
                        if existing["relevance"] == "medium" and relevance == "high":
                            existing["relevance"] = "high"
                    else:
                        feature_categories[category].append({
                            "type": "snippet",
                            "id": snippet_id,
                            "name": f"{snippet_type}: {snippet_name}",
                            "file_id": snippet_file_id,
                            "relevance": relevance
                        })
        
        # Get detailed features for selected categories
        for category, items in feature_categories.items():
            if len(items) > 0:
                # Filter out duplicates and low relevance items if we have enough
                high_relevance = [item for item in items if item["relevance"] == "high"]
                medium_relevance = [item for item in items if item["relevance"] == "medium"]
                
                # Prioritize high relevance items, add medium if needed
                selected_items = high_relevance[:8]  # Take up to 8 high relevance items (increased from 5)
                if len(selected_items) < 5 and medium_relevance:  # If we have less than 5 high relevance items, add some medium ones
                    selected_items.extend(medium_relevance[:5 - len(selected_items)])
                    
                feature_categories[category] = selected_items

        # Identify primary application type
        app_types = {
            "Web API": 0,
            "Full Stack Web App": 0,
            "Command Line Tool": 0,
            "Data Analysis Pipeline": 0,
            "Dashboard Application": 0,
            "Integration Service": 0,
            "Configuration Manager": 0,
            "Other": 0
        }
        
        # Scoring rules for application types
        if len(feature_categories["api"]) > 3:
            app_types["Web API"] += 5
        if len(feature_categories["database"]) > 3:
            app_types["Web API"] += 2
            app_types["Full Stack Web App"] += 2
            app_types["Data Analysis Pipeline"] += 3
        if len(feature_categories["user_interface"]) > 3:
            app_types["Full Stack Web App"] += 5
            app_types["Dashboard Application"] += 3
        if len(feature_categories["ui_components"]) > 3:
            app_types["Dashboard Application"] += 5
            app_types["Full Stack Web App"] += 3
        if len(feature_categories["data_processing"]) > 3:
            app_types["Data Analysis Pipeline"] += 5
            app_types["Dashboard Application"] += 3
        if "dashboard" in repository_purpose.lower() if repository_purpose else False:
            app_types["Dashboard Application"] += 10
        
        # Look for dashboard-specific keywords in files and snippets
        dashboard_keywords = ["dashboard", "chart", "graph", "analytics", "monitoring", "visualization", "metrics"]
        for file in files:
            if file.content and any(keyword in file.content.lower() for keyword in dashboard_keywords):
                app_types["Dashboard Application"] += 2
        
        # Determine primary application type
        primary_app_type = max(app_types.items(), key=lambda x: x[1])[0]
        if app_types[primary_app_type] == 0:
            primary_app_type = "Other"
        
        # Prepare result with enhanced metadata
        return {
            "application_type": primary_app_type,
            "features": feature_categories,
            "file_count": len(files),
            "snippet_count": len(snippets_to_analyze),
            "readme_content": readme_content,
            "repository_purpose": repository_purpose,
            "repository_description": repository_description,
            "ui_frameworks": {k: v for k, v in ui_frameworks.items() if v}
        }
    
    async def summarize_code(self, code: str, language: str = None) -> str:
        """Generate a natural language summary of code using the Ollama client.
        
        Args:
            code: The code content to summarize
            language: Optional language hint
            
        Returns:
            A string summary of the code functionality
        """
        if not code or not isinstance(code, str):
            return "No valid code content provided"
            
        # Initialize Ollama client on demand if it's not already initialized
        if not self.ollama_client:
            try:
                logger.info("Attempting to initialize Ollama client for code summarization")
                self.ollama_client = await OllamaClient.create()
                logger.info("Successfully initialized Ollama client on demand")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {str(e)}")
                return f"Code analysis unavailable - could not initialize language model: {str(e)}."
            
        # If language not provided, try to detect it
        if not language:
            language = self._detect_code_language_simple(code)
            
        # Truncate code if it's too large
        if len(code) > 8000:
            code = code[:8000] + "\n\n... [truncated due to length] ...\n"
            
        try:
            messages = [
                {"role": "system", "content": "You are an expert code analyst. Provide a clear, concise summary of the code's purpose and functionality."},
                {"role": "user", "content": f"Summarize this {language} code in a few sentences:\n\n```{language}\n{code}\n```"}
            ]
            
            response = await self.ollama_client.generate(
                messages=messages,
                temperature=0.1,
                max_tokens=500
            )
            
            if not response or "choices" not in response:
                return "Failed to generate code summary"
                
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error summarizing code: {str(e)}")
            return f"Error analyzing code: {str(e)}"
    
    def _detect_code_language_simple(self, code: str) -> str:
        """Simple language detection based on keywords and syntax patterns"""
        if not code:
            return "unknown"
            
        patterns = {
            "python": ["def ", "import ", "class ", "if __name__ == ", "print(", ".py"],
            "javascript": ["function ", "const ", "let ", "var ", "() =>", ".js", ".jsx"],
            "typescript": ["interface ", "type ", ": string", ": number", ": boolean", ".ts", ".tsx"],
            "html": ["<!DOCTYPE", "<html>", "<div>", "<body>", "<head>", ".html"],
            "css": ["@media", "@keyframes", "margin:", "padding:", "color:", ".css"],
            "java": ["public class", "private ", "protected ", "void ", "String[]", ".java"],
            "c": ["#include", "int main(", "void ", "char *", "printf(", ".c"],
            "cpp": ["std::", "namespace", "template<", "class ", "cout <<", ".cpp", ".hpp"],
            "go": ["package ", "func ", "import (", "type ", "struct {", ".go"],
            "rust": ["fn ", "let mut", "impl ", "trait ", "match ", ".rs"],
            "php": ["<?php", "function ", "$", "echo ", "->", ".php"],
            "ruby": ["def ", "require ", "class ", "end", "puts ", ".rb"],
            "markdown": ["# ", "## ", "**", "- ", "[", ".md"]
        }
        
        scores = {}
        for lang, patterns_list in patterns.items():
            score = 0
            for pattern in patterns_list:
                if pattern in code:
                    score += 1
            scores[lang] = score
            
        # Get the language with the highest score
        if scores:
            max_lang = max(scores.items(), key=lambda x: x[1])
            if max_lang[1] > 0:  # Only if we have at least one match
                return max_lang[0]
                
        return "unknown"
    
    def _extract_ui_component_details(self, file_path: str, content: str) -> Dict[str, Any]:
        """Extract detailed information about UI components"""
        details = {
            "element_types": [],
            "interactivity": [],
            "state_management": False,
            "data_display": False,
            "form_handling": False
        }
        
        # Extract element types
        elements = ["div", "span", "input", "button", "form", "table", "ul", "li", "select", "option"]
        for element in elements:
            if f"<{element}" in content.lower():
                if element not in details["element_types"]:
                    details["element_types"].append(element)
        
        # Check for interactivity
        interactive_patterns = ["onClick", "onChange", "onSubmit", "addEventListener", "@click", "v-on", "bindValue"]
        for pattern in interactive_patterns:
            if pattern in content:
                interaction = pattern.replace("on", "").replace("add", "").replace("Event", "").replace("Listener", "")
                interaction = interaction.replace("@", "").replace("v-", "").replace("bind", "")
                if interaction and interaction not in details["interactivity"]:
                    details["interactivity"].append(interaction.lower())
        
        # Check for state management
        state_patterns = ["useState", "useReducer", "this.state", "createStore", "reactive", "ref"]
        details["state_management"] = any(pattern in content for pattern in state_patterns)
        
        # Check for data display
        data_display_patterns = ["map(", "v-for", "ngFor", "forEach", ".filter(", "display:"]
        details["data_display"] = any(pattern in content for pattern in data_display_patterns)
        
        # Check for form handling
        form_patterns = ["<form", "onSubmit", "handleSubmit", "validation", "input", "form-control"]
        details["form_handling"] = any(pattern in content for pattern in form_patterns)
        
        return details
    
    def _extract_ui_handler_details(self, content: str) -> Dict[str, Any]:
        """Extract details from UI event handlers"""
        details = {
            "event_type": "unknown",
            "updates_state": False,
            "api_calls": False,
            "validation": False,
            "data_processing": False
        }
        
        # Detect event type
        if "click" in content.lower():
            details["event_type"] = "click"
        elif "submit" in content.lower():
            details["event_type"] = "submit"
        elif "change" in content.lower():
            details["event_type"] = "change"
        elif "input" in content.lower():
            details["event_type"] = "input"
        
        # Check if it updates state
        state_update_patterns = ["setState", "useState", "this.state", "ref.value", "store.dispatch"]
        details["updates_state"] = any(pattern in content for pattern in state_update_patterns)
        
        # Check if it makes API calls
        api_call_patterns = ["fetch(", "axios.", "http.", ".get(", ".post(", ".put(", ".delete(", "api."]
        details["api_calls"] = any(pattern in content for pattern in api_call_patterns)
        
        # Check if it does validation
        validation_patterns = ["valid", "error", "check", "isValid", "validation", "validator"]
        details["validation"] = any(pattern in content.lower() for pattern in validation_patterns)
        
        # Check if it processes data
        processing_patterns = ["map", "filter", "reduce", "forEach", "transform", "convert", "parse"]
        details["data_processing"] = any(pattern in content.lower() for pattern in processing_patterns)
        
        return details
    
    def _extract_styling_features(self, content: str) -> Dict[str, Any]:
        """Extract styling features from CSS/SCSS files"""
        if not content:
            return {}
            
        styling = {
            "animations": [],
            "responsive_design": False,
            "dark_mode": False,
            "color_scheme": [],
            "custom_components": []
        }
        
        # Check for animations
        animation_patterns = ["@keyframes", "animation:", "transition:", "transform:"]
        for pattern in animation_patterns:
            if pattern in content:
                # Extract animation name if possible
                if pattern == "@keyframes":
                    matches = re.findall(r"@keyframes\s+([a-zA-Z0-9_-]+)", content)
                    for match in matches:
                        styling["animations"].append(match)
                else:
                    styling["animations"].append(pattern.replace(":", ""))
        
        # Check for responsive design
        responsive_patterns = ["@media", "max-width", "min-width", "screen and"]
        styling["responsive_design"] = any(pattern in content for pattern in responsive_patterns)
        
        # Check for dark mode
        dark_mode_patterns = [".dark-mode", ".dark", "[data-theme='dark']", "[data-theme=\"dark\"]"]
        styling["dark_mode"] = any(pattern in content for pattern in dark_mode_patterns)
        
        # Extract color scheme
        color_matches = re.findall(r"(#[0-9A-Fa-f]{3,6}|rgba?\([^)]+\))", content)
        styling["color_scheme"] = list(set(color_matches))[:5]  # Limit to 5 unique colors
        
        # Look for custom component classes
        custom_components = re.findall(r"\.([a-zA-Z][a-zA-Z0-9_-]+)\s*{", content)
        styling["custom_components"] = list(set(custom_components))[:8]  # Limit to 8 unique components
        
        return styling

    def _parse_package_json(self, content: str) -> Dict[str, Any]:
        """
        Safely parse package.json content with enhanced error handling
        
        Args:
            content: String content of package.json file
            
        Returns:
            Dictionary containing parsed package.json data
        """
        if not content or not isinstance(content, str):
            return {"dependencies": {}, "devDependencies": {}}
            
        try:
            # Try direct parsing first
            return json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse package.json: {str(e)}")
            
            try:
                # Clean up the content and retry
                # 1. Remove comments (both single and multi-line)
                content_no_comments = re.sub(r'//.*?$|/\*.*?\*/', '', content, flags=re.MULTILINE|re.DOTALL)
                
                # 2. Fix trailing commas in arrays/objects
                content_fixed = re.sub(r',\s*([}\]])', r'\1', content_no_comments)
                
                # 3. Ensure all property names are properly quoted
                content_quoted = re.sub(r'([{,])\s*([a-zA-Z0-9_$]+)\s*:', r'\1"\2":', content_fixed)
                
                # Try parsing again after cleanup
                return json.loads(content_quoted)
            except json.JSONDecodeError as e2:
                logger.error(f"Failed to parse package.json even after cleanup: {str(e2)}")
                # Return empty structure as fallback
                return {"dependencies": {}, "devDependencies": {}}
            except Exception as e3:
                logger.error(f"Unexpected error parsing package.json: {str(e3)}")
                return {"dependencies": {}, "devDependencies": {}}
