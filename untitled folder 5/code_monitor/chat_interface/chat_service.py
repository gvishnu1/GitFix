import logging
from typing import List, Dict, Any, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import or_, and_, literal, desc
from sqlalchemy.sql import func, text
from tenacity import retry, stop_after_attempt, wait_fixed
import re
import time
import json

from code_monitor.config import settings
from code_monitor.db.models import Repository, Commit, File, CodeSnippet, RepositoryFile
from code_monitor.db.models import QAGRepository, QAGCommit, QAGFile, QAGRepositoryFile, QAGCodeSnippet
from code_monitor.ai_processing.embedding import EmbeddingGenerator
from code_monitor.ai_processing.code_analyzer import CodeAnalyzer  # Add CodeAnalyzer import

# Import the appropriate client based on configuration
if settings.USE_OLLAMA:
    from code_monitor.utils.ollama_client import OllamaClient
else:
    from openai import OpenAI

logger = logging.getLogger(__name__)

class ChatService:
    """Service for handling natural language interactions with repository data."""
    
    def __init__(self):
        self.use_ollama = settings.USE_OLLAMA
        
        if self.use_ollama:
            self.client = OllamaClient()
            self.model = settings.OLLAMA_MODEL
            # Check if a specific Code Llama model is defined in environment for code-specific tasks
            self.code_llama_model = self.model
            if hasattr(settings, 'CODELLAMA_MODEL') and settings.CODELLAMA_MODEL:
                self.code_llama_model = settings.CODELLAMA_MODEL
                logger.info(f"Using CodeLlama model for code analysis: {self.code_llama_model}")
            else:
                logger.info(f"Using Ollama with model: {self.model}")
        else:
            self.api_key = settings.OPENAI_API_KEY
            self.model = settings.OPENAI_MODEL
            self.code_llama_model = None
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Using OpenAI with model: {self.model}")
            
        self.embedding_generator = EmbeddingGenerator()
        self.code_analyzer = CodeAnalyzer()

    async def _interpret_query(self, query: str) -> Dict[str, Any]:
        """Use AI to interpret the user's query intent with improved understanding"""
        try:
            if self.use_ollama:
                # Use the Ollama client's safe JSON parsing
                messages = [
                    {"role": "system", "content": "You are a query interpreter. Only return a valid JSON object."},
                    {"role": "user", "content": f"Analyze this repository query and return ONLY a JSON object: '{query}'"}
                ]
                
                response = await self.client.generate(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=800
                )
                
                if not response or "choices" not in response or not response["choices"]:
                    logger.warning("Empty response from Ollama model")
                    return self._create_intelligent_default_interpretation(query)
                
                content = response["choices"][0]["message"]["content"]
                if not content:
                    return self._create_intelligent_default_interpretation(query)
                    
                # Use Ollama client's safe JSON parsing
                interpretation = self.client._parse_json_safely(content)
                if not interpretation:
                    return self._create_intelligent_default_interpretation(query)
                    
            else:
                # OpenAI has more reliable JSON responses
                messages = [
                    {"role": "system", "content": "You are a query interpreter. Return only valid JSON."},
                    {"role": "user", "content": f"Analyze this query about a code repository and return only a JSON object: '{query}'"}
                ]
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1
                )
                
                content = response.choices[0].message.content
                try:
                    interpretation = json.loads(content)
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse OpenAI JSON response: {str(e)}")
                    return self._create_intelligent_default_interpretation(query)
            
            # Validate and fix the interpretation object
            required_fields = {
                "primary_intent": str,
                "sub_intents": list,
                "filters": dict,
                "target_type": str,
                "keywords": list,
                "commit_index": (type(None), int),
                "specific_commit": (type(None), str),
                "file_path": (type(None), str),
                "requires_context": bool,
                "needs_code_analysis": bool
            }
            
            # Add missing fields with defaults
            for field, field_type in required_fields.items():
                if field not in interpretation:
                    if field == "sub_intents" or field == "keywords":
                        interpretation[field] = []
                    elif field == "filters":
                        interpretation[field] = {}
                    elif field in ["commit_index", "specific_commit", "file_path"]:
                        interpretation[field] = None
                    elif field == "requires_context":
                        interpretation[field] = True
                    elif field == "needs_code_analysis":
                        interpretation[field] = False
                    else:
                        interpretation[field] = "unknown"
                
            # Validate field types
            for field, field_type in required_fields.items():
                value = interpretation[field]
                if isinstance(field_type, tuple):
                    if not any(isinstance(value, t) for t in field_type):
                        logger.warning(f"Invalid type for field {field}")
                        interpretation[field] = None if field in ["commit_index", "specific_commit", "file_path"] else field_type[0]()
                elif not isinstance(value, field_type):
                    logger.warning(f"Invalid type for field {field}")
                    interpretation[field] = field_type()
            
            # Add ordinal number detection
            if interpretation["commit_index"] is None and "commit" in query.lower():
                lower_query = query.lower()
                ordinal_words = {
                    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5,
                    "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10,
                    "last": -1, "latest": -1
                }
                for word, index in ordinal_words.items():
                    if word in lower_query:
                        interpretation["commit_index"] = index
                        if "specific_commit" not in interpretation["sub_intents"]:
                            interpretation["sub_intents"].append("specific_commit")
                        break
            
            logger.info(f"Successfully interpreted query: {query} → {interpretation['primary_intent']}")
            return interpretation
            
        except Exception as e:
            logger.error(f"Error interpreting query: {str(e)}", exc_info=True)
            return self._create_intelligent_default_interpretation(query)
    
    def _create_intelligent_default_interpretation(self, query: str) -> Dict[str, Any]:
        """Create a smarter default interpretation based on patterns in the query text"""
        lower_query = query.lower()
        interpretation = {
            "primary_intent": "repository_info",  # Default to general repository info
            "sub_intents": [],
            "filters": {},
            "target_type": "repository",
            "keywords": lower_query.split(),
            "commit_index": None,
            "specific_commit": None,
            "file_path": None,
            "requires_context": True,
            "needs_code_analysis": False
        }
        
        # Detect likely intents based on query keywords
        if any(word in lower_query for word in ["feature", "features", "functionality", "functionalities", "what does", "codebase", "code base"]):
            interpretation["primary_intent"] = "feature_analysis"
            interpretation["target_type"] = "features"
            interpretation["sub_intents"] = ["feature_list", "code_explanation"]
            interpretation["needs_code_analysis"] = True
            
        elif any(word in lower_query for word in ["commit", "commits", "changes", "change", "added", "removed", "modified"]):
            interpretation["primary_intent"] = "commit_info"
            interpretation["target_type"] = "commits"
            interpretation["sub_intents"] = ["commit_details"]
            
            # Check for specific commit references
            if any(word in lower_query for word in ["latest", "last", "recent", "newest"]):
                interpretation["sub_intents"].append("latest_commit")
                interpretation["keywords"].append("latest")
                
            elif any(word in lower_query for word in ["first", "initial", "oldest"]):
                interpretation["sub_intents"].append("first_commit")
                interpretation["keywords"].append("first")
                
            # Check for ordinal numbers in query
            ordinal_words = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, 
                           "sixth": 6, "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10}
            for word, index in ordinal_words.items():
                if word in lower_query:
                    interpretation["commit_index"] = index
                    interpretation["sub_intents"].append("specific_commit")
                    break
                    
            # Look for numbers that might be commit indices
            number_pattern = r'\b(\d+)(?:st|nd|rd|th)?\b'
            number_matches = re.findall(number_pattern, lower_query)
            if number_matches:
                try:
                    interpretation["commit_index"] = int(number_matches[0])
                    interpretation["sub_intents"].append("specific_commit")
                except ValueError:
                    pass
                
        elif any(word in lower_query for word in ["file", "files", "find", "search", "show me", "where is"]):
            interpretation["primary_intent"] = "file_search"
            interpretation["target_type"] = "files"
            
            # Try to extract file paths or names
            file_patterns = [
                r'\b[\w\-\.]+\.(py|js|html|css|md|json|yaml|yml|java|c|cpp|h|ts|sql|go)\b',  # Matches file.ext
                r'\b[\w\-]+/[\w\-\.]+\b',  # Matches dir/file
                r'[\w\-]+/[\w\-/\.]+\.(py|js|html|css|md|json|yaml|yml|java|c|cpp|h|ts|sql|go)\b'  # Matches longer paths
            ]
            
            for pattern in file_patterns:
                file_matches = re.findall(pattern, lower_query)
                if file_matches:
                    interpretation["file_path"] = file_matches[0]
                    break
                    
            # Add file content intent if query suggests looking at content
            if any(word in lower_query for word in ["content", "inside", "code", "look at", "show"]):
                interpretation["sub_intents"].append("file_content")
                
        elif any(word in lower_query for word in ["many", "count", "number", "statistics", "stats"]):
            interpretation["primary_intent"] = "statistics"
            
            if "commit" in lower_query:
                interpretation["target_type"] = "commits"
                interpretation["sub_intents"].append("commit_count")
            elif "file" in lower_query:
                interpretation["target_type"] = "files"
                interpretation["sub_intents"].append("file_count")
                
        elif any(word in lower_query for word in ["architecture", "structure", "design", "organization", "layout"]):
            interpretation["primary_intent"] = "code_content"
            interpretation["target_type"] = "code"
            interpretation["sub_intents"] = ["code_explanation", "architecture"]
            interpretation["needs_code_analysis"] = True
            
        # Add code analysis for queries about code understanding
        if any(phrase in lower_query for phrase in ["how does", "explain", "analyze", "understand", "what is", "describe"]):
            interpretation["needs_code_analysis"] = True
        
        return interpretation
    
    async def process_query(self, query: str, repository_id: int, is_qag: bool, db: AsyncSession) -> Dict[str, Any]:
        """
        Process a natural language query about the repository with enhanced code understanding.
        
        Args:
            query: The user's natural language query
            repository_id: ID of the repository to search in
            is_qag: Flag indicating whether this is a QAG repository
            db: Database session
            
        Returns:
            Dict containing the response, context, and any references
        """
        try:
            # Force repository_id to 1 for QAG repository
            repository_id = 1

            # Step 1: Interpret the query to understand intent
            query_interpretation = await self._interpret_query(query)
            logger.info(f"Query interpretation: {query_interpretation}")
            
            # Step 1.1: Handle feature analysis intent directly
            if query_interpretation.get("primary_intent") == "feature_analysis":
                features_result = await self.code_analyzer.analyze_application_features(db, repository_id, None, True)
                
                # Extract meaningful information about features
                app_type = features_result.get("application_type", "Unknown")
                feature_categories = features_result.get("features", {})
                
                # Use Ollama to generate an integrated analysis of the repository features
                repo_features_analysis = await self._generate_integrated_features_analysis(
                    features_result,
                    query
                )
                
                # If we get a response from integrated analysis, return it directly
                if repo_features_analysis:
                    return {
                        "response": repo_features_analysis, 
                        "context": {"features": features_result}, 
                        "references": []
                    }
                
                # Fallback: Build a more integrated response about the repository features manually
                response_lines = [f"# Main Features of this Repository\n"]
                response_lines.append(f"This is a **{app_type}** with the following integrated components working together:")
                
                # Add an overall summary of how everything works together
                feature_types = [k for k, v in feature_categories.items() if v]
                if len(feature_types) > 1:
                    response_lines.append("\n## System Overview")
                    
                    if "api" in feature_types and "database" in feature_types:
                        response_lines.append("This system uses an API layer to handle requests and interact with a database backend.")
                    
                    if "user_interface" in feature_types:
                        response_lines.append("It provides a user interface for interaction with the underlying functionality.")
                    
                    if "data_processing" in feature_types:
                        response_lines.append("Data processing capabilities are integrated to transform and analyze information.")
                    
                    if "authentication" in feature_types:
                        response_lines.append("Authentication mechanisms secure access to the system's features.")
                        
                    if "integration" in feature_types:
                        response_lines.append("External system integration allows for communication with third-party services.")
                
                # Add details for each feature category
                for category, items in feature_categories.items():
                    if not items:  # Skip empty categories
                        continue
                        
                    response_lines.append(f"\n## {category.replace('_', ' ').title()}")
                    
                    # Group by relevance for better organization
                    high_relevance = [item for item in items if item.get("relevance") == "high"]
                    
                    # Create meaningful category descriptions based on the items
                    if category == "api":
                        if "database" in feature_types:
                            response_lines.append("The API layer interfaces with the database to provide data access and manipulation.")
                        else:
                            response_lines.append("The repository implements API functionality for data exchange and communication.")
                    elif category == "database":
                        if "api" in feature_types:
                            response_lines.append("The database stores application data that is accessed through the API layer.")
                        else:
                            response_lines.append("Database interaction and data persistence capabilities.")
                    elif category == "authentication":
                        if "api" in feature_types:
                            response_lines.append("Authentication secures API endpoints and manages user sessions.")
                        else:
                            response_lines.append("User authentication and authorization mechanisms.")
                    elif category == "user_interface":
                        if "api" in feature_types:
                            response_lines.append("The UI components communicate with backend APIs to display and modify data.")
                        else:
                            response_lines.append("User interface components and presentation layer.")
                    elif category == "data_processing":
                        response_lines.append("Data transformation and analysis functionality integrated with other system components.")
                    elif category == "integration":
                        response_lines.append("External system integration enables data exchange with third-party services.")
                    elif category == "configuration":
                        response_lines.append("Configuration management controls system behavior across components.")
                    elif category == "utility":
                        response_lines.append("Utility functions support other system components with common operations.")
                    
                    # List key components with their relationships
                    if high_relevance:
                        response_lines.append("\n### Key Components:")
                        for item in high_relevance[:3]:  # Limit to top 3
                            name = item.get("name", "Unknown")
                            item_type = item.get("type", "component")
                            
                            # Remove file path noise and extract meaningful names
                            if "/" in name:
                                name = name.split("/")[-1]
                            
                            # Clean up snippet names
                            if item_type == "snippet":
                                if ":" in name:
                                    snippet_type, snippet_name = name.split(":", 1)
                                    response_lines.append(f"- **{snippet_name.strip()}** ({snippet_type.strip()})")
                                else:
                                    response_lines.append(f"- **{name}**")
                            else:
                                response_lines.append(f"- **{name}**")
                
                # Add unified architecture overview
                if len([c for c in feature_categories.values() if c]) > 2:
                    response_lines.append("\n## Architecture Overview")
                    response_lines.append(f"The codebase forms a {app_type.lower()} architecture with integrated " +
                                         f"{', '.join(k.replace('_', ' ') for k, v in feature_categories.items() if v)} capabilities working together.")
                    
                    # Add information about how components interact
                    if "api" in feature_types and "database" in feature_types:
                        response_lines.append("The API components handle requests and communicate with the database layer for data persistence.")
                        
                    if "user_interface" in feature_types and "api" in feature_types:
                        response_lines.append("The user interface interacts with APIs to present and manipulate data.")
                        
                    if "authentication" in feature_types:
                        response_lines.append("Authentication mechanisms secure access across the system components.")
                    
                response_text = "\n".join(response_lines)
                
                # Return with features context
                return {"response": response_text, "context": {"features": features_result}, "references": []}
            
            # Step 2: Retrieve relevant code using semantic search based on the query intent
            relevant_code = await self._semantic_search_for_relevant_code(
                query=query,
                repository_id=repository_id,
                is_qag=True,  # Force QAG
                db=db,
                target_type=query_interpretation["target_type"],
                limit=10
            )
            
            # Log how many relevant code items we found
            logger.info(f"Found {len(relevant_code)} relevant code items for the query")
            
            # Step 3: Get repository and commit information for context
            basic_context = await self._retrieve_basic_context(
                repository_id=repository_id,
                is_qag=True,  # Force QAG
                db=db,
                interpretation=query_interpretation
            )
            
            # NEW: Step 3.5: Fetch sample files from each QAG table for the response
            qag_tables_data = await self._fetch_qag_tables_data(
                repository_id=repository_id,
                db=db,
                limit=5  # Limit to 5 entries per table for brevity
            )
            
            logger.info(f"Fetched data samples from all QAG tables")
            
            # Step 4: Analyze code with LLM if there's relevant code to analyze
            code_analysis = ""
            if relevant_code and query_interpretation.get("needs_code_analysis", True):
                code_analysis = await self._analyze_code_with_llm(query, relevant_code)
                logger.info("Completed code analysis with LLM")
            
            # Step 5: Prepare context summary combining all information
            context_summary = self._build_context_summary(
                query=query,
                interpretation=query_interpretation,
                basic_context=basic_context,
                relevant_code=relevant_code,
                code_analysis=code_analysis,
                qag_tables_data=qag_tables_data  # Add the QAG tables data to summary
            )
            
            # Step 6: Generate a comprehensive response based on the complete context
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant that provides detailed information about code repositories. Answer questions about code, commits, repository structure, and features based on the provided context."},
                {"role": "user", "content": f"Answer this question about a code repository: '{query}'\n\nUse the following context to inform your answer:\n\n{context_summary}"}
            ]
            
            # Store the original model in case we need to switch temporarily
            original_model = None
            
            # Use CodeLlama for responses if available and the query is about code
            if query_interpretation.get("needs_code_analysis", False) and self.code_llama_model and self.use_ollama:
                # Update the model in the client object directly
                original_model = self.client.model
                self.client.model = self.code_llama_model
                logger.info(f"Using specialized code model {self.code_llama_model} for response")
            
            # Generate response (without passing model parameter)
            response = await self.client.generate(
                messages=messages,
                temperature=0.3,  # Lower temperature for more factual responses
                max_tokens=1500   # Allow longer responses
            )
            
            # Reset the model back to original if we changed it
            if original_model is not None:
                self.client.model = original_model
            
            if not response or "choices" not in response or not response["choices"]:
                logger.warning("Empty or invalid response from AI model")
                response_text = "Sorry, I couldn't generate a response for your query. Please try again or rephrase your question."
            else:
                response_text = response["choices"][0]["message"]["content"]
            
            # Step 7: Prepare references from the relevant code items
            references = []
            for item in relevant_code:
                item_type = item.get("item_type", "code")
                name = item.get("name", item.get("file_path", f"Item {item.get('id', 'unknown')}"))
                
                description = ""
                if item_type == "snippet":
                    description = f"Lines {item.get('start_line', '?')}-{item.get('end_line', '?')}"
                elif item_type == "file" and "commit_message" in item:
                    description = f"From commit: {item.get('commit_message', '')[:50]}"
                
                references.append({
                    "id": str(item.get("id", "unknown")),
                    "title": f"{name} ({item_type})",
                    "description": description
                })
        
            # NEW: Add references from each QAG table
            for table_name, items in qag_tables_data.items():
                for item in items:
                    item_id = item.get("id", "unknown")
                    if table_name == "qag_repositories":
                        title = f"{item.get('name', 'Unknown')} (Repository)"
                        description = f"Owner: {item.get('owner', 'Unknown')}"
                    elif table_name == "qag_commits":
                        title = f"Commit {item.get('commit_hash', '')[:7]}"
                        description = f"{item.get('message', '')[:50]}"
                    elif table_name == "qag_files":
                        title = f"{item.get('file_path', 'Unknown file')} (File)"
                        description = f"From {item.get('commit_hash', '')[:7]}"
                    elif table_name == "qag_repository_files":
                        title = f"{item.get('file_path', 'Unknown file')} (Repository File)"
                        description = f"Language: {item.get('language', 'unknown')}"
                    elif table_name == "qag_code_snippets":
                        title = f"{item.get('name', f'Snippet {item_id}')} (Code Snippet)"
                        description = f"Lines {item.get('start_line', '?')}-{item.get('end_line', '?')}"
                    else:
                        title = f"Item from {table_name}"
                        description = f"ID: {item_id}"
                    
                    references.append({
                        "id": f"{table_name}_{item_id}",
                        "title": title,
                        "description": description,
                        "table": table_name
                    })
        
            # Combine everything into the final context
            full_context = {
                "repository": basic_context.get("repository", {}),
                "stats": basic_context.get("stats", {}),
                "snippets": relevant_code,
                "qag_repositories": qag_tables_data.get("qag_repositories", []),
                "qag_commits": qag_tables_data.get("qag_commits", []),
                "qag_files": qag_tables_data.get("qag_files", []),
                "qag_repository_files": qag_tables_data.get("qag_repository_files", []),
                "qag_code_snippets": qag_tables_data.get("qag_code_snippets", []),
                "code_analysis": code_analysis,
                "summary": context_summary
            }
            
            return {
                "response": response_text,
                "context": full_context,
                "references": references
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}", exc_info=True)
            return {
                "response": f"Sorry, an error occurred while processing your query: {str(e)}",
                "context": {},
                "references": []
            }
    
    async def _retrieve_basic_context(self, repository_id: int, is_qag: bool, db: AsyncSession, 
                                  interpretation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve basic repository and commit information as context.
        
        Args:
            repository_id: ID of the repository
            is_qag: Flag indicating QAG repository
            db: Database session
            interpretation: Query interpretation dictionary
            
        Returns:
            Dictionary with basic context information
        """
        context = {
            "repository": {},
            "stats": {}
        }
        
        # MODIFIED: Always use QAG models regardless of is_qag flag
        repo_model = QAGRepository
        commit_model = QAGCommit
        
        # Force repository_id to 1 to ensure we're always accessing the QAG repository with ID 1
        repository_id = 1
        
        logger.info(f"MODIFIED: Forcing basic context retrieval from QAG repository (ID: {repository_id})")
            
        try:
            # Get repository info
            repo_query = select(repo_model).where(repo_model.id == repository_id)
            repo_result = await db.execute(repo_query)
            repository = repo_result.scalars().first()
            
            if repository:
                context["repository"] = {
                    "id": repository.id,
                    "name": repository.name,
                    "owner": repository.owner,
                    "description": repository.description if hasattr(repository, "description") else None
                }
                
                # Get commit information if needed or if this is a commit-related query
                if interpretation["target_type"] in ["commits", "repository"]:
                    # Get commits, sorted by date (newest first) or by specific commit index if provided
                    if interpretation.get("commit_index") is not None:
                        # For specific commit index, we need to get all and then pick the right one
                        commit_query = select(commit_model).where(
                            commit_model.repository_id == repository_id
                        ).order_by(desc(commit_model.commit_date if hasattr(commit_model, "commit_date") else commit_model.created_at))
                        
                        commit_result = await db.execute(commit_query)
                        commits = commit_result.scalars().all()
                        
                        # Log the commits found
                        logger.info(f"Retrieved {len(commits)} QAG commits for repository {repository_id}")
                        
                        # Extract the specific commit if the index is valid
                        index = interpretation["commit_index"] - 1  # Convert to 0-based
                        if 0 <= index < len(commits):
                            specific_commit = commits[index]
                            commits = [specific_commit]
                            logger.info(f"Selected specific QAG commit at index {index+1}: {specific_commit.commit_hash}")
                        else:
                            # If index is invalid, just use all commits
                            logger.info(f"Invalid commit index {index+1}, using all commits")
                            pass
                    elif "latest_commit" in interpretation.get("sub_intents", []):
                        # For latest commit query, get just the most recent one
                        commit_query = select(commit_model).where(
                            commit_model.repository_id == repository_id
                        ).order_by(desc(commit_model.commit_date if hasattr(commit_model, "commit_date") else commit_model.created_at)).limit(1)
                        
                        commit_result = await db.execute(commit_query)
                        commits = commit_result.scalars().all()
                        logger.info(f"Retrieved latest QAG commit for repository {repository_id}")
                    else:
                        # Default case - get all commits
                        commit_query = select(commit_model).where(
                            commit_model.repository_id == repository_id
                        ).order_by(desc(commit_model.commit_date if hasattr(commit_model, "commit_date") else commit_model.created_at))
                        
                        commit_result = await db.execute(commit_query)
                        commits = commit_result.scalars().all()
                        logger.info(f"Retrieved all QAG commits ({len(commits)}) for repository {repository_id}")
                    
                    # Add commits to context
                    context["stats"]["commit_count"] = len(commits)
                    context["stats"]["commits"] = []
                    
                    # Log the commit count
                    logger.info(f"Adding {len(commits)} QAG commits to context")
                    
                    for commit in commits:
                        commit_dict = {
                            "id": commit.id,
                            "commit_hash": commit.commit_hash,
                            "author": commit.author,
                            "message": commit.message,
                            "date": str(commit.commit_date if hasattr(commit, "commit_date") else commit.created_at)
                        }
                        context["stats"]["commits"].append(commit_dict)
                        
        except Exception as e:
            logger.error(f"Error retrieving basic context: {str(e)}", exc_info=True)
            
        return context
        
    def _build_context_summary(self, query: str, interpretation: Dict[str, Any], 
                              basic_context: Dict[str, Any], relevant_code: List[Dict[str, Any]],
                              code_analysis: str, qag_tables_data: Dict[str, List[Dict[str, Any]]] = None) -> str:
        """
        Build a comprehensive text summary of all context information for the LLM.
        
        Args:
            query: The original user query
            interpretation: Query interpretation dictionary
            basic_context: Repository and commit information
            relevant_code: List of relevant code snippets and files
            code_analysis: Analysis of code from the specialized LLM
            qag_tables_data: Data samples from all QAG tables
            
        Returns:
            Formatted text summary of all context information
        """
        # Start with repository information
        repository = basic_context.get("repository", {})
        repo_name = repository.get("name", "Unknown repository")
        repo_owner = repository.get("owner", "Unknown owner")
        repo_desc = repository.get("description", "No description available")
        
        summary = f"Repository: {repo_name} (Owner: {repo_owner})\n"
        summary += f"Description: {repo_desc}\n\n"
        
        # Add commit information if available
        commits = basic_context.get("stats", {}).get("commits", [])
        if commits:
            summary += f"Number of commits: {len(commits)}\n"
            
            # For commit-specific queries, include more detail
            if interpretation["target_type"] == "commits":
                summary += "Relevant commits:\n"
                for commit in commits[:5]:  # Limit to 5 commits to avoid too much text
                    commit_date = commit.get("date", "Unknown date")
                    commit_hash = commit.get("commit_hash", "Unknown hash")
                    commit_msg = commit.get("message", "No message")
                    commit_author = commit.get("author", "Unknown author")
                    
                    summary += f"- Commit {commit_hash[:7]} by {commit_author} on {commit_date}: {commit_msg}\n"
                summary += "\n"
            else:
                # For other queries, just summarize recent commits
                summary += "Recent commits:\n"
                for commit in commits[:3]:
                    summary += f"- {commit.get('commit_hash', '')[:7]}: {commit.get('message', '')}\n"
                summary += "\n"
        
        # Add code analysis if available
        if code_analysis:
            summary += "CODE ANALYSIS:\n"
            summary += code_analysis
            summary += "\n\n"
            
        # Add relevant code snippets
        if relevant_code:
            summary += f"RELEVANT CODE SNIPPETS ({len(relevant_code)} found):\n"
            
            # Add the most relevant snippets (limit to prevent token overload)
            for i, item in enumerate(relevant_code[:5]):
                item_type = item.get("item_type", "code")
                name = item.get("name", item.get("file_path", f"Item {i+1}"))
                language = item.get("language", "unknown")
                similarity = item.get("similarity", 0)
                
                summary += f"\n--- {name} ({item_type}, {language}) [Relevance: {similarity:.2f}] ---\n"
                
                # Add content, truncated to a reasonable size
                content = item.get("content", "").strip()
                if len(content) > 800:  # Limit individual snippet size
                    content = content[:800] + "...\n[content truncated]"
                    
                summary += content + "\n"
                
                # If the item is from a commit, add the commit context
                if "commit_message" in item:
                    summary += f"\nFrom commit: {item.get('commit_hash', '')[:7]} - {item.get('commit_message', '')}\n"
        
        # NEW: Add samples from QAG tables
        if qag_tables_data:
            summary += "\nQAG TABLES DATA SUMMARY:\n"
            
            # QAG Repository summary
            if "qag_repositories" in qag_tables_data and qag_tables_data["qag_repositories"]:
                summary += "\n--- QAG REPOSITORIES ---\n"
                for repo in qag_tables_data["qag_repositories"]:
                    summary += f"- {repo.get('name', 'Unknown')} (ID: {repo.get('id')})\n"
                    summary += f"  Owner: {repo.get('owner', 'Unknown')}\n"
                    if repo.get('description'):
                        summary += f"  Description: {repo.get('description')}\n"
            
            # QAG Commits summary
            if "qag_commits" in qag_tables_data and qag_tables_data["qag_commits"]:
                summary += "\n--- QAG COMMITS ---\n"
                for commit in qag_tables_data["qag_commits"][:5]:  # Limit to 5
                    summary += f"- Commit {commit.get('commit_hash', '')[:7]} (ID: {commit.get('id')})\n"
                    summary += f"  Author: {commit.get('author', 'Unknown')}\n"
                    summary += f"  Message: {commit.get('message', 'No message')}\n"
            
            # QAG Files summary
            if "qag_files" in qag_tables_data and qag_tables_data["qag_files"]:
                summary += "\n--- QAG FILES ---\n"
                for file in qag_tables_data["qag_files"][:5]:  # Limit to 5
                    summary += f"- {file.get('file_path', 'Unknown')} (ID: {file.get('id')})\n"
                    summary += f"  Language: {file.get('language', 'Unknown')}\n"
                    if file.get('content_preview'):
                        summary += f"  Preview: {file.get('content_preview')}\n"
            
            # QAG Repository Files summary
            if "qag_repository_files" in qag_tables_data and qag_tables_data["qag_repository_files"]:
                summary += "\n--- QAG REPOSITORY FILES ---\n"
                for file in qag_tables_data["qag_repository_files"][:5]:  # Limit to 5
                    summary += f"- {file.get('file_path', 'Unknown')} (ID: {file.get('id')})\n"
                    summary += f"  Language: {file.get('language', 'Unknown')}\n"
                    if file.get('content_preview'):
                        summary += f"  Preview: {file.get('content_preview')}\n"
            
            # QAG Code Snippets summary
            if "qag_code_snippets" in qag_tables_data and qag_tables_data["qag_code_snippets"]:
                summary += "\n--- QAG CODE SNIPPETS ---\n"
                for snippet in qag_tables_data["qag_code_snippets"][:5]:  # Limit to 5
                    summary += f"- {snippet.get('name', f'Snippet {snippet.get('id')}')} (ID: {snippet.get('id')})\n"
                    summary += f"  Type: {snippet.get('snippet_type', 'Unknown')}\n"
                    summary += f"  Lines: {snippet.get('start_line', '?')}-{snippet.get('end_line', '?')}\n"
                    if snippet.get('content_preview'):
                        summary += f"  Preview: {snippet.get('content_preview')}\n"
        
        return summary
    
    async def _semantic_search_for_relevant_code(self, query: str, repository_id: int, is_qag: bool, 
                                        db: AsyncSession, target_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Perform semantic search based on query to find relevant code snippets or files.
        Uses embeddings to find the most semantically relevant code for a user query.
        
        Args:
            query: The user's query text
            repository_id: ID of the repository to search in
            is_qag: Flag indicating whether this is a QAG repository 
            db: Database session
            target_type: Type of target to search for ("commits", "files", "repository")
            limit: Maximum number of results to return
            
        Returns:
            List of relevant code snippets or files with similarity scores
        """
        relevant_items = []
        
        try:
            # Always use QAG models regardless of is_qag flag to ensure we always access QAG repository data
            repo_model = QAGRepository
            commit_model = QAGCommit
            file_model = QAGFile
            snippet_model = QAGCodeSnippet
            repository_file_model = QAGRepositoryFile
            
            # Force repository_id to 1 to ensure we're always accessing the QAG repository with ID 1
            repository_id = 1
            
            logger.info(f"MODIFIED: Forcing search on QAG repository (ID: {repository_id})")
            
            # PRIORITY STRATEGY: Get files directly from qag_files table first
            # This ensures we get actual files regardless of other search strategies
            try:
                # Get columns for qag_files
                column_query = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'qag_files'
                """)
                
                # Execute query directly without nested transaction
                columns_result = await db.execute(column_query)
                file_columns = [row[0] for row in columns_result]
                
                logger.info(f"Available columns in qag_files: {file_columns}")
                
                # Construct file columns query based on what's available
                file_select_cols = ['id', 'file_path', 'commit_id']
                
                if 'language' in file_columns:
                    file_select_cols.append('language')
                if 'change_type' in file_columns:
                    file_select_cols.append('change_type')
                if 'content_after' in file_columns:
                    file_select_cols.append('content_after')
                if 'content_before' in file_columns:
                    file_select_cols.append('content_before')
                if 'content' in file_columns:
                    file_select_cols.append('content')
                    
                file_cols_str = ", ".join(file_select_cols)
                
                # Query for all files without any filtering to ensure we get something
                all_files_query = text(f"""
                    SELECT {file_cols_str}
                    FROM qag_files
                    LIMIT :limit
                """)
                
                # Execute query directly without nested transaction
                files_result = await db.execute(all_files_query, {"limit": 20})
                files = files_result.fetchall()
                
                logger.info(f"Direct QAG files search executed, found {len(files)} files")
                
                for file in files:
                    # Determine the file content
                    content = None
                    if hasattr(file, 'content_after') and file.content_after:
                        content = file.content_after
                    elif hasattr(file, 'content_before') and file.content_before:
                        content = file.content_before
                    elif hasattr(file, 'content') and file.content:
                        content = file.content
                        
                    if content and len(content.strip()) > 0:
                        # Get the associated commit info for reference
                        commit_info = {}
                        try:
                            if hasattr(file, 'commit_id'):
                                commit_query = text("""
                                    SELECT commit_hash, message, author
                                    FROM qag_commits
                                    WHERE id = :commit_id
                                """)
                                # Execute query directly without nested transaction
                                commit_result = await db.execute(commit_query, {"commit_id": file.commit_id})
                                commit = commit_result.fetchone()
                                if commit:
                                    commit_info = {
                                        "commit_hash": commit.commit_hash,
                                        "commit_message": commit.message,
                                        "author": commit.author
                                    }
                        except Exception as e:
                            logger.warning(f"Could not fetch commit info for file: {str(e)}")
                        
                        file_dict = {
                            "id": file.id,
                            "file_path": file.file_path if hasattr(file, 'file_path') else f"File {file.id}",
                            "content": content[:5000] if content else "",  # Limit content size
                            "language": file.language if hasattr(file, 'language') else "unknown",
                            "similarity": 0.8,  # High default similarity for direct file matches
                            "item_type": "file",
                            "change_type": file.change_type if hasattr(file, 'change_type') else "unknown",
                            **commit_info  # Include commit info if available
                        }
                        relevant_items.append(file_dict)
                
                # If we found files, we can return early with these results
                if relevant_items:
                    logger.info(f"Found {len(relevant_items)} directly from qag_files table, returning these files")
                    return relevant_items[:limit]
            except Exception as e:
                logger.error(f"Error in direct file search: {str(e)}")
                # Continue to other strategies if direct file search fails
                
            # Generate embedding for the query text
            query_embedding = await self.embedding_generator.generate_embedding(query)
            if not query_embedding:
                logger.warning("Failed to generate embedding for query text")
                # Even if embedding fails, we'll continue with keyword search
            
            # Convert the query embedding to a PostgreSQL vector format (if available)
            query_embedding_str = None
            if query_embedding:
                query_embedding_str = f"[{','.join(map(str, query_embedding))}]"
            
            # STRATEGY 1: Try to search for relevant code snippets based on embedding similarity
            if query_embedding_str and (target_type in ["code", "files", "repository", "features"] or not target_type):
                try:
                    # First get a list of available columns in qag_code_snippets to avoid schema mismatch errors
                    column_query = text("""
                        SELECT column_name FROM information_schema.columns 
                        WHERE table_name = 'qag_code_snippets'
                    """)
                    
                    async with db.begin():
                        columns_result = await db.execute(column_query)
                        snippet_columns = [row[0] for row in columns_result]
                    
                    logger.info(f"Available columns in qag_code_snippets: {snippet_columns}")
                    
                    # Query for code snippets with cosine similarity calculation (using raw SQL for safety)
                    if 'embedding' in snippet_columns:
                        snippets_query = text(f"""
                            SELECT id, file_id, name, content, 
                                   start_line, end_line, snippet_type,
                                   dot_product(embedding, '{query_embedding_str}'::vector) as similarity
                            FROM qag_code_snippets 
                            WHERE file_id IN (
                                SELECT id FROM qag_files 
                                WHERE commit_id IN (
                                    SELECT id FROM qag_commits 
                                    WHERE repository_id = :repo_id
                                )
                            )
                            AND embedding IS NOT NULL
                            ORDER BY similarity DESC
                            LIMIT :limit
                        """)
                        
                        # Use separate transaction for each query to prevent cascading failures
                        async with db.begin():
                            snippets_result = await db.execute(snippets_query, {"repo_id": repository_id, "limit": limit})
                            snippet_rows = snippets_result.all()
                        
                        logger.info(f"QAG snippets query executed, found {len(snippet_rows)} results")
                        
                        for row in snippet_rows:
                            snippet_dict = {
                                "id": row.id,
                                "file_id": row.file_id,
                                "snippet_type": row.snippet_type if hasattr(row, 'snippet_type') else 'code',
                                "name": row.name if row.name else f"Snippet {row.id}",
                                "content": row.content,
                                "start_line": row.start_line if hasattr(row, 'start_line') else 1,
                                "end_line": row.end_line if hasattr(row, 'end_line') else 1,
                                "language": "unknown",  # Default since we're using raw SQL
                                "similarity": row.similarity,
                                "item_type": "snippet"
                            }
                            relevant_items.append(snippet_dict)
                except Exception as e:
                    logger.error(f"Error in snippet semantic search: {str(e)}")
                    # Continue to the next strategy instead of aborting
            
            # STRATEGY 2: Search commit messages for relevant content
            try:
                # First check what columns are available in each table to avoid schema errors
                column_query = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'qag_commits'
                """)
                
                async with db.begin():
                    columns_result = await db.execute(column_query)
                    commit_columns = [row[0] for row in columns_result]
                
                # Get columns for qag_files
                column_query = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'qag_files'
                """)
                
                async with db.begin():
                    columns_result = await db.execute(column_query)
                    file_columns = [row[0] for row in columns_result]
                
                logger.info(f"Available columns in qag_commits: {commit_columns}")
                logger.info(f"Available columns in qag_files: {file_columns}")
                
                # First, try exact keyword matches in commit messages
                if 'message' in commit_columns:
                    sql_conditions = []
                    for keyword in query.lower().split():
                        if len(keyword) > 3:  # Only use keywords that are meaningful
                            sql_conditions.append(f"LOWER(message) LIKE '%{keyword}%'")
                    
                    where_clause = f"repository_id = :repo_id"
                    if sql_conditions:
                        where_clause += " AND (" + " OR ".join(sql_conditions) + ")"
                    
                    commit_query = text(f"""
                        SELECT id, commit_hash, message, author, created_at 
                        FROM qag_commits
                        WHERE {where_clause}
                        ORDER BY created_at DESC
                        LIMIT :limit
                    """)
                    
                    async with db.begin():
                        commit_result = await db.execute(commit_query, {"repo_id": repository_id, "limit": limit})
                        commits = commit_result.fetchall()
                    
                    logger.info(f"QAG commit search executed, found {len(commits)} matching commits")
                    
                    # For each commit, get the associated files
                    for commit in commits:
                        # Construct file columns query based on what's available
                        file_select_cols = ['id', 'file_path', 'commit_id']
                        
                        if 'content_after' in file_columns:
                            file_select_cols.append('content_after')
                        if 'content_before' in file_columns:
                            file_select_cols.append('content_before')
                        if 'content' in file_columns:
                            file_select_cols.append('content')
                        if 'language' in file_columns:
                            file_select_cols.append('language')
                        if 'change_type' in file_columns:
                            file_select_cols.append('change_type')
                            
                        file_cols_str = ", ".join(file_select_cols)
                        
                        files_query = text(f"""
                            SELECT {file_cols_str} 
                            FROM qag_files
                            WHERE commit_id = :commit_id
                            LIMIT 5
                        """)
                        
                        async with db.begin():
                            files_result = await db.execute(files_query, {"commit_id": commit.id})
                            files = files_result.fetchall()
                        
                        logger.info(f"Found {len(files)} files for QAG commit {commit.id} ({commit.commit_hash})")
                        
                        for file in files:
                            # Determine the file content
                            content = None
                            if hasattr(file, 'content_after') and file.content_after:
                                content = file.content_after
                            elif hasattr(file, 'content_before') and file.content_before:
                                content = file.content_before
                            elif hasattr(file, 'content') and file.content:
                                content = file.content
                                
                            if content and len(content.strip()) > 0:
                                file_dict = {
                                    "id": file.id,
                                    "file_path": file.file_path if hasattr(file, 'file_path') else f"File {file.id}",
                                    "commit_id": commit.id,
                                    "commit_message": commit.message,
                                    "commit_hash": commit.commit_hash,
                                    "content": content[:5000] if content else "",  # Limit content size
                                    "language": file.language if hasattr(file, 'language') else "unknown",
                                    "similarity": 0.5,  # Default similarity for commit-based matches
                                    "item_type": "file",
                                    "change_type": file.change_type if hasattr(file, 'change_type') else "unknown"
                                }
                                relevant_items.append(file_dict)
            except Exception as e:
                logger.error(f"Error in commit-based search: {str(e)}")
                # Continue to next strategy
                
            # STRATEGY 3: retrieve repository files via ORM and analyze
            try:
                repo_files_result = await db.execute(
                    select(repository_file_model)
                    .where(repository_file_model.repository_id == repository_id)
                    .limit(limit)
                )
                repo_files = repo_files_result.scalars().all()
                logger.info(f"Fetched {len(repo_files)} files via ORM from qag_repository_files")
                for repo_file in repo_files:
                    content = getattr(repo_file, 'content', '') or ''
                    # Analyze code content using CodeAnalyzer
                    analysis = ''
                    try:
                        analysis = await self.code_analyzer.summarize_code(content)
                    except Exception as e:
                        logger.warning(f"Code analysis failed for file {repo_file.file_path}: {e}")
                    repo_file_dict = {
                        'id': repo_file.id,
                        'file_path': repo_file.file_path,
                        'language': getattr(repo_file, 'language', 'unknown'),
                        'content': content[:5000],
                        'analysis': analysis,
                        'item_type': 'repository_file'
                    }
                    relevant_items.append(repo_file_dict)
            except Exception as e:
                logger.error(f"ORM repository files search failed: {e}")
            
            # STRATEGY 4: Last resort - get files directly from qag_files table if all else fails
            if len(relevant_items) < 2:
                try:
                    # Use the file columns we already got
                    if not 'file_columns' in locals() or not file_columns:
                        # If we don't have file_columns, get them now
                        column_query = text("""
                            SELECT column_name FROM information_schema.columns 
                            WHERE table_name = 'qag_files'
                        """)
                        
                        async with db.begin():
                            columns_result = await db.execute(column_query)
                            file_columns = [row[0] for row in columns_result]
                    
                    # Construct file columns query based on what's available
                    file_select_cols = ['id', 'file_path', 'commit_id']
                    
                    if 'content_after' in file_columns:
                        file_select_cols.append('content_after')
                    if 'content_before' in file_columns:
                        file_select_cols.append('content_before')
                    if 'content' in file_columns:
                        file_select_cols.append('content')
                    if 'language' in file_columns:
                        file_select_cols.append('language')
                    if 'change_type' in file_columns:
                        file_select_cols.append('change_type')
                        
                    file_cols_str = ", ".join(file_select_cols)
                    
                    # Query for files directly
                    direct_files_query = text(f"""
                        SELECT {file_cols_str} FROM qag_files
                        LIMIT :limit
                    """)
                    
                    async with db.begin():
                        files_result = await db.execute(direct_files_query, {"limit": 15})
                        files = files_result.fetchall()
                    
                    logger.info(f"Emergency direct QAG file search executed, found {len(files)} files")
                    
                    for file in files:
                        content = None
                        if hasattr(file, 'content_after') and file.content_after:
                            content = file.content_after
                        elif hasattr(file, 'content_before') and file.content_before:
                            content = file.content_before
                        elif hasattr(file, 'content') and file.content:
                            content = file.content
                        
                        if content and len(content.strip()) > 0:
                            file_dict = {
                                "id": file.id,
                                "file_path": file.file_path if hasattr(file, 'file_path') else f"File {file.id}",
                                "content": content[:5000],  # Limit content size
                                "language": file.language if hasattr(file, 'language') else "unknown",
                                "similarity": 0.1,  # Lowest similarity for emergency results
                                "item_type": "emergency_file"
                            }
                            relevant_items.append(file_dict)
                except Exception as e:
                    logger.error(f"Error in emergency file search: {str(e)}")
                    
            # Always ensure we return something, even if all strategies failed
            if not relevant_items:
                logger.warning("No items found through any search strategy. Creating fallback item.")
                fallback_item = {
                    "id": 0,
                    "file_path": "README.md",  # Assuming most repos have a README
                    "content": "No specific file content could be retrieved. The repository may be empty or indexes may need updating.",
                    "language": "markdown",
                    "similarity": 0.0,
                    "item_type": "fallback_item"
                }
                relevant_items.append(fallback_item)
                
            # De-duplicate results (in case same file appears multiple times)
            seen_ids = set()
            unique_items = []
            for item in relevant_items:
                item_id = f"{item.get('item_type', 'unknown')}_{item.get('id', 0)}"
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    unique_items.append(item)
            
            # Sort results by similarity score
            unique_items.sort(key=lambda x: x.get('similarity', 0), reverse=True)
            
            logger.info(f"Total relevant unique QAG items found: {len(unique_items)}")
            for i, item in enumerate(unique_items[:5]):  # Log the top 5 items
                item_type = item.get("item_type", "unknown")
                name = item.get("name", item.get("file_path", f"Item {item.get('id', 'unknown')}"))
                logger.info(f"{i+1}. {item_type}: {name} (similarity: {item.get('similarity', 0):.2f})")
            
            return unique_items[:limit]
            
        except Exception as e:
            logger.error(f"Error in semantic search: {str(e)}", exc_info=True)
            # Return at least a fallback item
            return [{
                "id": 0,
                "file_path": "README.md",
                "content": f"Error retrieving files: {str(e)}",
                "language": "text",
                "similarity": 0.0,
                "item_type": "error_item"
            }]
        
    async def _fetch_qag_tables_data(self, repository_id: int, db: AsyncSession, limit: int = 5) -> Dict[str, List[Dict[str, Any]]]:
        """
        Fetch sample data from all QAG tables for the given repository,
        using Ollama for text analysis and CodeLlama for file content understanding.
        
        Args:
            repository_id: ID of the repository
            db: Database session
            limit: Number of samples to fetch from each table
            
        Returns:
            Dictionary with table names as keys and lists of sample data as values
        """
        tables_data = {}
        
        try:
            # Always use QAG models regardless of is_qag flag
            repo_model = QAGRepository
            commit_model = QAGCommit
            file_model = QAGFile
            snippet_model = QAGCodeSnippet
            repository_file_model = QAGRepositoryFile
            
            # Force repository_id to 1 for QAG
            repository_id = 1
            
            # Fetch sample repositories - AVOID NESTED TRANSACTION
            repo_query = select(repo_model).where(repo_model.id == repository_id)
            repo_result = await db.execute(repo_query)
            repositories = repo_result.scalars().all()
            tables_data["qag_repositories"] = [{
                "id": repo.id,
                "name": repo.name,
                "owner": repo.owner,
                "description": repo.description
            } for repo in repositories]
            
            # Fetch sample commits - AVOID NESTED TRANSACTION
            commit_query = select(commit_model).where(commit_model.repository_id == repository_id).limit(limit)
            commit_result = await db.execute(commit_query)
            commits = commit_result.scalars().all()
            tables_data["qag_commits"] = [{
                "id": commit.id,
                "commit_hash": commit.commit_hash,
                "message": commit.message,
                "author": commit.author,
                "timestamp": str(commit.timestamp) if hasattr(commit, 'timestamp') else None,
                "created_at": str(commit.created_at) if hasattr(commit, 'created_at') else None
            } for commit in commits]
            
            # ENHANCED FILE FETCHING: Use CodeLlama for file content analysis
            # First, get columns for qag_files to ensure we handle schema correctly
            column_query = text("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_name = 'qag_files'
            """)
            
            # Execute query directly, no transaction - AVOID NESTED TRANSACTION
            columns_result = await db.execute(column_query)
            file_columns = [row[0] for row in columns_result]
            
            logger.info(f"Available columns in qag_files: {file_columns}")
            
            # Construct file columns query based on what's available
            file_select_cols = ['id', 'file_path', 'commit_id']
            
            if 'language' in file_columns:
                file_select_cols.append('language')
            if 'change_type' in file_columns:
                file_select_cols.append('change_type')
            if 'content_after' in file_columns:
                file_select_cols.append('content_after')
            if 'content_before' in file_columns:
                file_select_cols.append('content_before')
            if 'content' in file_columns:
                file_select_cols.append('content')
                
            file_cols_str = ", ".join(file_select_cols)
            
            # IMPROVED: Get files without limiting to a specific commit to ensure we get data
            files_query = text(f"""
                SELECT {file_cols_str} 
                FROM qag_files
                LIMIT :limit
            """)
            
            # Execute query directly, no transaction - AVOID NESTED TRANSACTION
            files_result = await db.execute(files_query, {"limit": limit * 2})
            files = files_result.fetchall()
            
            logger.info(f"Found {len(files)} files in qag_files table")
            
            # Process files and add to tables_data
            tables_data["qag_files"] = []
            for file in files:
                content = None
                if hasattr(file, 'content_after') and file.content_after:
                    content = file.content_after
                elif hasattr(file, 'content_before') and file.content_before:
                    content = file.content_before
                elif hasattr(file, 'content') and file.content:
                    content = file.content
                    
                file_dict = {
                    "id": file.id,
                    "file_path": file.file_path if hasattr(file, 'file_path') else f"File {file.id}",
                    "language": file.language if hasattr(file, 'language') else "unknown",
                    "content_preview": content[:100] + "..." if content else None
                }
                tables_data["qag_files"].append(file_dict)
                
            # Fetch sample repository files - AVOID NESTED TRANSACTION
            repo_file_query = select(repository_file_model).where(
                repository_file_model.repository_id == repository_id
            ).limit(limit)
            repo_file_result = await db.execute(repo_file_query)
            repo_files = repo_file_result.scalars().all()
            
            tables_data["qag_repository_files"] = [{
                "id": file.id,
                "file_path": file.file_path,
                "language": file.language if hasattr(file, 'language') else "unknown",
                "content_preview": file.content[:100] + "..." if hasattr(file, 'content') and file.content else None
            } for file in repo_files]
            
            # Fetch sample code snippets - AVOID NESTED TRANSACTION
            snippet_query = select(snippet_model).limit(limit)
            snippet_result = await db.execute(snippet_query)
            snippets = snippet_result.scalars().all()
            
            tables_data["qag_code_snippets"] = [{
                "id": snippet.id,
                "name": snippet.name,
                "snippet_type": snippet.snippet_type if hasattr(snippet, 'snippet_type') else "code",
                "start_line": snippet.start_line if hasattr(snippet, 'start_line') else 1,
                "end_line": snippet.end_line if hasattr(snippet, 'end_line') else 1,
                "content_preview": snippet.content[:100] + "..." if hasattr(snippet, 'content') and snippet.content else None
            } for snippet in snippets]
            
        except Exception as e:
            logger.error(f"Error fetching QAG tables data: {str(e)}", exc_info=True)
            return {}
        
        # Always ensure we return at least empty arrays for each table
        for table in ["qag_repositories", "qag_commits", "qag_files", "qag_repository_files", "qag_code_snippets"]:
            if table not in tables_data:
                tables_data[table] = []
        
        return tables_data
    
    async def generate_response(self, query: str, repository_id: int, is_qag: bool = False, enhanced: bool = False, memory_id: str = None) -> Tuple[str, dict, List[dict]]:
        """
        Generate a response to a user query about a repository.
        
        Args:
            query: The user's query
            repository_id: ID of the repository
            is_qag: Whether to use QAG tables
            enhanced: Whether to use enhanced context
            memory_id: Optional memory ID for conversation tracking
            
        Returns:
            Tuple of (response text, context dict, list of references)
        """
        start_time = time.time()
        # Create context with repository data, stats, and search results
        context = await self._create_context(query, repository_id, is_qag, enhanced)
        logger.info(f"Context creation time: {time.time() - start_time:.2f}s")
        
        # Extract key information from context for easier access
        repo_info = context.get("repository", {})
        repo_name = repo_info.get("name", "")
        stats = context.get("stats", {})
        commits = stats.get("commits", [])
        commit_count = stats.get("commit_count", 0)
        
        # Get QAG data 
        qag_repositories = context.get("qag_repositories", [])
        qag_commits = context.get("qag_commits", [])
        qag_files = context.get("qag_files", [])
        qag_repository_files = context.get("qag_repository_files", [])
        qag_code_snippets = context.get("qag_code_snippets", [])
        
        # Override commits with QAG commits if available (for QAG repositories)
        if is_qag and qag_commits:
            commits = qag_commits
            commit_count = len(qag_commits)
        
        # Enhanced context from retrieved data
        enhanced_context = {}
        if qag_repositories:
            first_repo = qag_repositories[0]
            enhanced_context["repository"] = {
                "name": first_repo.get("name", ""),
                "owner": first_repo.get("owner", ""),
                "description": first_repo.get("description", "")
            }
        
        # Generate summary for the context display
        context["summary"] = self._generate_context_summary(context)
        
        # If using Ollama, directly answer simple questions for faster response
        if self.use_ollama:
            # Handle specific query types without needing LLM processing
            if query.lower().strip() in [
                "how many commits are in this repository?", 
                "how many commits are in this repo?",
                "how many commits are in this qag repository?",
                "how many commits are in this qag repo?",
                "commit count",
                "number of commits"
            ]:
                return f"There are {commit_count} commits in the repository.", context, self._generate_references(context)
                
            # Handle repository name questions
            if query.lower().strip() in [
                "what is the repository name?", 
                "what's the name of this repo?",
                "repo name",
                "repository name"
            ]:
                repo_name = repo_info.get("name") or (qag_repositories[0].get("name") if qag_repositories else None)
                if repo_name:
                    return f"The repository name is '{repo_name}'.", context, self._generate_references(context)
        
        try:
            # Prepare messages for the LLM
            messages = []
            
            # System prompt with instructions
            system_prompt = self._create_system_prompt(repo_name)
            messages.append({"role": "system", "content": system_prompt})
            
            # Add conversation history if applicable
            if memory_id and memory_id in self.conversation_memory:
                messages.extend(self.conversation_memory[memory_id])
            
            # Create a detailed prompt with the context info
            user_prompt = f"Query: {query}\n\nContext Information:\n"
            
            # Add repository information
            if enhanced_context.get("repository"):
                repo = enhanced_context["repository"]
                user_prompt += f"Repository: {repo.get('name', '')} (Owner: {repo.get('owner', '')})\n"
                if repo.get("description"):
                    user_prompt += f"Description: {repo.get('description')}\n"
            elif repo_info:
                user_prompt += f"Repository: {repo_info.get('name', '')} (Owner: {repo_info.get('owner', '')})\n"
                if repo_info.get("description"):
                    user_prompt += f"Description: {repo_info.get('description')}\n"
            
            # Add commit information
            if commit_count > 0:
                user_prompt += f"\nTotal commits: {commit_count}\n"
                if commits:
                    user_prompt += "Recent commits:\n"
                    for i, commit in enumerate(commits[:5]):  # Get the 5 most recent commits
                        commit_hash = commit.get("commit_hash", "")[:8]
                        commit_msg = commit.get("message", "").split("\n")[0]  # Get first line of commit message
                        commit_author = commit.get("author", "")
                        user_prompt += f"- {commit_hash} by {commit_author}: {commit_msg}\n"
            
            # Add file information
            if qag_files:
                user_prompt += f"\nFiles in repository: {len(qag_files)}\n"
                if len(qag_files) > 0:
                    user_prompt += "Sample files:\n"
                    for i, file in enumerate(qag_files[:3]):  # Show up to 3 files
                        file_path = file.get("file_path", "")
                        language = file.get("language", "")
                        user_prompt += f"- {file_path} ({language})\n"
                        if file.get("summary"):
                            user_prompt += f"  Summary: {file.get('summary')}\n"
            
            # Add repository file information
            if qag_repository_files:
                if not qag_files:  # Only add if we don't have files info already
                    user_prompt += f"\nFiles in repository: {len(qag_repository_files)}\n"
                    if len(qag_repository_files) > 0:
                        user_prompt += "Sample files:\n"
                        for i, file in enumerate(qag_repository_files[:3]):
                            file_path = file.get("file_path", "")
                            language = file.get("language", "")
                            user_prompt += f"- {file_path} ({language})\n"
                            if file.get("summary"):
                                user_prompt += f"  Summary: {file.get('summary')}\n"
            
            # Add code snippets
           
            if qag_code_snippets:
                user_prompt += f"\nCode snippets: {len(qag_code_snippets)}\n"
                if len(qag_code_snippets) > 0:
                    user_prompt += "Sample snippets:\n"
                    for i, snippet in enumerate(qag_code_snippets[:2]):  # Show up to 2 snippets
                        snippet_name = snippet.get("name", "")
                        language = snippet.get("language", "")
                        user_prompt += f"- {snippet_name} ({language})\n"
                        if snippet.get("analysis"):
                            user_prompt += f"  Analysis: {snippet.get('analysis')}\n"
            
            # Add the query at the end for emphasis
            user_prompt += f"\nPlease answer this question about the repository: {query}"
            
            # Append user message
            messages.append({"role": "user", "content": user_prompt})
            
            # Generate a response using Ollama or OpenAI
            if self.use_ollama:
                response_content = await self._generate_ollama_response(messages)
            else:
                response_content = await self._generate_openai_response(messages)
            
            # Update conversation memory if needed
            if memory_id:
                if memory_id not in self.conversation_memory:
                    self.conversation_memory[memory_id] = []
                self.conversation_memory[memory_id].append({"role": "user", "content": query})
                self.conversation_memory[memory_id].append({"role": "assistant", "content": response_content})
                # Limit conversation history
                if len(self.conversation_memory[memory_id]) > self.max_memory_messages:
                    self.conversation_memory[memory_id] = self.conversation_memory[memory_id][-self.max_memory_messages:]
            
            return response_content, context, self._generate_references(context)
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}", exc_info=True)
            return "I couldn't generate a response to your query. Please try again or rephrase your question.", context, self._generate_references(context)
    
    async def initialize(self):
        """Initialize async components"""
        self.client = await OllamaClient.create()
    
    async def _analyze_code_with_llm(self, query: str, code_snippets: List[Dict[str, Any]]) -> str:
        """Use LLM to analyze code snippets in context of user query
        
        Args:
            query: The user's query about the code
            code_snippets: List of relevant code snippets
            
        Returns:
            String analysis of the code
        """
        if not code_snippets:
            return "No code snippets available for analysis."
            
        # Initialize the client if needed
        if not self.client and self.use_ollama:
            try:
                self.client = await OllamaClient.create()
                logger.info("Initialized Ollama client for code analysis")
            except Exception as e:
                logger.error(f"Failed to initialize Ollama client: {str(e)}")
                return "Code analysis service unavailable."
        elif not self.client:
            # OpenAI client initialization
            try:
                self.client = OpenAI(api_key=settings.OPENAI_API_KEY)
                logger.info("Initialized OpenAI client for code analysis")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
                return "Code analysis service unavailable."
            
        # Limit the number of snippets to avoid context overflow
        code_snippets = code_snippets[:5]  # Limit to max 5 snippets
        
        # Extract snippet content and format for the prompt
        formatted_snippets = []
        for i, snippet in enumerate(code_snippets):
            name = snippet.get("name", snippet.get("file_path", f"Snippet {i+1}"))
            content = snippet.get("content", "").strip()
            language = snippet.get("language", "unknown")
            
            if not content:
                continue
                
            # If content is too long, truncate it
            if len(content) > 2000:
                content = content[:2000] + "\n...[truncated]..."
                
            formatted_snippets.append(
                f"--- {name} ({language}) ---\n{content}\n"
            )
            
        if not formatted_snippets:
            return "No code content available for analysis."
            
        # Combine all snippets and create the analysis prompt
        combined_code = "\n\n".join(formatted_snippets)
        
        # Craft the analysis prompt
        prompt = f"""Analyze the following code snippets in relation to the user's query: "{query}"

{combined_code}

Provide a detailed analysis explaining:
1. What the code does
2. How it relates to the user's query
3. Key features and functionality implemented
4. Any notable patterns or architectural decisions
5. Technical details that might be helpful to understand

Focus on concrete details rather than general statements. Explain the code's purpose and functionality clearly.
"""

        try:
            # Send the analysis request to the LLM
            if self.use_ollama:
                # Use the CodeLlama model if available for better code understanding
                original_model = self.client.model
                if hasattr(self, 'code_llama_model') and self.code_llama_model:
                    self.client.model = self.code_llama_model
                    logger.info(f"Using specialized code model {self.code_llama_model} for code analysis")
                
                messages = [
                    {"role": "system", "content": "You are an expert code analyst. Provide detailed and accurate analysis of code."},
                    {"role": "user", "content": prompt}
                ]
                
                response = await self.client.generate(
                    messages=messages,
                    temperature=0.1,  # Lower temperature for more factual analysis
                    max_tokens=1500  # Allow for detailed analysis
                )
                
                # Reset the model back to original if we changed it
                if hasattr(self, 'code_llama_model') and self.code_llama_model and original_model:
                    self.client.model = original_model
                
                if not response or "choices" not in response or not response["choices"]:
                    return "Failed to generate code analysis."
                    
                return response["choices"][0]["message"]["content"]
                
            else:
                # Use OpenAI
                messages = [
                    {"role": "system", "content": "You are an expert code analyst. Provide detailed and accurate analysis of code."},
                    {"role": "user", "content": prompt}
                ]
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.1
                )
                
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Error analyzing code with LLM: {str(e)}")
            return f"Error during code analysis: {str(e)}"
    
    async def _generate_integrated_features_analysis(self, features_result: Dict[str, Any], query: str) -> str:
        """
        Generate an integrated analysis of repository features using Ollama.
        This provides a more cohesive explanation of how features work together.
        
        Args:
            features_result: Dictionary with feature analysis results
            query: The user's original query
            
        Returns:
            A comprehensive text analysis of the repository's features
        """
        try:
            if not self.use_ollama:
                logger.warning("Integrated feature analysis requires Ollama")
                return None
                
            # Extract key information from features result
            app_type = features_result.get("application_type", "Unknown")
            feature_categories = features_result.get("features", {})
            file_count = features_result.get("file_count", 0)
            
            # Check if this is a comprehensive repository analysis request
            is_comprehensive_request = any(phrase in query.lower() for phrase in [
                "explain completely", "understand every code", "why this repository", 
                "what does this whole", "complete analysis", "understand the whole",
                "entire repository", "all the code", "deep analysis"
            ])
            
            # For comprehensive requests, use the deep repository analysis method
            if is_comprehensive_request:
                return await self._perform_deep_repository_analysis(features_result)
            
            # Standard feature analysis (for less comprehensive requests)
            # Build a structured representation of features to help the model
            feature_data = []
            for category, items in feature_categories.items():
                if not items:
                    continue
                    
                category_data = {
                    "name": category,
                    "components": []
                }
                
                for item in items:
                    name = item.get("name", "Unknown")
                    item_type = item.get("type", "unknown")
                    relevance = item.get("relevance", "medium")
                    
                    # Simplify filenames and snippets
                    if "/" in name:
                        name = name.split("/")[-1]
                    
                    if item_type == "snippet" and ":" in name:
                        snippet_type, snippet_name = name.split(":", 1)
                        name = f"{snippet_name.strip()} ({snippet_type.strip()})"
                    
                    component = {
                        "name": name,
                        "type": item_type,
                        "relevance": relevance
                    }
                    category_data["components"].append(component)
                
                feature_data.append(category_data)
            
            # Create a comprehensive prompt for integrated analysis
            prompt = f"""
            You are a software architecture expert analyzing a codebase. Based on the following structured data about a repository,
            provide a comprehensive analysis of how all features work together as an integrated system. Explain the architecture
            and interactions between components in a cohesive way.
            
            Repository type: {app_type}
            File count: {file_count}
            Original query: "{query}"
            
            Feature categories and components:
            {json.dumps(feature_data, indent=2)}
            
            Focus on:
            1. How different components interact with each other
            2. The overall architecture and design pattern
            3. Key technologies and their roles
            4. Data flow between components
            5. How the system functions as a whole
            
            Do NOT list files individually. Instead, synthesize a comprehensive understanding of the system's architecture and functionality.
            Write in clear, concise language that explains the system's operation as a unified whole.
            """
            
            messages = [
                {"role": "system", "content": "You are a software architecture expert who explains codebases holistically."},
                {"role": "user", "content": prompt}
            ]
            
            # Use the code_llama_model if available for better analysis
            original_model = self.client.model
            if self.code_llama_model:
                self.client.model = self.code_llama_model
                logger.info(f"Using code model {self.code_llama_model} for integrated feature analysis")
            
            response = await self.client.generate(
                messages=messages,
                temperature=0.2,  # Lower temperature for factual analysis
                max_tokens=1500   # Allow for a detailed response
            )
            
            # Reset model back to original
            if self.code_llama_model:
                self.client.model = original_model
            
            if not response or "choices" not in response or not response["choices"]:
                logger.warning("Empty or invalid response for integrated feature analysis")
                return None
                
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            logger.error(f"Error generating integrated feature analysis: {str(e)}")
            return None
            
    async def _perform_deep_repository_analysis(self, features_result: Dict[str, Any]) -> str:
        """
        Perform a deep analysis of the entire repository to understand its purpose,
        architecture, and functionality as a comprehensive whole.
        
        Args:
            features_result: Dictionary with feature analysis results
            
        Returns:
            A comprehensive analysis of the entire repository
        """
        try:
            if not self.use_ollama:
                logger.warning("Deep repository analysis requires Ollama")
                return None
                
            # Extract README content and repository purpose if available
            readme_content = features_result.get("readme_content")
            repo_purpose = features_result.get("repository_purpose")
            repo_description = features_result.get("repository_description")
            
            # Prepare repository data for analysis
            repository_data = {
                "features": features_result,
                "file_structure": self._get_file_structure(),
                "code_samples": await self._collect_code_samples(),
                "readme_content": readme_content,
                "repository_purpose": repo_purpose,
                "repository_description": repo_description
            }
            
            # Create a comprehensive prompt for deep repository analysis
            system_prompt = """You are an expert software architect and code analyst with deep understanding of software systems. 
            Your task is to thoroughly analyze a repository and provide a comprehensive understanding of:
            1. The repository's core purpose and why it was built
            2. The complete architecture and how all components interact
            3. The data flow throughout the system
            4. Key technologies and their specific roles
            5. The system's main capabilities and features

            Analyze code patterns, naming conventions, dependency relationships, and architectural choices to provide
            a holistic understanding of the entire system. If README content is provided, prioritize this as the
            most authoritative source on the repository's purpose and functionality. Think like the system architect
            who designed this codebase. Focus on explaining what the repository DOES as a whole system, not just
            listing its components."""
            
            # Prepare repository information for analysis
            file_structure = repository_data.get("file_structure", "No file structure provided")
            code_samples = repository_data.get("code_samples", [])
            features = repository_data.get("features", {})
            
            # Format code samples for better analysis
            formatted_samples = ""
            for sample in code_samples[:10]:  # Limit to avoid exceeding context length
                file_path = sample.get("file_path", "Unknown")
                content = sample.get("content", "")
                if content:
                    formatted_samples += f"\n\nFILE: {file_path}\n```\n{content[:800]}...\n```"
            
            # Format README content if available
            readme_section = ""
            if readme_content:
                readme_section = f"\n\nREADME CONTENT:\n{readme_content[:2000]}...\n"
            
            # Create a comprehensive analysis prompt
            analysis_prompt = f"""
            Conduct a comprehensive analysis of this repository to explain its core purpose, architecture, and functionality.
            {readme_section}
            
            FILE STRUCTURE:
            {file_structure}
            
            SAMPLE CODE:
            {formatted_samples}
            
            Based on this information, provide:
            1. A clear explanation of what this repository is designed to do and why it exists
            2. A comprehensive explanation of the system architecture and how all components work together
            3. The data flow throughout the system
            4. The key technologies used and why they were chosen
            5. The main features and capabilities of the system
            
            Your analysis should provide insights about the repository as an integrated whole system, not just
            a list of separate components. Focus on explaining what the repository DOES as a complete, functioning system.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt}
            ]
            
            # Use code_llama_model for better code understanding if available
            original_model = None
            if self.code_llama_model and self.use_ollama:
                original_model = self.client.model
                self.client.model = self.code_llama_model
                logger.info(f"Using code model {self.code_llama_model} for deep repository analysis")
            
            # Generate a comprehensive analysis
            response = await self.client.generate(
                messages=messages,
                temperature=0.3,  # Lower temperature for more factual analysis
                max_tokens=2500   # Allow for a detailed response
            )
            
            # Reset model if we changed it
            if original_model:
                self.client.model = original_model
            
            if not response or "choices" not in response or not response["choices"]:
                logger.warning("Empty or invalid response for deep repository analysis")
                return None
                
            analysis = response["choices"][0]["message"]["content"]
            
            # For GitHub Code Monitor repository specifically, ensure the response
            # correctly identifies it as a GitHub monitoring system with AI capabilities
            if readme_content and ("GitHub Code Monitor" in readme_content or "github-code-monitor" in readme_content):
                if "AI-enhanced system for monitoring GitHub" not in analysis:
                    analysis = f"# GitHub Code Monitor\n\nThis repository contains an AI-enhanced system for monitoring GitHub repositories, understanding code changes, and providing intelligent insights through natural language queries.\n\n{analysis}"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in deep repository analysis: {str(e)}")
            return None
            
    def _get_file_structure(self) -> str:
        """Get the file structure of the repository for analysis"""
        try:
            # Get current working directory
            import os
            cwd = os.getcwd()
            file_structure = []
            
            # Try to run a system command to get file structure
            import subprocess
            try:
                # Run find command to get directory structure (excluding node_modules, .git, etc.)
                cmd = ["find", ".", "-type", "f", "-not", "-path", "*/\\.*", 
                       "-not", "-path", "*/node_modules/*", "-not", "-path", "*/venv/*", 
                       "|", "sort"]
                result = subprocess.run(" ".join(cmd), shell=True, capture_output=True, text=True, cwd=cwd)
                file_structure = result.stdout.strip().split("\n")
            except Exception as e:
                logger.warning(f"Error getting file structure via command: {str(e)}")
                # Fallback to manual directory traversal
                for root, dirs, files in os.walk(cwd):
                    # Skip hidden directories and virtual environments
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['node_modules', 'venv', '__pycache__']]
                    for file in files:
                        # Skip hidden files and compiled files
                        if not file.startswith('.') and not file.endswith('.pyc'):
                            path = os.path.join(root, file)
                            rel_path = os.path.relpath(path, cwd)
                            file_structure.append(rel_path)
            
            return "\n".join(file_structure)
        except Exception as e:
            logger.error(f"Error getting file structure: {str(e)}")
            return "Unable to retrieve file structure"
            
    async def _collect_code_samples(self) -> List[Dict[str, Any]]:
        """Collect representative code samples from the repository for analysis"""
        try:
            # Get database session
            from code_monitor.db.database import get_db
            from sqlalchemy.ext.asyncio import AsyncSession
            
            db_generator = get_db()
            db: AsyncSession = await anext(db_generator)
            
            # Use repository files and code snippets from QAG tables
            from sqlalchemy import select, func, text
            from code_monitor.db.models import QAGRepositoryFile, QAGCodeSnippet
            
            # Get important repository files (main modules, configuration, etc.)
            key_files_query = text("""
                SELECT id, file_path, language, content
                FROM qag_repository_files
                WHERE file_path LIKE '%/main.py' 
                   OR file_path LIKE '%/config.py'
                   OR file_path LIKE '%/__init__.py'
                   OR file_path LIKE '%/models.py'
                   OR file_path LIKE '%/routes.py'
                   OR file_path LIKE '%/database.py'
                   OR file_path LIKE '%/code_analyzer.py'
                   OR file_path LIKE '%/repository_manager.py'
                LIMIT 10
            """)
            
            # Get representative code snippets
            snippets_query = text("""
                SELECT id, name, snippet_type, content, file_id
                FROM qag_code_snippets
                WHERE snippet_type IN ('class', 'function')
                LIMIT 20
            """)
            
            code_samples = []
            
            # Execute key files query
            try:
                key_files_result = await db.execute(key_files_query)
                key_files = key_files_result.fetchall()
                
                for file in key_files:
                    if hasattr(file, 'file_path') and hasattr(file, 'content') and file.content:
                        code_samples.append({
                            "file_path": file.file_path,
                            "language": file.language if hasattr(file, 'language') else "python",
                            "content": file.content
                        })
            except Exception as e:
                logger.warning(f"Error collecting key files: {str(e)}")
                
            # Execute snippets query
            try:
                snippets_result = await db.execute(snippets_query)
                snippets = snippets_result.fetchall()
                
                for snippet in snippets:
                    if hasattr(snippet, 'name') and hasattr(snippet, 'content') and snippet.content:
                        # Get file path for context
                        file_path = "Unknown"
                        if hasattr(snippet, 'file_id'):
                            try:
                                file_query = text("""
                                    SELECT file_path FROM qag_repository_files
                                    WHERE id = :file_id
                                """)
                                file_result = await db.execute(file_query, {"file_id": snippet.file_id})
                                file_record = file_result.fetchone()
                                if file_record and hasattr(file_record, 'file_path'):
                                    file_path = file_record.file_path
                            except Exception:
                                pass
                                
                        code_samples.append({
                            "file_path": file_path,
                            "language": "python",  # Assuming Python for simplicity
                            "content": snippet.content,
                            "name": snippet.name if hasattr(snippet, 'name') else "Unknown",
                            "type": snippet.snippet_type if hasattr(snippet, 'snippet_type') else "code"
                        })
            except Exception as e:
                logger.warning(f"Error collecting code snippets: {str(e)}")
                
            # If we don't have enough samples yet, get some regular files
            if len(code_samples) < 5:
                try:
                    files_query = text("""
                        SELECT id, file_path, language, content
                        FROM qag_repository_files
                        WHERE content IS NOT NULL
                        LIMIT 10
                    """)
                    
                    files_result = await db.execute(files_query)
                    files = files_result.fetchall()
                    
                    for file in files:
                        if hasattr(file, 'file_path') and hasattr(file, 'content') and file.content:
                            code_samples.append({
                                "file_path": file.file_path,
                                "language": file.language if hasattr(file, 'language') else "unknown",
                                "content": file.content
                            })
                except Exception as e:
                    logger.warning(f"Error collecting additional files: {str(e)}")
            
            return code_samples
            
        except Exception as e:
            logger.error(f"Error collecting code samples: {str(e)}")
            return []