import logging
import difflib
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from code_monitor.db.models import Commit, File, RepositoryFile, CodeSnippet
from code_monitor.db.models import QAGFile, QAGRepositoryFile, QAGCodeSnippet
from code_monitor.ai_processing.embedding import EmbeddingGenerator
from code_monitor.ai_processing.code_parser import CodeParser
from code_monitor.ai_processing.summarizer import CommitSummarizer

logger = logging.getLogger(__name__)

class ChangeProcessor:
    """Process code changes from GitHub."""
    
    def __init__(self):
        self.embedding_generator = EmbeddingGenerator()
        self.code_parser = CodeParser()
        self.commit_summarizer = CommitSummarizer()
        
    async def process_commit_files(self, 
                                  db: AsyncSession, 
                                  commit_id: int, 
                                  commit_detail: Dict[str, Any],
                                  repo_owner: str,
                                  repo_name: str,
                                  table_prefix: str = "") -> None:
        """
        Process files in a commit and store them in the database.
        
        Args:
            db: Database session
            commit_id: ID of the commit in the database
            commit_detail: Commit details from GitHub API
            repo_owner: Repository owner
            repo_name: Repository name
            table_prefix: Optional prefix for table names (e.g., "qag_" for QAG tables)
        """
        files = commit_detail.get("files", [])
        
        # Select the appropriate model class based on table_prefix
        FileModel = QAGFile if table_prefix == "qag_" else File
        RepositoryFileModel = QAGRepositoryFile if table_prefix == "qag_" else RepositoryFile
        CodeSnippetModel = QAGCodeSnippet if table_prefix == "qag_" else CodeSnippet
        
        for file_change in files:
            try:
                file_path = file_change["filename"]
                change_type = self._determine_change_type(file_change)
                
                # Get file contents before and after the change
                content_before, content_after = await self._get_file_contents(file_change)
                
                # Generate diff if both versions are available
                diff = None
                if content_before and content_after:
                    diff = self._generate_diff(content_before, content_after, file_path)
                
                # Determine language
                language = self._detect_language(file_path)
                
                # Create file record
                file_record = FileModel(
                    commit_id=commit_id,
                    file_path=file_path,
                    change_type=change_type,
                    content_before=content_before,
                    content_after=content_after,
                    diff=diff,
                    language=language,
                    file_metadata={}
                )
                
                # Generate embedding for the file (either content_after or content_before)
                content_for_embedding = content_after if content_after else content_before
                if content_for_embedding:
                    embedding = await self.embedding_generator.generate_embedding(content_for_embedding)
                    file_record.embedding = embedding
                
                db.add(file_record)
                await db.flush()  # Flush to get the file_record.id
                
                # Extract code snippets if this is a code file
                if content_after and language in ["python", "javascript", "typescript", "java", "c", "cpp", "go", "rust"]:
                    await self._extract_code_snippets(db, file_record.id, content_after, language, CodeSnippetModel)
                
                # Update repository file if this is not a deletion
                if change_type != "deleted" and content_after:
                    await self._update_repository_file(
                        db, 
                        repo_owner, 
                        repo_name, 
                        file_path, 
                        content_after, 
                        language,
                        RepositoryFileModel
                    )
                
            except Exception as e:
                logger.error(f"Error processing file {file_change.get('filename', 'unknown')}: {str(e)}")
                # Continue with next file
                continue
                
    def _determine_change_type(self, file_change: Dict[str, Any]) -> str:
        """Determine the type of file change."""
        status = file_change.get("status", "")
        
        if status == "added":
            return "added"
        elif status == "removed":
            return "deleted"
        elif status == "renamed":
            return "renamed"
        else:
            return "modified"
            
    async def _get_file_contents(self, file_change: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
        """Get the file contents before and after the change."""
        content_before = None
        content_after = None
        
        # For binary files, skip content retrieval
        if file_change.get("status") == "removed":
            # File was deleted, only get before content
            content_before = file_change.get("patch", "")
        elif file_change.get("status") == "added":
            # File was added, only get after content
            content_after = file_change.get("patch", "")
        else:
            # File was modified or renamed
            patch = file_change.get("patch", "")
            if patch:
                # This is simplified, in a real scenario you'd need to parse the patch
                # to reconstruct before and after content accurately
                content_before = patch
                content_after = patch
            
        return content_before, content_after
        
    def _generate_diff(self, content_before: str, content_after: str, file_path: str) -> str:
        """Generate a diff between before and after content."""
        before_lines = content_before.splitlines()
        after_lines = content_after.splitlines()
        
        diff = difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{file_path}",
            tofile=f"b/{file_path}",
            lineterm=""
        )
        
        return "\n".join(diff)
        
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file path."""
        extensions = {
            ".py": "python",
            ".js": "javascript",
            ".jsx": "javascript",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".java": "java",
            ".c": "c",
            ".cpp": "cpp",
            ".h": "c",
            ".hpp": "cpp",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
            ".html": "html",
            ".css": "css",
            ".scss": "scss",
            ".sass": "sass",
            ".json": "json",
            ".xml": "xml",
            ".yml": "yaml",
            ".yaml": "yaml",
            ".md": "markdown"
        }
        
        for ext, lang in extensions.items():
            if file_path.lower().endswith(ext):
                return lang
                
        return "unknown"
        
    async def _extract_code_snippets(self, db: AsyncSession, file_id: int, content: str, 
                                   language: str, CodeSnippetModel: Any) -> None:
        """Extract code snippets from file content and store them in the database."""
        try:
            snippets = await self.code_parser.parse_code(content, language)
            
            for snippet in snippets:
                snippet_record = CodeSnippetModel(
                    file_id=file_id,
                    snippet_type=snippet["type"],
                    name=snippet["name"],
                    content=snippet["content"],
                    start_line=snippet["start_line"],
                    end_line=snippet["end_line"]
                )
                
                # Generate embedding for the snippet
                embedding = await self.embedding_generator.generate_embedding(snippet["content"])
                snippet_record.embedding = embedding
                
                db.add(snippet_record)
                
        except Exception as e:
            logger.error(f"Error extracting code snippets: {str(e)}")
            
    async def _update_repository_file(self, db: AsyncSession, repo_owner: str, repo_name: str, 
                                    file_path: str, content: str, language: str, 
                                    RepositoryFileModel: Any) -> None:
        """Update or create repository file record with the latest content."""
        from sqlalchemy.future import select
        
        try:
            # First get the repository ID
            from code_monitor.db.models import Repository, QAGRepository
            RepoModel = QAGRepository if RepositoryFileModel == QAGRepositoryFile else Repository
            
            query = select(RepoModel).where(
                RepoModel.owner == repo_owner,
                RepoModel.name == repo_name
            )
            result = await db.execute(query)
            repo = result.scalars().first()
            
            if not repo:
                logger.warning(f"Repository {repo_owner}/{repo_name} not found in database")
                return
                
            # Check if file already exists
            query = select(RepositoryFileModel).where(
                RepositoryFileModel.repository_id == repo.id,
                RepositoryFileModel.file_path == file_path
            )
            result = await db.execute(query)
            repo_file = result.scalars().first()
            
            # Generate embedding for the file
            embedding = await self.embedding_generator.generate_embedding(content)
            
            if repo_file:
                # Update existing file
                repo_file.content = content
                repo_file.embedding = embedding
                repo_file.language = language
                repo_file.last_modified_at = datetime.now()
                db.add(repo_file)
            else:
                # Create new file
                new_file = RepositoryFileModel(
                    repository_id=repo.id,
                    file_path=file_path,
                    content=content,
                    embedding=embedding,
                    language=language,
                    last_modified_at=datetime.now(),
                    file_metadata={}
                )
                db.add(new_file)
                
        except Exception as e:
            logger.error(f"Error updating repository file {file_path}: {str(e)}")