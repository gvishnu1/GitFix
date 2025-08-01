#!/usr/bin/env python
"""
Import a complete repository into the database with embeddings.
This script clones the repository, processes all files, and adds them to the database.
"""

import argparse
import asyncio
import os
import sys
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional, AsyncGenerator
from datetime import datetime

# Fix the module import path by adding the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from code_monitor.db.database import get_db
from code_monitor.github_integration.github_client import GitHubClient
from code_monitor.db.models import Repository, RepositoryFile
from code_monitor.ai_processing.embedding import EmbeddingGenerator
from code_monitor.ai_processing.code_parser import CodeParser

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Helper function to create an async generator from get_db
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """Wrapper around get_db to provide compatibility with the code"""
    async for session in get_db():
        yield session

class RepositoryImporter:
    """Class to handle importing a complete repository with embeddings"""
    
    def __init__(self):
        self.github_client = GitHubClient()
        self.embedding_generator = EmbeddingGenerator()
        self.code_parser = CodeParser()
        
    async def import_repository(self, repo_url: str, branch: Optional[str] = None) -> Dict[str, Any]:
        """
        Import a complete repository into the database with embeddings
        
        Args:
            repo_url: URL of the GitHub repository
            branch: Branch to import (defaults to the default branch)
            
        Returns:
            Dictionary with import statistics
        """
        logger.info(f"Starting import of repository: {repo_url}")
        
        stats = {
            "total_files": 0,
            "processed_files": 0,
            "failed_files": 0,
            "languages": {}
        }
        
        # Parse repo URL to get owner and name
        owner, name = self.github_client.parse_repo_url(repo_url)
        if not owner or not name:
            raise ValueError(f"Invalid repository URL: {repo_url}")
        
        # Initialize the GitHub client session    
        await self.github_client._ensure_session()
            
        # Get or create repository record
        async for db in get_async_session():
            repo = await self._get_or_create_repository(db, owner, name, repo_url)
            
            try:
                # Get all files from repository
                files = await self.github_client.get_repository_contents(owner, name, branch)
                stats["total_files"] = len(files)
                
                # Process each file
                for file_info in files:
                    try:
                        # Skip binary files, images, etc.
                        if not self._is_text_file(file_info["path"]):
                            logger.info(f"Skipping binary file: {file_info['path']}")
                            continue
                        
                        # Get file content
                        content = await self.github_client.get_file_content(owner, name, file_info["path"], branch)
                        if not content:
                            logger.warning(f"Could not get content for {file_info['path']}")
                            stats["failed_files"] += 1
                            continue
                        
                        # Determine language
                        language = self._detect_language(file_info["path"])
                        stats["languages"][language] = stats["languages"].get(language, 0) + 1
                        
                        # Create repository file record with embedding
                        await self._create_repository_file(db, repo.id, file_info["path"], content, language)
                        stats["processed_files"] += 1
                        
                        if stats["processed_files"] % 10 == 0:
                            logger.info(f"Processed {stats['processed_files']} files out of {stats['total_files']}")
                            
                    except Exception as e:
                        logger.error(f"Error processing file {file_info['path']}: {str(e)}")
                        stats["failed_files"] += 1
            finally:
                # Clean up the session
                await self.github_client.close()
            
            logger.info(f"Import complete. Processed {stats['processed_files']} files.")
            return stats
            
    async def _get_or_create_repository(self, db: AsyncSession, owner: str, name: str, url: str) -> Repository:
        """Get or create a repository record"""
        from sqlalchemy.future import select
        
        # Check if repository already exists by querying owner and name
        query = select(Repository).where(
            Repository.owner == owner,
            Repository.name == name
        )
        result = await db.execute(query)
        existing_repo = result.scalars().first()
        
        if existing_repo:
            logger.info(f"Repository {owner}/{name} already exists in database")
            return existing_repo
            
        # Create new repository
        repository = Repository(
            owner=owner,
            name=name,
            url=url,
            track_commits=True,
            track_merged_prs=True
        )
        db.add(repository)
        await db.commit()
        await db.refresh(repository)
        
        logger.info(f"Created new repository record for {owner}/{name}")
        return repository
        
    async def _create_repository_file(self, db: AsyncSession, repository_id: int, file_path: str, 
                                     content: str, language: str) -> RepositoryFile:
        """Create a repository file record with embedding"""
        from sqlalchemy.future import select
        
        # Check if the file already exists for this repository
        query = select(RepositoryFile).where(
            RepositoryFile.repository_id == repository_id,
            RepositoryFile.file_path == file_path
        )
        result = await db.execute(query)
        existing_file = result.scalars().first()
        
        # Generate embedding for the file
        embedding = await self.embedding_generator.generate_embedding(content)
        
        if existing_file:
            # Update existing file if content changed
            existing_file.content = content
            existing_file.language = language
            existing_file.embedding = embedding
            existing_file.updated_at = datetime.now()
            existing_file.last_modified_at = datetime.now()
            db.add(existing_file)
            await db.commit()
            await db.refresh(existing_file)
            repo_file = existing_file
        else:
            # Create new file record
            repo_file = RepositoryFile(
                repository_id=repository_id,
                file_path=file_path,
                content=content,
                language=language,
                embedding=embedding,
                last_modified_at=datetime.now(),
                file_metadata={}
            )
            db.add(repo_file)
            await db.commit()
            await db.refresh(repo_file)
        
        # If this is a code file, extract code snippets
        if language in ["python", "javascript", "typescript", "java", "c", "cpp", "go", "rust"]:
            try:
                # TODO: Implement code snippet extraction for repository files
                pass
            except Exception as e:
                logger.warning(f"Failed to extract code snippets from {file_path}: {str(e)}")
        
        return repo_file
        
    def _is_text_file(self, file_path: str) -> bool:
        """Check if a file is likely a text file based on extension"""
        binary_extensions = [
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".ico", ".webp",  # Images
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",  # Documents
            ".zip", ".tar", ".gz", ".rar", ".7z",  # Archives
            ".exe", ".dll", ".so", ".dylib",  # Binaries
            ".pyc", ".pyo", ".o", ".obj",  # Compiled code
            ".ttf", ".otf", ".woff", ".woff2"  # Fonts
        ]
        
        return not any(file_path.lower().endswith(ext) for ext in binary_extensions)
        
    def _detect_language(self, file_path: str) -> str:
        """Detect programming language from file path"""
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

async def main():
    parser = argparse.ArgumentParser(description="Import a complete repository into the database with embeddings")
    parser.add_argument("repo_url", help="URL of the GitHub repository")
    parser.add_argument("--branch", help="Branch to import (defaults to the default branch)")
    
    args = parser.parse_args()
    
    importer = RepositoryImporter()
    stats = await importer.import_repository(args.repo_url, args.branch)
    
    print("\nImport Complete!")
    print(f"Total files: {stats['total_files']}")
    print(f"Processed files: {stats['processed_files']}")
    print(f"Failed files: {stats['failed_files']}")
    
    print("\nLanguage breakdown:")
    for lang, count in sorted(stats["languages"].items(), key=lambda x: x[1], reverse=True):
        print(f"  - {lang}: {count} files")

if __name__ == "__main__":
    asyncio.run(main())