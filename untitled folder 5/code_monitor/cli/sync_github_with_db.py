#!/usr/bin/env python
"""
Simple script to synchronize GitHub files with database files.
This script skips embedding generation entirely to avoid dimension issues.
It also skips binary files to avoid UTF-8 encoding errors.
"""

import asyncio
import sys
import getpass
import logging
import os
from datetime import datetime

# Add the project root directory to sys.path
sys.path.append('.')

from sqlalchemy import select, delete
from code_monitor.db.database import SessionLocal
from code_monitor.db.models import QAGRepository, QAGRepositoryFile, QAGFile
from code_monitor.github_integration.github_client import GitHubClient
from code_monitor.config import settings

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ANSI colors for terminal output
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
BLUE = '\033[94m'
ENDC = '\033[0m'
BOLD = '\033[1m'

# Define text file extensions
TEXT_FILE_EXTENSIONS = [
    '.py', '.js', '.jsx', '.ts', '.tsx', '.html', '.css', '.scss', '.json',
    '.md', '.txt', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
    '.sh', '.bash', '.env', '.gitignore', '.gitattributes'
]

def is_likely_text_file(file_path):
    """Check if a file is likely to be a text file based on its extension"""
    _, ext = os.path.splitext(file_path.lower())
    # Include files without extensions that are commonly text (like README, LICENSE)
    basename = os.path.basename(file_path)
    if ext == '' and basename.upper() in ['README', 'LICENSE', 'CHANGELOG', 'CONTRIBUTING', 'AUTHORS', 'CONTRIBUTORS']:
        return True
    return ext in TEXT_FILE_EXTENSIONS

async def sync_github_with_db():
    """Synchronize GitHub files with database files"""
    print(f"{BLUE}=== Syncing GitHub Files with Database ==={ENDC}")
    
    # Initialize GitHub client
    github_client = GitHubClient(access_token=settings.GITHUB_ACCESS_TOKEN)
    repo_url = "https://github.com/statsperform/qag-uat-arena-dashboard"
    owner, name = github_client.parse_repo_url(repo_url)
    
    if not owner or not name:
        print(f"{RED}Invalid repository URL: {repo_url}{ENDC}")
        return
    
    print(f"Repository: {owner}/{name}")
    
    try:
        # Initialize GitHub session
        await github_client._ensure_session()
        
        # Get GitHub repository files
        print(f"{BLUE}Fetching files from GitHub...{ENDC}")
        try:
            github_files = await github_client.get_repository_contents(owner, name)
            # Filter files (exclude node_modules, .git, binary files)
            github_files = [
                f for f in github_files
                if not f["path"].startswith("node_modules/") and
                not f["path"].startswith(".git/")
            ]
            
            # Only keep text files and skip binary files
            text_github_files = [f for f in github_files if is_likely_text_file(f["path"])]
            
            print(f"Found {len(github_files)} total files in GitHub repository")
            print(f"Found {len(text_github_files)} text files in GitHub repository")
            
            # Use only text files for database sync
            github_files = text_github_files
            
        except Exception as e:
            print(f"{RED}Error retrieving files from GitHub: {str(e)}{ENDC}")
            return
            
        async with SessionLocal() as db:
            # Get repository ID
            repo_query = select(QAGRepository).where(
                QAGRepository.owner == owner,
                QAGRepository.name == name
            )
            repo_result = await db.execute(repo_query)
            repo = repo_result.scalars().first()
            
            if not repo:
                print(f"{RED}Repository not found in database{ENDC}")
                return
                
            print(f"Found repository in database with ID: {repo.id}")
            
            # Get database repository files
            repo_files_query = select(QAGRepositoryFile).where(
                QAGRepositoryFile.repository_id == repo.id
            )
            repo_files_result = await db.execute(repo_files_query)
            db_repo_files = list(repo_files_result.scalars().all())
            
            # Filter repository files to exclude node_modules and only include text files
            filtered_db_files = [
                f for f in db_repo_files
                if not f.file_path.startswith("node_modules/") and
                not f.file_path.startswith(".git/") and
                is_likely_text_file(f.file_path)
            ]
            
            print(f"Found {len(filtered_db_files)} text files in database repository table")
            
            # Create sets of file paths
            github_file_paths = {f["path"] for f in github_files}
            db_file_paths = {f.file_path for f in filtered_db_files}
            
            # Identify files to add and remove
            files_to_add = github_file_paths - db_file_paths
            files_to_remove = db_file_paths - github_file_paths
            
            print(f"Text files to add: {len(files_to_add)}")
            print(f"Text files to remove: {len(files_to_remove)}")
            
            # Add missing files
            for file_path in files_to_add:
                print(f"Adding {file_path} to database...")
                try:
                    # Get file content
                    content = await github_client.get_file_content(owner, name, file_path)
                    if content:
                        # Determine language based on extension
                        language = "unknown"
                        if file_path.endswith(".py"):
                            language = "python"
                        elif file_path.endswith(".js"):
                            language = "javascript"
                        elif file_path.endswith(".jsx"):
                            language = "javascript"
                        elif file_path.endswith(".ts"):
                            language = "typescript"
                        elif file_path.endswith(".tsx"):
                            language = "typescript"
                        elif file_path.endswith(".json"):
                            language = "json"
                        elif file_path.endswith(".md"):
                            language = "markdown"
                        elif file_path.endswith(".html"):
                            language = "html"
                        elif file_path.endswith(".css"):
                            language = "css"
                        elif file_path.endswith(".scss"):
                            language = "scss"
                            
                        # Create file record WITHOUT embedding
                        new_file = QAGRepositoryFile(
                            repository_id=repo.id,
                            file_path=file_path,
                            content=content,
                            language=language,
                            last_modified_at=datetime.now(),
                            file_metadata={}
                        )
                        db.add(new_file)
                        await db.flush()  # Flush after each file to avoid large transaction
                    else:
                        print(f"{YELLOW}Could not get content for {file_path}{ENDC}")
                except Exception as e:
                    print(f"{YELLOW}Error adding file {file_path}: {str(e)}{ENDC}")
            
            # Remove files that don't exist in GitHub
            for file_path in files_to_remove:
                print(f"Removing {file_path} from database...")
                # Find file ID
                file_id = next((f.id for f in filtered_db_files if f.file_path == file_path), None)
                if file_id:
                    file_delete_stmt = delete(QAGRepositoryFile).where(
                        QAGRepositoryFile.id == file_id
                    )
                    await db.execute(file_delete_stmt)
            
            # Get all commit files with a separate transaction to avoid autoflush issues
            print(f"{BLUE}Checking commit files...{ENDC}")
            commit_files_query = select(QAGFile)
            commit_files_result = await db.execute(commit_files_query)
            commit_files = list(commit_files_result.scalars().all())
            
            # Filter excess commit files - only looking at text files
            filtered_commit_files = [
                f for f in commit_files
                if f.file_path not in github_file_paths and
                not f.file_path.startswith("node_modules/") and
                not f.file_path.startswith(".git/") and
                is_likely_text_file(f.file_path)
            ]
            
            print(f"Found {len(filtered_commit_files)} excess text commit files to remove")
            
            # Delete excess commit files
            if filtered_commit_files:
                delete_count = 0
                for commit_file in filtered_commit_files:
                    try:
                        file_delete_stmt = delete(QAGFile).where(
                            QAGFile.id == commit_file.id
                        )
                        await db.execute(file_delete_stmt)
                        await db.flush()  # Flush after each delete to avoid large transaction
                        delete_count += 1
                    except Exception as e:
                        print(f"{YELLOW}Error deleting commit file {commit_file.file_path}: {str(e)}{ENDC}")
                
                print(f"Removed {delete_count} excess commit files")
            
            # Commit changes
            await db.commit()
            print(f"{GREEN}Successfully synchronized text files between GitHub and database{ENDC}")
            
    except Exception as e:
        print(f"{RED}Error: {str(e)}{ENDC}")
        import traceback
        traceback.print_exc()
    finally:
        # Close GitHub client session
        await github_client.close()
        
    print(f"{BLUE}Next step: Run check_qag_file_counts.py to verify file counts{ENDC}")

if __name__ == "__main__":
    asyncio.run(sync_github_with_db())