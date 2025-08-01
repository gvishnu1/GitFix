#!/usr/bin/env python
"""
Analyze the QAG repository files in the database to get a clear view of the actual codebase.
This script excludes node_modules and other non-essential files.
"""

import asyncio
import sys
import os
from collections import Counter
from sqlalchemy import select
from tabulate import tabulate

# Add the project root directory to sys.path
sys.path.append('.')  # Add the current directory to path for imports

from code_monitor.db.database import SessionLocal
from code_monitor.db.models import QAGRepository, QAGFile, QAGRepositoryFile, QAGCommit

async def analyze_qag_files():
    """Analyze QAG repository files to get a clear view of the actual codebase"""
    async with SessionLocal() as db:
        # Get all repository files (current state)
        repo_files_query = select(QAGRepositoryFile)
        repo_files_result = await db.execute(repo_files_query)
        repo_files = repo_files_result.scalars().all()
        
        # Get all commit files (historical changes)
        commit_files_query = select(QAGFile)
        commit_files_result = await db.execute(commit_files_query)
        commit_files = commit_files_result.scalars().all()
        
        # Filter out node_modules and other non-essential files
        filtered_repo_files = [f for f in repo_files if not f.file_path.startswith('node_modules/') and 
                               not f.file_path.startswith('.git/')]
        filtered_commit_files = [f for f in commit_files if not f.file_path.startswith('node_modules/') and 
                                 not f.file_path.startswith('.git/')]
        
        # Count files by extension
        repo_extensions = Counter([os.path.splitext(f.file_path)[1] for f in filtered_repo_files])
        commit_extensions = Counter([os.path.splitext(f.file_path)[1] for f in filtered_commit_files])
        
        # Count files by language
        repo_languages = Counter([f.language for f in filtered_repo_files])
        commit_languages = Counter([f.language for f in filtered_commit_files])
        
        # Count files by directory
        repo_directories = Counter([os.path.dirname(f.file_path) or 'root' for f in filtered_repo_files])
        
        # Get commit changes
        commits_query = select(QAGCommit)
        commits_result = await db.execute(commits_query)
        commits = commits_result.scalars().all()
        
        # Print summary
        print("\n=== QAG Repository Analysis ===\n")
        
        print("File Counts:")
        print(f"Total repository files in DB: {len(repo_files)}")
        print(f"Actual codebase files (excluding node_modules): {len(filtered_repo_files)}")
        print(f"Total commit file changes in DB: {len(commit_files)}")
        print(f"Actual commit file changes (excluding node_modules): {len(filtered_commit_files)}")
        
        print("\nRepository Files by Directory:")
        directory_data = [[dir, count] for dir, count in repo_directories.most_common()]
        print(tabulate(directory_data, headers=["Directory", "Count"], tablefmt="grid"))
        
        print("\nRepository Files by Language:")
        language_data = [[lang, count] for lang, count in repo_languages.most_common()]
        print(tabulate(language_data, headers=["Language", "Count"], tablefmt="grid"))
        
        print("\nRepository Files by Extension:")
        extension_data = [[ext or '(no extension)', count] for ext, count in repo_extensions.most_common()]
        print(tabulate(extension_data, headers=["Extension", "Count"], tablefmt="grid"))
        
        print("\nCommit History:")
        for commit in commits:
            # Get files for this commit
            commit_files_query = select(QAGFile).where(QAGFile.commit_id == commit.id)
            commit_files_result = await db.execute(commit_files_query)
            files = commit_files_result.scalars().all()
            filtered_files = [f for f in files if not f.file_path.startswith('node_modules/') and 
                             not f.file_path.startswith('.git/')]
            
            print(f"\nCommit: {commit.commit_hash[:10]}... ({commit.id})")
            print(f"Author: {commit.author}")
            print(f"Message: {commit.message}")
            print(f"Date: {commit.timestamp}")
            print(f"Files changed: {len(filtered_files)} (excluding node_modules)")
            
            # Show the first 10 files changed
            if filtered_files:
                print("\nSample of files changed:")
                for i, file in enumerate(filtered_files[:10]):
                    print(f"  {i+1}. {file.file_path} ({file.change_type})")
                if len(filtered_files) > 10:
                    print(f"  ... and {len(filtered_files) - 10} more files")
        
        # List actual codebase files with embeddings
        print("\nActual Codebase Files with Embeddings:")
        actual_files_with_embeddings = [f for f in filtered_repo_files if f.embedding is not None]
        print(f"Count: {len(actual_files_with_embeddings)} out of {len(filtered_repo_files)}")
        
        if actual_files_with_embeddings:
            file_data = [[f.id, f.file_path, f.language or 'unknown', 'Yes' if f.embedding is not None else 'No'] 
                        for f in filtered_repo_files[:20]]
            print(tabulate(file_data, headers=["ID", "File Path", "Language", "Has Embedding"], tablefmt="grid"))
            if len(filtered_repo_files) > 20:
                print(f"... and {len(filtered_repo_files) - 20} more files")

async def main():
    await analyze_qag_files()

if __name__ == "__main__":
    asyncio.run(main())