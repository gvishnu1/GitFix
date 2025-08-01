#!/usr/bin/env python
"""
Compare the actual GitHub repository files with what's stored in the database.
This helps identify any discrepancies between GitHub and the database.
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
from code_monitor.db.models import QAGRepository, QAGRepositoryFile
from code_monitor.github_integration.github_client import GitHubClient
from code_monitor.config import settings

async def compare_github_with_db(repo_url=None, access_token=None):
    """Compare GitHub repository files with what's stored in the database"""
    
    # If no repo URL provided, use the QAG repo URL
    if not repo_url:
        repo_url = "https://github.com/statsperform/qag-uat-arena-dashboard"
    
    # Use provided access token or get from settings
    if not access_token:
        access_token = settings.GITHUB_ACCESS_TOKEN
        if not access_token:
            print("No GitHub access token found in settings. Please provide one with --token.")
            print("You can set it in the .env file as GITHUB_ACCESS_TOKEN or pass it as an argument.")
            return
    
    github_client = GitHubClient(access_token=access_token)
    owner, name = github_client.parse_repo_url(repo_url)
    if not owner or not name:
        print(f"Invalid repository URL: {repo_url}")
        return
    
    print(f"\n=== Comparing GitHub Repository with Database ===")
    print(f"Repository: {owner}/{name}")
    print(f"Using authentication: {'Yes' if access_token else 'No'}")
    
    # Initialize the GitHub client session
    await github_client._ensure_session()
    
    try:
        # Get files from GitHub
        github_files = await github_client.get_repository_contents(owner, name)
        
        # Filter out files we typically don't care about
        filtered_github_files = [
            f for f in github_files
            if not f["path"].startswith("node_modules/")
            and not f["path"].startswith(".git/")
        ]
        
        # Get file paths only
        github_file_paths = set(f["path"] for f in filtered_github_files)
        
        # Get files from database
        async with SessionLocal() as db:
            # Get repository ID
            repo_query = select(QAGRepository).where(
                QAGRepository.owner == owner,
                QAGRepository.name == name
            )
            repo_result = await db.execute(repo_query)
            repo = repo_result.scalars().first()
            
            if not repo:
                print(f"Repository {owner}/{name} not found in database.")
                return
            
            # Get all repository files
            repo_files_query = select(QAGRepositoryFile).where(QAGRepositoryFile.repository_id == repo.id)
            repo_files_result = await db.execute(repo_files_query)
            db_files = repo_files_result.scalars().all()
            
            # Filter out files we typically don't care about
            filtered_db_files = [
                f for f in db_files
                if not f.file_path.startswith("node_modules/")
                and not f.file_path.startswith(".git/")
            ]
            
            # Get file paths only
            db_file_paths = set(f.file_path for f in filtered_db_files)
        
        # Compare the sets
        in_github_not_in_db = github_file_paths - db_file_paths
        in_db_not_in_github = db_file_paths - github_file_paths
        in_both = github_file_paths.intersection(db_file_paths)
        
        # Print results
        print(f"\nSummary:")
        print(f"Total files in GitHub (excluding node_modules, .git): {len(github_file_paths)}")
        print(f"Total files in Database (excluding node_modules, .git): {len(db_file_paths)}")
        print(f"Files in both GitHub and Database: {len(in_both)}")
        print(f"Files in GitHub but not in Database: {len(in_github_not_in_db)}")
        print(f"Files in Database but not in GitHub: {len(in_db_not_in_github)}")
        
        # Show files that exist in GitHub but not in the database
        if in_github_not_in_db:
            print(f"\nFiles in GitHub that are missing from the Database:")
            for i, file_path in enumerate(sorted(in_github_not_in_db)):
                print(f"  {i+1}. {file_path}")
        
        # Show files that exist in the database but not in GitHub
        if in_db_not_in_github:
            print(f"\nFiles in Database that are not in GitHub:")
            for i, file_path in enumerate(sorted(in_db_not_in_github)):
                print(f"  {i+1}. {file_path}")
        
        # Print list of files in both
        if in_both:
            print(f"\nFiles that exist in both GitHub and Database ({len(in_both)}):")
            for i, file_path in enumerate(sorted(in_both)[:20]):
                print(f"  {i+1}. {file_path}")
            if len(in_both) > 20:
                print(f"  ... and {len(in_both) - 20} more files")
                
        # Check if DB files have embeddings
        if filtered_db_files:
            db_files_with_embeddings = [f for f in filtered_db_files if f.embedding is not None]
            db_files_without_embeddings = [f for f in filtered_db_files if f.embedding is None]
            
            print(f"\nDatabase files with embeddings: {len(db_files_with_embeddings)} of {len(filtered_db_files)}")
            if db_files_without_embeddings:
                print(f"Database files missing embeddings ({len(db_files_without_embeddings)}):")
                for i, file in enumerate(db_files_without_embeddings[:10]):
                    print(f"  {i+1}. {file.file_path}")
                if len(db_files_without_embeddings) > 10:
                    print(f"  ... and {len(db_files_without_embeddings) - 10} more files")
    
    finally:
        # Clean up the session
        await github_client.close()

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare GitHub repository files with database records")
    parser.add_argument("--repo-url", help="GitHub repository URL to check")
    parser.add_argument("--token", help="GitHub access token for authentication")
    
    args = parser.parse_args()
    await compare_github_with_db(args.repo_url, args.token)

if __name__ == "__main__":
    asyncio.run(main())