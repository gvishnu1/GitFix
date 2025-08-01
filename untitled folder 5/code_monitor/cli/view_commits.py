#!/usr/bin/env python
import asyncio
import argparse
import sys
from tabulate import tabulate
from sqlalchemy import select
from sqlalchemy.orm import selectinload

sys.path.append('.')  # Add the current directory to path for imports

from code_monitor.db.database import SessionLocal
from code_monitor.db.models import Repository, Commit, File
from code_monitor.config import settings

async def display_commits_and_files(repo_id=None, limit=10, show_content=False):
    """Display commits and their associated files from the database"""
    async with SessionLocal() as db:
        # Build the query - select commits with their associated files
        query = (
            select(Commit)
            .options(selectinload(Commit.files))
            .order_by(Commit.created_at.desc())
            .limit(limit)
        )
        
        # If repo_id is provided, filter by repository
        if repo_id:
            query = query.filter(Commit.repository_id == repo_id)
            
        # Execute the query
        result = await db.execute(query)
        commits = result.scalars().all()
        
        if not commits:
            repo_text = f"repository with ID {repo_id}" if repo_id else "any repository"
            print(f"No commits found for {repo_text} in the database.")
            return
        
        print(f"\nFound {len(commits)} commits in the database:")
        print("-" * 80)
        
        for commit in commits:
            # Get repository name
            repo_result = await db.execute(
                select(Repository).filter(Repository.id == commit.repository_id)
            )
            repo = repo_result.scalars().first()
            repo_name = repo.name if repo else "Unknown"
            
            # Display commit info
            print(f"Commit: {commit.commit_hash}")
            print(f"Repository: {repo_name} (ID: {commit.repository_id})")
            print(f"Author: {commit.author} <{commit.author_email}>")
            print(f"Date: {commit.timestamp}")
            print(f"Message: {commit.message}")
            
            if commit.summary:
                print(f"\nAI Summary: {commit.summary}")
            
            # Display files
            if commit.files:
                print("\nFiles changed:")
                file_data = []
                for file in commit.files:
                    file_data.append([
                        file.id,
                        file.file_path,
                        file.change_type,
                        file.language or 'Unknown',
                        "Yes" if file.embedding is not None else "No"
                    ])
                
                headers = ["ID", "File Path", "Change Type", "Language", "Has Embedding"]
                print(tabulate(file_data, headers=headers, tablefmt="grid"))
                
                # If show_content flag is set, display file content for each file
                if show_content:
                    for file in commit.files:
                        print(f"\nFile: {file.file_path} (ID: {file.id})")
                        print("-" * 40)
                        if file.content_after:
                            print("Content:")
                            print(file.content_after[:1000] + "..." if len(file.content_after) > 1000 else file.content_after)
                        else:
                            print("No content available for this file")
                        print("-" * 40)
            else:
                print("\nNo files associated with this commit.")
            
            print("\n" + "-" * 80)

async def main():
    parser = argparse.ArgumentParser(description='View commits and files in the database')
    parser.add_argument('--repo-id', type=int, help='Filter by repository ID')
    parser.add_argument('--limit', type=int, default=10, help='Maximum number of commits to show')
    parser.add_argument('--show-content', action='store_true', help='Display file content')
    
    args = parser.parse_args()
    
    try:
        await display_commits_and_files(args.repo_id, args.limit, args.show_content)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())