#!/usr/bin/env python3
"""Script to delete repository with ID 2 from the database"""

import asyncio
import logging
from sqlalchemy import delete
from code_monitor.db.database import SessionLocal
from code_monitor.db.models import Repository, Commit, File, RepositoryFile

# Set up logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def delete_repository():
    """Delete repository with ID 2 and all associated records"""
    try:
        repo_id = 2
        print(f"Deleting repository with ID {repo_id}...")
        
        async with SessionLocal() as db:
            # Get repository info first for confirmation
            from sqlalchemy.future import select
            stmt = select(Repository).where(Repository.id == repo_id)
            result = await db.execute(stmt)
            repo = result.scalars().first()
            
            if not repo:
                print(f"Repository with ID {repo_id} not found in database.")
                return
                
            print(f"Found repository: {repo.name} (ID: {repo.id})")
            print(f"URL: {repo.url}")
            print("Preparing to delete...")
            
            # Get associated commits
            stmt = select(Commit).where(Commit.repository_id == repo_id)
            result = await db.execute(stmt)
            commits = result.scalars().all()
            commit_ids = [commit.id for commit in commits]
            print(f"Found {len(commits)} commits to delete")
            
            # Delete all files associated with these commits
            try:
                if commit_ids:
                    # Delete files
                    files_stmt = delete(File).where(File.commit_id.in_(commit_ids))
                    result = await db.execute(files_stmt)
                    print(f"Deleted files associated with commits")
            except Exception as e:
                print(f"Error deleting files: {str(e)}")
            
            # Delete repository files
            try:
                repo_files_stmt = delete(RepositoryFile).where(RepositoryFile.repository_id == repo_id)
                result = await db.execute(repo_files_stmt)
                print(f"Deleted repository files")
            except Exception as e:
                print(f"Error deleting repository files: {str(e)}")
            
            # Delete commits
            try:
                commits_stmt = delete(Commit).where(Commit.repository_id == repo_id)
                result = await db.execute(commits_stmt)
                print(f"Deleted commits")
            except Exception as e:
                print(f"Error deleting commits: {str(e)}")
            
            # Finally delete the repository
            try:
                repo_stmt = delete(Repository).where(Repository.id == repo_id)
                result = await db.execute(repo_stmt)
                print(f"Deleted repository")
            except Exception as e:
                print(f"Error deleting repository record: {str(e)}")
            
            # Commit the transaction
            await db.commit()
            print(f"Successfully deleted repository ID {repo_id} and all associated records")
            
    except Exception as e:
        print(f"Error deleting repository: {str(e)}")

if __name__ == "__main__":
    print("Starting repository deletion script...")
    asyncio.run(delete_repository())
    print("Script execution completed.")
