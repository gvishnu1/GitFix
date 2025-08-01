#!/usr/bin/env python
import asyncio
import sys
import logging
from sqlalchemy import delete
from sqlalchemy.future import select

sys.path.append('.')  # Add the current directory to path for imports

from code_monitor.db.database import SessionLocal
from code_monitor.db.models import Repository, Commit, File, CodeSnippet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def reset_repository(repo_id):
    """Delete all commits for a repository to allow reprocessing"""
    async with SessionLocal() as db:
        try:
            # Get repository
            result = await db.execute(
                select(Repository).where(Repository.id == repo_id)
            )
            repo = result.scalars().first()
            
            if not repo:
                logger.error(f"Repository with ID {repo_id} not found")
                return
                
            logger.info(f"Resetting repository: {repo.name} (ID: {repo_id})")
            
            # Get all commits for this repository
            result = await db.execute(
                select(Commit).where(Commit.repository_id == repo_id)
            )
            commits = result.scalars().all()
            
            # Get all file IDs for these commits
            file_ids = []
            for commit in commits:
                result = await db.execute(
                    select(File).where(File.commit_id == commit.id)
                )
                files = result.scalars().all()
                file_ids.extend([f.id for f in files])
            
            # Delete code snippets associated with these files
            if file_ids:
                await db.execute(
                    delete(CodeSnippet).where(CodeSnippet.file_id.in_(file_ids))
                )
                logger.info(f"Deleted code snippets for {len(file_ids)} files")
                
            # Delete files associated with these commits
            for commit in commits:
                await db.execute(
                    delete(File).where(File.commit_id == commit.id)
                )
            logger.info(f"Deleted files for {len(commits)} commits")
            
            # Delete commits
            await db.execute(
                delete(Commit).where(Commit.repository_id == repo_id)
            )
            logger.info(f"Deleted {len(commits)} commits from repository {repo.name}")
            
            await db.commit()
            logger.info("Reset complete")
            
        except Exception as e:
            logger.error(f"Error resetting repository: {str(e)}")
            await db.rollback()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python reset_repo.py <repository_id>")
        sys.exit(1)
        
    repo_id = int(sys.argv[1])
    asyncio.run(reset_repository(repo_id))