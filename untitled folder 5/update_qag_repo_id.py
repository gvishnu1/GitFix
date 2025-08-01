#!/usr/bin/env python3
"""Script to update QAG repository ID from 1 to 2"""

import asyncio
import logging
from sqlalchemy import update
from sqlalchemy.future import select
from code_monitor.db.database import SessionLocal
from code_monitor.db.models import QAGRepository, QAGCommit, QAGFile, QAGRepositoryFile, QAGCodeSnippet

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def update_qag_repository_id():
    """Update QAG repository ID from 1 to 2"""
    try:
        old_id = 1
        new_id = 2
        print(f"Updating QAG repository ID from {old_id} to {new_id}...")
        
        async with SessionLocal() as db:
            # First check if repository exists with ID 1
            repo_query = await db.execute(
                select(QAGRepository).where(QAGRepository.id == old_id)
            )
            repo = repo_query.scalars().first()
            
            if not repo:
                print(f"QAG Repository with ID {old_id} not found in database.")
                return
                
            print(f"Found QAG repository: {repo.name} (Current ID: {repo.id})")
            
            # Check if ID 2 already exists
            check_query = await db.execute(
                select(QAGRepository).where(QAGRepository.id == new_id)
            )
            existing_repo = check_query.scalars().first()
            
            if existing_repo:
                print(f"Error: A QAG repository with ID {new_id} already exists: {existing_repo.name}")
                print("Cannot update ID. Please delete the existing repository with ID 2 first.")
                return
            
            # Update the repository ID
            try:
                # Update repository ID
                repo_stmt = update(QAGRepository).where(
                    QAGRepository.id == old_id
                ).values(id=new_id)
                await db.execute(repo_stmt)
                print(f"Updated QAG repository ID from {old_id} to {new_id}")
                
                # Update all related records
                
                # 1. Update commits
                commits_stmt = update(QAGCommit).where(
                    QAGCommit.repository_id == old_id
                ).values(repository_id=new_id)
                result = await db.execute(commits_stmt)
                print(f"Updated repository_id in QAG commits")
                
                # 2. Update repository files
                repo_files_stmt = update(QAGRepositoryFile).where(
                    QAGRepositoryFile.repository_id == old_id
                ).values(repository_id=new_id)
                result = await db.execute(repo_files_stmt)
                print(f"Updated repository_id in QAG repository files")
                
                # Commit the transaction
                await db.commit()
                print(f"Successfully updated QAG repository ID from {old_id} to {new_id}")
                
                # Verify the update
                verify_query = await db.execute(
                    select(QAGRepository).where(QAGRepository.id == new_id)
                )
                updated_repo = verify_query.scalars().first()
                
                if updated_repo:
                    print(f"Verified: QAG repository now has ID {updated_repo.id}, Name: {updated_repo.name}")
                else:
                    print(f"Warning: Could not verify the updated repository with ID {new_id}")
                    
            except Exception as e:
                await db.rollback()
                print(f"Error updating repository ID: {str(e)}")
                print("Rolling back all changes.")
            
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Starting QAG repository ID update script...")
    asyncio.run(update_qag_repository_id())
    print("Script execution completed.")
