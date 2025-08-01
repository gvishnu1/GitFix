#!/usr/bin/env python3
import asyncio
from sqlalchemy.future import select
from code_monitor.db.database import SessionLocal
from code_monitor.db.models import QAGRepository, QAGCommit, QAGFile, QAGRepositoryFile

async def update_qag_repository_id():
    """Update QAG repository ID from 1 to 2 and update all related records"""
    try:
        old_id = 1
        new_id = 2
        print(f"Starting update of QAG repository ID from {old_id} to {new_id}...")
        
        async with SessionLocal() as db:
            async with db.begin():
                # First, verify the repository exists
                repo_stmt = select(QAGRepository).where(QAGRepository.id == old_id)
                result = await db.execute(repo_stmt)
                repo = result.scalars().first()
                
                if not repo:
                    print(f"QAG Repository with ID {old_id} not found!")
                    return
                    
                print(f"Found repository: {repo.name}")
                
                # Update QAGRepositoryFile records first (they reference repository_id)
                await db.execute(
                    f"""UPDATE qag_repository_files 
                        SET repository_id = {new_id} 
                        WHERE repository_id = {old_id}"""
                )
                print("Updated qag_repository_files")
                
                # Update QAGCommit records (they reference repository_id)
                await db.execute(
                    f"""UPDATE qag_commits 
                        SET repository_id = {new_id} 
                        WHERE repository_id = {old_id}"""
                )
                print("Updated qag_commits")
                
                # Finally update the repository itself
                await db.execute(
                    f"""UPDATE qag_repositories 
                        SET id = {new_id}, 
                            updated_at = now() 
                        WHERE id = {old_id}"""
                )
                print("Updated qag_repositories")
                
                print(f"Successfully updated QAG repository ID from {old_id} to {new_id}")
                
    except Exception as e:
        print(f"Error updating repository ID: {str(e)}")
        print("Rolling back all changes.")
        raise
    finally:
        print("Script execution completed.")

if __name__ == "__main__":
    asyncio.run(update_qag_repository_id())
