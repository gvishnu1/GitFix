#!/usr/bin/env python3
import asyncio
from code_monitor.db.database import SessionLocal

async def update_qag_repository_id():
    print("Starting QAG repository ID update process with raw SQL...")
    async with SessionLocal() as db:
        try:
            # Begin transaction
            async with db.begin():
                # Use a single SQL statement with multiple updates
                sql = """
                DO $$
                BEGIN
                    -- Update files first
                    UPDATE qag_files SET repository_id = 2 WHERE repository_id = 1;
                    
                    -- Update repository files
                    UPDATE qag_repository_files SET repository_id = 2 WHERE repository_id = 1;
                    
                    -- Update commits
                    UPDATE qag_commits SET repository_id = 2 WHERE repository_id = 1;
                    
                    -- Finally update the repository itself
                    UPDATE qag_repositories SET id = 2 WHERE id = 1;
                END $$;
                """
                
                print("Executing SQL updates...")
                await db.execute(sql)
                print("SQL updates completed successfully")
                
                # Verify the update
                result = await db.execute("SELECT id, name FROM qag_repositories WHERE id = 2")
                repo = result.first()
                if repo:
                    print(f"Successfully verified repository update. New ID: 2, Name: {repo.name}")
                else:
                    print("Warning: Could not verify repository update")
                
            print("All changes committed successfully")
            
        except Exception as e:
            print(f"Error updating repository ID: {str(e)}")
            print("Rolling back all changes.")
            raise

if __name__ == "__main__":
    asyncio.run(update_qag_repository_id())
