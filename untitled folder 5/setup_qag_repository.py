#!/usr/bin/env python
"""
Script to set up the QAG Arena Dashboard repository in the separate tables.
This performs a complete workflow:
1. Add the QAG repository
2. Import all repository files
3. Load commit history
"""

import asyncio
import logging
import sys
import os
import argparse
from datetime import datetime

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from code_monitor.cli.repo_manager_cli import add_qag_repository, load_qag_commits
from code_monitor.cli.import_qag_repository import QAGRepositoryImporter
from code_monitor.db.database import SessionLocal

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def setup_qag_repository():
    """Set up the QAG Arena Dashboard repository"""
    try:
        # Step 1: Add the QAG repository to database
        logger.info("Step 1: Adding QAG repository to database...")
        repo = await add_qag_repository()
        if not repo:
            logger.error("Failed to add QAG repository")
            return False
        
        logger.info(f"Successfully added QAG repository with ID: {repo.id}")
        
        # Step 2: Import all repository files
        logger.info(f"Step 2: Importing QAG repository files...")
        importer = QAGRepositoryImporter()
        stats = await importer.import_repository()
        
        logger.info(f"Imported {stats['processed_files']} files from QAG repository")
        logger.info(f"Language breakdown: {stats['languages']}")
        
        # Step 3: Load commit history
        logger.info(f"Step 3: Loading QAG commit history...")
        await load_qag_commits(repo.id, limit=20)
        
        logger.info("QAG repository setup completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error setting up QAG repository: {str(e)}")
        return False

async def test_qag_setup():
    """Test if the QAG repository was set up correctly"""
    try:
        from sqlalchemy.future import select
        from code_monitor.db.models import QAGRepository, QAGCommit, QAGFile, QAGRepositoryFile
        
        print("\n" + "="*50)
        print("🔍 QAG REPOSITORY TEST RESULTS")
        print("="*50)
        
        async with SessionLocal() as db:
            # Get repository
            query = select(QAGRepository)
            result = await db.execute(query)
            repo = result.scalars().first()
            
            if not repo:
                print("❌ ERROR: No QAG repository found in the database")
                logger.error("No QAG repository found in the database")
                return False
                
            print(f"✅ QAG Repository found: {repo.name} (ID: {repo.id})")
            logger.info(f"QAG Repository found: {repo.name} (ID: {repo.id})")
            
            # Get commit count
            query = select(QAGCommit).where(QAGCommit.repository_id == repo.id)
            result = await db.execute(query)
            commits = result.scalars().all()
            commit_count = len(commits)
            print(f"✅ Found {commit_count} commits in QAG repository")
            logger.info(f"Found {commit_count} commits in QAG repository")
            
            # Get file count
            query = select(QAGRepositoryFile).where(QAGRepositoryFile.repository_id == repo.id)
            result = await db.execute(query)
            files = result.scalars().all()
            file_count = len(files)
            print(f"✅ Found {file_count} files in QAG repository")
            logger.info(f"Found {file_count} files in QAG repository")
            
            # Show sample files by language
            languages = {}
            for file in files:
                if not file.language:
                    continue
                if file.language not in languages:
                    languages[file.language] = 0
                languages[file.language] += 1
                
            print("\nLanguage breakdown:")
            logger.info("Language breakdown:")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {lang}: {count} files")
                logger.info(f"  - {lang}: {count} files")
                
            # Get most recent commit
            if commits:
                latest_commit = commits[0]  # Assuming commits are ordered by date desc
                commit_time = latest_commit.timestamp
                if isinstance(commit_time, str):
                    # Parse the timestamp if it's a string
                    from dateutil import parser
                    commit_time = parser.parse(commit_time)
                    
                print(f"\nLatest commit: {latest_commit.commit_hash[:7]} - {latest_commit.message[:50]}")
                print(f"Commit date: {commit_time}")
                logger.info(f"Latest commit: {latest_commit.commit_hash[:7]} - {latest_commit.message[:50]}")
                logger.info(f"Commit date: {commit_time}")
                
                # Get files in most recent commit
                query = select(QAGFile).where(QAGFile.commit_id == latest_commit.id)
                result = await db.execute(query)
                commit_files = result.scalars().all()
                print(f"Files in latest commit: {len(commit_files)}")
                logger.info(f"Files in latest commit: {len(commit_files)}")
                
                # Show changed files
                if commit_files:
                    print("\nChanged files:")
                    logger.info("Changed files:")
                    for file in commit_files[:5]:  # Show first 5 files
                        print(f"  - {file.file_path} ({file.change_type})")
                        logger.info(f"  - {file.file_path} ({file.change_type})")
            
            # Verification success criteria
            print("\n" + "-"*50)
            if commit_count > 0 and file_count > 0:
                print("✅ QAG Repository setup verification PASSED")
                logger.info("✅ QAG Repository setup verification PASSED")
                print("-"*50 + "\n")
                return True
            else:
                print("❌ QAG Repository setup verification FAILED: Missing commits or files")
                logger.warning("❌ QAG Repository setup verification FAILED: Missing commits or files")
                print("-"*50 + "\n")
                return False
                        
    except Exception as e:
        print(f"❌ ERROR: Error testing QAG repository setup: {str(e)}")
        logger.error(f"Error testing QAG repository setup: {str(e)}")
        print("-"*50 + "\n")
        return False

async def main():
    """Main function to run setup and test"""
    parser = argparse.ArgumentParser(description="Set up and test QAG repository")
    parser.add_argument("--test-only", action="store_true", help="Only run the test without setting up")
    args = parser.parse_args()
    
    if args.test_only:
        logger.info("Running QAG repository test only...")
        success = await test_qag_setup()
    else:
        success = await setup_qag_repository()
        
        if success:
            logger.info("\nTesting QAG repository data...\n")
            test_success = await test_qag_setup()
            
            if test_success:
                logger.info("\nSetup and verification complete!")
                logger.info("You can now query the QAG repository through the chat interface.")
                logger.info("Run the application with: python start.sh")
            else:
                logger.error("\nQAG repository was set up but verification failed.")
        else:
            logger.error("QAG repository setup failed.")

if __name__ == "__main__":
    asyncio.run(main())