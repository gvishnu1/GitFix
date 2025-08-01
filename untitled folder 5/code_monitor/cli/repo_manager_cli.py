#!/usr/bin/env python
import asyncio
import argparse
import logging
import sys
from sqlalchemy.ext.asyncio import AsyncSession

sys.path.append('.')  # Add the current directory to path for imports

from code_monitor.config import settings
from code_monitor.db.database import SessionLocal
from code_monitor.github_integration.github_client import GitHubClient
from code_monitor.github_integration.processor import ChangeProcessor
from code_monitor.github_integration.repository_manager import RepositoryManager
from code_monitor.db.models import QAGRepository, QAGCommit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize components
github_client = GitHubClient(settings.GITHUB_ACCESS_TOKEN)
change_processor = ChangeProcessor()
repository_manager = RepositoryManager(github_client=github_client, change_processor=change_processor)

async def get_db_session():
    """Get a database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        await db.close()

async def list_repositories():
    """List all repositories and their tracking settings"""
    async with SessionLocal() as db:
        repositories = await repository_manager.list_repositories(db)
        
        if not repositories:
            print("No repositories found.")
            return
            
        print("\nRepository List:")
        print("-" * 80)
        print(f"{'ID':<4} {'Name':<25} {'Owner':<20} {'Track Commits':<15} {'Track Merged PRs':<15} {'Active'}")
        print("-" * 80)
        
        for repo in repositories:
            print(f"{repo['id']:<4} {repo['name'][:25]:<25} {repo['owner'][:20]:<20} "
                  f"{'Yes' if repo['track_commits'] else 'No':<15} "
                  f"{'Yes' if repo['track_merged_prs'] else 'No':<15} "
                  f"{'Yes' if repo['is_active'] else 'No'}")

async def add_repository(repo_url, track_commits=True, track_merged_prs=True):
    """Add a new repository to monitor"""
    async with SessionLocal() as db:
        try:
            result = await repository_manager.add_repository(
                db=db,
                repo_url=repo_url,
                track_commits=track_commits,
                track_merged_prs=track_merged_prs
            )
            
            print(f"Successfully added repository: {result['repository']['name']}")
            print(f"Repository ID: {result['repository']['id']}")
            print(f"Track commits: {'Yes' if result['repository']['track_commits'] else 'No'}")
            print(f"Track merged PRs: {'Yes' if result['repository']['track_merged_prs'] else 'No'}")
            
        except Exception as e:
            print(f"Error adding repository: {str(e)}")
            
async def update_repository_settings(repo_id, track_commits=None, track_merged_prs=None, is_active=None):
    """Update repository tracking settings"""
    async with SessionLocal() as db:
        try:
            result = await repository_manager.update_repository_settings(
                db=db,
                repo_id=repo_id,
                track_commits=track_commits,
                track_merged_prs=track_merged_prs,
                is_active=is_active
            )
            
            print(f"Successfully updated repository: {result['name']}")
            print(f"Track commits: {'Yes' if result['track_commits'] else 'No'}")
            print(f"Track merged PRs: {'Yes' if result['track_merged_prs'] else 'No'}")
            print(f"Active: {'Yes' if result['is_active'] else 'No'}")
            
        except Exception as e:
            print(f"Error updating repository: {str(e)}")

async def load_merged_prs(repo_id, limit=10):
    """Load merged pull requests for a repository"""
    async with SessionLocal() as db:
        try:
            print(f"Loading up to {limit} merged PRs for repository ID {repo_id}...")
            result = await repository_manager.load_merged_pull_requests(
                db=db,
                repo_id=repo_id,
                limit=limit
            )
            
            if result['success']:
                print(f"Success: {result['message']}")
            else:
                print(f"Failed: {result['message']}")
                
        except Exception as e:
            print(f"Error loading merged PRs: {str(e)}")

async def load_commits(repo_id, limit=10):
    """Load regular commits for a repository"""
    async with SessionLocal() as db:
        try:
            print(f"Loading up to {limit} commits for repository ID {repo_id}...")
            result = await repository_manager.load_repository_commits(
                db=db,
                repo_id=repo_id,
                limit=limit
            )
            
            if result['success']:
                print(f"Success: {result['message']}")
            else:
                print(f"Failed: {result['message']}")
                
        except Exception as e:
            print(f"Error loading commits: {str(e)}")

async def add_qag_repository(repo_url="https://github.com/statsperform/qag-uat-arena-dashboard", track_commits=True, track_merged_prs=True):
    """Add the QAG repository to monitor"""
    db = await get_db_session()
    try:
        logger.info(f"Adding QAG repository: {repo_url}")
        
        # Parse URL to get owner and name
        owner, name = repository_manager.github_client.parse_repo_url(repo_url)
        if not owner or not name:
            logger.error(f"Invalid repository URL: {repo_url}")
            return
            
        # Check if repository already exists in the QAG table
        from sqlalchemy.future import select
        query = select(QAGRepository).where(
            QAGRepository.owner == owner,
            QAGRepository.name == name
        )
        result = await db.execute(query)
        existing_repo = result.scalars().first()
        
        if existing_repo:
            logger.info(f"QAG Repository {owner}/{name} already exists in database")
            return existing_repo
            
        # Create new QAG repository record
        repo = QAGRepository(
            owner=owner,
            name=name,
            url=repo_url,
            track_commits=track_commits,
            track_merged_prs=track_merged_prs
        )
        db.add(repo)
        await db.commit()
        await db.refresh(repo)
        
        logger.info(f"Successfully added QAG repository: {owner}/{name} with ID {repo.id}")
        return repo
        
    except Exception as e:
        logger.error(f"Error adding QAG repository: {str(e)}")
        raise
    finally:
        await db.close()

async def load_qag_commits(repo_id, limit=10):
    """Load regular commits for a QAG repository"""
    db = await get_db_session()
    try:
        # Get repository
        from sqlalchemy.future import select
        from datetime import datetime
        import dateutil.parser
        
        query = select(QAGRepository).where(QAGRepository.id == repo_id)
        result = await db.execute(query)
        repo = result.scalars().first()
        
        if not repo:
            logger.error(f"QAG Repository with ID {repo_id} not found")
            return
            
        logger.info(f"Loading commits for QAG repository {repo.name} (ID: {repo_id})")
        
        # Initialize the client
        await repository_manager.github_client._ensure_session()
        
        # Get commits
        commits = await repository_manager.github_client.get_commits(f"{repo.owner}/{repo.name}", limit=limit)
        
        # Process and store each commit to the QAG_commits table
        for commit_data in commits:
            # Check if commit already exists
            commit_hash = commit_data["sha"]
            query = select(QAGCommit).where(
                QAGCommit.repository_id == repo_id,
                QAGCommit.commit_hash == commit_hash
            )
            result = await db.execute(query)
            existing_commit = result.scalars().first()
            
            if existing_commit:
                logger.info(f"Commit {commit_hash[:7]} already exists, skipping")
                continue
                
            # Extract commit info
            author_name = commit_data["commit"]["author"]["name"]
            author_email = commit_data["commit"]["author"]["email"]
            message = commit_data["commit"]["message"]
            timestamp_str = commit_data["commit"]["author"]["date"]
            
            # Parse the timestamp string into a datetime object
            try:
                timestamp = dateutil.parser.parse(timestamp_str)
            except Exception as e:
                logger.error(f"Failed to parse timestamp {timestamp_str}: {str(e)}")
                timestamp = datetime.now()  # Use current time as fallback
            
            # Create commit record in QAGCommit table
            new_commit = QAGCommit(
                repository_id=repo_id,
                commit_hash=commit_hash,
                author=author_name,
                author_email=author_email,
                message=message,
                timestamp=timestamp  # Now using a proper datetime object
            )
            db.add(new_commit)
            await db.flush()  # Ensure the commit is saved before processing files
            
            # Get commit details and process files
            try:
                commit_detail = await repository_manager.github_client.get_commit_detail(
                    repo.owner, repo.name, commit_hash
                )
                
                # Process file changes for this commit and store in QAGFile
                await repository_manager.change_processor.process_commit_files(
                    db, new_commit.id, commit_detail, repo.owner, repo.name, 
                    table_prefix="qag_"  # This tells the processor to use the QAG tables
                )
                
                logger.info(f"Processed commit {commit_hash[:7]}: {message[:50]}")
            except Exception as e:
                logger.error(f"Error processing commit {commit_hash[:7]}: {str(e)}")
                
        await db.commit()
        logger.info(f"Loaded {len(commits)} commits for repository {repo.name}")
        
    except Exception as e:
        logger.error(f"Error loading commits: {str(e)}")
        raise
    finally:
        # Clean up
        await repository_manager.github_client.close()
        await db.close()

async def main():
    parser = argparse.ArgumentParser(description='GitHub Code Monitor Repository Manager CLI')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # List repositories command
    list_parser = subparsers.add_parser('list', help='List all repositories')
    
    # Add repository command
    add_parser = subparsers.add_parser('add', help='Add a new repository')
    add_parser.add_argument('repo_url', help='GitHub repository URL')
    add_parser.add_argument('--no-track-commits', action='store_true', help='Disable tracking of regular commits')
    add_parser.add_argument('--no-track-merged-prs', action='store_true', help='Disable tracking of merged PR commits')
    
    # Update repository settings command
    update_parser = subparsers.add_parser('update', help='Update repository settings')
    update_parser.add_argument('repo_id', type=int, help='Repository ID')
    update_parser.add_argument('--track-commits', action='store_true', help='Enable tracking of regular commits')
    update_parser.add_argument('--no-track-commits', action='store_true', help='Disable tracking of regular commits')
    update_parser.add_argument('--track-merged-prs', action='store_true', help='Enable tracking of merged PR commits')
    update_parser.add_argument('--no-track-merged-prs', action='store_true', help='Disable tracking of merged PR commits')
    update_parser.add_argument('--active', action='store_true', help='Set repository as active')
    update_parser.add_argument('--inactive', action='store_true', help='Set repository as inactive')
    
    # Load merged PRs command
    load_parser = subparsers.add_parser('load-prs', help='Load merged PRs for a repository')
    load_parser.add_argument('repo_id', type=int, help='Repository ID')
    load_parser.add_argument('--limit', type=int, default=10, help='Maximum number of PRs to load')
    
    # Load commits command
    load_commits_parser = subparsers.add_parser('load-commits', help='Load regular commits for a repository')
    load_commits_parser.add_argument('repo_id', type=int, help='Repository ID')
    load_commits_parser.add_argument('--limit', type=int, default=10, help='Maximum number of commits to load')
    
    # Add QAG repository command
    add_qag_parser = subparsers.add_parser('add-qag', help='Add the QAG Arena Dashboard repository')
    add_qag_parser.add_argument('--no-track-commits', action='store_true', help='Disable tracking of regular commits')
    add_qag_parser.add_argument('--no-track-merged-prs', action='store_true', help='Disable tracking of merged PR commits')
    
    # Load QAG commits command
    load_qag_commits_parser = subparsers.add_parser('load-qag-commits', help='Load commits for QAG repository')
    load_qag_commits_parser.add_argument('repo_id', type=int, help='QAG Repository ID')
    load_qag_commits_parser.add_argument('--limit', type=int, default=10, help='Maximum number of commits to load')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        await list_repositories()
        
    elif args.command == 'add':
        track_commits = not args.no_track_commits
        track_merged_prs = not args.no_track_merged_prs
        await add_repository(args.repo_url, track_commits, track_merged_prs)
        
    elif args.command == 'update':
        track_commits = True if args.track_commits else (False if args.no_track_commits else None)
        track_merged_prs = True if args.track_merged_prs else (False if args.no_track_merged_prs else None)
        is_active = True if args.active else (False if args.inactive else None)
        
        await update_repository_settings(args.repo_id, track_commits, track_merged_prs, is_active)
        
    elif args.command == 'load-prs':
        await load_merged_prs(args.repo_id, args.limit)
        
    elif args.command == 'load-commits':
        await load_commits(args.repo_id, args.limit)
        
    elif args.command == 'add-qag':
        track_commits = not args.no_track_commits
        track_merged_prs = not args.no_track_merged_prs
        await add_qag_repository("https://github.com/statsperform/qag-uat-arena-dashboard", track_commits, track_merged_prs)
        
    elif args.command == 'load-qag-commits':
        await load_qag_commits(args.repo_id, args.limit)
        
    else:
        parser.print_help()

if __name__ == '__main__':
    asyncio.run(main())