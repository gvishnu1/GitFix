import logging
import asyncio
import httpx
from typing import List, Dict, Any, Optional, Union
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import update

from code_monitor.db.models import Repository, Commit  # Add Commit import
from code_monitor.github_integration.github_client import GitHubClient
from code_monitor.github_integration.processor import ChangeProcessor

logger = logging.getLogger(__name__)

class RepositoryManager:
    """
    Manager for controlling which repositories have their commits loaded into the database
    and for retroactively loading merged PRs from GitHub repositories.
    """
    
    def __init__(self, github_client: GitHubClient, change_processor: ChangeProcessor):
        self.github_client = github_client
        self.change_processor = change_processor
    
    async def list_repositories(self, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        List all repositories in the database with their tracking settings
        
        Args:
            db: Database session
            
        Returns:
            List of repositories with their tracking settings
        """
        result = await db.execute(select(Repository))
        repositories = result.scalars().all()
        
        return [
            {
                "id": repo.id,
                "name": repo.name, 
                "owner": repo.owner,
                "url": repo.url,
                "is_active": repo.is_active,
                "track_commits": repo.track_commits,
                "track_merged_prs": repo.track_merged_prs
            } 
            for repo in repositories
        ]
    
    async def update_repository_settings(
        self, 
        db: AsyncSession, 
        repo_id: int, 
        track_commits: Optional[bool] = None,
        track_merged_prs: Optional[bool] = None,
        is_active: Optional[bool] = None
    ) -> Dict[str, Any]:
        """
        Update a repository's commit tracking settings
        
        Args:
            db: Database session
            repo_id: Repository ID
            track_commits: Whether to track regular commits
            track_merged_prs: Whether to track merged PR commits
            is_active: Whether the repository is active
            
        Returns:
            Updated repository information
        """
        # Build update dict with only provided values
        update_data = {}
        if track_commits is not None:
            update_data["track_commits"] = track_commits
        if track_merged_prs is not None:
            update_data["track_merged_prs"] = track_merged_prs
        if is_active is not None:
            update_data["is_active"] = is_active
            
        if not update_data:
            # No updates provided
            result = await db.execute(select(Repository).where(Repository.id == repo_id))
            repo = result.scalars().first()
            if not repo:
                raise ValueError(f"Repository with ID {repo_id} not found")
            return {
                "id": repo.id,
                "name": repo.name,
                "track_commits": repo.track_commits,
                "track_merged_prs": repo.track_merged_prs,
                "is_active": repo.is_active
            }
        
        # Update repository
        await db.execute(
            update(Repository)
            .where(Repository.id == repo_id)
            .values(**update_data)
        )
        await db.commit()
        
        # Get updated repository
        result = await db.execute(select(Repository).where(Repository.id == repo_id))
        repo = result.scalars().first()
        if not repo:
            raise ValueError(f"Repository with ID {repo_id} not found")
        
        return {
            "id": repo.id,
            "name": repo.name,
            "track_commits": repo.track_commits,
            "track_merged_prs": repo.track_merged_prs,
            "is_active": repo.is_active
        }
    
    async def load_merged_pull_requests(
        self, 
        db: AsyncSession, 
        repo_id: int, 
        limit: int = 10
    ) -> Dict[str, Any]:
        """
        Retroactively load merged PRs for a repository
        
        Args:
            db: Database session
            repo_id: Repository ID
            limit: Max number of PRs to load (to avoid rate limiting)
            
        Returns:
            Result information
        """
        # Get repository
        result = await db.execute(select(Repository).where(Repository.id == repo_id))
        repo = result.scalars().first()
        if not repo:
            raise ValueError(f"Repository with ID {repo_id} not found")
        
        # Ensure repository has tracking enabled
        if not repo.is_active or not repo.track_merged_prs:
            return {
                "success": False,
                "message": f"Repository {repo.name} has tracking disabled. Enable track_merged_prs first."
            }
        
        # Format repo name for GitHub API
        repo_full_name = f"{repo.owner}/{repo.name}"
        
        # Get merged PRs
        try:
            merged_prs = await self.github_client.get_merged_pull_requests(repo_full_name)
            
            # Limit to avoid overwhelming the system
            merged_prs = merged_prs[:limit]
            
            processed_count = 0
            for pr in merged_prs:
                # Get PR commits
                pr_number = pr["number"]
                pr_commits = await self.github_client.get_pull_request_commits(
                    repo_full_name, pr_number
                )
                
                # Process each commit
                for commit in pr_commits:
                    commit_hash = commit["sha"]
                    
                    # Skip if commit already exists
                    existing_result = await db.execute(
                        select(Repository).join(
                            Repository.commits
                        ).where(
                            Repository.id == repo.id, 
                            Repository.commits.any(commit_hash=commit_hash)
                        )
                    )
                    if existing_result.scalars().first():
                        logger.info(f"Commit {commit_hash} already exists in database, skipping")
                        continue
                    
                    # Get detailed commit info
                    commit_detail = await self.github_client.get_commit(repo_full_name, commit_hash)
                    
                    # Process commit
                    await self.change_processor.process_commit(
                        db=db,
                        repo_data={
                            "name": repo.name,
                            "url": repo.url,
                            "owner": repo.owner
                        },
                        commit_data={
                            "commit_hash": commit_hash,
                            "message": commit["commit"]["message"],
                            "author": commit["commit"]["author"]["name"],
                            "author_email": commit["commit"]["author"]["email"], 
                            "timestamp": commit["commit"]["author"]["date"],
                            "added": [],
                            "modified": [],
                            "removed": []
                        },
                        commit_detail=commit_detail
                    )
                    processed_count += 1
            
            return {
                "success": True,
                "message": f"Successfully processed {processed_count} commits from {len(merged_prs)} merged PRs"
            }
            
        except Exception as e:
            logger.error(f"Error loading merged PRs for {repo_full_name}: {str(e)}")
            return {
                "success": False,
                "message": f"Error loading merged PRs: {str(e)}"
            }
    
    async def add_repository(
        self, 
        db: AsyncSession, 
        repo_url: str, 
        track_commits: bool = True,
        track_merged_prs: bool = True
    ) -> Dict[str, Any]:
        """
        Add a new repository to monitor
        
        Args:
            db: Database session
            repo_url: Repository URL (https://github.com/owner/repo)
            track_commits: Whether to track regular commits
            track_merged_prs: Whether to track merged PR commits
            
        Returns:
            Added repository info
        """
        # Extract owner and name from URL
        parts = repo_url.strip("/").split("/")
        if len(parts) < 5 or "github.com" not in repo_url:
            raise ValueError("Invalid GitHub repository URL")
            
        owner = parts[-2]
        name = parts[-1]
        repo_full_name = f"{owner}/{name}"
        
        try:
            # Verify repository exists
            repo_info = await self.github_client.get_repository(repo_full_name)
            
            # Create repository in database
            repo = Repository(
                name=name,
                url=repo_url,
                owner=owner,
                description=repo_info.get("description", ""),
                track_commits=track_commits,
                track_merged_prs=track_merged_prs
            )
            
            db.add(repo)
            await db.commit()
            await db.refresh(repo)
            
            return {
                "success": True,
                "repository": {
                    "id": repo.id,
                    "name": repo.name,
                    "owner": repo.owner,
                    "url": repo.url,
                    "track_commits": repo.track_commits,
                    "track_merged_prs": repo.track_merged_prs,
                    "is_active": repo.is_active
                }
            }
            
        except Exception as e:
            logger.error(f"Error adding repository {repo_url}: {str(e)}")
            raise
        
    async def load_repository_commits(
        self, 
        db: AsyncSession, 
        repo_id: int, 
        limit: int = 10
    ) -> Dict[str, Any]:
        """Load regular commits for a repository"""
        repo_full_name = None
        try:
            # Get repository info - outside transaction
            result = await db.execute(select(Repository).filter(Repository.id == repo_id))
            repo = result.scalars().first()
            
            if not repo:
                return {
                    "success": False,
                    "message": f"Repository with ID {repo_id} not found"
                }
                
            repo_full_name = f"{repo.owner}/{repo.name}"
            logger.info(f"Loading commits for {repo_full_name}")
            
            # Get commits from GitHub
            response = await self.github_client.get_commits(repo_full_name)
            
            if not response:
                return {
                    "success": False,
                    "message": f"Failed to get commits from GitHub"
                }
                
            commits = response
            processed_count = 0
            
            # Process each commit
            for commit_data in commits[:limit]:
                try:
                    commit_hash = commit_data["sha"]
                    
                    # Check if commit exists - outside transaction
                    existing = await db.execute(
                        select(Commit).where(Commit.commit_hash == commit_hash)
                    )
                    if existing.scalars().first():
                        logger.info(f"Commit {commit_hash} already exists, skipping")
                        continue
                    
                    # Get detailed commit info
                    commit_detail = await self.github_client.get_commit(repo_full_name, commit_hash)
                    if not commit_detail:
                        logger.warning(f"Could not get details for commit {commit_hash}")
                        continue
                        
                    # Create a new session for each commit to avoid transaction conflicts
                    async with db.begin_nested():
                        await self.change_processor.process_commit(
                            db=db,
                            repo_data={
                                "name": repo.name,
                                "url": repo.url,
                                "owner": repo.owner
                            },
                            commit_data={
                                "commit_hash": commit_hash,
                                "message": commit_data["commit"]["message"],
                                "author": commit_data["commit"]["author"]["name"],
                                "author_email": commit_data["commit"]["author"]["email"],
                                "timestamp": commit_data["commit"]["author"]["date"],
                                "added": [],
                                "modified": [],
                                "removed": []
                            },
                            commit_detail=commit_detail
                        )
                        processed_count += 1
                        logger.info(f"Successfully processed commit {commit_hash}")
                        
                except Exception as e:
                    logger.error(f"Error processing commit: {str(e)}")
                    continue
            
            # Commit all changes
            await db.commit()
            
            return {
                "success": True,
                "message": f"Successfully processed {processed_count} new commits from repository {repo.name}"
            }
            
        except Exception as e:
            logger.error(f"Error loading commits for {repo_full_name}: {str(e)}")
            await db.rollback()
            return {
                "success": False,
                "message": f"Error loading commits: {str(e)}"
            }