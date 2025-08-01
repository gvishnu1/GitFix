import hmac
import hashlib
import json
from fastapi import Request, HTTPException, Depends
from sqlalchemy.ext.asyncio import AsyncSession
import logging

from code_monitor.config import settings
from code_monitor.db.database import get_db
from code_monitor.github_integration.github_client import GitHubClient
from code_monitor.github_integration.processor import ChangeProcessor

logger = logging.getLogger(__name__)

class WebhookHandler:
    def __init__(self):
        self.secret = settings.GITHUB_WEBHOOK_SECRET
        self.github_client = GitHubClient(settings.GITHUB_ACCESS_TOKEN)
        self.change_processor = ChangeProcessor()
    
    def verify_signature(self, payload_body: bytes, signature_header: str) -> bool:
        """Verify that the webhook is from GitHub using the webhook secret."""
        if not self.secret:
            logger.warning("No webhook secret configured, skipping signature verification")
            return True
            
        if not signature_header:
            raise HTTPException(status_code=401, detail="Missing signature header")
            
        signature = "sha1=" + hmac.new(
            key=self.secret.encode(),
            msg=payload_body,
            digestmod=hashlib.sha1
        ).hexdigest()
            
        return hmac.compare_digest(signature, signature_header)
    
    async def handle_webhook(self, request: Request, db: AsyncSession = Depends(get_db)):
        """Process incoming GitHub webhook events."""
        # Get the signature from the header
        signature_header = request.headers.get("X-Hub-Signature")
        event_type = request.headers.get("X-GitHub-Event")
        
        # Read and verify the request body
        payload_body = await request.body()
        
        # Verify signature
        if not self.verify_signature(payload_body, signature_header):
            raise HTTPException(status_code=401, detail="Invalid signature")
        
        payload = json.loads(payload_body)
        
        # Handle different event types
        if event_type == "push":
            return await self.handle_push_event(payload, db)
        elif event_type == "pull_request":
            return await self.handle_pull_request_event(payload, db)
        else:
            logger.info(f"Received unsupported event type: {event_type}")
            return {"message": f"Event {event_type} received but not processed"}
    
    async def handle_push_event(self, payload: dict, db: AsyncSession):
        """Process a push event from GitHub."""
        try:
            # Extract repository information
            repo_data = {
                "name": payload["repository"]["name"],
                "url": payload["repository"]["html_url"],
                "owner": payload["repository"]["owner"]["name"]
            }
            
            logger.info(f"Processing push event for repository: {repo_data['name']}")
            
            # Get or create repository first
            repo = await self.change_processor._get_or_create_repository(db, repo_data)
            
            # Check if we should track commits for this repository
            if not repo.track_commits:
                logger.info(f"Skipping commits for repository {repo.name} (commit tracking disabled)")
                return {"message": f"Repository {repo.name} has commit tracking disabled"}
                
            # Extract commits
            commits = payload.get("commits", [])
            if not commits:
                logger.warning("No commits found in the push event")
                return {"message": "No commits to process"}
                
            logger.info(f"Found {len(commits)} commits to process")
            
            # Process each commit
            processed_commits = 0
            for commit_data in commits:
                try:
                    # Get full commit data including diffs
                    commit_hash = commit_data["id"]
                    repo_full_name = payload["repository"]["full_name"]
                    
                    logger.info(f"Processing commit {commit_hash} for {repo_full_name}")
                    
                    # Get detailed commit info from GitHub API
                    commit_detail = await self.github_client.get_commit(repo_full_name, commit_hash)
                    if not commit_detail:
                        logger.error(f"Failed to get commit details for {commit_hash}")
                        continue
                    
                    # Process changes
                    await self.change_processor.process_commit(
                        db=db,
                        repo_data=repo_data,
                        commit_data={
                            "commit_hash": commit_hash,
                            "message": commit_data["message"],
                            "author": commit_data["author"]["name"],
                            "author_email": commit_data["author"]["email"],
                            "timestamp": commit_data["timestamp"],
                            "added": commit_data.get("added", []),
                            "modified": commit_data.get("modified", []),
                            "removed": commit_data.get("removed", [])
                        },
                        commit_detail=commit_detail
                    )
                    processed_commits += 1
                    logger.info(f"Successfully processed commit {commit_hash}")
                except Exception as commit_error:
                    logger.error(f"Error processing commit {commit_hash}: {str(commit_error)}")
                    # Continue processing other commits even if one fails
                    continue
            
            return {
                "message": f"Successfully processed {processed_commits} out of {len(commits)} commits",
                "total_commits": len(commits),
                "processed_commits": processed_commits
            }
            
        except Exception as e:
            logger.error(f"Error processing push event: {str(e)}")
            # Roll back the database session on error
            await db.rollback()
            raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")
    
    async def handle_pull_request_event(self, payload: dict, db: AsyncSession):
        """Process a pull request event from GitHub."""
        try:
            action = payload.get("action")
            
            # We're mainly interested in opened, synchronize (updated), or closed+merged PRs
            if action not in ["opened", "synchronize", "closed"]:
                return {"message": f"Pull request action {action} not processed"}
            
            # If PR was merged, process it
            if action == "closed" and payload["pull_request"].get("merged"):
                # Extract repository information
                repo_data = {
                    "name": payload["repository"]["name"],
                    "url": payload["repository"]["html_url"],
                    "owner": payload["repository"]["owner"]["login"]
                }
                
                # Get or create repository first
                repo = await self.change_processor._get_or_create_repository(db, repo_data)
                
                # Check if we should track merged PRs for this repository
                if not repo.track_merged_prs:
                    logger.info(f"Skipping merged PR commits for repository {repo.name} (merged PR tracking disabled)")
                    return {"message": f"Repository {repo.name} has merged PR tracking disabled"}
                
                # Get PR data
                pr_number = payload["number"]
                repo_full_name = payload["repository"]["full_name"]
                
                # Get detailed PR info from GitHub API
                pr_commits = await self.github_client.get_pull_request_commits(
                    repo_full_name, pr_number
                )
                
                # Process each commit in the PR
                for commit in pr_commits:
                    commit_hash = commit["sha"]
                    commit_detail = await self.github_client.get_commit(repo_full_name, commit_hash)
                    
                    await self.change_processor.process_commit(
                        db=db,
                        repo_data=repo_data,
                        commit_data={
                            "commit_hash": commit_hash,
                            "message": commit["commit"]["message"],
                            "author": commit["commit"]["author"]["name"],
                            "author_email": commit["commit"]["author"]["email"],
                            "timestamp": commit["commit"]["author"]["date"],
                            # These will be filled from commit_detail
                            "added": [],
                            "modified": [],
                            "removed": []
                        },
                        commit_detail=commit_detail
                    )
                
                return {"message": f"Successfully processed {len(pr_commits)} commits from PR #{pr_number}"}
            
            return {"message": f"Pull request action {action} received"}
            
        except Exception as e:
            logger.error(f"Error processing pull request event: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing webhook: {str(e)}")