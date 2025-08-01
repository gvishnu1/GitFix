import httpx
import logging
import re
from typing import Dict, List, Any, Optional, Tuple
from asyncio import Lock

logger = logging.getLogger(__name__)

class RateLimitExceededException(Exception):
    """Exception raised when GitHub API rate limit is exceeded"""
    pass

class GitHubClient:
    """Client for interacting with GitHub API"""
    
    def __init__(self, access_token=None):
        self.access_token = access_token
        self.api_base = "https://api.github.com"
        self.session = None
        self._session_lock = Lock()
        
        # Set up headers
        self.headers = {
            "Accept": "application/vnd.github.v3+json"
        }
        
        if access_token:
            self.headers["Authorization"] = f"token {access_token}"
    
    async def __aenter__(self):
        await self._ensure_session()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def _ensure_session(self):
        """Ensure an HTTP session exists"""
        async with self._session_lock:
            if self.session is None:
                self.session = httpx.AsyncClient()
                
    async def close(self):
        """Close the HTTP session"""
        async with self._session_lock:
            if self.session:
                await self.session.aclose()
                self.session = None
    
    async def get_commit(self, repo_full_name: str, commit_hash: str) -> Dict[str, Any]:
        """
        Get detailed commit information including diffs
        """
        url = f"{self.api_base}/repos/{repo_full_name}/commits/{commit_hash}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get commit {commit_hash}: {response.text}")
                response.raise_for_status()
            
            return response.json()
    
    async def get_file_content(self, repo_full_name: str, file_path: str, ref: Optional[str] = None) -> str:
        """
        Get file content from GitHub repository
        """
        url = f"{self.api_base}/repos/{repo_full_name}/contents/{file_path}"
        params = {"ref": ref} if ref else {}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            
            if response.status_code == 404:
                logger.warning(f"File not found: {file_path} at ref {ref}")
                return ""
            
            if response.status_code != 200:
                logger.error(f"Failed to get file content for {file_path}: {response.text}")
                response.raise_for_status()
            
            content_data = response.json()
            
            if content_data.get("type") != "file":
                logger.error(f"Path {file_path} is not a file")
                return ""
            
            # GitHub API returns base64 encoded content
            import base64
            content = base64.b64decode(content_data["content"]).decode("utf-8")
            return content
    
    async def get_pull_request_commits(self, repo_full_name: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        Get all commits from a pull request
        """
        url = f"{self.api_base}/repos/{repo_full_name}/pulls/{pr_number}/commits"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get commits for PR #{pr_number}: {response.text}")
                response.raise_for_status()
            
            return response.json()
    
    async def get_repository(self, repo_full_name: str) -> Dict[str, Any]:
        """
        Get repository information
        """
        url = f"{self.api_base}/repos/{repo_full_name}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get repository {repo_full_name}: {response.text}")
                response.raise_for_status()
            
            return response.json()
    
    async def fetch_all_repo_contents(self, repo_full_name: str, path: str = "", ref: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Recursively fetch all files in a repository
        """
        url = f"{self.api_base}/repos/{repo_full_name}/contents/{path}"
        params = {"ref": ref} if ref else {}
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers, params=params)
            
            if response.status_code != 200:
                logger.error(f"Failed to get repository contents for {repo_full_name}: {response.text}")
                response.raise_for_status()
            
            contents = response.json()
            
            if not isinstance(contents, list):
                # Single file was returned
                return [contents]
            
            all_contents = []
            for item in contents:
                if item["type"] == "file":
                    all_contents.append(item)
                elif item["type"] == "dir":
                    # Recursively get contents of subdirectory
                    sub_contents = await self.fetch_all_repo_contents(
                        repo_full_name, item["path"], ref
                    )
                    all_contents.extend(sub_contents)
            
            return all_contents
        
    async def get_merged_pull_requests(self, repo_full_name: str, state: str = "closed", per_page: int = 100) -> List[Dict[str, Any]]:
        """
        Get merged pull requests from a repository
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            state: PR state ('closed', 'all', or 'open')
            per_page: Number of results per page
            
        Returns:
            List of merged pull requests
        """
        url = f"{self.api_base}/repos/{repo_full_name}/pulls"
        params = {"state": state, "per_page": per_page}
        merged_prs = []
        
        async with httpx.AsyncClient() as client:
            # GitHub API uses pagination
            page = 1
            while True:
                params["page"] = page
                response = await client.get(url, headers=self.headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Failed to get pull requests for {repo_full_name}: {response.text}")
                    response.raise_for_status()
                
                pull_requests = response.json()
                
                if not pull_requests:
                    break
                    
                # Filter for merged PRs only
                for pr in pull_requests:
                    if pr.get("merged_at"):
                        merged_prs.append(pr)
                
                # Check if we've reached the last page
                if len(pull_requests) < per_page:
                    break
                    
                page += 1
        
        return merged_prs

    async def get_pull_request(self, repo_full_name: str, pr_number: int) -> Dict[str, Any]:
        """
        Get detailed information about a specific pull request
        
        Args:
            repo_full_name: Repository full name (owner/repo)
            pr_number: Pull request number
            
        Returns:
            Pull request details
        """
        url = f"{self.api_base}/repos/{repo_full_name}/pulls/{pr_number}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get pull request #{pr_number} for {repo_full_name}: {response.text}")
                response.raise_for_status()
            
            return response.json()

    async def get_repository_list(self, username: str = None, org: str = None) -> List[Dict[str, Any]]:
        """
        Get list of repositories for a user or organization
        
        Args:
            username: GitHub username (optional)
            org: GitHub organization name (optional)
            
        Returns:
            List of repositories
        """
        if org:
            url = f"{self.api_base}/orgs/{org}/repos"
        elif username:
            url = f"{self.api_base}/users/{username}/repos"
        else:
            # Get authenticated user's repositories
            url = f"{self.api_base}/user/repos"
            
        params = {"per_page": 100}
        repos = []
        
        async with httpx.AsyncClient() as client:
            # GitHub API uses pagination
            page = 1
            while True:
                params["page"] = page
                response = await client.get(url, headers=self.headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Failed to get repositories: {response.text}")
                    response.raise_for_status()
                
                page_repos = response.json()
                
                if not page_repos:
                    break
                    
                repos.extend(page_repos)
                
                # Check if we've reached the last page
                if len(page_repos) < 100:
                    break
                    
                page += 1
        
        return repos
    
    async def get_commits(self, repo_full_name: str, limit: int = 30) -> List[Dict[str, Any]]:
        """
        Get commits for a repository
        
        Args:
            owner: Repository owner
            repo: Repository name
            limit: Maximum number of commits to return
            
        Returns:
            List of commit information
        """
        if not self.access_token:
            raise ValueError("GitHub token not configured")

        url = f"https://api.github.com/repos/{repo_full_name}/commits"
        params = {"per_page": min(limit, 100)}  # GitHub API max per_page is 100
        
        commits = []
        
        async with httpx.AsyncClient() as client:
            # GitHub API uses pagination
            page = 1
            while len(commits) < limit:
                params["page"] = page
                response = await client.get(url, headers=self.headers, params=params)
                
                if response.status_code != 200:
                    logger.error(f"Failed to get commits: {response.text}")
                    return commits
                
                page_commits = response.json()
                
                if not page_commits:
                    break
                    
                commits.extend(page_commits)
                
                # Check if we've reached the last page or limit
                if len(page_commits) < params["per_page"] or len(commits) >= limit:
                    break
                    
                page += 1
        
        return commits[:limit]  # Make sure we don't return more than the limit
    
    def parse_repo_url(self, repo_url: str) -> Tuple[Optional[str], Optional[str]]:
        """Parse repository URL to extract owner and name"""
        if not repo_url:
            return None, None
            
        # Handle different URL formats
        patterns = [
            r"github.com/([^/]+)/([^/]+)",  # https://github.com/owner/repo
            r"github.com:([^/]+)/([^/]+)",  # git@github.com:owner/repo
        ]
        
        for pattern in patterns:
            match = re.search(pattern, repo_url)
            if match:
                owner, repo_name = match.groups()
                # Remove .git extension if present
                repo_name = repo_name.replace(".git", "")
                return owner, repo_name
                
        return None, None
        
    async def get_repository_contents(self, owner: str, repo: str, branch: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get all file paths in a repository recursively
        
        Args:
            owner: Repository owner
            repo: Repository name
            branch: Branch name (optional, defaults to the default branch)
            
        Returns:
            List of dictionaries with file information
        """
        all_files = []
        
        async def get_contents(path: str):
            nonlocal all_files
            
            url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
            params = {}
            if branch:
                params["ref"] = branch
            
            # Create a new async client for this request
            async with httpx.AsyncClient() as client:    
                response = await client.get(url, headers=self.headers, params=params)
                
                if response.status_code == 403 and "rate limit exceeded" in response.text:
                    raise RateLimitExceededException("GitHub API rate limit exceeded")
                elif response.status_code != 200:
                    logger.error(f"Failed to get repository contents: {response.text}")
                    return
                    
                contents = response.json()
                
                # If it's a single file (not a list), wrap it in a list
                if not isinstance(contents, list):
                    contents = [contents]
                
                for item in contents:
                    if item["type"] == "file":
                        all_files.append({
                            "path": item["path"],
                            "sha": item["sha"],
                            "size": item["size"],
                            "url": item["url"]
                        })
                    elif item["type"] == "dir":
                        # Recursively get contents of subdirectories
                        await get_contents(item["path"])
        
        try:
            # Start with the root directory
            await get_contents("")
            return all_files
        except Exception as e:
            logger.error(f"Error getting repository contents: {str(e)}")
            return []
            
    async def get_file_content(self, owner: str, repo: str, path: str, branch: Optional[str] = None) -> Optional[str]:
        """
        Get the content of a file from GitHub
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: Path to the file
            branch: Branch name (optional, defaults to the default branch)
            
        Returns:
            File content as string, or None if not found
        """
        url = f"{self.api_base}/repos/{owner}/{repo}/contents/{path}"
        params = {}
        if branch:
            params["ref"] = branch
            
        try:
            # Use a separate async client instead of self.session
            async with httpx.AsyncClient() as client:
                response = await client.get(url, params=params, headers=self.headers)
                
                if response.status_code == 403 and "rate limit exceeded" in response.text:
                    raise RateLimitExceededException("GitHub API rate limit exceeded")
                elif response.status_code != 200:
                    logger.error(f"Failed to get file content: {response.text}")
                    return None
                    
                data = response.json()
                
                # GitHub API returns content as base64 encoded string
                if data.get("encoding") == "base64" and data.get("content"):
                    import base64
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                    return content
                    
                return None
        except Exception as e:
            logger.error(f"Error getting file content: {str(e)}")
            return None
    
    async def get_commit_detail(self, owner: str, repo: str, commit_hash: str) -> Dict[str, Any]:
        """
        Get detailed commit information including files changed
        
        Args:
            owner: Repository owner
            repo: Repository name
            commit_hash: Commit hash
            
        Returns:
            Dict with commit details
        """
        url = f"{self.api_base}/repos/{owner}/{repo}/commits/{commit_hash}"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=self.headers)
            
            if response.status_code != 200:
                logger.error(f"Failed to get commit details for {commit_hash}: {response.text}")
                response.raise_for_status()
            
            return response.json()