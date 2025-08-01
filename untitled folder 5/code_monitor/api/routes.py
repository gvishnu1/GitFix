"""API routes for code monitor."""

from fastapi import APIRouter, Depends, Request, HTTPException, Body, Query, FastAPI
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Dict, Any, Optional
import logging

from code_monitor.db.database import get_db
from code_monitor.db.models import Repository, Commit, File, CodeSnippet
from code_monitor.db.models import QAGRepository, QAGCommit, QAGFile  # Import QAG models
from code_monitor.github_integration.webhook_handler import WebhookHandler
from code_monitor.api import schemas
from code_monitor.chat_interface.chat_service import ChatService
from code_monitor.github_integration.repository_manager import RepositoryManager
from code_monitor.github_integration.github_client import GitHubClient
from code_monitor.github_integration.processor import ChangeProcessor
from code_monitor.config import settings
from code_monitor.ai_processing.code_analyzer import CodeAnalyzer
from code_monitor.utils.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# Create API router
api_router = APIRouter()

# Create FastAPI app
app = FastAPI()

# Create webhook handler
webhook_handler = WebhookHandler()

# Create chat service
chat_service = ChatService()

# Create repository manager
github_client = GitHubClient(settings.GITHUB_ACCESS_TOKEN)
change_processor = ChangeProcessor()
repository_manager = RepositoryManager(github_client=github_client, change_processor=change_processor)

# Initialize Ollama client at module level
ollama_client = None

async def initialize_ollama():
    """Initialize Ollama client"""
    global ollama_client
    ollama_client = await OllamaClient.create()

# Create code analyzer for CodeLlama testing
code_analyzer = CodeAnalyzer()

async def initialize_services():
    """Initialize all services that require async initialization"""
    global ollama_client
    global chat_service
    global code_analyzer
    
    # Initialize Ollama client
    await initialize_ollama()
    
    # Initialize chat service
    await chat_service.initialize()
    
    # Initialize code analyzer
    await code_analyzer.initialize()

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    await initialize_services()

# Add router to app
app.include_router(api_router)

@api_router.post("/webhook/github", response_model=Dict[str, Any])
async def github_webhook(request: Request, db: AsyncSession = Depends(get_db)):
    """
    Handle GitHub webhook events
    """
    return await webhook_handler.handle_webhook(request, db)

@api_router.get("/repositories", response_model=List[schemas.Repository])
async def get_repositories(
    skip: int = 0, 
    limit: int = 100, 
    include_qag: bool = True,  # New parameter to control including QAG repositories
    db: AsyncSession = Depends(get_db)
):
    """
    Get all repositories, including QAG repositories if specified
    """
    from sqlalchemy.future import select
    result = await db.execute(select(Repository).offset(skip).limit(limit))
    repositories = result.scalars().all()
    
    # Convert SQLAlchemy ORM objects to dictionaries that match the schema
    repo_list = []
    for repo in repositories:
        repo_dict = {
            "id": repo.id,
            "name": repo.name,
            "url": repo.url,
            "owner": repo.owner,
            "description": repo.description,
            "created_at": repo.created_at,
            "updated_at": repo.updated_at,
            "is_active": repo.is_active,
            "is_qag": False  # Standard repositories are not QAG
        }
        repo_list.append(repo_dict)
    
    # Include QAG repositories if requested
    if include_qag:
        qag_result = await db.execute(select(QAGRepository).offset(skip).limit(limit))
        qag_repositories = qag_result.scalars().all()
        
        # Convert QAG repositories to standard repository format
        for qag_repo in qag_repositories:
            repo_dict = {
                "id": 2 if qag_repo.id == 1 else qag_repo.id,  # Ensure QAG repo with ID 1 is presented as ID 2
                "name": f"QAG: {qag_repo.name}",  # Prefix with QAG for clarity
                "url": qag_repo.url,
                "owner": qag_repo.owner,
                "description": qag_repo.description or "",
                "created_at": qag_repo.created_at,
                "updated_at": qag_repo.updated_at,
                "is_active": qag_repo.is_active,
                "is_qag": True  # Mark as QAG repository
            }
            # Add to repositories list as a dictionary, not an ORM object
            repo_list.append(repo_dict)
    
    return repo_list

@api_router.get("/repositories/{repo_id}", response_model=schemas.Repository)
async def get_repository(repo_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get repository by ID
    """
    from sqlalchemy.future import select
    result = await db.execute(select(Repository).filter(Repository.id == repo_id))
    repository = result.scalars().first()
    
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    return repository

@api_router.get("/repositories/{repo_id}/commits", response_model=List[schemas.Commit])
async def get_repository_commits(
    repo_id: int, 
    skip: int = 0, 
    limit: int = 100, 
    db: AsyncSession = Depends(get_db)
):
    """
    Get all commits for a repository
    """
    from sqlalchemy.future import select
    result = await db.execute(
        select(Commit)
        .filter(Commit.repository_id == repo_id)
        .order_by(Commit.timestamp.desc())
        .offset(skip)
        .limit(limit)
    )
    commits = result.scalars().all()
    return commits

@api_router.get("/commits/{commit_id}", response_model=schemas.CommitDetail)
async def get_commit(commit_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get commit by ID with associated files
    """
    from sqlalchemy.future import select
    result = await db.execute(select(Commit).filter(Commit.id == commit_id))
    commit = result.scalars().first()
    
    if commit is None:
        raise HTTPException(status_code=404, detail="Commit not found")
    
    # Get files for this commit
    file_result = await db.execute(select(File).filter(File.commit_id == commit_id))
    files = file_result.scalars().all()
    
    return {
        "commit": commit,
        "files": files
    }

@api_router.get("/files/{file_id}", response_model=schemas.File)
async def get_file(file_id: int, db: AsyncSession = Depends(get_db)):
    """
    Get file by ID
    """
    from sqlalchemy.future import select
    result = await db.execute(select(File).filter(File.id == file_id))
    file = result.scalars().first()
    
    if file is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    return file

@api_router.post("/chat", response_model=schemas.ChatResponse)
async def chat_with_repository(
    chat_request: schemas.ChatRequest = Body(...),  # Use Body to ensure proper request parsing
    db: AsyncSession = Depends(get_db)
):
    """
    Chat interface for querying repository information
    """
    chat_service = ChatService()
    
    try:
        # Check if this is a QAG repository
        is_qag = False
        repository_id = chat_request.repository_id
        actual_repo_id = repository_id
        
        # Special handling for QAG repository with ID 2 (displayed ID)
        if repository_id == 2:
            from sqlalchemy.future import select
            # Check if this is really the QAG repository with original ID 1
            qag_repo_query = await db.execute(select(QAGRepository).where(QAGRepository.id == 1))
            qag_repo = qag_repo_query.scalars().first()
            if qag_repo:
                is_qag = True
                actual_repo_id = 1  # Use the actual ID 1 in the database
                logger.info(f"Using QAG repository: {qag_repo.name} (ID: 1, displayed as ID: 2)")
        # Check if we explicitly want the QAG repo with ID 1
        elif repository_id == 1 and "QAG" in chat_request.query:
            from sqlalchemy.future import select
            # Check if a QAG repo with ID 1 exists
            qag_repo_query = await db.execute(select(QAGRepository).where(QAGRepository.id == 1))
            qag_repo = qag_repo_query.scalars().first()
            if qag_repo:
                is_qag = True
                logger.info(f"Using QAG repository by direct mention: {qag_repo.name} (ID: {repository_id})")
        # Normal repository ID handling
        elif repository_id:
            from sqlalchemy.future import select
            # First check standard repositories
            std_repo_query = await db.execute(select(Repository).where(Repository.id == repository_id))
            std_repo = std_repo_query.scalars().first()
            
            # If not found in standard repositories, check QAG repositories
            if not std_repo:
                qag_repo_query = await db.execute(select(QAGRepository).where(QAGRepository.id == repository_id))
                qag_repo = qag_repo_query.scalars().first()
                if qag_repo:
                    is_qag = True
                    logger.info(f"Using QAG repository: {qag_repo.name} (ID: {repository_id})")
        
        # Add is_qag flag to the process_query call
        response = await chat_service.process_query(
            query=chat_request.query,
            repository_id=actual_repo_id,
            is_qag=is_qag,  # Pass the is_qag flag
            db=db
        )
        
        return schemas.ChatResponse(
            response=response["response"],
            context=response.get("context", {}),
            references=response.get("references", [])
        )
    except Exception as e:
        logger.error(f"Error processing chat query: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat query: {str(e)}"
        )

@api_router.get("/search", response_model=List[schemas.SearchResult])
async def search_code(
    query: str,
    repository_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Search code repository using semantic search
    """
    results = await chat_service.search_code(
        query=query,
        repository_id=repository_id,
        db=db
    )
    
    return results

@api_router.post("/repositories", response_model=schemas.Repository)
async def create_repository(
    repository: schemas.RepositoryCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Create a new repository
    """
    repo_data = repository.dict()
    new_repository = Repository(**repo_data)
    
    db.add(new_repository)
    await db.commit()
    await db.refresh(new_repository)
    
    return new_repository

@api_router.put("/repositories/{repo_id}", response_model=schemas.Repository)
async def update_repository(
    repo_id: int,
    repository: schemas.RepositoryUpdate,
    db: AsyncSession = Depends(get_db)
):
    """
    Update an existing repository
    """
    repo_data = repository.dict(exclude_unset=True)
    from sqlalchemy.future import select
    result = await db.execute(select(Repository).filter(Repository.id == repo_id))
    existing_repository = result.scalars().first()
    
    if existing_repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    for key, value in repo_data.items():
        setattr(existing_repository, key, value)
    
    db.add(existing_repository)
    await db.commit()
    await db.refresh(existing_repository)
    
    return existing_repository

@api_router.delete("/repositories/{repo_id}", response_model=schemas.Repository)
async def delete_repository(repo_id: int, db: AsyncSession = Depends(get_db)):
    """
    Delete a repository
    """
    from sqlalchemy.future import select
    result = await db.execute(select(Repository).filter(Repository.id == repo_id))
    repository = result.scalars().first()
    
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    db.delete(repository)
    await db.commit()
    
    return repository

@api_router.post("/repositories/{repo_id}/sync", response_model=schemas.Repository)
async def sync_repository(repo_id: int, db: AsyncSession = Depends(get_db)):
    """
    Sync a repository with the remote source
    """
    from sqlalchemy.future import select
    result = await db.execute(select(Repository).filter(Repository.id == repo_id))
    repository = result.scalars().first()
    
    if repository is None:
        raise HTTPException(status_code=404, detail="Repository not found")
    
    # Perform the sync operation
    await repository_manager.sync_repository(repository)
    
    return repository

@api_router.post("/repository-management/add", response_model=Dict[str, Any])
async def add_repository_to_monitor(
    repo_url: str,
    track_commits: bool = True,
    track_merged_prs: bool = True,
    db: AsyncSession = Depends(get_db)
):
    """
    Add a new repository to monitor and specify tracking settings
    """
    try:
        result = await repository_manager.add_repository(
            db=db,
            repo_url=repo_url,
            track_commits=track_commits,
            track_merged_prs=track_merged_prs
        )
        return result
    except Exception as e:
        logger.error(f"Error adding repository: {str(e)}")
        raise HTTPException(status_code=400, detail=f"Error adding repository: {str(e)}")

@api_router.get("/repository-management/list", response_model=List[Dict[str, Any]])
async def list_monitored_repositories(db: AsyncSession = Depends(get_db)):
    """
    List all repositories with their tracking settings
    """
    try:
        repositories = await repository_manager.list_repositories(db)
        return repositories
    except Exception as e:
        logger.error(f"Error listing repositories: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing repositories: {str(e)}")

@api_router.put("/repository-management/{repo_id}/settings", response_model=Dict[str, Any])
async def update_repository_tracking_settings(
    repo_id: int,
    track_commits: Optional[bool] = None,
    track_merged_prs: Optional[bool] = None,
    is_active: Optional[bool] = None,
    db: AsyncSession = Depends(get_db)
):
    """
    Update tracking settings for a repository
    """
    try:
        result = await repository_manager.update_repository_settings(
            db=db,
            repo_id=repo_id,
            track_commits=track_commits,
            track_merged_prs=track_merged_prs,
            is_active=is_active
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating repository settings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error updating repository settings: {str(e)}")

@api_router.post("/repository-management/{repo_id}/load-merged-prs", response_model=Dict[str, Any])
async def load_merged_pull_requests(
    repo_id: int,
    limit: int = 10,
    db: AsyncSession = Depends(get_db)
):
    """
    Retroactively load merged pull requests for a repository
    """
    try:
        result = await repository_manager.load_merged_pull_requests(
            db=db,
            repo_id=repo_id,
            limit=limit
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error loading merged PRs: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error loading merged PRs: {str(e)}")

@api_router.post(
    "/repository-management/import-full-repository",
    response_model=schemas.RepositoryImportResponse,
    summary="Import a complete repository with embeddings"
)
async def import_full_repository(
    repo_url: str = Query(..., description="URL of the GitHub repository"),
    branch: Optional[str] = Query(None, description="Branch to import (defaults to the default branch)"),
    db: AsyncSession = Depends(get_db)
):
    """
    Import all files from a repository into the database with embeddings.
    This creates a comprehensive knowledge base of the repository's code.
    
    - **repo_url**: URL of the GitHub repository (https://github.com/owner/repo)
    - **branch**: Specific branch to import (optional)
    """
    from code_monitor.cli.import_repository import RepositoryImporter
    
    try:
        # Create repository importer
        importer = RepositoryImporter()
        
        # Start the import process
        stats = await importer.import_repository(repo_url, branch)
        
        return {
            "status": "success",
            "message": f"Repository imported successfully. Processed {stats['processed_files']} files.",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Error importing repository: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error importing repository: {str(e)}"
        )

@api_router.post("/code-understanding/test", response_model=Dict[str, Any])
async def test_code_understanding(
    request: schemas.CodeUnderstandingRequest,
    db: AsyncSession = Depends(get_db)
):
    """
    Test CodeLlama's code understanding capabilities
    """
    try:
        # Extract code snippets from the request
        code_snippets = request.code_snippets
        
        # Analyze each code snippet using CodeLlama
        results = []
        for snippet in code_snippets:
            analysis_result = await code_analyzer.analyze_code(snippet)
            results.append(analysis_result)
        
        return {"results": results}
    except Exception as e:
        logger.error(f"Error in code understanding test: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in code understanding test: {str(e)}")