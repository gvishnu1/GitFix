from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime

class Repository(BaseModel):
    id: int
    name: str
    url: str
    owner: str
    description: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    is_active: bool
    is_qag: Optional[bool] = False  # Add flag to identify QAG repositories
    
    class Config:
        from_attributes = True

class Commit(BaseModel):
    id: int
    commit_hash: str
    repository_id: int
    author: str
    author_email: str
    message: str
    timestamp: datetime
    summary: Optional[str] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class File(BaseModel):
    id: int
    commit_id: int
    file_path: str
    change_type: str
    content_before: Optional[str] = None
    content_after: Optional[str] = None
    diff: Optional[str] = None
    language: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

class CodeSnippet(BaseModel):
    id: int
    file_id: int
    snippet_type: str
    name: str
    content: str
    start_line: int
    end_line: int
    
    class Config:
        from_attributes = True

class CommitDetail(BaseModel):
    commit: Commit
    files: List[File]

class ChatRequest(BaseModel):
    query: str
    repository_id: Optional[int] = None
    context: Optional[Dict[str, Any]] = None
    
class ChatResponse(BaseModel):
    response: str
    context: Optional[Dict[str, Any]] = None
    references: Optional[List[Dict[str, Any]]] = None

    class Config:
        from_attributes = True
        arbitrary_types_allowed = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class SearchResult(BaseModel):
    id: int
    type: str = Field(..., description="Type of result: 'file', 'commit', 'snippet'")
    relevance_score: float
    content: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class RepositoryCreate(BaseModel):
    name: str
    url: str
    owner: str
    description: Optional[str] = None
    
class UserAuth(BaseModel):
    username: str
    password: str
    
class Token(BaseModel):
    access_token: str
    token_type: str
    
class TokenData(BaseModel):
    username: Optional[str] = None
    
class User(BaseModel):
    id: int
    username: str
    email: str
    is_active: bool
    
    class Config:
        from_attributes = True

class RepositoryUpdate(BaseModel):
    name: Optional[str] = None
    url: Optional[str] = None
    owner: Optional[str] = None
    description: Optional[str] = None
    is_active: Optional[bool] = None
    track_commits: Optional[bool] = None
    track_merged_prs: Optional[bool] = None

class RepositorySettings(BaseModel):
    id: int
    name: str
    track_commits: bool
    track_merged_prs: bool
    is_active: bool

class RepositoryManagementResponse(BaseModel):
    success: bool
    message: Optional[str] = None
    repository: Optional[Dict[str, Any]] = None

class LoadMergedPRsRequest(BaseModel):
    limit: Optional[int] = 10

class AddRepositoryRequest(BaseModel):
    repo_url: str
    track_commits: Optional[bool] = True
    track_merged_prs: Optional[bool] = True

class RepositoryImportResponse(BaseModel):
    """Response schema for repository import operations"""
    status: str
    message: str
    stats: Dict[str, Any]
    
class CodeSnippetForAnalysis(BaseModel):
    """A code snippet to be analyzed by CodeLlama"""
    content: str
    file_path: Optional[str] = None
    language: Optional[str] = None
    
class CodeUnderstandingRequest(BaseModel):
    """Request schema for testing CodeLlama's code understanding"""
    code_snippets: List[CodeSnippetForAnalysis]
    repository_id: Optional[int] = None