from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, Boolean, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from code_monitor.db.database import Base

class Repository(Base):
    __tablename__ = "repositories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    url = Column(String, unique=True)
    owner = Column(String)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    track_commits = Column(Boolean, default=True)
    track_merged_prs = Column(Boolean, default=True)
    
    # Relationships
    commits = relationship("Commit", back_populates="repository", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Repository {self.name}>"

class Commit(Base):
    __tablename__ = "commits"
    
    id = Column(Integer, primary_key=True, index=True)
    commit_hash = Column(String, index=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"))
    author = Column(String)
    author_email = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime(timezone=True))
    summary = Column(Text, nullable=True)  # AI-generated summary
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    repository = relationship("Repository", back_populates="commits")
    files = relationship("File", back_populates="commit", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Commit {self.commit_hash[:7]}>"

class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True, index=True)
    commit_id = Column(Integer, ForeignKey("commits.id"))
    file_path = Column(String)
    change_type = Column(String)  # "added", "modified", "deleted"
    content_before = Column(Text, nullable=True)
    content_after = Column(Text, nullable=True)
    diff = Column(Text, nullable=True)
    language = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Vector embedding for semantic search
    embedding = Column(Vector(1536), nullable=True)
    
    # JSON field to store metadata about affected functions/classes/etc.
    file_metadata = Column(JSON, default={})
    
    # Relationships
    commit = relationship("Commit", back_populates="files")
    
    def __repr__(self):
        return f"<File {self.file_path}>"

class RepositoryFile(Base):
    """Stores repository files separate from commits for a complete view of the codebase"""
    __tablename__ = "repository_files"
    
    id = Column(Integer, primary_key=True, index=True)
    repository_id = Column(Integer, ForeignKey("repositories.id"))
    file_path = Column(String)
    content = Column(Text, nullable=True)
    language = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_modified_at = Column(DateTime(timezone=True))
    
    # Vector embedding for semantic search
    embedding = Column(Vector(1536), nullable=True)
    
    # JSON field to store metadata about the file
    file_metadata = Column(JSON, default={})
    
    # Relationships
    repository = relationship("Repository", backref="repository_files")
    
    def __repr__(self):
        return f"<RepositoryFile {self.file_path}>"

class CodeSnippet(Base):
    """Stores individual code snippets (functions, classes, methods) for more granular tracking"""
    __tablename__ = "code_snippets"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("files.id"))
    snippet_type = Column(String)  # "function", "class", "method", etc.
    name = Column(String)
    content = Column(Text)
    start_line = Column(Integer)
    end_line = Column(Integer)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<CodeSnippet {self.snippet_type}:{self.name}>"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<User {self.username}>"

# New models for QAG Arena Dashboard repository
class QAGRepository(Base):
    """Stores the QAG Arena Dashboard repository information separate from other repositories"""
    __tablename__ = "qag_repositories"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    url = Column(String, unique=True)
    owner = Column(String)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)
    track_commits = Column(Boolean, default=True)
    track_merged_prs = Column(Boolean, default=True)
    
    # Relationships
    commits = relationship("QAGCommit", back_populates="repository", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<QAGRepository {self.name}>"

class QAGCommit(Base):
    """Stores commits specifically for the QAG Arena Dashboard repository"""
    __tablename__ = "qag_commits"
    
    id = Column(Integer, primary_key=True, index=True)
    commit_hash = Column(String, index=True)
    repository_id = Column(Integer, ForeignKey("qag_repositories.id"))
    author = Column(String)
    author_email = Column(String)
    message = Column(Text)
    timestamp = Column(DateTime(timezone=True))
    summary = Column(Text, nullable=True)  # AI-generated summary
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    repository = relationship("QAGRepository", back_populates="commits")
    files = relationship("QAGFile", back_populates="commit", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<QAGCommit {self.commit_hash[:7]}>"

class QAGFile(Base):
    """Stores file changes for commits in the QAG Arena Dashboard repository"""
    __tablename__ = "qag_files"
    
    id = Column(Integer, primary_key=True, index=True)
    commit_id = Column(Integer, ForeignKey("qag_commits.id"))
    file_path = Column(String)
    change_type = Column(String)  # "added", "modified", "deleted"
    content_before = Column(Text, nullable=True)
    content_after = Column(Text, nullable=True)
    diff = Column(Text, nullable=True)
    language = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Vector embedding for semantic search
    embedding = Column(Vector(1536), nullable=True)
    
    # JSON field to store metadata about affected functions/classes/etc.
    file_metadata = Column(JSON, default={})
    
    # Enhanced analysis fields
    technical_analysis = Column(JSON, default={})  # Stores detailed code analysis
    code_quality_metrics = Column(JSON, default={})  # Code quality assessment
    dependencies = Column(JSON, default={})  # Technical dependencies
    architectural_impact = Column(JSON, default={})  # Impact on system architecture
    change_summary = Column(Text, nullable=True)  # AI-generated change summary
    testing_requirements = Column(JSON, default={})  # Required tests for changes
    review_comments = Column(JSON, default=[])  # AI-generated review comments
    
    # Relationships
    commit = relationship("QAGCommit", back_populates="files")
    
    def __repr__(self):
        return f"<QAGFile {self.file_path}>"

class QAGRepositoryFile(Base):
    """Stores QAG Arena Dashboard repository files separate from commits"""
    __tablename__ = "qag_repository_files"
    
    id = Column(Integer, primary_key=True, index=True)
    repository_id = Column(Integer, ForeignKey("qag_repositories.id"))
    file_path = Column(String)
    content = Column(Text, nullable=True)
    language = Column(String, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_modified_at = Column(DateTime(timezone=True))
    
    # Vector embedding for semantic search
    embedding = Column(Vector(1536), nullable=True)
    
    # JSON field to store metadata about the file
    file_metadata = Column(JSON, default={})
    
    # Relationships
    repository = relationship("QAGRepository", backref="repository_files")
    
    def __repr__(self):
        return f"<QAGRepositoryFile {self.file_path}>"

class QAGCodeSnippet(Base):
    """Stores code snippets from QAG Arena Dashboard repository files"""
    __tablename__ = "qag_code_snippets"
    
    id = Column(Integer, primary_key=True, index=True)
    file_id = Column(Integer, ForeignKey("qag_files.id"))
    snippet_type = Column(String)  # "function", "class", "method", etc.
    name = Column(String)
    content = Column(Text)
    start_line = Column(Integer)
    end_line = Column(Integer)
    embedding = Column(Vector(1536), nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Enhanced analysis fields
    functionality_analysis = Column(JSON, default={})  # Core functionality details
    code_patterns = Column(JSON, default={})  # Design patterns used
    complexity_metrics = Column(JSON, default={})  # Code complexity metrics
    dependencies = Column(JSON, default={})  # Code dependencies
    api_endpoints = Column(JSON, nullable=True)  # API details if applicable
    usage_examples = Column(JSON, default=[])  # Example usages
    test_coverage = Column(JSON, nullable=True)  # Test coverage info
    documentation = Column(Text, nullable=True)  # Generated documentation
    
    def __repr__(self):
        return f"<QAGCodeSnippet {self.snippet_type}:{self.name}>"