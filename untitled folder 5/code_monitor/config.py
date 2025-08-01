import os
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # General settings
    PROJECT_NAME: str = "GitHub Code Monitor"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Database settings
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "code_monitor"
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/code_monitor"
    
    # GitHub settings
    GITHUB_WEBHOOK_SECRET: str = ""
    GITHUB_ACCESS_TOKEN: str = ""
    GITHUB_TOKEN: str = ""
    
    # AI settings
    USE_OLLAMA: bool = True
    USE_INSTRUCTOR: bool = False
    
    # Ollama settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "codellama:7b-python"  # Updated to correct model name
    OLLAMA_EMBEDDING_MODEL: str = "nomic-embed-text"
    OLLAMA_EMBEDDING_DIMENSION: int = 768
    OLLAMA_TIMEOUT: int = 30
    
    # JWT settings
    SECRET_KEY: str = "super-secret-key"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 1 week

    # Webhook configuration
    WEBHOOK_EVENTS: list[str] = ["push", "pull_request"]

    # Code Analysis Settings
    MIN_SIMILARITY_SCORE: float = 0.75
    MAX_ANALYSIS_CHUNK_SIZE: int = 8000
    ENABLE_DEEP_ANALYSIS: bool = True
    CODE_UNDERSTANDING_TEMPERATURE: float = 0.1
    MODEL_CONTEXT_LENGTH: int = 16000
    
    # File Analysis Settings
    MAX_FILE_SIZE: int = 1024 * 1024  # 1MB
    SUPPORTED_LANGUAGES: list[str] = [
        "python", "javascript", "typescript", "java", "c", "cpp", 
        "go", "rust", "ruby", "php", "csharp", "html", "css", "json"
    ]
    
    # Cache Settings
    ENABLE_EMBEDDING_CACHE: bool = True
    CACHE_TTL: int = 3600  # 1 hour
    
    model_config = {
        "case_sensitive": True,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore"
    }

settings = Settings()