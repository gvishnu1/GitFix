from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from contextlib import asynccontextmanager
import logging
from code_monitor.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO if settings.DEBUG else logging.WARNING)
logger = logging.getLogger(__name__)

# Convert the standard PostgreSQL URL to an async version
async_db_url = settings.DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine with echo for debugging
engine = create_async_engine(
    async_db_url,
    echo=settings.DEBUG,
    pool_pre_ping=True,
    isolation_level="READ COMMITTED"
)

# Create sessionmaker with autoflush and expire_on_commit=False
SessionLocal = sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=True
)

# Base class for SQLAlchemy models
Base = declarative_base()

@asynccontextmanager
async def async_session_scope():
    """Provide an async transactional scope around a series of operations."""
    session = SessionLocal()
    try:
        yield session
        await session.commit()
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        await session.rollback()
        raise
    finally:
        await session.close()

async def get_db():
    """Get a fresh database session with async context management."""
    session = SessionLocal()
    try:
        # Check if there's already an active transaction and rollback if needed
        if session.in_transaction():
            await session.rollback()
        yield session
        # Only commit if there were no exceptions
        await session.commit()
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        await session.rollback()
        raise
    finally:
        await session.close()

async def init_db():
    """Initialize the database, creating all tables and enabling pgvector."""
    try:
        async with engine.begin() as conn:
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise