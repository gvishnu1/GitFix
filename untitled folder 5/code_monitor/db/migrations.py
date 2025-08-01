import logging
import asyncio
import os
import sys
from pathlib import Path

# Add parent directory to path to allow imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine
from code_monitor.db.database import engine, Base
from code_monitor.db.models import (
    Repository, Commit, File, RepositoryFile, CodeSnippet, User,
    QAGRepository, QAGCommit, QAGFile, QAGRepositoryFile, QAGCodeSnippet
)
from code_monitor.config import settings

logger = logging.getLogger(__name__)

async def add_qag_code_snippets_columns():
    """Add new columns to qag_code_snippets table"""
    async with engine.begin() as conn:
        # Add functionality_analysis column
        await conn.execute(text('''
            ALTER TABLE qag_code_snippets 
            ADD COLUMN IF NOT EXISTS functionality_analysis JSONB DEFAULT '{}'::jsonb;
        '''))
        
        # Add code_patterns column
        await conn.execute(text('''
            ALTER TABLE qag_code_snippets 
            ADD COLUMN IF NOT EXISTS code_patterns JSONB DEFAULT '{}'::jsonb;
        '''))
        
        # Add complexity_metrics column
        await conn.execute(text('''
            ALTER TABLE qag_code_snippets 
            ADD COLUMN IF NOT EXISTS complexity_metrics JSONB DEFAULT '{}'::jsonb;
        '''))
        
        # Add dependencies column
        await conn.execute(text('''
            ALTER TABLE qag_code_snippets 
            ADD COLUMN IF NOT EXISTS dependencies JSONB DEFAULT '{}'::jsonb;
        '''))
        
        # Add api_endpoints column
        await conn.execute(text('''
            ALTER TABLE qag_code_snippets 
            ADD COLUMN IF NOT EXISTS api_endpoints JSONB DEFAULT NULL;
        '''))
        
        # Add usage_examples column
        await conn.execute(text('''
            ALTER TABLE qag_code_snippets 
            ADD COLUMN IF NOT EXISTS usage_examples JSONB DEFAULT '[]'::jsonb;
        '''))
        
        # Add test_coverage column
        await conn.execute(text('''
            ALTER TABLE qag_code_snippets 
            ADD COLUMN IF NOT EXISTS test_coverage JSONB DEFAULT NULL;
        '''))
        
        # Add documentation column
        await conn.execute(text('''
            ALTER TABLE qag_code_snippets 
            ADD COLUMN IF NOT EXISTS documentation TEXT DEFAULT NULL;
        '''))

async def run_migrations():
    """Run database migrations to add new columns and update schema"""
    try:
        logger.info("Running database migrations...")
        
        # Enable pgvector extension if not already enabled
        async with engine.begin() as conn:
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            
        # Create all QAG tables if they don't exist
        async with engine.begin() as conn:
            logger.info("Creating QAG tables if they don't exist...")
            
            # Check if QAG tables exist
            result = await conn.execute(text(
                """
                SELECT EXISTS (
                    SELECT FROM pg_tables
                    WHERE tablename = 'qag_repositories'
                );
                """
            ))
            qag_tables_exist = result.scalar()
            
            if not qag_tables_exist:
                logger.info("QAG tables don't exist, creating them...")
                
                # Create QAG tables
                await conn.run_sync(Base.metadata.create_all)
                
                logger.info("QAG tables created successfully.")
            else:
                logger.info("QAG tables already exist.")
                
                # Add new columns to qag_code_snippets table
                logger.info("Adding new columns to qag_code_snippets table...")
                await add_qag_code_snippets_columns()
                logger.info("New columns added successfully.")
        
        logger.info("Migrations completed successfully.")
        
    except Exception as e:
        logger.error(f"Error running migrations: {str(e)}")
        raise

async def apply_migrations():
    """Apply all database migrations"""
    logger.info("Applying database migrations...")
    
    # Create async engine
    engine = create_async_engine(settings.DATABASE_URL, echo=False)
    
    try:
        async with engine.begin() as conn:
            # Apply vector functions
            await apply_vector_functions(conn)
            
        logger.info("Migrations applied successfully")
    except Exception as e:
        logger.error(f"Error applying migrations: {str(e)}")
        raise
    finally:
        await engine.dispose()

async def apply_vector_functions(conn):
    """Apply vector functions needed for similarity search"""
    logger.info("Applying vector functions...")
    
    # Get the path to the SQL file
    base_dir = Path(__file__).parent
    sql_file_path = base_dir / "sql" / "vector_functions.sql"
    
    # Check if the file exists
    if not os.path.exists(sql_file_path):
        logger.warning(f"Vector functions SQL file not found at {sql_file_path}")
        return
        
    # Read the SQL file
    with open(sql_file_path, 'r') as f:
        sql = f.read()
        
    # Execute the SQL
    logger.info("Creating vector similarity functions in database")
    await conn.execute(text(sql))
    logger.info("Vector functions applied successfully")

if __name__ == "__main__":
    # Run migrations when script is executed directly
    asyncio.run(run_migrations())
    asyncio.run(apply_migrations())