#!/usr/bin/env python
import asyncio
import sys
import argparse
import logging
from sqlalchemy import select
from sqlalchemy.future import select
from tqdm import tqdm

sys.path.append('.')  # Add the current directory to path for imports

from code_monitor.db.database import SessionLocal, engine
from code_monitor.db.models import QAGRepository, QAGFile, QAGRepositoryFile, QAGCodeSnippet
from code_monitor.ai_processing.embedding import EmbeddingGenerator
from code_monitor.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("qag_embeddings_regenerator")

# Force Instructor model to be used
settings.USE_INSTRUCTOR = True

async def regenerate_file_embeddings(repository_id=None, force=False, limit=None):
    """
    Regenerate embeddings for QAG repository files using the Instructor model
    
    Args:
        repository_id (int): Optional repository ID to filter by
        force (bool): If True, regenerate all embeddings, otherwise only process files with no embedding
        limit (int): Optional limit on number of files to process
    """
    embedding_generator = EmbeddingGenerator()
    
    async with SessionLocal() as db:
        # Build query for QAGFiles
        query = select(QAGFile)
        if not force:
            # Only files with no embeddings
            query = query.filter(QAGFile.embedding == None)
        if repository_id:
            query = query.join(QAGFile.commit).filter(QAGFile.repository_id == repository_id)
        if limit:
            query = query.limit(limit)
            
        # Execute query
        result = await db.execute(query)
        files = result.scalars().all()
        
        if not files:
            logger.info("No QAG files found to process.")
            return
        
        logger.info(f"Found {len(files)} QAG files to process.")
        
        # Process each file
        for file in tqdm(files, desc="Processing QAG files"):
            if not file.content_after:
                logger.warning(f"File ID {file.id} has no content, skipping.")
                continue
                
            # Generate embedding
            logger.info(f"Generating embedding for file ID {file.id}: {file.file_path}")
            content = file.content_after
            embedding = await embedding_generator.generate_embedding(content)
            
            if embedding:
                # Update file with new embedding
                file.embedding = embedding
                await db.commit()
                logger.info(f"Updated embedding for file ID {file.id}")
            else:
                logger.error(f"Failed to generate embedding for file ID {file.id}")

async def regenerate_repo_file_embeddings(repository_id=None, force=False, limit=None):
    """
    Regenerate embeddings for QAG repository files (current state) using the Instructor model
    
    Args:
        repository_id (int): Optional repository ID to filter by
        force (bool): If True, regenerate all embeddings, otherwise only process files with no embedding
        limit (int): Optional limit on number of files to process
    """
    embedding_generator = EmbeddingGenerator()
    
    async with SessionLocal() as db:
        # Build query for QAGRepositoryFiles
        query = select(QAGRepositoryFile)
        if not force:
            # Only files with no embeddings
            query = query.filter(QAGRepositoryFile.embedding == None)
        if repository_id:
            query = query.filter(QAGRepositoryFile.repository_id == repository_id)
        if limit:
            query = query.limit(limit)
            
        # Execute query
        result = await db.execute(query)
        repo_files = result.scalars().all()
        
        if not repo_files:
            logger.info("No QAG repository files found to process.")
            return
        
        logger.info(f"Found {len(repo_files)} QAG repository files to process.")
        
        # Process each file
        for repo_file in tqdm(repo_files, desc="Processing QAG repository files"):
            if not repo_file.content:
                logger.warning(f"Repository file ID {repo_file.id} has no content, skipping.")
                continue
                
            # Generate embedding
            logger.info(f"Generating embedding for repository file ID {repo_file.id}: {repo_file.file_path}")
            content = repo_file.content
            embedding = await embedding_generator.generate_embedding(content)
            
            if embedding:
                # Update file with new embedding
                repo_file.embedding = embedding
                await db.commit()
                logger.info(f"Updated embedding for repository file ID {repo_file.id}")
            else:
                logger.error(f"Failed to generate embedding for repository file ID {repo_file.id}")

async def regenerate_code_snippet_embeddings(repository_id=None, force=False, limit=None):
    """
    Regenerate embeddings for QAG code snippets using the Instructor model
    
    Args:
        repository_id (int): Optional repository ID to filter by
        force (bool): If True, regenerate all embeddings, otherwise only process snippets with no embedding
        limit (int): Optional limit on number of snippets to process
    """
    embedding_generator = EmbeddingGenerator()
    
    async with SessionLocal() as db:
        # Build query for QAGCodeSnippets
        query = select(QAGCodeSnippet)
        if not force:
            # Only snippets with no embeddings
            query = query.filter(QAGCodeSnippet.embedding == None)
        if repository_id:
            # This is trickier as we need to join through files and commits
            # For simplicity we'll handle filtering in Python
            pass
        if limit:
            query = query.limit(limit)
            
        # Execute query
        result = await db.execute(query)
        snippets = result.scalars().all()
        
        if repository_id:
            # Filter by repository_id through file -> commit -> repository
            filtered_snippets = []
            for snippet in snippets:
                # Get file associated with snippet
                file_result = await db.execute(select(QAGFile).filter(QAGFile.id == snippet.file_id))
                file = file_result.scalars().first()
                if file and file.commit and file.commit.repository_id == repository_id:
                    filtered_snippets.append(snippet)
            snippets = filtered_snippets
        
        if not snippets:
            logger.info("No QAG code snippets found to process.")
            return
        
        logger.info(f"Found {len(snippets)} QAG code snippets to process.")
        
        # Process each snippet
        for snippet in tqdm(snippets, desc="Processing QAG code snippets"):
            if not snippet.content:
                logger.warning(f"Code snippet ID {snippet.id} has no content, skipping.")
                continue
                
            # Generate embedding
            logger.info(f"Generating embedding for code snippet ID {snippet.id}: {snippet.name}")
            content = snippet.content
            embedding = await embedding_generator.generate_embedding(content)
            
            if embedding:
                # Update snippet with new embedding
                snippet.embedding = embedding
                await db.commit()
                logger.info(f"Updated embedding for code snippet ID {snippet.id}")
            else:
                logger.error(f"Failed to generate embedding for code snippet ID {snippet.id}")

async def main():
    parser = argparse.ArgumentParser(description='Regenerate embeddings for QAG repository using Instructor model')
    parser.add_argument('--repo-id', type=int, help='Repository ID to process')
    parser.add_argument('--force', action='store_true', help='Force regeneration of all embeddings')
    parser.add_argument('--limit', type=int, help='Limit number of items to process')
    parser.add_argument('--files-only', action='store_true', help='Only process commit files')
    parser.add_argument('--repo-files-only', action='store_true', help='Only process repository files')
    parser.add_argument('--snippets-only', action='store_true', help='Only process code snippets')
    
    args = parser.parse_args()
    
    try:
        logger.info("Starting QAG embeddings regeneration with Instructor model")
        
        # Process each type of embedding unless specific flags are set
        process_all = not (args.files_only or args.repo_files_only or args.snippets_only)
        
        if process_all or args.files_only:
            await regenerate_file_embeddings(args.repo_id, args.force, args.limit)
            
        if process_all or args.repo_files_only:
            await regenerate_repo_file_embeddings(args.repo_id, args.force, args.limit)
            
        if process_all or args.snippets_only:
            await regenerate_code_snippet_embeddings(args.repo_id, args.force, args.limit)
            
        logger.info("QAG embeddings regeneration complete!")
        
    except Exception as e:
        logger.error(f"Error during embeddings regeneration: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())