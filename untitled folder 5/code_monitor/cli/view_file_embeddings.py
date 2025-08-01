#!/usr/bin/env python
import asyncio
import argparse
import sys
import numpy as np
from tabulate import tabulate
from sqlalchemy import select, or_
from sqlalchemy.orm import selectinload

sys.path.append('.')  # Add the current directory to path for imports

from code_monitor.db.database import SessionLocal
from code_monitor.db.models import Repository, Commit, File, QAGFile, QAGCommit, QAGRepository

async def display_file_embeddings(file_id=None, commit_id=None, repo_id=None, qag=False, limit=5, dimensions=5):
    """Display file embeddings from committed files"""
    async with SessionLocal() as db:
        # Determine which models to use based on qag flag
        file_model = QAGFile if qag else File
        commit_model = QAGCommit if qag else Commit
        repo_model = QAGRepository if qag else Repository
        
        # Build query to get files with embeddings
        query = (
            select(file_model)
            .filter(file_model.embedding.is_not(None))
            .options(selectinload(file_model.commit))
            .order_by(file_model.created_at.desc())
            .limit(limit)
        )
        
        # Apply filters if provided
        if file_id:
            query = query.filter(file_model.id == file_id)
        if commit_id:
            query = query.filter(file_model.commit_id == commit_id)
        if repo_id:
            # Need to join with commits to filter by repository
            query = query.join(commit_model).filter(commit_model.repository_id == repo_id)
            
        # Execute query
        result = await db.execute(query)
        files = result.scalars().all()
        
        if not files:
            repo_str = f"repository ID {repo_id}" if repo_id else "any repository"
            commit_str = f"commit ID {commit_id}" if commit_id else "any commit"
            file_str = f"file ID {file_id}" if file_id else "any file"
            qag_str = "QAG" if qag else "standard"
            print(f"No {qag_str} files with embeddings found for {repo_str}, {commit_str}, {file_str}.")
            return
        
        print(f"\nFound {len(files)} {'QAG' if qag else ''} files with embeddings:")
        print("-" * 80)
        
        for file in files:
            try:
                # Get commit and repository info
                commit = file.commit
                if commit:
                    repo_result = await db.execute(
                        select(repo_model).filter(repo_model.id == commit.repository_id)
                    )
                    repo = repo_result.scalars().first()
                    repo_name = repo.name if repo else "Unknown"
                    
                    print(f"File ID: {file.id}")
                    print(f"Path: {file.file_path}")
                    print(f"Commit: {commit.commit_hash[:10]}... ({commit.id})")
                    print(f"Repository: {repo_name} ({commit.repository_id})")
                    print(f"Change Type: {file.change_type}")
                    print(f"Language: {file.language or 'Unknown'}")
                    
                    # Display embedding information
                    embedding = file.embedding
                    if embedding is not None:
                        embedding_length = len(embedding)
                        print(f"Embedding: Vector with {embedding_length} dimensions")
                        
                        # Safely convert to numpy array and calculate statistics
                        embedding_array = np.array(embedding, dtype=float)
                        print(f"Mean: {np.mean(embedding_array):.6f}")
                        print(f"Std Dev: {np.std(embedding_array):.6f}")
                        print(f"Min: {np.min(embedding_array):.6f}")
                        print(f"Max: {np.max(embedding_array):.6f}")
                        
                        # Show first few dimensions of the embedding
                        show_dims = min(dimensions, embedding_length)
                        print(f"\nFirst {show_dims} dimensions:")
                        for i in range(show_dims):
                            print(f"  [{i}]: {embedding[i]:.6f}")
                        
                        # Show last few dimensions
                        if embedding_length > show_dims * 2:
                            print(f"\nLast {show_dims} dimensions:")
                            for i in range(embedding_length - show_dims, embedding_length):
                                print(f"  [{i}]: {embedding[i]:.6f}")
                    else:
                        print("Warning: Embedding field exists but contains None value")
                        
                    print("\n" + "-" * 80)
                else:
                    print(f"File ID {file.id} has no associated commit information.")
            except Exception as e:
                print(f"Error processing file ID {file.id}: {str(e)}")

async def main():
    parser = argparse.ArgumentParser(description='View embeddings for committed files')
    parser.add_argument('--file-id', type=int, help='Filter by specific file ID')
    parser.add_argument('--commit-id', type=int, help='Filter by commit ID')
    parser.add_argument('--repo-id', type=int, help='Filter by repository ID')
    parser.add_argument('--qag', action='store_true', help='View QAG files instead of standard files')
    parser.add_argument('--limit', type=int, default=5, help='Maximum number of files to show')
    parser.add_argument('--dimensions', type=int, default=5, help='Number of embedding dimensions to display')
    
    args = parser.parse_args()
    
    try:
        await display_file_embeddings(
            args.file_id, 
            args.commit_id, 
            args.repo_id, 
            args.qag, 
            args.limit, 
            args.dimensions
        )
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    asyncio.run(main())