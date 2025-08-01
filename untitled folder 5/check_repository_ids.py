#!/usr/bin/env python3

import asyncio
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from code_monitor.db.database import get_db, init_db
from code_monitor.db.models import Repository, QAGRepository

async def check_repository_ids():
    # Initialize the database connection
    await init_db()
    
    # Get a database session
    async for db in get_db():
        print("==== Standard Repositories ====")
        result = await db.execute(select(Repository))
        std_repos = result.scalars().all()
        
        for repo in std_repos:
            print(f"ID: {repo.id}, Name: {repo.name}, Owner: {repo.owner}, URL: {repo.url}")
            
        print("\n==== QAG Repositories ====")
        result = await db.execute(select(QAGRepository))
        qag_repos = result.scalars().all()
        
        for repo in qag_repos:
            print(f"ID: {repo.id}, Name: {repo.name}, Owner: {repo.owner}, URL: {repo.url}")
        
        # Exit the loop after successful query
        break

if __name__ == "__main__":
    asyncio.run(check_repository_ids())