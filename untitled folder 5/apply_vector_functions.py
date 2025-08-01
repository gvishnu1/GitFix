#!/usr/bin/env python3
"""
Script to apply vector functions to the database
This fixes the 'function dot_product(vector, character varying) does not exist' error
"""

import asyncio
import logging
import sys
from code_monitor.db.migrations import apply_migrations

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

async def main():
    """Apply vector functions to the database"""
    try:
        logger.info("Starting application of vector functions...")
        await apply_migrations()
        logger.info("Successfully applied vector functions")
    except Exception as e:
        logger.error(f"Error applying vector functions: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())