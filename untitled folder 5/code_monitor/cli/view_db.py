#!/usr/bin/env python
import asyncio
import sys
from tabulate import tabulate
from sqlalchemy.future import select
from sqlalchemy import inspect, text

sys.path.append('.')  # Add the current directory to path for imports

from code_monitor.db.database import SessionLocal
from code_monitor.db.models import Repository, Commit, File, CodeSnippet, RepositoryFile
from code_monitor.db.models import QAGRepository, QAGCommit, QAGFile, QAGRepositoryFile, QAGCodeSnippet
from code_monitor.config import settings

async def view_table(table_name):
    """View all records in a specified table with full content display"""
    
    async with SessionLocal() as db:
        # Get table class
        tables = {
            'repositories': Repository,
            'commits': Commit,
            'files': File,
            'code_snippets': CodeSnippet,
            'repository_files': RepositoryFile,
            # Add QAG tables
            'qag_repositories': QAGRepository,
            'qag_commits': QAGCommit,
            'qag_files': QAGFile,
            'qag_repository_files': QAGRepositoryFile,
            'qag_code_snippets': QAGCodeSnippet
        }
        
        if table_name not in tables:
            print(f"Table '{table_name}' not found. Available tables: {', '.join(tables.keys())}")
            return
            
        try:
            # Use raw SQL to fetch actual columns from the database
            query = text(f"SELECT column_name FROM information_schema.columns WHERE table_name = '{table_name}' ORDER BY ordinal_position;")
            result = await db.execute(query)
            actual_columns = [row[0] for row in result]
            
            if not actual_columns:
                print(f"Table '{table_name}' not found in database.")
                return
                
            print(f"\nColumns in {table_name}: {', '.join(actual_columns)}")
            
            # Fetch the data using raw SQL to avoid ORM schema mismatches
            columns_str = ', '.join(actual_columns)
            
            # For qag_files table specifically, show a custom display with better formatting for code content
            if table_name == 'qag_files':
                # Get all records but limit content display
                query = text(f"SELECT {columns_str} FROM {table_name} ORDER BY id")
                result = await db.execute(query)
                records = result.fetchall()
                
                if not records:
                    print(f"No records found in the {table_name} table.")
                    return
                
                # Format data for display in a special way for qag_files
                print(f"\nTable: {table_name} ({len(records)} records)")
                
                # Prepare the tabular display with all columns
                table_data = []
                for record in records:
                    row_data = {}
                    # Process each column
                    for col_name, value in zip(actual_columns, record):
                        # For content columns, show the first few lines or a placeholder
                        if col_name in ('content_before', 'content_after') and value:
                            # Extract first few lines (up to 3) for display
                            lines = value.split('\n')[:5]
                            preview = '\n'.join(lines)
                            if len(lines) < 5 and len(preview) < 100:
                                row_data[col_name] = preview
                            else:
                                row_data[col_name] = preview + '...' if len(value) > len(preview) else preview
                        # For embedding column, show dimensions
                        elif col_name == 'embedding' and value is not None:
                            try:
                                row_data[col_name] = f"[Embedding vector with {len(value)} dimensions]"
                            except:
                                row_data[col_name] = "[Embedding vector]"
                        # For other columns, use as is
                        else:
                            row_data[col_name] = value
                    
                    table_data.append(row_data)
                
                # Display the table
                headers = actual_columns
                rows = [[row.get(col, '') for col in headers] for row in table_data]
                print(tabulate(rows, headers=headers, tablefmt="grid"))
                
                # Offer instructions to view complete content
                print("\nTo view the complete content of a file, run:")
                print(f"python view_file_content.py {table_name} <id> <column_name>")
                print(f"Example: python view_file_content.py {table_name} {records[0][0]} content_after")
                
                # Create the helper script
                create_content_viewer_script()
                return
            
            # For other tables, use standard display
            query = text(f"SELECT {columns_str} FROM {table_name}")
            result = await db.execute(query)
            records = result.fetchall()
            
            if not records:
                print(f"No records found in the {table_name} table.")
                return
                
            # Format data for display, handling large fields like embeddings
            table_data = []
            for record in records:
                row = []
                for column_name, value in zip(actual_columns, record):
                    # Handle special fields
                    if column_name == 'embedding' and value is not None:
                        try:
                            value = f"[Embedding vector with {len(value)} dimensions]"
                        except:
                            value = "[Embedding vector]"
                    elif isinstance(value, (list, dict)) and value is not None:
                        value = str(value)
                    elif isinstance(value, str) and len(value) > 100:
                        value = value[:100] + "..."
                    
                    row.append(value)
                table_data.append(row)
            
            # Display table
            print(f"\nTable: {table_name} ({len(records)} records)")
            print(tabulate(table_data, headers=actual_columns, tablefmt="grid"))
            
            # For tables with content field, offer to display complete content
            content_columns = [col for col in actual_columns if col in ('content', 'content_before', 'content_after')]
            if content_columns and records:
                print("\nTo view complete content for a specific record, run:")
                print(f"python view_file_content.py {table_name} <record_id> <column_name>")
                print(f"Example: python view_file_content.py {table_name} {records[0][0]} {content_columns[0]}")
                
                # Create a helper script for viewing complete content
                create_content_viewer_script()
                
        except Exception as e:
            print(f"Error querying table {table_name}: {str(e)}")
            print("Attempting alternative approach with manual column selection...")
            
            # Fallback to a simpler query that only selects ID and common columns
            try:
                common_cols = ['id', 'file_path', 'commit_id', 'repository_id', 'language', 'change_type', 'created_at']
                safe_cols = [col for col in common_cols if col in actual_columns]
                
                if not safe_cols:
                    print(f"Could not identify any safe columns to query for table {table_name}")
                    return
                    
                cols_str = ', '.join(safe_cols)
                safe_query = text(f"SELECT {cols_str} FROM {table_name}")
                safe_result = await db.execute(safe_query)
                safe_records = safe_result.fetchall()
                
                if not safe_records:
                    print(f"No records found in the {table_name} table.")
                    return
                
                # Display limited data
                safe_data = []
                for record in safe_records:
                    safe_data.append(list(record))
                
                print(f"\nTable: {table_name} ({len(safe_records)} records) - LIMITED COLUMNS")
                print(tabulate(safe_data, headers=safe_cols, tablefmt="grid"))
            except Exception as e2:
                print(f"Alternative approach also failed: {str(e2)}")

def create_content_viewer_script():
    """Create a helper script for viewing complete file content"""
    script_path = 'view_file_content.py'
    
    # Skip if the script already exists
    if os.path.exists(script_path):
        return
        
    with open(script_path, 'w') as f:
        f.write('''#!/usr/bin/env python
import asyncio
import sys
from sqlalchemy import text
import os
sys.path.append('.')
from code_monitor.db.database import SessionLocal

async def view_file_content(table_name, record_id, column_name):
    async with SessionLocal() as db:
        query = text(f"SELECT {column_name} FROM {table_name} WHERE id = :id")
        result = await db.execute(query, {"id": record_id})
        record = result.fetchone()
        
        if not record:
            print(f"No record found with ID {record_id} in table {table_name}")
            return
            
        content = record[0]
        if not content:
            print(f"No content found in column {column_name} for record {record_id}")
            return
            
        print(f"\\nContent from {table_name}.{column_name} (ID: {record_id}):\\n")
        print("=" * 80)
        print(content)
        print("=" * 80)
        
        # Optionally save to a file
        save_to_file = input("\\nSave this content to a file? (y/n): ").lower() == 'y'
        if save_to_file:
            file_query = text(f"SELECT file_path FROM {table_name} WHERE id = :id")
            file_result = await db.execute(file_query, {"id": record_id})
            file_record = file_result.fetchone()
            
            if file_record and file_record[0]:
                default_filename = os.path.basename(file_record[0])
            else:
                default_filename = f"{table_name}_{record_id}_{column_name}.txt"
                
            filename = input(f"Enter filename (default: {default_filename}): ")
            if not filename:
                filename = default_filename
                
            with open(filename, 'w') as f:
                f.write(content)
            print(f"Content saved to {filename}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python view_file_content.py <table_name> <record_id> <column_name>")
        print("Example: python view_file_content.py qag_files 1 content_after")
        sys.exit(1)
        
    table_name = sys.argv[1]
    record_id = int(sys.argv[2])
    column_name = sys.argv[3]
    
    asyncio.run(view_file_content(table_name, record_id, column_name))
'''
        )
    
    # Make the script executable
    try:
        import os
        os.chmod(script_path, 0o755)
        print(f"\nCreated helper script: {script_path}")
    except:
        pass

async def list_tables():
    """List all tables in the database"""
    async with SessionLocal() as db:
        try:
            # Query information_schema for table names
            query = text("SELECT table_name FROM information_schema.tables WHERE table_schema='public';")
            result = await db.execute(query)
            tables = result.scalars().all()
            
            print("\nAvailable tables in the database:")
            standard_tables = ['repositories', 'commits', 'files', 'code_snippets', 'repository_files']
            qag_tables = ['qag_repositories', 'qag_commits', 'qag_files', 'qag_repository_files', 'qag_code_snippets']
            
            print("\nStandard Repository Tables:")
            for table in standard_tables:
                if table in tables:
                    print(f"- {table}")
            
            print("\nQAG Repository Tables:")
            for table in qag_tables:
                if table in tables:
                    print(f"- {table}")
            
            print("\nOther Tables:")
            other_tables = [table for table in tables if table not in standard_tables and table not in qag_tables]
            for table in other_tables:
                print(f"- {table}")
                
            print("\nTo view a specific table, run: python view_db.py <table_name>")
            print("Example: python view_db.py repositories")
            print("Example: python view_db.py qag_repositories")
            
        except Exception as e:
            print(f"Error listing tables: {str(e)}")

async def view_embeddings(table_name, limit=5):
    """View records with embeddings in a specified table"""
    async with SessionLocal() as db:
        tables = {
            'repository_files': RepositoryFile,
            'qag_repository_files': QAGRepositoryFile
        }
        
        if table_name not in tables:
            print(f"Table '{table_name}' not supported for embedding view. Available tables: {', '.join(tables.keys())}")
            return
        
        table_class = tables[table_name]
        
        # Get records with embeddings
        query = select(table_class).where(table_class.embedding.is_not(None)).limit(limit)
        result = await db.execute(query)
        records = result.scalars().all()
        
        if not records:
            print(f"No records with embeddings found in the {table_name} table.")
            return
        
        print(f"\nSample records with embeddings from {table_name} ({len(records)} records):")
        
        for record in records:
            print(f"\nID: {record.id}")
            print(f"File Path: {record.file_path}")
            print(f"Language: {record.language}")
            
            if hasattr(record, 'repository_id'):
                print(f"Repository ID: {record.repository_id}")
                
            if record.embedding is not None:
                embedding = record.embedding
                embedding_length = len(embedding)
                print(f"Embedding: Vector with {embedding_length} dimensions")
                print(f"First 5 dimensions: {embedding[:5]}")
                
            print("-" * 40)

async def show_connection_info():
    """Display database connection information"""
    print("\nDatabase connection information:")
    print(f"Host: {settings.POSTGRES_HOST}")
    print(f"Port: {settings.POSTGRES_PORT}")
    print(f"Database: {settings.POSTGRES_DB}")
    print(f"User: {settings.POSTGRES_USER}")
    print(f"URL: {settings.DATABASE_URL.replace(settings.POSTGRES_PASSWORD, '********')}")

async def main():
    print("GitHub Code Monitor Database Viewer")
    print("=" * 40)
    
    await show_connection_info()
    
    if len(sys.argv) < 2:
        # If no table specified, list all tables
        await list_tables()
    elif sys.argv[1] == "embeddings" and len(sys.argv) >= 3:
        # View embeddings for a specific table
        table_name = sys.argv[2].lower()
        limit = int(sys.argv[3]) if len(sys.argv) >= 4 else 5
        await view_embeddings(table_name, limit)
    else:
        table_name = sys.argv[1].lower()
        await view_table(table_name)

if __name__ == "__main__":
    asyncio.run(main())