#!/usr/bin/env python
"""
Analyze repository code using AI-powered embeddings.
This script uses embeddings to understand code structure, dependencies, and patterns.
"""

import asyncio
import argparse
import sys
import os
import json
import logging
from tabulate import tabulate
from sqlalchemy import select
from datetime import datetime

# Fix the module import path by adding the project root directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

from code_monitor.db.database import SessionLocal
from code_monitor.db.models import Repository, RepositoryFile, QAGRepositoryFile
from code_monitor.ai_processing.code_analyzer import CodeAnalyzer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("repository_analyzer")

async def analyze_full_repository(repo_id: int, is_qag: bool = False):
    """Analyze an entire repository and print the results"""
    async with SessionLocal() as db:
        analyzer = CodeAnalyzer()
        analysis = await analyzer.analyze_repository(db, repo_id, is_qag)
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        print("\n" + "=" * 80)
        print(f"Repository Analysis: {analysis['repository_name']}")
        print("=" * 80)
        
        # Print summary
        print("\nSummary:")
        print("-" * 80)
        print(analysis["summary"])
        print()
        
        # Print language breakdown
        print("Language Distribution:")
        print("-" * 80)
        language_data = [[lang, count] for lang, count in analysis["languages"].items()]
        print(tabulate(language_data, headers=["Language", "Count"], tablefmt="grid"))
        print()
        
        # Print key concepts
        print("Key Concepts in Codebase:")
        print("-" * 80)
        if analysis["key_concepts"]:
            concept_data = []
            for concept in analysis["key_concepts"]:
                sample_files = ", ".join(concept["sample_files"][:2])
                if len(concept["sample_files"]) > 2:
                    sample_files += f", ... ({concept['file_count'] - 2} more)"
                concept_data.append([concept["name"], concept["file_count"], sample_files])
            print(tabulate(concept_data, headers=["Concept", "Files", "Examples"], tablefmt="grid"))
        else:
            print("No distinct concepts identified.")
        print()
        
        # Print core modules
        print("Core Modules:")
        print("-" * 80)
        if analysis["core_modules"]:
            module_data = []
            for module in analysis["core_modules"]:
                sample_files = ", ".join(module["sample_files"][:2])
                if len(module["sample_files"]) > 2:
                    sample_files += f", ... ({module['file_count'] - 2} more)"
                module_data.append([module["name"], module["file_count"], sample_files])
            print(tabulate(module_data, headers=["Module", "Files", "Examples"], tablefmt="grid"))
        else:
            print("No core modules identified.")
        print()
        
        # Print application features
        if "application_features" in analysis and analysis["application_features"]:
            features = analysis["application_features"]
            
            print("Application Type:")
            print("-" * 80)
            print(f"This appears to be a {features['application_type']} application")
            print()
            
            print("Application Features:")
            print("-" * 80)
            
            for category, items in features.get("features", {}).items():
                if items:
                    print(f"\n{category.capitalize()} Features:")
                    feature_data = []
                    for item in items:
                        name = item["name"]
                        type_str = item["type"]
                        relevance = item["relevance"]
                        feature_data.append([name, type_str, relevance])
                    print(tabulate(feature_data, headers=["Name", "Type", "Relevance"], tablefmt="simple"))
            print()
        
        # Identify code patterns
        patterns = await analyzer.identify_code_patterns(db, repo_id, is_qag)
        if "error" not in patterns:
            print("Common Code Patterns:")
            print("-" * 80)
            if patterns["function_patterns"]:
                print("\nFunction Patterns:")
                function_data = []
                for pattern in patterns["function_patterns"]:
                    examples = ", ".join(pattern["examples"])
                    function_data.append([pattern["pattern"], pattern["count"], examples])
                print(tabulate(function_data, headers=["Pattern", "Count", "Examples"], tablefmt="simple"))
            
            if patterns["class_patterns"]:
                print("\nClass Patterns:")
                class_data = []
                for pattern in patterns["class_patterns"]:
                    examples = ", ".join(pattern["examples"])
                    class_data.append([pattern["pattern"], pattern["count"], examples])
                print(tabulate(class_data, headers=["Pattern", "Count", "Examples"], tablefmt="simple"))
            
            if patterns["method_patterns"]:
                print("\nMethod Patterns:")
                method_data = []
                for pattern in patterns["method_patterns"]:
                    examples = ", ".join(pattern["examples"])
                    method_data.append([pattern["pattern"], pattern["count"], examples])
                print(tabulate(method_data, headers=["Pattern", "Count", "Examples"], tablefmt="simple"))
            
            if not (patterns["function_patterns"] or patterns["class_patterns"] or patterns["method_patterns"]):
                print("No significant code patterns identified.")

async def find_similar_code(repo_id: int, query_text: str, limit: int = 5, is_qag: bool = False):
    """
    Find code similar to the query text using embeddings
    
    Args:
        repo_id: Repository ID
        query_text: Text to search for
        limit: Maximum number of results
        is_qag: Whether it's a QAG repository
    """
    analyzer = CodeAnalyzer()
    
    async with SessionLocal() as db:
        # Get repository info
        query = select(Repository).where(Repository.id == repo_id)
        result = await db.execute(query)
        repo = result.scalars().first()
        
        if not repo:
            print(f"Error: Repository with ID {repo_id} not found")
            return
        
        print(f"\n=== Finding Similar Code in {repo.name} (ID: {repo_id}) ===\n")
        print(f"Query: '{query_text}'")
        print("-" * 80)
        
        # Search for similar code
        results = await analyzer.find_similar_code(db, repo_id, query_text, limit, is_qag)
        
        if not results:
            print("No similar code found.")
            return
        
        # Print results
        for i, result in enumerate(results):
            print(f"\nMatch #{i+1} - Similarity: {result['similarity']:.4f}")
            print(f"Type: {result['type']}")
            
            if result['type'] == 'file':
                print(f"File: {result['file_path']}")
                print(f"Language: {result['language']}")
            else:
                print(f"Name: {result['name']}")
                print(f"Type: {result['type']}")
                
                # Get the file path for this snippet
                file_model = QAGRepositoryFile if is_qag else RepositoryFile
                file_query = select(file_model).where(file_model.id == result['file_id'])
                file_result = await db.execute(file_query)
                file = file_result.scalars().first()
                if file:
                    print(f"File: {file.file_path}")
            
            print("\nCode Preview:")
            print("-" * 40)
            lines = result['content'].split('\n')[:10]  # Show up to 10 lines
            print('\n'.join(lines))
            if len(result['content'].split('\n')) > 10:
                print("...")

async def analyze_code_relations(file_id: int, is_qag: bool = False):
    """
    Analyze the relationships between a file and other files in the repository
    
    Args:
        file_id: File ID
        is_qag: Whether it's a QAG repository file
    """
    analyzer = CodeAnalyzer()
    
    async with SessionLocal() as db:
        # Get file info
        file_model = QAGRepositoryFile if is_qag else RepositoryFile
        file_query = select(file_model).where(file_model.id == file_id)
        file_result = await db.execute(file_query)
        file = file_result.scalars().first()
        
        if not file:
            print(f"Error: File with ID {file_id} not found")
            return
        
        print(f"\n=== Analyzing Code Relations for {file.file_path} (ID: {file_id}) ===\n")
        
        # Analyze code relations
        analysis = await analyzer.analyze_code_relations(db, file_id, is_qag)
        
        if "error" in analysis:
            print(f"Error: {analysis['error']}")
            return
        
        # Print imports
        print("Imports:")
        print("-" * 80)
        if analysis["imports"]:
            for imp in analysis["imports"]:
                print(f"- {imp}")
        else:
            print("No imports detected.")
        print()
        
        # Print related files
        print("Related Files:")
        print("-" * 80)
        if analysis["related_files"]:
            related_data = []
            for related in analysis["related_files"]:
                related_data.append([related["file_path"], f"{related['similarity']:.4f}"])
            print(tabulate(related_data, headers=["File Path", "Similarity"], tablefmt="grid"))
        else:
            print("No related files found.")

async def generate_documentation(file_id: int, is_qag: bool = False, output_file: str = None):
    """
    Generate documentation for a file based on its embedding and content
    
    Args:
        file_id: File ID
        is_qag: Whether it's a QAG repository file
        output_file: Optional file path to write the documentation to
    """
    analyzer = CodeAnalyzer()
    
    async with SessionLocal() as db:
        # Get file info
        file_model = QAGRepositoryFile if is_qag else RepositoryFile
        file_query = select(file_model).where(file_model.id == file_id)
        file_result = await db.execute(file_query)
        file = file_result.scalars().first()
        
        if not file:
            print(f"Error: File with ID {file_id} not found")
            return
        
        print(f"\n=== Generating Documentation for {file.file_path} (ID: {file_id}) ===\n")
        
        # Generate documentation
        documentation = await analyzer.generate_documentation(db, file_id, is_qag)
        
        if documentation.startswith("Error:"):
            print(documentation)
            return
        
        # Output documentation
        if output_file:
            with open(output_file, 'w') as f:
                f.write(documentation)
            print(f"Documentation written to {output_file}")
        else:
            print(documentation)

async def main():
    parser = argparse.ArgumentParser(description='Analyze repository code using AI-powered embeddings')
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')
    
    # Analyze repository command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze an entire repository')
    analyze_parser.add_argument('--repo-id', type=int, required=True, help='Repository ID')
    analyze_parser.add_argument('--qag', action='store_true', help='Use QAG-specific models')
    
    # Find similar code command
    find_parser = subparsers.add_parser('find', help='Find code similar to query')
    find_parser.add_argument('--repo-id', type=int, required=True, help='Repository ID')
    find_parser.add_argument('--query', type=str, required=True, help='Query text')
    find_parser.add_argument('--limit', type=int, default=5, help='Maximum number of results')
    find_parser.add_argument('--qag', action='store_true', help='Use QAG-specific models')
    
    # Analyze code relations command
    relations_parser = subparsers.add_parser('relations', help='Analyze code relations between files')
    relations_parser.add_argument('--file-id', type=int, required=True, help='File ID')
    relations_parser.add_argument('--qag', action='store_true', help='Use QAG-specific models')
    
    # Generate documentation command
    docs_parser = subparsers.add_parser('docs', help='Generate documentation for a file')
    docs_parser.add_argument('--file-id', type=int, required=True, help='File ID')
    docs_parser.add_argument('--qag', action='store_true', help='Use QAG-specific models')
    docs_parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    try:
        if args.command == 'analyze':
            await analyze_full_repository(args.repo_id, args.qag)
        elif args.command == 'find':
            await find_similar_code(args.repo_id, args.query, args.limit, args.qag)
        elif args.command == 'relations':
            await analyze_code_relations(args.file_id, args.qag)
        elif args.command == 'docs':
            await generate_documentation(args.file_id, args.qag, args.output)
        else:
            parser.print_help()
    except Exception as e:
        logger.error(f"Error during repository analysis: {str(e)}")
        raise

if __name__ == "__main__":
    asyncio.run(main())