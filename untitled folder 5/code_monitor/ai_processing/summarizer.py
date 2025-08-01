import logging
from typing import List, Dict, Any, Optional
from tenacity import retry, stop_after_attempt, wait_fixed
from code_monitor.config import settings
from code_monitor.utils.ollama_client import OllamaClient

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

logger = logging.getLogger(__name__)

class CommitSummarizer:
    """Generate natural language summaries of code changes."""
    
    def __init__(self):
        self.use_ollama = settings.USE_OLLAMA
        if self.use_ollama:
            self.ollama_client = None
        else:
            self.api_key = settings.OPENAI_API_KEY
            self.model = settings.OPENAI_MODEL
            if self.api_key:
                self.client = OpenAI(api_key=self.api_key)
                logger.info(f"Using OpenAI with model: {settings.OPENAI_MODEL}")
            else:
                logger.warning("Neither Ollama nor OpenAI is configured")
    
    async def initialize(self):
        """Initialize async components"""
        if self.use_ollama:
            self.ollama_client = await OllamaClient.create()
    
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def summarize_commit(
        self, 
        commit_message: str, 
        files_changed: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a detailed summary of a commit using LLM
        
        Args:
            commit_message: The original commit message
            files_changed: List of files changed in the commit with their diffs
            
        Returns:
            A detailed summary of the changes
        """
        if not (self.use_ollama or self.api_key):
            logger.warning("No LLM configured, skipping commit summarization")
            return commit_message
            
        try:
            # Prepare the input
            file_changes = []
            total_diff_size = 0
            
            for file in files_changed:
                file_path = file.get("filename", "unknown")
                diff = file.get("patch", "")
                
                # Limit the size of diffs to avoid exceeding token limits
                if diff:
                    max_diff_size = 1000
                    if len(diff) > max_diff_size:
                        diff = diff[:max_diff_size] + "... [truncated]"
                    
                    total_diff_size += len(diff)
                    file_changes.append(f"File: {file_path}\nChanges:\n{diff}")
            
            # Further limit if the total is too large
            max_total_diff = 6000
            if total_diff_size > max_total_diff:
                file_changes = file_changes[:3]
                file_changes.append(f"... and {len(files_changed) - 3} more files (truncated)")
                
            changes_text = "\n\n".join(file_changes)
            
            prompt = f"""
            Please provide a comprehensive analysis of the following code changes:

            Original commit message: {commit_message}
            
            Changes:
            {changes_text}
            
            Provide a detailed explanation that includes:
            1. A technical summary of all changes made across files
            2. The purpose and rationale behind these changes
            3. How the changes work together and their relationships
            4. Any architectural or design pattern changes
            5. Notable code quality improvements or potential issues
            6. Impact on existing functionality and system behavior
            7. Important implementation details developers should know
            
            Format your response in clear sections and explain technical concepts thoroughly.
            Focus on helping developers understand both what changed and why it matters.
            """
            
            if self.use_ollama:
                messages = [
                    {"role": "system", "content": "You are a code review assistant that explains code changes clearly and concisely."},
                    {"role": "user", "content": prompt}
                ]
                response = await self.ollama_client.generate(
                    messages=messages,
                    temperature=0.3,
                    max_tokens=800
                )
                summary = response["choices"][0]["message"]["content"]
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a code review assistant that explains code changes clearly and concisely."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=800
                )
                summary = response.choices[0].message.content
                
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing commit: {str(e)}")
            return commit_message  # Return original message on error
            
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def generate_change_impact(self, 
        file_changes: List[Dict[str, Any]], 
        codebase_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Enhanced change impact analysis for better code understanding"""
        if not (self.use_ollama or self.api_key):
            logger.warning("No LLM configured, skipping impact analysis")
            return {"impact": "unknown"}
            
        try:
            changes_summary = []
            
            for file in file_changes:
                file_path = file.get("file_path", "unknown")
                change_type = file.get("change_type", "modified")
                content = file.get("content_after", "")
                
                if content:
                    content = content[:2000] + "..." if len(content) > 2000 else content
                    changes_summary.append(f"File: {file_path} ({change_type})\nContent:\n{content}")
            
            context = f"Codebase context: {codebase_context}\n\n" if codebase_context else ""
            changes_text = "\n\n".join(changes_summary)
            
            prompt = f"""
            {context}
            Perform a detailed technical analysis of these code changes and their impact:
            
            {changes_text}
            
            Provide your analysis as a JSON object with these sections:

            1. code_changes: {
                "summary": "High-level overview of changes",
                "technical_details": [
                    {
                        "component": "affected component/module",
                        "changes": "specific technical changes",
                        "complexity": "high/medium/low",
                        "impact_scope": "local/module/system-wide"
                    }
                ],
                "architectural_impact": {
                    "patterns_affected": ["design patterns affected"],
                    "interfaces_changed": ["API/interface changes"],
                    "data_flow_changes": ["changes to data flow"]
                }
            }

            2. implementation_analysis: {
                "algorithms_changed": ["affected algorithms"],
                "data_structures": ["affected data structures"],
                "performance_impact": {
                    "time_complexity_changes": "description",
                    "memory_usage_changes": "description",
                    "bottlenecks": ["potential bottlenecks"]
                },
                "code_quality": {
                    "maintainability_impact": "description",
                    "readability_changes": "description",
                    "technical_debt": ["new technical debt items"]
                }
            }

            3. testing_implications: {
                "required_tests": ["specific test cases needed"],
                "affected_scenarios": ["test scenarios to update"],
                "edge_cases": ["edge cases to verify"],
                "integration_tests": ["needed integration tests"]
            }

            4. dependency_analysis: {
                "internal_dependencies": ["affected internal modules"],
                "external_dependencies": ["affected external dependencies"],
                "version_constraints": ["version requirements"],
                "compatibility_issues": ["potential compatibility problems"]
            }

            5. security_impact: {
                "vulnerability_introduction": "yes/no with details",
                "attack_vectors": ["potential security risks"],
                "data_exposure": ["sensitive data concerns"],
                "mitigation_needed": ["required security measures"]
            }

            6. deployment_considerations: {
                "required_steps": ["deployment steps"],
                "configuration_changes": ["config updates needed"],
                "migration_requirements": ["data migration needs"],
                "rollback_plan": "rollback strategy",
                "monitoring_updates": ["metrics to track"]
            }

            Focus on technical accuracy and provide specific, actionable insights."""

            if self.use_ollama:
                messages = [
                    {"role": "system", "content": "You are an expert code analyst specializing in understanding code changes and their implications."},
                    {"role": "user", "content": prompt}
                ]
                response = await self.ollama_client.generate(
                    messages=messages,
                    temperature=0.1,
                    max_tokens=2000
                )
                analysis = response["choices"][0]["message"]["content"]
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are an expert code analyst specializing in understanding code changes and their implications."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.1,
                    max_tokens=2000,
                    response_format={"type": "json_object"}
                )
                analysis = response.choices[0].message.content
            
            # Parse the JSON response
            import json
            try:
                return json.loads(analysis)
            except json.JSONDecodeError:
                logger.error("Failed to parse JSON from impact analysis")
                return {"error": "Failed to parse impact analysis", "raw_response": analysis}
                
        except Exception as e:
            logger.error(f"Error generating change impact: {str(e)}")
            return {"error": str(e)}