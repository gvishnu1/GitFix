import os
import logging
import json
import asyncio
import re
from typing import Dict, List, Any, Optional
import httpx

from code_monitor.config import settings

logger = logging.getLogger(__name__)

class OllamaClient:
    """Client for interacting with Ollama API"""
    
    def __init__(self, model: str = None):
        self.base_url = settings.OLLAMA_BASE_URL
        self.model = model or settings.OLLAMA_MODEL
        self._client = None
        
    @classmethod
    async def create(cls, model: str = None):
        """Create and initialize the Ollama client"""
        client = cls(model)
        # Ensure client is initialized
        await client._ensure_client()
        return client
        
    async def _ensure_client(self):
        """Ensure the HTTP client is initialized"""
        if not self._client:
            try:
                self._client = httpx.AsyncClient(timeout=60.0)  # Increase timeout for longer prompts
                logger.info(f"Initialized httpx AsyncClient for Ollama API at {self.base_url}")
                
                # Test connection to Ollama API
                try:
                    test_response = await self._client.get(f"{self.base_url}/api/version")
                    if test_response.status_code == 200:
                        version_data = test_response.json()
                        logger.info(f"Successfully connected to Ollama API, version: {version_data.get('version', 'unknown')}")
                    else:
                        logger.warning(f"Connected to Ollama API, but received unexpected status code: {test_response.status_code}")
                except Exception as e:
                    logger.warning(f"Failed to verify Ollama API connection: {str(e)}")
            except Exception as e:
                logger.error(f"Failed to initialize httpx AsyncClient: {str(e)}")
                # Set to None to allow retry later
                self._client = None
                raise
                
    async def close(self):
        """Close the underlying HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None
            
    async def generate(self, messages: List[Dict[str, str]], 
                      temperature: float = 0.7, 
                      max_tokens: int = 1000,
                      system: str = None) -> Dict[str, Any]:
        """
        Generate a response from Ollama based on the provided messages.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (higher = more random)
            max_tokens: Maximum number of tokens to generate
            system: Optional system message to override any in messages
            
        Returns:
            Dictionary containing the generated response
        """
        try:
            # Ensure the client is initialized
            await self._ensure_client()
            
            if not self._client:
                logger.error("Failed to initialize HTTP client for Ollama API")
                return {"error": "HTTP client initialization failed", 
                        "choices": [{"message": {"role": "assistant", "content": "I'm having trouble connecting to the language model. Please try again later."}}]}
            
            # Extract the system message if present
            system_message = None
            user_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    user_messages.append(msg)
                    
            # Override with provided system message if any
            if system:
                system_message = system
                
            # Format for Ollama API - it expects a different format than OpenAI
            prompt = ""
            
            # If we have a conversation
            if len(user_messages) > 1:
                for i, msg in enumerate(user_messages):
                    role_prefix = "User: " if msg['role'] == 'user' else "Assistant: "
                    prompt += f"{role_prefix}{msg['content']}\n\n"
                    
                # Add the final role prefix for the response
                prompt += "Assistant: "
            else:
                # Simple query
                prompt = user_messages[0]['content'] if user_messages else ""
            
            # Prepare the request - note Ollama API is different from OpenAI
            request_data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            }
            
            # Add system message if present
            if system_message:
                request_data["system"] = system_message
                
            # Make the request
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/generate",
                    json=request_data,
                    timeout=120.0  # Extended timeout for longer responses
                )
            except httpx.TimeoutException:
                logger.error(f"Timeout while calling Ollama API")
                return {"error": "API Timeout", 
                        "choices": [{"message": {"role": "assistant", "content": "The request to the language model timed out. Please try again with a shorter query."}}]}
            except httpx.ConnectError:
                logger.error(f"Connection error to Ollama API at {self.base_url}")
                # Reset client so we'll try to reconnect next time
                self._client = None
                return {"error": "Connection Error", 
                        "choices": [{"message": {"role": "assistant", "content": "I couldn't connect to the language model. Please check if the Ollama service is running."}}]}
            except Exception as e:
                logger.error(f"Error making request to Ollama API: {str(e)}")
                return {"error": f"Request Error: {str(e)}", 
                        "choices": [{"message": {"role": "assistant", "content": f"An error occurred while communicating with the language model: {str(e)}"}}]}
            
            if response.status_code != 200:
                logger.error(f"Error from Ollama API: {response.status_code}, {response.text}")
                return {"error": f"API Error: {response.status_code}", 
                        "choices": [{"message": {"role": "assistant", "content": f"The language model returned an error (status code {response.status_code}). Please try again later."}}]}
                
            # Parse the response
            try:
                result = response.json()
            except json.JSONDecodeError:
                logger.error(f"Failed to parse Ollama API response as JSON: {response.text[:200]}")
                return {"error": "Invalid JSON response", 
                        "choices": [{"message": {"role": "assistant", "content": "I received an invalid response from the language model. Please try again."}}]}
            
            # Format the response to match OpenAI's structure
            openai_formatted_response = {
                "id": "ollama-" + self.model,
                "object": "chat.completion",
                "created": 0,  # Ollama doesn't provide this
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": result.get("response", "")
                        },
                        "finish_reason": "stop"  # Ollama doesn't provide this
                    }
                ]
            }
            
            return openai_formatted_response
            
        except Exception as e:
            logger.error(f"Error in Ollama generate request: {str(e)}")
            return {"error": f"Request failed: {str(e)}", 
                    "choices": [{"message": {"role": "assistant", "content": f"An unexpected error occurred: {str(e)}"}}]}
    
    async def generate_with_json(self, messages: List[Dict[str, str]], 
                               temperature: float = 0.2) -> Dict[str, Any]:
        """
        Generate a response with JSON formatting, with retries for proper JSON output.
        
        Args:
            messages: A list of message dictionaries with 'role' and 'content' keys
            temperature: Controls randomness (lower is better for JSON)
            
        Returns:
            Dictionary containing the parsed JSON response
        """
        # Add a strong instruction to return valid JSON
        system_message = "You are a helpful assistant that always responds with valid JSON only. No explanations, just JSON."
        
        try:
            # Try to generate valid JSON (up to 3 attempts)
            for attempt in range(3):
                response = await self.generate(
                    messages=messages,
                    temperature=temperature,
                    system=system_message,
                    max_tokens=4000  # Increase max tokens for JSON responses
                )
                
                if "error" in response:
                    logger.error(f"Error in generate_with_json: {response['error']}")
                    return {"error": response["error"]}
                
                content = response["choices"][0]["message"]["content"]
                
                # Try to parse the response as JSON
                try:
                    # Clean up markdown code blocks if present
                    if "```json" in content:
                        content = content.split("```json")[1].split("```")[0].strip()
                    elif "```" in content:
                        content = content.split("```")[1].split("```")[0].strip()
                    
                    # Try to extract JSON object using regex
                    json_match = re.search(r'(\{.*\})', content, re.DOTALL)
                    if json_match:
                        content = json_match.group(1)
                        
                    # Parse it
                    result = json.loads(content)
                    return result
                except json.JSONDecodeError as e:
                    logger.error(f"JSON decode error: {str(e)}")
                    if attempt < 2:  # Only log if not the last attempt
                        logger.info(f"Retrying JSON generation (attempt {attempt+2})")
                        
            # If we got here, all attempts failed
            logger.error("Failed to generate valid JSON after multiple attempts")
            return {"error": "Failed to generate valid JSON"}
            
        except Exception as e:
            logger.error(f"Error in generate_with_json: {str(e)}")
            return {"error": f"JSON generation failed: {str(e)}"}

    async def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for the given text using Ollama.
        
        Args:
            text: The text to generate embeddings for
            
        Returns:
            List of float values representing the embedding vector
        """
        try:
            # Ensure the client is initialized
            await self._ensure_client()
            
            if not self._client:
                logger.error("Failed to initialize HTTP client for Ollama API")
                raise RuntimeError("HTTP client initialization failed")
                
            # Prepare the request for embeddings
            request_data = {
                "model": self.model,
                "prompt": text,
            }
            
            # Make the request to Ollama embeddings API
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/embeddings",
                    json=request_data,
                    timeout=60.0
                )
            except httpx.TimeoutException:
                logger.error(f"Timeout while calling Ollama embeddings API")
                raise TimeoutError("API Timeout for embeddings request")
            except httpx.ConnectError:
                logger.error(f"Connection error to Ollama API at {self.base_url}")
                # Reset client so we'll try to reconnect next time
                self._client = None
                raise ConnectionError(f"Failed to connect to Ollama API at {self.base_url}")
                
            if response.status_code != 200:
                logger.error(f"Error from Ollama embeddings API: {response.status_code}, {response.text}")
                raise RuntimeError(f"API Error: {response.status_code}")
                
            # Parse the response
            result = response.json()
            
            # Extract the embedding
            embedding = result.get("embedding", [])
            
            if not embedding:
                logger.error("No embedding returned from Ollama API")
                raise ValueError("Empty embedding returned from API")
                
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating Ollama embedding: {str(e)}")
            raise RuntimeError(f"Failed to generate embedding: {str(e)}")

    async def get_repository_analysis(self, repository_data: Dict) -> Dict:
        """
        Deeply analyze a repository to understand its purpose and architecture.
        
        Args:
            repository_data: Dictionary with repository data including files and code
            
        Returns:
            Dictionary containing comprehensive repository analysis
        """
        try:
            # Create a comprehensive prompt for deep repository analysis
            system_prompt = """You are an expert software architect and code analyst with deep understanding of software systems. 
            Your task is to thoroughly analyze a repository and provide a comprehensive understanding of:
            1. The repository's core purpose and why it was built
            2. The complete architecture and how all components interact
            3. The data flow throughout the system
            4. Key technologies and their specific roles
            5. The system's main capabilities and features

            Analyze code patterns, naming conventions, dependency relationships, and architectural choices to provide
            a holistic understanding of the entire system. Think like the system architect who designed this codebase.
            Your response should be thorough, insightful, and explain the repository as a cohesive whole rather than 
            isolated components."""
            
            # Prepare repository information for analysis
            file_structure = repository_data.get("file_structure", "No file structure provided")
            code_samples = repository_data.get("code_samples", [])
            features = repository_data.get("features", {})
            
            # Format code samples for better analysis
            formatted_samples = ""
            for sample in code_samples[:15]:  # Limit to avoid exceeding context length
                file_path = sample.get("file_path", "Unknown")
                content = sample.get("content", "")
                if content:
                    formatted_samples += f"\n\nFILE: {file_path}\n```\n{content[:1000]}...\n```"
            
            # Create a comprehensive analysis prompt
            analysis_prompt = f"""
            Conduct a comprehensive analysis of this repository to explain its core purpose, architecture, and functionality.
            
            FILE STRUCTURE:
            {file_structure}
            
            SAMPLE CODE:
            {formatted_samples}
            
            FEATURE INFORMATION:
            {json.dumps(features, indent=2)}
            
            Based on this information, provide:
            1. A clear explanation of what this repository is designed to do and why it exists
            2. A comprehensive explanation of the system architecture and how all components work together
            3. The data flow throughout the system
            4. The key technologies used and why they were chosen
            5. The main features and capabilities of the system
            
            Your analysis should be thorough and provide insights about the repository as an integrated whole, not just isolated components.
            """
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt}
            ]
            
            # Generate a comprehensive analysis
            response = await self.generate(
                messages=messages,
                temperature=0.3,  # Lower temperature for more factual analysis
                max_tokens=4000   # Allow for a detailed response
            )
            
            if "error" in response:
                logger.error(f"Error in repository analysis: {response['error']}")
                return {"error": response["error"]}
            
            content = response["choices"][0]["message"]["content"]
            return {"analysis": content}
            
        except Exception as e:
            logger.error(f"Error in repository analysis: {str(e)}")
            return {"error": f"Repository analysis failed: {str(e)}"}
            
    def _parse_json_safely(self, text: str) -> Dict:
        """
        Parse JSON from text, handling various formats and cleaning issues.
        
        Args:
            text: Text containing JSON data
            
        Returns:
            Parsed JSON data or None if parsing fails
        """
        try:
            # First try direct parsing
            return json.loads(text)
        except json.JSONDecodeError:
            # Clean up the text and try again
            try:
                # Remove markdown formatting
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0].strip()
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0].strip()
                
                # Remove any text before the first { and after the last }
                start_idx = text.find('{')
                end_idx = text.rfind('}')
                
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    cleaned_text = text[start_idx:end_idx+1]
                    return json.loads(cleaned_text)
                
                # If that fails, try to be more aggressive with cleaning
                cleaned_text = ''.join(char for char in text if char.isprintable()).strip()
                logger.info("Failed to recover JSON after cleaning")
                return None
            except Exception as e:
                logger.error(f"Failed to recover JSON after cleaning: {str(e)}")
                return None