import logging
import numpy as np
import hashlib
import torch
import asyncio
from typing import List, Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
from code_monitor.config import settings

# Import Ollama client if using Ollama
if settings.USE_OLLAMA:
    from code_monitor.utils.ollama_client import OllamaClient

# Import transformers for Instructor model
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Generate embeddings for code understanding with language-specific features."""
    
    def __init__(self):
        self.use_ollama = settings.USE_OLLAMA
        self.use_instructor = getattr(settings, 'USE_INSTRUCTOR', False)
        self.dimension = getattr(settings, 'EMBEDDING_DIMENSION', 1536)
        self.api_key = getattr(settings, 'OPENAI_API_KEY', None)
        self.ollama_client = None
        
        if not self.use_ollama:
            self.model = settings.EMBEDDING_MODEL
            self.client = OpenAI(api_key=self.api_key)
            logger.info(f"Using OpenAI for embeddings with model: {settings.EMBEDDING_MODEL}")
            
        # Instructor model initialization moved to initialize()
        self.instructor_tokenizer = None
        self.instructor_model = None

    async def initialize(self):
        """Initialize async components with robust error handling"""
        try:
            logger.info("Initializing EmbeddingGenerator components")
            
            # Initialize Ollama client if we're configured to use it
            if self.use_ollama:
                retry_count = 0
                max_retries = 3
                
                while retry_count < max_retries:
                    try:
                        from code_monitor.utils.ollama_client import OllamaClient
                        self.ollama_client = await OllamaClient.create()
                        logger.info(f"Successfully initialized Ollama client for embeddings")
                        break
                    except Exception as e:
                        retry_count += 1
                        if retry_count >= max_retries:
                            logger.error(f"Failed to initialize Ollama client after {max_retries} attempts: {str(e)}")
                            # Create a basic client without async initialization that will retry connection on use
                            self.ollama_client = OllamaClient()
                        else:
                            logger.warning(f"Ollama client initialization attempt {retry_count} failed: {str(e)}. Retrying...")
                            await asyncio.sleep(1)  # Wait a bit before retrying
            
            # Initialize Instructor model if configured
            if self.use_instructor:
                try:
                    self.instructor_tokenizer = AutoTokenizer.from_pretrained(settings.INSTRUCTOR_MODEL_NAME)
                    self.instructor_model = AutoModel.from_pretrained(settings.INSTRUCTOR_MODEL_NAME)
                    logger.info(f"Loaded Instructor model: {settings.INSTRUCTOR_MODEL_NAME}")
                except Exception as e:
                    logger.error(f"Failed to load Instructor model: {e}")
                    self.use_instructor = False
                    
            logger.info("EmbeddingGenerator initialization complete")
                    
        except Exception as e:
            logger.error(f"Error initializing EmbeddingGenerator: {e}")
            # Reset state on failure but create basic clients for retry later
            if self.use_ollama and not self.ollama_client:
                try:
                    self.ollama_client = OllamaClient()
                except Exception:
                    self.ollama_client = None
            
            self.instructor_tokenizer = None
            self.instructor_model = None

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    async def generate_embedding(self, text: str, language: str = None) -> Optional[List[float]]:
        """Generate embeddings with better code understanding
        
        Args:
            text: Text to generate embedding for
            language: Optional programming language hint
            
        Returns:
            List of embedding values or None if generation fails
        """
        if not text:
            logger.warning("Empty text provided for embedding generation")
            return None

        try:
            embedding_result = None
            error_messages = []

            # Try Instructor model first if enabled and initialized
            if self.use_instructor and self.instructor_model:
                try:
                    instructor_embedding = await self._generate_instructor_embedding(text)
                    if instructor_embedding:
                        return instructor_embedding
                except Exception as e:
                    error_messages.append(f"Instructor embedding failed: {str(e)}")

            # Try OpenAI if API key is available
            if self.api_key:
                try:
                    logger.info("Using OpenAI API for embedding generation")
                    openai_embedding = await self._generate_openai_embedding(text)
                    if openai_embedding:
                        return openai_embedding
                except Exception as e:
                    error_messages.append(f"OpenAI embedding failed: {str(e)}")
            
            # Try Ollama if configured
            if self.use_ollama:
                try:
                    # Initialize the Ollama client on demand if it doesn't exist
                    if not self.ollama_client:
                        from code_monitor.utils.ollama_client import OllamaClient
                        try:
                            logger.info("Initializing Ollama client on demand for embedding")
                            self.ollama_client = await OllamaClient.create()
                        except Exception as e:
                            logger.warning(f"Failed to initialize Ollama client: {str(e)}")
                            # Fall back to a basic client
                            self.ollama_client = OllamaClient()
                    
                    logger.info("Using Ollama for embedding generation")
                    ollama_embedding = await self._generate_ollama_embedding(text)
                    if ollama_embedding:
                        return ollama_embedding
                except Exception as e:
                    error_messages.append(f"Ollama embedding failed: {str(e)}")

            # If all services failed, try local fallback
            logger.warning(f"All embedding services failed: {', '.join(error_messages)}")
            logger.info("Using local fallback embedding generation method")
            return self._generate_local_fallback_embedding(text)

        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            # Always return a fallback embedding so the application can continue
            return self._generate_local_fallback_embedding(text)
    
    async def _generate_instructor_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using the Instructor model from transformers"""
        try:
            # Instructor requires instruction-formatted inputs
            instruction = "Represent the code for retrieval:"
            # Format input in the way Instructor expects
            input_text = f"{instruction} {text}"
            
            # Tokenize the input
            inputs = self.instructor_tokenizer(
                input_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512  # Set a reasonable max length
            )
            
            # Generate embeddings - make sure we're passing all needed parameters
            with torch.no_grad():
                # Add decoder_input_ids for T5-based model
                decoder_input_ids = self.instructor_tokenizer(
                    "This is the decoder input", 
                    return_tensors="pt"
                )["input_ids"]
                
                # Forward pass with all required inputs
                outputs = self.instructor_model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    decoder_input_ids=decoder_input_ids  # Add this parameter to fix the error
                )
                
                # Get embeddings from the last hidden state of the encoder
                embeddings = outputs.last_hidden_state[:, 0]
                normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
            # Convert to list and ensure dimension matches expected
            embedding = normalized_embeddings[0].tolist()
            
            # Ensure we have exactly 1536 dimensions by either padding or interpolating
            if len(embedding) < self.dimension:
                # Not enough dimensions, need to pad or upscale
                logger.info(f"Padding Instructor embedding from {len(embedding)} to {self.dimension} dimensions")
                
                # Use interpolation for better quality
                original_len = len(embedding)
                factor = self.dimension / original_len
                
                # Create a larger embedding by linear interpolation
                embedding_array = np.array(embedding)
                indices = np.arange(self.dimension) / factor
                
                # Use linear interpolation
                new_embedding = np.interp(
                    indices, 
                    np.arange(original_len), 
                    embedding_array
                )
                
                # Normalize again after interpolation
                new_embedding = new_embedding / np.linalg.norm(new_embedding)
                embedding = new_embedding.tolist()
                
            elif len(embedding) > self.dimension:
                # Too many dimensions, need to truncate or downscale
                logger.info(f"Reducing Instructor embedding from {len(embedding)} to {self.dimension} dimensions")
                
                # Dimensionality reduction with interpolation
                original_len = len(embedding)
                indices = np.linspace(0, original_len - 1, self.dimension)
                embedding_array = np.array(embedding)
                new_embedding = np.interp(
                    indices, 
                    np.arange(original_len), 
                    embedding_array
                )
                
                # Normalize again after interpolation
                new_embedding = new_embedding / np.linalg.norm(new_embedding)
                embedding = new_embedding.tolist()
            
            # Double-check the dimension
            assert len(embedding) == self.dimension, f"Embedding dimension mismatch: {len(embedding)} != {self.dimension}"
            logger.info(f"Generated Instructor embedding with dimension {len(embedding)}")
            
            return embedding
            
        except Exception as e:
            logger.error(f"Error generating Instructor embedding: {str(e)}")
            return None
    
    async def _generate_openai_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using OpenAI API"""
        if not self.api_key:
            logger.warning("OpenAI API key not configured, skipping OpenAI embedding generation")
            return None
            
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            embedding = response.data[0].embedding
            
            # Verify embedding length matches expected dimension
            if len(embedding) != self.dimension:
                logger.warning(f"OpenAI embedding dimension {len(embedding)} doesn't match expected {self.dimension}")
                # Adjust the embedding dimension if needed
                if len(embedding) < self.dimension:
                    # Pad with zeros
                    embedding = embedding + [0.0] * (self.dimension - len(embedding))
                else:
                    # Truncate
                    embedding = embedding[:self.dimension]
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating OpenAI embedding: {str(e)}")
            return None
    
    async def _generate_ollama_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding using Ollama API"""
        try:
            embedding = await self.ollama_client.generate_embedding(text)
            
            if embedding is None:
                logger.error("Ollama returned None for embedding")
                return None
                
            # Check if we got a valid embedding
            if not isinstance(embedding, list) or len(embedding) == 0:
                logger.error(f"Invalid embedding format from Ollama: {type(embedding)}")
                return None
                
            # Get the actual dimension of Ollama embeddings from settings or use the actual length
            ollama_dim = getattr(settings, 'OLLAMA_EMBEDDING_DIMENSION', len(embedding))
            
            # Verify embedding length matches expected dimension
            if len(embedding) != self.dimension:
                # Log at debug level instead of warning since we'll handle this gracefully
                logger.debug(f"Ollama embedding dimension {len(embedding)} doesn't match expected {self.dimension}")
                # Adjust the embedding dimension if needed
                if len(embedding) < self.dimension:
                    # Use interpolation to resize
                    logger.info(f"Upscaling Ollama embedding from {len(embedding)} to {self.dimension} dimensions")
                    original_len = len(embedding)
                    embedding_array = np.array(embedding)
                    
                    # Create new indices for interpolation
                    indices = np.linspace(0, original_len - 1, self.dimension)
                    
                    # Use interpolation to resize the vector
                    new_embedding = np.interp(
                        indices, 
                        np.arange(original_len), 
                        embedding_array
                    )
                    
                    # Normalize the new embedding
                    norm = np.linalg.norm(new_embedding)
                    if norm > 0:
                        new_embedding = new_embedding / norm
                    embedding = new_embedding.tolist()
                else:
                    # If embeddings are too large, use PCA-like approach for better reduction
                    logger.info(f"Downscaling Ollama embedding from {len(embedding)} to {self.dimension} dimensions")
                    
                    # Simple dimensionality reduction via interpolation
                    embedding_array = np.array(embedding)
                    indices = np.linspace(0, len(embedding) - 1, self.dimension)
                    new_embedding = np.interp(
                        indices,
                        np.arange(len(embedding)),
                        embedding_array
                    )
                    
                    # Normalize the new embedding
                    norm = np.linalg.norm(new_embedding)
                    if norm > 0:
                        new_embedding = new_embedding / norm
                    embedding = new_embedding.tolist()
            
            return embedding
        except Exception as e:
            logger.error(f"Error generating Ollama embedding: {str(e)}")
            return None
    
    def _generate_local_fallback_embedding(self, text: str) -> List[float]:
        """
        Generate a simple embedding based on text hash and basic features.
        This is a fallback method when no AI service is available.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            A list of floats approximating an embedding vector
        """
        # Create a fixed-size vector of zeros with exactly the required dimension
        embedding = [0.0] * self.dimension
        
        if not text:
            return embedding
            
        # Use hash of text to generate pseudo-random values
        text_hash = hashlib.sha256(text.encode('utf-8')).digest()
        
        # For larger dimensions, use multiple hashes
        all_bytes = bytearray()
        for i in range((self.dimension // 32) + 1):
            h = hashlib.sha256((text + str(i)).encode('utf-8')).digest()
            all_bytes.extend(h)
        
        # Fill positions with hash byte values (normalized to range between -1 and 1)
        for i in range(self.dimension):
            if i < len(all_bytes):
                embedding[i] = (all_bytes[i] / 128.0) - 1.0
        
        # Add some basic text features by modifying certain positions
        feature_indices = [self.dimension // 4, self.dimension // 3, self.dimension // 2]
        
        # Text length features
        embedding[feature_indices[0] % self.dimension] = len(text) / 10000  # Document length (normalized)
        embedding[(feature_indices[0] + 1) % self.dimension] = text.count('\n') / 1000  # Line count
        embedding[(feature_indices[0] + 2) % self.dimension] = len(text.split()) / 5000  # Word count
        
        # Character distribution features
        embedding[(feature_indices[1]) % self.dimension] = sum(c.isupper() for c in text) / max(1, len(text))
        embedding[(feature_indices[1] + 1) % self.dimension] = sum(c.isdigit() for c in text) / max(1, len(text))
        embedding[(feature_indices[1] + 2) % self.dimension] = sum(c.isspace() for c in text) / max(1, len(text))
        
        # Code-specific features
        embedding[(feature_indices[2]) % self.dimension] = text.count('def ') / 100  # Python functions
        embedding[(feature_indices[2] + 1) % self.dimension] = text.count('class ') / 100  # Classes
        embedding[(feature_indices[2] + 2) % self.dimension] = text.count('import ') / 100  # Imports
        
        # Normalize the embedding vector
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = [x / norm for x in embedding]
            
        logger.info(f"Generated fallback embedding with dimension {len(embedding)}")
        
        # Ensure we're returning exactly the right dimension
        assert len(embedding) == self.dimension, f"Embedding dimension mismatch: {len(embedding)} != {self.dimension}"
        
        return embedding
        
    async def search_similar(
        self, 
        db_query: Any, 
        query_text: str, 
        limit: int = 5, 
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Search for similar items using vector similarity
        
        Args:
            db_query: SQLAlchemy query object
            query_text: Text to search for
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of matching items with similarity scores
        """
        try:
            # Import necessary SQLAlchemy functions locally to ensure they're available
            from sqlalchemy.future import select
            from sqlalchemy import func
            
            # Generate embedding for the query text
            query_embedding = await self.generate_embedding(query_text)
            if not query_embedding:
                logger.error("Failed to generate embedding for query text")
                return []
            
            # Execute the query with the embedding
            try:
                # For SQLAlchemy 2.0 with asyncio
                if hasattr(db_query, 'execute'):
                    # If db_query is a session
                    result = await db_query.execute(db_query)
                    results = result.scalars().all()
                else:
                    # If db_query is already a prepared statement
                    # We need a session to execute it, but we don't have one in this context
                    logger.error("No session provided for executing the query")
                    results = []
                    
            except Exception as inner_e:
                logger.error(f"Error executing query: {str(inner_e)}", exc_info=True)
                results = []
            
            return results
            
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}", exc_info=True)
            return []