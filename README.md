# GitHub Code Monitor

An AI-enhanced system for monitoring GitHub repositories, understanding code changes, and providing intelligent insights through natural language queries.

## Features

- **GitHub Webhook Integration**: Automatically captures commits and code changes from GitHub repositories
- **AI-Powered Code Analysis**: Uses AI to understand code semantics and generate summaries of changes
- **Natural Language Chat Interface**: Ask questions about your codebase in plain English
- **Semantic Code Search**: Find relevant code using natural language queries
- **Change Impact Analysis**: Analyze potential impacts of code changes
- **Vector-Based Code Search**: Uses pgvector in PostgreSQL to store and search vector embeddings of code

## Architecture

GitHub Code Monitor consists of several key components:

1. **GitHub Integration & Data Ingestion**
   - GitHub webhooks to trigger on repository events
   - Change processor to analyze code changes
   - PostgreSQL database to store repository data

2. **AI Knowledge Base & Understanding**
   - Code embeddings for semantic search
   - Automatic detection of code structures
   - Vector search capabilities

3. **Chat Interface with AI**
   - Natural language interface for querying the codebase
   - Context retrieval system to find relevant information
   - LLM-powered responses (supports OpenAI GPT models or Ollama for local LLM support)

## Getting Started

### Prerequisites

- Docker and Docker Compose
- GitHub account with repositories you want to monitor
- OpenAI API key (optional, can use Ollama for local inference)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/github-code-monitor.git
   cd github-code-monitor
   ```

2. Set up environment variables by creating a `.env` file based on `sample.env`:
   ```bash
   cp sample.env .env
   ```
   Then edit the `.env` file with your specific configuration.

3. Start the services:
   ```bash
   ./start.sh
   ```

   This script will:
   - Create a virtual environment if it doesn't exist
   - Install dependencies
   - Start the FastAPI backend and Streamlit frontend

4. Access the UI at http://localhost:8501

### Setting up GitHub Webhooks

1. Go to your GitHub repository settings
2. Select "Webhooks" and click "Add webhook"
3. Set the Payload URL to `http://your-server-address:8000/webhook/github`
4. Select "application/json" as the content type
5. Choose which events to trigger the webhook (recommend: Pushes and Pull Requests)
6. Save the webhook

## Using the Chat Interface

Once the application is running, you can access the chat interface at http://localhost:8501. Here, you can ask questions about your codebase in natural language, such as:

- "What does this repository do?"
- "Show me the most recent changes to authentication.py"
- "Explain how the database connection works"
- "Find all functions related to user authentication"
- "What are the main features of this codebase?"

## Configuration Options

Key configuration options in `.env` include:

- `DATABASE_URL`: PostgreSQL connection string
- `GITHUB_TOKEN`: GitHub personal access token for API access
- `OPENAI_API_KEY`: Your OpenAI API key (if using OpenAI)
- `USE_OLLAMA`: Set to "true" to use Ollama instead of OpenAI
- `OLLAMA_BASE_URL`: URL for your Ollama instance (default: http://localhost:11434)
- `OLLAMA_MODEL`: The model to use with Ollama (e.g., "llama2")

## Repository Management

You can manage which repositories are monitored using either:

- CLI tools in `code_monitor/cli/repo_manager_cli.py`
- API endpoints at `/repository-management/...`

## Development

### Project Structure

- `code_monitor/api/`: REST API endpoints
- `code_monitor/ai_processing/`: Code analysis, summarization, and embedding generation
- `code_monitor/chat_interface/`: Natural language interaction with the codebase
- `code_monitor/db/`: Database models and connection management
- `code_monitor/github_integration/`: Webhook handling and GitHub API interaction
- `code_monitor/cli/`: Command-line utilities for repository management

### Running Tests

```bash
# Run the test suite
python -m pytest
```

## Docker Deployment

For production deployment, use Docker Compose:

```bash
docker-compose up -d
```

This will start:
- PostgreSQL database with pgvector extension
- FastAPI backend service
- Streamlit frontend service

## License

[MIT License](LICENSE)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
