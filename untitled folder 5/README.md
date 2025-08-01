# GitHub Code Monitor

A powerful AI-enhanced system for monitoring GitHub repositories, understanding code changes, and providing intelligent insights through natural language queries.

## 🌟 Features

- **GitHub Webhook Integration**: Automatically capture all commits and code changes
- **AI-Powered Code Analysis**: Understand code semantics and generate summaries of changes
- **Natural Language Interface**: Chat with your codebase and get insights about changes
- **Semantic Code Search**: Find relevant code using natural language queries
- **Change Impact Analysis**: Understand the potential impact of code changes
- **PostgreSQL with pgvector**: Store and search through vector embeddings of code
- **Selective Commit Tracking**: Choose which repository commits get loaded into the database
- **Automatic Merged PR Loading**: Automatically load merged pull request commits from repositories

## 📋 System Architecture

The system consists of several key components:

1. **GitHub Integration & Data Ingestion**
   - GitHub webhooks to trigger on push events
   - Change processor to analyze code changes
   - PostgreSQL storage for repository data

2. **AI Knowledge Base & Understanding**
   - Code embeddings for semantic search
   - Automatic detection and labeling of code structures
   - Vector search capabilities

3. **Chat Interface with AI**
   - Natural language interface for querying the codebase
   - Context retrieval system to find relevant information
   - LLM-powered responses using OpenAI GPT models

## 🚀 Getting Started

### Prerequisites

- Docker and Docker Compose
- GitHub repository with admin access (to set up webhooks)
- OpenAI API key for AI capabilities

### Installation

1. **Clone this repository**:
   ```bash
   git clone https://github.com/yourusername/github-code-monitor.git
   cd github-code-monitor
   ```

2. **Set up environment variables**:
   ```bash
   cp sample.env .env
   ```
   
   Edit the `.env` file and add your GitHub and OpenAI API credentials.

3. **Launch the services with Docker Compose**:
   ```bash
   docker-compose up -d
   ```

4. **Set up GitHub webhook**:
   - Go to your GitHub repository → Settings → Webhooks → Add webhook
   - Set Payload URL to `http://your-server-address:8000/webhook/github`
   - Set Content type to `application/json`
   - Set Secret to match your `GITHUB_WEBHOOK_SECRET` in the `.env` file
   - Select events: Choose `Push` events (or select specific events you want to track)
   - Enable the webhook

5. **Access the UI**:
   - Open `http://localhost:8501` in your browser to access the chat interface

## 💬 Using the Chat Interface

You can ask questions about your codebase such as:

- "What was added in the last commit?"
- "Show me all commits affecting the login flow"
- "Explain what changed in the authentication module"
- "Who modified the user service in the past month?"
- "What functions were affected by the recent refactoring?"

## 🔄 Repository Management

You can control which repositories have their commits tracked and loaded into the database.

### Via CLI

The system includes a command-line interface for managing repositories:

```bash
# List all monitored repositories with their tracking settings
python code_monitor/cli/repo_manager_cli.py list

# Add a new repository to monitor (by default tracks both regular commits and merged PRs)
python code_monitor/cli/repo_manager_cli.py add https://github.com/owner/repo

# Add a repo but disable tracking for regular commits
python code_monitor/cli/repo_manager_cli.py add https://github.com/owner/repo --no-track-commits

# Update repository settings (enable/disable tracking)
python code_monitor/cli/repo_manager_cli.py update 1 --no-track-commits --track-merged-prs

# Load all merged PRs for a repository (up to the specified limit)
python code_monitor/cli/repo_manager_cli.py load-prs 1 --limit 20
```

### Via API

Repository management can also be done through API endpoints:

- `GET /repository-management/list`: List all repositories with tracking settings
- `POST /repository-management/add`: Add a new repository to monitor
- `PUT /repository-management/{repo_id}/settings`: Update repository tracking settings
- `POST /repository-management/{repo_id}/load-merged-prs`: Retroactively load merged PR commits

Example API usage:

```bash
# Add a new repository
curl -X POST "http://localhost:8000/repository-management/add?repo_url=https://github.com/owner/repo&track_commits=true&track_merged_prs=true"

# Update tracking settings
curl -X PUT "http://localhost:8000/repository-management/1/settings?track_commits=false&track_merged_prs=true"

# Load merged PRs for a repository
curl -X POST "http://localhost:8000/repository-management/1/load-merged-prs?limit=20"
```

## 🧩 API Endpoints

The system provides several REST API endpoints:

- `POST /webhook/github`: Handles GitHub webhook events
- `GET /repositories`: Lists all repositories
- `GET /repositories/{repo_id}`: Gets repository details
- `GET /repositories/{repo_id}/commits`: Lists commits for a repository
- `GET /commits/{commit_id}`: Gets commit details with associated files
- `POST /chat`: Processes natural language queries about the codebase
- `GET /search`: Performs semantic search over the codebase

## 🛠️ Development Setup

For local development without Docker:

1. **Set up a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up PostgreSQL with pgvector**:
   Follow the [pgvector installation guide](https://github.com/pgvector/pgvector) to set up locally.

4. **Create a .env file**:
   Copy `sample.env` to `.env` and update the values.

5. **Run the application**:
   ```bash
   uvicorn main:app --reload
   ```
   
   In another terminal, run the UI:
   ```bash
   streamlit run code_monitor/chat_interface/streamlit_app.py
   ```

## 📊 Database Schema

The system uses the following database models:

- **Repository**: Stores repository metadata
- **Commit**: Tracks individual commits
- **File**: Stores file changes with content before and after
- **CodeSnippet**: Tracks functions, classes, and methods
- **User**: For authentication (when enabled)

## 📝 Configuration Options

See `sample.env` for all available configuration options. Key settings include:

- Database connection parameters
- GitHub integration settings
- OpenAI API configuration
- Application host and port

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.